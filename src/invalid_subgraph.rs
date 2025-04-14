use crate::derivative::Derivative;
use crate::matrix::*;
use crate::plugin::EchelonMatrix;
use crate::util::*;
use crate::{decoding_hypergraph::*, invalid_subgraph};
use std::cmp::Ordering;
// use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::dual_module::DualModuleImpl;
use crate::dual_module_pq::{EdgePtr, VertexPtr};
#[cfg(feature = "unsafe_pointer")]
use crate::pointers::UnsafePtr;

/// an invalid subgraph $S = (V_S, E_S)$, also store the hair $\delta(S)$
#[derive(Clone, PartialEq, Eq, Derivative, Default)]
#[derivative(Debug)]
pub struct InvalidSubgraph {
    /// the hash value calculated by other fields
    #[derivative(Debug = "ignore")]
    pub hash_value: u64,
    /// subset of vertices
    pub vertices: FastIterSet<VertexPtr>,
    /// subset of edges
    pub edges: FastIterSet<EdgePtr>,
    /// the hair of the invalid subgraph, to avoid repeated computation
    pub hair: FastIterSet<EdgePtr>,
}

impl Hash for InvalidSubgraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_value.hash(state);
    }
}

impl Ord for InvalidSubgraph {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.hash_value != other.hash_value {
            self.hash_value.cmp(&other.hash_value)
        } else if self == other {
            Ordering::Equal
        } else {
            // rare cases: same hash value but different state
            (&self.vertices, &self.edges, &self.hair).cmp(&(&other.vertices, &other.edges, &other.hair))
        }
    }
}

impl PartialOrd for InvalidSubgraph {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl InvalidSubgraph {
    /// construct an invalid subgraph using only $E_S$, and constructing the $V_S$ by $\cup E_S$
    #[allow(clippy::unnecessary_cast)]
    pub fn new(edges: &FastIterSet<EdgePtr>) -> Self {
        let mut vertices: BTreeSet<VertexPtr> = FastIterSet::new();
        for edge_ptr in edges.iter() {
            for vertex_ptr in edge_ptr.read_recursive().vertices.iter() {
                vertices.insert(vertex_ptr.upgrade_force().clone());
            }
        }
        Self::new_complete(&vertices, edges)
    }

    /// complete definition of invalid subgraph $S = (V_S, E_S)$
    #[allow(clippy::unnecessary_cast)]
    pub fn new_complete(vertices: &FastIterSet<VertexPtr>, edges: &FastIterSet<EdgePtr>) -> Self {
        let mut hair = FastIterSet::new();
        for vertex_ptr in vertices.iter() {
            for edge_weak in vertex_ptr.read_recursive().edges.iter() {
                let edge_ptr = edge_weak.upgrade_force();
                if !edges.contains(&edge_ptr) {
                    hair.insert(edge_ptr);
                }
            }
        }
        let invalid_subgraph = Self::new_raw(vertices, edges, &hair);
        // debug_assert_eq!(invalid_subgraph.sanity_check(decoding_graph), Ok(()));
        invalid_subgraph
    }

    /// create $S = (V_S, E_S)$ and $\delta(S)$ directly, without any checks
    pub fn new_raw(vertices: &FastIterSet<VertexPtr>, edges: &FastIterSet<EdgePtr>, hair: &FastIterSet<EdgePtr>) -> Self {
        let mut invalid_subgraph = Self {
            hash_value: 0,
            vertices: vertices.clone(),
            edges: edges.clone(),
            hair: hair.clone(),
        };
        invalid_subgraph.update_hash();
        invalid_subgraph
    }

    pub fn update_hash(&mut self) {
        let mut hasher = DefaultHasher::default();
        self.vertices.hash(&mut hasher);
        self.edges.hash(&mut hasher);
        self.hair.hash(&mut hasher);
        self.hash_value = hasher.finish();
    }

    // note that only new_from_indices and new_complete_from_indices have sanity_check
    pub fn new_from_indices(edges: BTreeSet<EdgeIndex>, dual_module: &mut impl DualModuleImpl) -> Self {
        let edges_ptr = edges.iter().map(|i| dual_module.get_edge_ptr(*i)).collect::<BTreeSet<_>>();
        let invalid_subgraph = Self::new(&edges_ptr);
        debug_assert_eq!(invalid_subgraph.sanity_check(dual_module), Ok(()));
        invalid_subgraph
    }

    // note that only new_from_indices and new_complete_from_indices have sanity_check
    pub fn new_complete_from_indices(
        vertices: BTreeSet<VertexIndex>,
        edges: BTreeSet<EdgeIndex>,
        dual_module: &mut impl DualModuleImpl,
    ) -> Self {
        let vertices_ptr = vertices
            .iter()
            .map(|i| dual_module.get_vertex_ptr(*i))
            .collect::<BTreeSet<_>>();
        let edges_ptr = edges.iter().map(|i| dual_module.get_edge_ptr(*i)).collect::<BTreeSet<_>>();
        let invalid_subgraph = Self::new_complete(&vertices_ptr, &edges_ptr);
        debug_assert_eq!(invalid_subgraph.sanity_check(dual_module), Ok(()));
        invalid_subgraph
    }

    pub fn new_raw_from_indices(
        vertices: BTreeSet<VertexIndex>,
        edges: BTreeSet<EdgeIndex>,
        hair: BTreeSet<EdgeIndex>,
        dual_module: &mut impl DualModuleImpl,
    ) -> Self {
        let vertices_ptr = vertices
            .iter()
            .map(|i| dual_module.get_vertex_ptr(*i))
            .collect::<BTreeSet<_>>();
        let edges_ptr = edges.iter().map(|i| dual_module.get_edge_ptr(*i)).collect::<BTreeSet<_>>();
        let hair_ptr = hair.iter().map(|i| dual_module.get_edge_ptr(*i)).collect::<BTreeSet<_>>();
        Self::new_raw(&vertices_ptr, &edges_ptr, &hair_ptr)
    }

    // check whether this invalid subgraph is indeed invalid, this is costly and should be disabled in release runs
    #[allow(clippy::unnecessary_cast)]
    pub fn sanity_check(&self, dual_module: &mut impl DualModuleImpl) -> Result<(), String> {
        if self.vertices.is_empty() {
            return Err("an invalid subgraph must contain at least one vertex".to_string());
        }
        // check if all vertices are valid
        for vertex_ptr in self.vertices.iter() {
            let vertex_index = vertex_ptr.read_recursive().vertex_index;
            if vertex_index >= dual_module.get_vertex_num() {
                return Err(format!("vertex {vertex_index} is not a vertex in the model graph"));
            }
        }
        // check if every edge is subset of its vertices
        for edge_ptr in self.edges.iter() {
            let edge = edge_ptr.read_recursive();
            let edge_index = edge.edge_index;
            if edge_index as usize >= dual_module.get_edge_num() {
                return Err(format!("edge {edge_index} is not an edge in the model graph"));
            }

            for vertex_weak in edge.vertices.iter() {
                if !self.vertices.contains(&vertex_weak.upgrade_force()) {
                    return Err(format!(
                        "hyperedge {edge_index} connects vertices {:?}, \
                    but vertex {:?} is not in the invalid subgraph vertices {:?}",
                        edge.vertices,
                        vertex_weak.upgrade_force().read_recursive().vertex_index,
                        self.vertices
                    ));
                }
            }
        }
        // check the edges indeed cannot satisfy the requirement of the vertices
        let mut matrix = Echelon::<CompleteMatrix>::new();
        for edge_ptr in self.edges.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }
        for vertex_ptr in self.vertices.iter() {
            let vertex_weak = vertex_ptr.downgrade();
            let vertex = vertex_ptr.read_recursive();
            let incident_edges = &vertex.edges;
            let parity = vertex.is_defect;
            matrix.add_constraint(vertex_weak, incident_edges, parity);
        }
        if matrix.get_echelon_info().satisfiable {
            return Err(format!(
                "it's a valid subgraph because edges {:?} ⊆ {:?} can satisfy the parity requirement from vertices {:?}",
                matrix.get_solution().unwrap(),
                self.edges,
                self.vertices
            ));
        }
        Ok(())
    }

    pub fn generate_matrix(&self) -> EchelonMatrix {
        let mut matrix = EchelonMatrix::new();
        for edge_ptr in self.hair.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }
        for vertex_ptr in self.vertices.iter() {
            let vertex_weak = vertex_ptr.downgrade();
            let vertex = vertex_ptr.read_recursive();
            let incident_edges = &vertex.edges;
            let parity = vertex.is_defect;
            matrix.add_constraint(vertex_weak, incident_edges, parity);
        }
        matrix
    }
}

// shortcuts for easier code writing at debugging
impl InvalidSubgraph {
    pub fn new_ptr(edges: &FastIterSet<EdgePtr>) -> Arc<Self> {
        Arc::new(Self::new(edges))
    }
    pub fn new_vec_ptr(edges: &[EdgePtr]) -> Arc<Self> {
        let strong_edges: BTreeSet<EdgePtr> = edges.iter().cloned().collect::<BTreeSet<_>>();
        Self::new_ptr(&strong_edges)
    }
    pub fn new_complete_ptr(vertices: &FastIterSet<VertexPtr>, edges: &FastIterSet<EdgePtr>) -> Arc<Self> {
        Arc::new(Self::new_complete(vertices, edges))
    }
    pub fn new_complete_vec_ptr(vertices: &FastIterSet<VertexPtr>, edges: &[EdgePtr]) -> Arc<Self> {
        let strong_edges: FastIterSet<EdgePtr> = edges.iter().cloned().collect::<FastIterSet<_>>();
        Self::new_complete_ptr(vertices, &strong_edges)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::decoding_hypergraph::tests::*;
    use crate::dual_module::DualModuleInterfacePtr;
    use crate::dual_module_pq::DualModulePQ;
    use crate::dual_module_pq::{Edge, EdgePtr, Vertex, VertexPtr};
    use std::collections::HashSet;
    use sugar::*;

    #[test]
    fn invalid_subgraph_good() {
        // cargo test invalid_subgraph_good -- --nocapture
        let visualize_filename = "invalid_subgraph_good.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern.clone(), &mut dual_module, 0); // this is needed to load the defect vertices

        let invalid_subgraph_1 = InvalidSubgraph::new_from_indices(btreeset! {13}, &mut dual_module);
        println!("invalid_subgraph_1: {invalid_subgraph_1:?}");
        assert_eq!(
            invalid_subgraph_1
                .vertices
                .iter()
                .map(|v| v.read_recursive().vertex_index)
                .collect::<HashSet<_>>(),
            vec![2, 6, 7].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            invalid_subgraph_1
                .edges
                .iter()
                .map(|e| e.read_recursive().edge_index)
                .collect::<HashSet<_>>(),
            vec![13].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            invalid_subgraph_1
                .hair
                .iter()
                .map(|e| e.read_recursive().edge_index)
                .collect::<HashSet<_>>(),
            vec![5, 6, 9, 10, 11, 12, 14, 15, 16, 17].into_iter().collect::<HashSet<_>>()
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn invalid_subgraph_bad() {
        // cargo test invalid_subgraph_bad -- --nocapture
        let visualize_filename = "invalid_subgraph_bad.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer);
        let invalid_subgraph = InvalidSubgraph::new_from_indices(btreeset! {6, 10}, &mut dual_module);
        println!("invalid_subgraph: {invalid_subgraph:?}"); // should not print because it panics
    }

    pub fn get_default_hash_value(object: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::default();
        object.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn invalid_subgraph_hash() {
        // cargo test invalid_subgraph_hash -- --nocapture
        let visualize_filename = "invalid_subgraph_good.json".to_string();
        // we use an arbitrary decoding graph, the defect vertices here are not loaded
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer);

        let vertices: FastIterSet<VertexIndex> = [1, 2, 3].into();
        let edges: FastIterSet<EdgeIndex> = [4, 5].into();
        let hair: FastIterSet<EdgeIndex> = [6, 7, 8].into();

        let invalid_subgraph_1 =
            InvalidSubgraph::new_raw_from_indices(vertices.clone(), edges.clone(), hair.clone(), &mut dual_module);
        let invalid_subgraph_2 =
            InvalidSubgraph::new_raw_from_indices(vertices.clone(), edges.clone(), hair.clone(), &mut dual_module);
        assert_eq!(invalid_subgraph_1, invalid_subgraph_2);
        // they should have the same hash value
        assert_eq!(
            get_default_hash_value(&invalid_subgraph_1),
            get_default_hash_value(&invalid_subgraph_1.hash_value)
        );
        assert_eq!(
            get_default_hash_value(&invalid_subgraph_1),
            get_default_hash_value(&invalid_subgraph_2)
        );
        // the pointer should also have the same hash value
        let ptr_1 = Arc::new(invalid_subgraph_1.clone());
        let ptr_2 = Arc::new(invalid_subgraph_2);
        assert_eq!(get_default_hash_value(&ptr_1), get_default_hash_value(&ptr_1.hash_value));
        assert_eq!(get_default_hash_value(&ptr_1), get_default_hash_value(&ptr_2));
        // any different value would generate a different invalid subgraph
        assert_ne!(
            invalid_subgraph_1,
            InvalidSubgraph::new_raw_from_indices(btreeset! {1, 2}, edges.clone(), hair.clone(), &mut dual_module)
        );
        assert_ne!(
            invalid_subgraph_1,
            InvalidSubgraph::new_raw_from_indices(vertices.clone(), btreeset! {4, 5, 6}, hair.clone(), &mut dual_module)
        );
        assert_ne!(
            invalid_subgraph_1,
            InvalidSubgraph::new_raw_from_indices(vertices.clone(), edges.clone(), btreeset! {6, 7}, &mut dual_module)
        );
    }
}
