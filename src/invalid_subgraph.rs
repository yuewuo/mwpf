use crate::decoding_hypergraph::*;
use crate::derivative::Derivative;
use crate::matrix::*;
use crate::plugin::EchelonMatrix;
use crate::util::*;
use std::cmp::Ordering;
// use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// an invalid subgraph $S = (V_S, E_S)$, also store the hair $\delta(S)$
#[derive(Clone, PartialEq, Eq, Derivative, Default)]
#[derivative(Debug)]
pub struct InvalidSubgraph {
    /// the hash value calculated by other fields
    #[derivative(Debug = "ignore")]
    pub hash_value: u64,
    /// subset of vertices
    pub vertices: FastIterSet<VertexIndex>,
    /// subset of edges
    pub edges: FastIterSet<EdgeIndex>,
    /// the hair of the invalid subgraph, to avoid repeated computation
    pub hair: FastIterSet<EdgeIndex>,
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
    pub fn new(edges: FastIterSet<EdgeIndex>, decoding_graph: &DecodingHyperGraph) -> Self {
        let mut vertices = FastIterSet::new();
        for &edge_index in edges.iter() {
            let hyperedge = &decoding_graph.model_graph.initializer.weighted_edges[edge_index as usize];
            for &vertex_index in hyperedge.vertices.iter() {
                vertices.insert(vertex_index);
            }
        }
        Self::new_complete(vertices, edges, decoding_graph)
    }

    /// complete definition of invalid subgraph $S = (V_S, E_S)$
    #[allow(clippy::unnecessary_cast)]
    pub fn new_complete(
        vertices: FastIterSet<VertexIndex>,
        edges: FastIterSet<EdgeIndex>,
        decoding_graph: &DecodingHyperGraph,
    ) -> Self {
        let mut hair = FastIterSet::new();
        for &vertex_index in vertices.iter() {
            let vertex = &decoding_graph.model_graph.vertices[vertex_index as usize];
            for &edge_index in vertex.edges.iter() {
                if !edges.contains(&edge_index) {
                    hair.insert(edge_index);
                }
            }
        }
        let invalid_subgraph = Self::new_raw(vertices, edges, hair);
        debug_assert_eq!(invalid_subgraph.sanity_check(decoding_graph), Ok(()));
        invalid_subgraph
    }

    /// create $S = (V_S, E_S)$ and $\delta(S)$ directly, without any checks
    pub fn new_raw(vertices: FastIterSet<VertexIndex>, edges: FastIterSet<EdgeIndex>, hair: FastIterSet<EdgeIndex>) -> Self {
        let mut invalid_subgraph = Self {
            hash_value: 0,
            vertices,
            edges,
            hair,
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

    // check whether this invalid subgraph is indeed invalid, this is costly and should be disabled in release runs
    #[allow(clippy::unnecessary_cast)]
    pub fn sanity_check(&self, decoding_graph: &DecodingHyperGraph) -> Result<(), String> {
        if self.vertices.is_empty() {
            return Err("an invalid subgraph must contain at least one vertex".to_string());
        }
        // check if all vertices are valid
        for &vertex_index in self.vertices.iter() {
            if vertex_index >= decoding_graph.model_graph.initializer.vertex_num {
                return Err(format!("vertex {vertex_index} is not a vertex in the model graph"));
            }
        }
        // check if every edge is subset of its vertices
        for &edge_index in self.edges.iter() {
            if edge_index as usize >= decoding_graph.model_graph.initializer.weighted_edges.len() {
                return Err(format!("edge {edge_index} is not an edge in the model graph"));
            }
            let hyperedge = &decoding_graph.model_graph.initializer.weighted_edges[edge_index as usize];
            for &vertex_index in hyperedge.vertices.iter() {
                if !self.vertices.contains(&vertex_index) {
                    return Err(format!(
                        "hyperedge {edge_index} connects vertices {:?}, \
                    but vertex {vertex_index} is not in the invalid subgraph vertices {:?}",
                        hyperedge.vertices, self.vertices
                    ));
                }
            }
        }
        // check the edges indeed cannot satisfy the requirement of the vertices
        let mut matrix = Echelon::<CompleteMatrix>::new();
        for &edge_index in self.edges.iter() {
            matrix.add_variable(edge_index);
        }
        for &vertex_index in self.vertices.iter() {
            let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
            let parity = decoding_graph.is_vertex_defect(vertex_index);
            matrix.add_constraint(vertex_index, incident_edges, parity);
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

    pub fn generate_matrix(&self, decoding_graph: &DecodingHyperGraph) -> EchelonMatrix {
        let mut matrix = EchelonMatrix::new();
        for &edge_index in self.hair.iter() {
            matrix.add_variable(edge_index);
        }
        for &vertex_index in self.vertices.iter() {
            let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
            let parity = decoding_graph.is_vertex_defect(vertex_index);
            matrix.add_constraint(vertex_index, incident_edges, parity);
        }
        matrix
    }
}

// shortcuts for easier code writing at debugging
impl InvalidSubgraph {
    pub fn new_ptr(edges: FastIterSet<EdgeIndex>, decoding_graph: &DecodingHyperGraph) -> Arc<Self> {
        Arc::new(Self::new(edges, decoding_graph))
    }
    pub fn new_vec_ptr(edges: &[EdgeIndex], decoding_graph: &DecodingHyperGraph) -> Arc<Self> {
        Self::new_ptr(edges.iter().cloned().collect(), decoding_graph)
    }
    pub fn new_complete_ptr(
        vertices: FastIterSet<VertexIndex>,
        edges: FastIterSet<EdgeIndex>,
        decoding_graph: &DecodingHyperGraph,
    ) -> Arc<Self> {
        Arc::new(Self::new_complete(vertices, edges, decoding_graph))
    }
    pub fn new_complete_vec_ptr(
        vertices: FastIterSet<VertexIndex>,
        edges: &[EdgeIndex],
        decoding_graph: &DecodingHyperGraph,
    ) -> Arc<Self> {
        Self::new_complete_ptr(
            vertices.iter().cloned().collect(),
            edges.iter().cloned().collect(),
            decoding_graph,
        )
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::decoding_hypergraph::tests::*;

    #[test]
    fn invalid_subgraph_good() {
        // cargo test invalid_subgraph_good -- --nocapture
        let visualize_filename = "invalid_subgraph_good.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let invalid_subgraph_1 = InvalidSubgraph::new(vec![13].into_iter().collect(), decoding_graph.as_ref());
        println!("invalid_subgraph_1: {invalid_subgraph_1:?}");
        assert_eq!(sorted_vec(invalid_subgraph_1.vertices.into_iter().collect()), vec![2, 6, 7]);
        assert_eq!(sorted_vec(invalid_subgraph_1.edges.into_iter().collect()), vec![13]);
        assert_eq!(
            sorted_vec(invalid_subgraph_1.hair.into_iter().collect()),
            vec![5, 6, 9, 10, 11, 12, 14, 15, 16, 17]
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn invalid_subgraph_bad() {
        // cargo test invalid_subgraph_bad -- --nocapture
        let visualize_filename = "invalid_subgraph_bad.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let invalid_subgraph = InvalidSubgraph::new(vec![6, 10].into_iter().collect(), decoding_graph.as_ref());
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
        let vertices: FastIterSet<VertexIndex> = [1, 2, 3].into();
        let edges: FastIterSet<EdgeIndex> = [4, 5].into();
        let hair: FastIterSet<EdgeIndex> = [6, 7, 8].into();
        let invalid_subgraph_1 = InvalidSubgraph::new_raw(vertices.clone(), edges.clone(), hair.clone());
        let invalid_subgraph_2 = InvalidSubgraph::new_raw(vertices.clone(), edges.clone(), hair.clone());
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
            InvalidSubgraph::new_raw([1, 2].into(), edges.clone(), hair.clone())
        );
        assert_ne!(
            invalid_subgraph_1,
            InvalidSubgraph::new_raw(vertices.clone(), [4, 5, 6].into(), hair.clone())
        );
        assert_ne!(
            invalid_subgraph_1,
            InvalidSubgraph::new_raw(vertices.clone(), edges.clone(), [6, 7].into())
        );
    }
}
