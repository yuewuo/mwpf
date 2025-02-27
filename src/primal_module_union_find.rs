//! Serial Primal Module like Union-Find decoder
//!
//! This implementation is an approximate MWPF solver, which is essentially what hypergraph UF decoder does.
//! Delfosse, Nicolas, Vivien Londe, and Michael E. Beverland. "Toward a Union-Find decoder for quantum LDPC codes." IEEE Transactions on Information Theory 68.5 (2022): 3187-3199.
//!
//! there might be some minor difference with Delfosse's paper, but the idea is the same
//!

use crate::derivative::Derivative;
use crate::dual_module::*;
use crate::invalid_subgraph::*;
use crate::num_traits::Zero;
use crate::pointers::*;
use crate::primal_module::*;
use crate::union_find::*;
use crate::util::*;
use crate::visualize::*;

use std::sync::Arc;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct PrimalModuleUnionFind {
    /// union find data structure
    union_find: UnionFind,
}

type UnionFind = UnionFindGeneric<PrimalModuleUnionFindNode>;

/// define your own union-find node data structure like this
#[derive(Debug, Clone)]
pub struct PrimalModuleUnionFindNode {
    /// all the internal edges
    pub internal_edges: FastIterSet<EdgeIndex>,
    /// the corresponding node index with these internal edges
    pub node_index: NodeIndex,
}

/// example trait implementation
impl UnionNodeTrait for PrimalModuleUnionFindNode {
    #[inline]
    fn union(left: &Self, right: &Self) -> (bool, Self) {
        let mut internal_edges = FastIterSet::new();
        internal_edges.extend(left.internal_edges.iter().cloned());
        internal_edges.extend(right.internal_edges.iter().cloned());
        let result = Self {
            internal_edges,
            node_index: NodeIndex::MAX, // waiting for assignment
        };
        // if left size is larger, choose left (weighted union)
        (true, result)
    }
    #[inline]
    fn clear(&mut self) {
        panic!("clear a node is meaningless here, call `remove_all` instead");
    }
    #[inline]
    fn default() -> Self {
        Self {
            internal_edges: FastIterSet::new(),
            node_index: NodeIndex::MAX, // waiting for assignment
        }
    }
}

impl PrimalModuleImpl for PrimalModuleUnionFind {
    fn new_empty(_initializer: &Arc<SolverInitializer>) -> Self {
        Self {
            union_find: UnionFind::new(0),
        }
    }

    fn clear(&mut self) {
        self.union_find.remove_all();
    }

    #[allow(clippy::unnecessary_cast)]
    fn load<D: DualModuleImpl>(&mut self, interface_ptr: &DualModuleInterfacePtr, _dual_module: &mut D) {
        let interface = interface_ptr.read_recursive();
        for index in 0..interface.nodes.len() as NodeIndex {
            let node_ptr = &interface.nodes[index as usize];
            let node = node_ptr.read_recursive();
            debug_assert!(
                node.invalid_subgraph.edges.is_empty(),
                "must load a fresh dual module interface, found a complex node; did you forget to call solver.clear()?"
            );
            debug_assert!(
                node.invalid_subgraph.vertices.len() == 1,
                "must load a fresh dual module interface, found invalid defect node; did you forget to call solver.clear()?"
            );
            debug_assert_eq!(
                node.index, index,
                "must load a fresh dual module interface, found index out of order; did you forget to call solver.clear()?"
            );
            assert_eq!(
                node.index as usize,
                self.union_find.size(),
                "must load defect nodes in order, did you forget to call solver.clear()?"
            );
            self.union_find.insert(PrimalModuleUnionFindNode {
                internal_edges: FastIterSet::new(),
                node_index: node.index,
            });
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn resolve(
        &mut self,
        mut dual_report: DualReport,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        debug_assert!(!dual_report.is_unbounded() && dual_report.get_valid_growth().is_none());
        let mut active_clusters = FastIterSet::<NodeIndex>::new();
        while let Some(obstacle) = dual_report.pop() {
            match obstacle {
                Obstacle::Conflict { edge_index } => {
                    // union all the dual nodes in the edge index and create new dual node by adding this edge to `internal_edges`
                    let dual_nodes = dual_module.get_edge_nodes(edge_index);
                    debug_assert!(
                        !dual_nodes.is_empty(),
                        "should not conflict if no dual nodes are contributing"
                    );
                    let cluster_index = dual_nodes[0].read_recursive().index;
                    for dual_node_ptr in dual_nodes.iter() {
                        dual_module.set_grow_rate(dual_node_ptr, Rational::zero());
                        let node_index = dual_node_ptr.read_recursive().index;
                        active_clusters.remove(&(self.union_find.find(node_index as usize) as NodeIndex));
                        self.union_find.union(cluster_index as usize, node_index as usize);
                    }
                    self.union_find
                        .get_mut(cluster_index as usize)
                        .internal_edges
                        .insert(edge_index);
                    active_clusters.insert(self.union_find.find(cluster_index as usize) as NodeIndex);
                }
                _ => {
                    unreachable!()
                }
            }
        }
        for &cluster_index in active_clusters.iter() {
            if interface_ptr
                .read_recursive()
                .decoding_graph
                .is_valid_cluster_auto_vertices(&self.union_find.get(cluster_index as usize).internal_edges)
            {
                // do nothing
            } else {
                let new_cluster_node_index = self.union_find.size() as NodeIndex;
                self.union_find.insert(PrimalModuleUnionFindNode {
                    internal_edges: FastIterSet::new(),
                    node_index: new_cluster_node_index,
                });
                self.union_find.union(cluster_index as usize, new_cluster_node_index as usize);
                let invalid_subgraph = InvalidSubgraph::new_ptr(
                    self.union_find.get(cluster_index as usize).internal_edges.clone(),
                    &interface_ptr.read_recursive().decoding_graph,
                );
                interface_ptr.create_node(invalid_subgraph, dual_module);
            }
        }
        false
    }

    fn subgraph(
        &mut self,
        interface_ptr: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> OutputSubgraph {
        let mut valid_clusters = FastIterSet::new();
        let mut subgraph = vec![];
        for i in 0..self.union_find.size() {
            let root_index = self.union_find.find(i);
            if !valid_clusters.contains(&root_index) {
                valid_clusters.insert(root_index);
                let cluster_subgraph = interface_ptr
                    .read_recursive()
                    .decoding_graph
                    .find_valid_subgraph_auto_vertices(&self.union_find.get(root_index).internal_edges)
                    .expect("must be valid cluster");
                subgraph.extend(cluster_subgraph.iter());
            }
        }

        // let mut subgraph_set = subgraph.into_iter().collect::<hashbrown::HashSet<EdgeIndex>>();
        // for to_flip in _dual_module.get_negative_edges().iter() {
        //     if subgraph_set.contains(to_flip) {
        //         subgraph_set.remove(to_flip);
        //     } else {
        //         subgraph_set.insert(*to_flip);
        //     }
        // }
        // OutputSubgraph::new(subgraph_set.into_iter().collect(), Default::default())

        // note: note implmented to handle negative weights yet
        OutputSubgraph::new(subgraph, _dual_module.get_negative_edges())
    }
}

/*
Implementing visualization functions
*/

impl MWPSVisualizer for PrimalModuleUnionFind {
    fn snapshot(&self, _abbrev: bool) -> serde_json::Value {
        json!({})
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dual_module_pq::*;
    use crate::example_codes::*;
    use crate::num_traits::ToPrimitive;
    use std::sync::Arc;

    pub fn primal_module_union_find_basic_standard_syndrome_optional_viz(
        mut code: impl ExampleCode,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        mut dual_module: impl DualModuleImpl + MWPSVisualizer,
        model_graph: Arc<crate::model_hypergraph::ModelHyperGraph>,
        mut visualizer: Option<Visualizer>,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleUnionFind,
        impl DualModuleImpl + MWPSVisualizer,
    ) {
        // create primal module
        let mut primal_module = PrimalModuleUnionFind::new_empty(&model_graph.initializer);
        // try to work on a simple syndrome
        code.set_defect_vertices(&defect_vertices);
        let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());
        primal_module.solve_visualizer(
            &interface_ptr,
            Arc::new(code.get_syndrome()),
            &mut dual_module,
            visualizer.as_mut(),
        );
        let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
        if let Some(visualizer) = visualizer.as_mut() {
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &subgraph, &weight_range],
                )
                .unwrap();
        }

        let v = visualizer.as_mut().unwrap();
        v.save_html_along_json();
        println!("open visualizer at {}", v.html_along_json_path());

        assert!(
            model_graph.initializer.matches_subgraph_syndrome(&subgraph, &defect_vertices),
            "the result subgraph is invalid"
        );
        println!("final_dual: {final_dual:?}");
        println!("upper: {:?}", weight_range.upper);
        assert!(
            rational_approx_le(&final_dual, &weight_range.upper),
            "unmatched sum dual variables"
        );
        assert!(
            rational_approx_ge(&final_dual, &weight_range.lower),
            "unexpected final dual variable sum"
        );
        println!(
            "weight range: [{}, {}]",
            weight_range.lower.to_i64().unwrap(),
            weight_range.upper.to_i64().unwrap()
        );
        (interface_ptr, primal_module, dual_module)
    }

    pub fn primal_module_union_find_basic_standard_syndrome(
        code: impl ExampleCode,
        visualize_filename: String,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleUnionFind,
        impl DualModuleImpl + MWPSVisualizer,
    ) {
        println!("{defect_vertices:?}");
        let visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();

        // create dual module
        let model_graph = code.get_model_graph();

        primal_module_union_find_basic_standard_syndrome_optional_viz(
            code,
            defect_vertices,
            final_dual,
            DualModulePQ::new_empty(&model_graph.initializer),
            model_graph,
            Some(visualizer),
        )
    }

    /// test a simple case
    #[test]
    fn primal_module_union_find_basic_1() {
        // cargo test primal_module_union_find_basic_1 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_1.json".to_string();
        let defect_vertices = vec![23, 24, 29, 30];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(4.59511985013459).unwrap(),
        );
    }

    #[test]
    fn primal_module_union_find_basic_2() {
        // cargo test primal_module_union_find_basic_2 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_2.json".to_string();
        let defect_vertices = vec![16, 17, 23, 25, 29, 30];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(9.19023970026918).unwrap(),
        );
    }

    #[test]
    fn primal_module_union_find_basic_3() {
        // cargo test primal_module_union_find_basic_3 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_3.json".to_string();
        let defect_vertices = vec![14, 15, 16, 17, 22, 25, 28, 31, 36, 37, 38, 39];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(22.975599250672953).unwrap(),
        );
    }

    #[test]
    fn primal_module_union_find_basic_4() {
        // cargo test primal_module_union_find_basic_4 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_4.json".to_string();
        let defect_vertices = vec![3, 12];
        let code = CodeCapacityColorCode::new(7, 0.01);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(9.19023970026918).unwrap(),
        );
    }

    #[test]
    fn primal_module_union_find_basic_5() {
        // cargo test primal_module_union_find_basic_5 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_5.json".to_string();
        let defect_vertices = vec![3, 5, 10, 12];
        let code = CodeCapacityColorCode::new(7, 0.01);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(12.25365293369224).unwrap(),
        );
    }

    #[test]
    fn primal_module_union_find_basic_6() {
        // cargo test primal_module_union_find_basic_6 -- --nocapture
        let visualize_filename = "primal_module_union_find_basic_6.json".to_string();
        let defect_vertices = vec![22];
        let code = CodeCapacityTailoredCode::new(5, 0., 0.05);
        primal_module_union_find_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(11.777755916665761).unwrap(),
        );
    }
}
