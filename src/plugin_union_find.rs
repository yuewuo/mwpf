//! if the tight edges doesn't constitute a feasible solution, then report a relaxer,
//! just like in the hypergraph UF decoder paper: https://arxiv.org/abs/2103.08049
//!
//! This plugin will always be appended to the end of the plugin sequence to make
//! sure there is a feasible MINLP solution.
//!

use crate::decoding_hypergraph::*;
use crate::dual_module::*;
use crate::invalid_subgraph::*;
use crate::matrix::*;
use crate::num_traits::One;
use crate::plugin::*;
use crate::relaxer::*;
use crate::util::*;

#[derive(Debug, Clone, Default)]
pub struct PluginUnionFind {}

impl PluginUnionFind {
    /// check if the cluster is valid (hypergraph union-find decoder)
    pub fn find_single_relaxer(decoding_graph: &DecodingHyperGraph, matrix: &mut EchelonMatrix) -> Option<Relaxer> {
        if matrix.get_echelon_info().satisfiable {
            return None; // cannot find any relaxer
        }
        let invalid_subgraph = InvalidSubgraph::new_complete_ptr(
            matrix.get_vertices(),
            FastIterSet::from_iter(matrix.get_view_edges()),
            decoding_graph,
        );
        Some(Relaxer::new([(invalid_subgraph, Rational::one())].into()))
    }
}

impl PluginImpl for PluginUnionFind {
    fn find_relaxers(
        &self,
        decoding_graph: &DecodingHyperGraph,
        matrix: &mut EchelonMatrix,
        _positive_dual_nodes: &[DualNodePtr],
    ) -> Vec<Relaxer> {
        if let Some(relaxer) = Self::find_single_relaxer(decoding_graph, matrix) {
            vec![relaxer]
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::example_codes::*;
    use crate::primal_module_serial::tests::*;

    #[test]
    fn plugin_union_find_basic_1() {
        // cargo test plugin_union_find_basic_1 -- --nocapture
        let visualize_filename = "plugin_union_find_basic_1.json".to_string();
        let defect_vertices = vec![10, 11, 16, 17];
        let code = CodeCapacityTailoredCode::new(5, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(4.59511985013459).unwrap(),
            vec![PluginUnionFind::entry()],
        );
    }
}
