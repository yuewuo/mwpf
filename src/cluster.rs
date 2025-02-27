use crate::dual_module::*;
use crate::matrix::*;
use crate::util::*;

use derivative::Derivative;

#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Cluster {
    /// vertices of the cluster
    pub vertices: FastIterSet<VertexIndex>,
    /// tight edges of the cluster
    pub edges: FastIterSet<EdgeIndex>,
    /// edges incident to the vertices but are not tight
    pub hair: FastIterSet<EdgeIndex>,
    /// dual variables of the cluster
    pub nodes: FastIterSet<OrderedDualNodePtr>,
    /// parity matrix of the cluster
    #[derivative(Debug = "ignore")]
    pub parity_matrix: Tight<BasicMatrix>,
}

impl Default for Cluster {
    fn default() -> Self {
        Self::new()
    }
}

impl Cluster {
    /// Create a new cluster
    pub fn new() -> Self {
        Cluster {
            vertices: FastIterSet::new(),
            edges: FastIterSet::new(),
            hair: FastIterSet::new(),
            nodes: FastIterSet::new(),
            parity_matrix: Tight::<BasicMatrix>::new(),
        }
    }

    /// Add a vertex to the cluster
    pub fn add_vertex(&mut self, vertex: VertexIndex) {
        self.vertices.insert(vertex);
    }

    /// Add an edge to the cluster
    pub fn add_edge(&mut self, edge: EdgeIndex) {
        self.edges.insert(edge);
    }

    /// Add a hair to the cluster
    pub fn add_hair(&mut self, hair: EdgeIndex) {
        self.hair.insert(hair);
    }

    /// Add a dual variable to the cluster
    pub fn add_node(&mut self, node: OrderedDualNodePtr) {
        self.nodes.insert(node);
    }

    /// set the parity matrix of the cluster
    pub fn set_parity_matrix(&mut self, parity_matrix: Tight<BasicMatrix>) {
        self.parity_matrix = parity_matrix;
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::example_codes::*;
    use crate::mwpf_solver::*;
    use crate::visualize::*;
    use num_traits::One;
    use std::sync::Arc;

    fn cluster_test_common(
        code: &impl ExampleCode,
        syndrome: SyndromePattern,
        visualize_filename: &str,
        expected_vertices: FastIterSet<VertexIndex>,
        expected_edges: FastIterSet<EdgeIndex>,
        expected_hair: FastIterSet<EdgeIndex>,
    ) -> Cluster {
        let visualizer_path = visualize_data_folder() + visualize_filename;
        let mut visualizer = Visualizer::new(Some(visualizer_path.clone()), code.get_positions(), true).unwrap();
        let mut initializer = code.get_initializer();
        initializer.uniform_weights(Rational::one());
        let mut solver = SolverSerialJointSingleHair::new(&Arc::new(initializer), json!({}));
        solver.solve_visualizer(syndrome, Some(&mut visualizer));
        if cfg!(feature = "embed_visualizer") {
            let html = visualizer.generate_html(json!({}));
            assert!(visualizer_path.ends_with(".json"));
            let html_path = format!("{}.html", &visualizer_path.as_str()[..visualizer_path.len() - 5]);
            std::fs::write(&html_path, html).expect("Unable to write file");
            println!("visualizer path: {}", &html_path);
        }
        // generate the cluster
        let cluster = solver.get_cluster(2);
        println!("cluster: {cluster:?}");
        assert_eq!(cluster.vertices, expected_vertices);
        assert_eq!(cluster.edges, expected_edges);
        assert_eq!(cluster.hair, expected_hair);
        cluster
    }

    #[test]
    fn cluster_example_1() {
        // cargo test cluster_example_1 -- --nocapture
        let name = "cluster_example_1.json";
        let code = CodeCapacityColorCode::new(5, 0.005);
        let syndrome = SyndromePattern::new_vertices(vec![2, 3, 7]);
        let mut cluster = cluster_test_common(
            &code,
            syndrome,
            name,
            fast_iter_set! { 2, 7, 3 },
            fast_iter_set! { 10 },
            fast_iter_set! { 9, 5, 6, 2, 3, 7, 11, 16, 15, 13, 12, 14 },
        );
        let node_indices = cluster.nodes.iter().map(|d| d.index).collect::<Vec<_>>();
        assert_eq!(sorted_vec(node_indices), vec![0, 1, 2]);
        cluster.parity_matrix.printstd();
        assert_eq!(
            cluster.parity_matrix.clone().printstd_str(),
            "\
┌─┬─┬───┐
┊ ┊1┊ = ┊
┊ ┊0┊   ┊
╞═╪═╪═══╡
┊0┊1┊ 1 ┊
├─┼─┼───┤
┊1┊1┊ 1 ┊
├─┼─┼───┤
┊2┊1┊ 1 ┊
└─┴─┴───┘
"
        );
        cluster.parity_matrix.get_base().clone().printstd();
        //         assert_eq!(
        //             cluster.parity_matrix.get_base().clone().printstd_str(),
        //             "\
        // ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┐
        // ┊ ┊1┊5┊6┊9┊1┊1┊2┊3┊7┊1┊1┊1┊1┊ = ┊
        // ┊ ┊0┊ ┊ ┊ ┊2┊3┊ ┊ ┊ ┊1┊4┊5┊6┊   ┊
        // ╞═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═══╡
        // ┊0┊1┊1┊1┊1┊1┊1┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ 1 ┊
        // ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼───┤
        // ┊1┊1┊ ┊1┊ ┊ ┊ ┊1┊1┊1┊1┊ ┊ ┊ ┊ 1 ┊
        // ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼───┤
        // ┊2┊1┊ ┊ ┊ ┊ ┊1┊ ┊ ┊ ┊1┊1┊1┊1┊ 1 ┊
        // └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┘
        // "
        //         );
    }
}
