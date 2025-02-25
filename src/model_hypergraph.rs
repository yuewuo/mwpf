use crate::util::*;
use crate::visualize::*;

use std::sync::Arc;

/// hyper model graph that contains static information regardless of the syndrome
#[derive(Debug, Clone)]
pub struct ModelHyperGraph {
    /// initializer
    pub initializer: Arc<SolverInitializer>,
    /// the data structure for each vertex
    pub vertices: Vec<ModelHyperGraphVertex>,
}

#[derive(Default, Debug, Clone)]
pub struct ModelHyperGraphVertex {
    /// the incident edges
    pub edges: Vec<EdgeIndex>,
}

impl ModelHyperGraph {
    #[allow(clippy::unnecessary_cast)]
    pub fn new(initializer: Arc<SolverInitializer>) -> Self {
        let mut vertices: Vec<ModelHyperGraphVertex> =
            vec![ModelHyperGraphVertex::default(); initializer.vertex_num as usize];
        for (edge_index, hyperedge) in initializer.weighted_edges.iter().enumerate() {
            for &vertex_index in hyperedge.vertices.iter() {
                vertices[vertex_index as usize].edges.push(edge_index as EdgeIndex);
            }
        }
        Self { initializer, vertices }
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_edge_neighbors(&self, edge_index: EdgeIndex) -> &Vec<VertexIndex> {
        &self.initializer.weighted_edges[edge_index as usize].vertices
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_vertex_neighbors(&self, vertex_index: VertexIndex) -> &Vec<EdgeIndex> {
        &self.vertices[vertex_index as usize].edges
    }

    pub fn get_edges_neighbors(&self, edges: &FastIterSet<EdgeIndex>) -> FastIterSet<VertexIndex> {
        let mut vertices = FastIterSet::new();
        for &edge_index in edges.iter() {
            vertices.extend(self.get_edge_neighbors(edge_index));
        }
        vertices
    }

    pub fn matches_subgraph_syndrome(&self, subgraph: &OutputSubgraph, defect_vertices: &[VertexIndex]) -> bool {
        self.initializer.matches_subgraph_syndrome(subgraph, defect_vertices)
    }
}

impl MWPSVisualizer for ModelHyperGraph {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        self.initializer.snapshot(abbrev)
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::example_codes::*;
    use super::*;

    pub fn color_code_5_model_graph(visualize_filename: String) -> (Arc<ModelHyperGraph>, Visualizer) {
        let code = CodeCapacityColorCode::new(5, 0.1);
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        visualizer.snapshot_combined("code".to_string(), vec![&code]).unwrap();
        let model_graph = code.get_model_graph();
        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());
        (model_graph, visualizer)
    }

    #[test]
    fn hyper_model_graph_basic_1() {
        // cargo test hyper_model_graph_basic_1 -- --nocapture
        let visualize_filename = "hyper_model_graph_basic_1.json".to_string();
        let (model_graph, ..) = color_code_5_model_graph(visualize_filename);
        println!("model_graph: {model_graph:?}");
        let mut edge_reference_initializer = 0;
        let mut edge_reference_hyper_model_graph = 0;
        for hyperedge in model_graph.initializer.weighted_edges.iter() {
            edge_reference_initializer += hyperedge.vertices.len();
        }
        for vertex in model_graph.vertices.iter() {
            edge_reference_hyper_model_graph += vertex.edges.len();
        }
        assert_eq!(edge_reference_initializer, edge_reference_hyper_model_graph);
    }
}
