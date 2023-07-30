use crate::hyper_model_graph::*;
use crate::matrix::*;
use crate::util::*;
use crate::visualize::*;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct HyperDecodingGraph {
    /// model graph
    pub model_graph: Arc<HyperModelGraph>,
    /// syndrome
    pub syndrome_pattern: Arc<SyndromePattern>,
    /// fast check whether a vertex is defect
    pub defect_vertices_hashset: HashSet<VertexIndex>,
    /// fast check whether an edge is erased
    pub erasures_hashset: HashSet<EdgeIndex>,
}

impl HyperDecodingGraph {
    pub fn new(model_graph: Arc<HyperModelGraph>, syndrome_pattern: Arc<SyndromePattern>) -> Self {
        let mut decoding_graph = Self {
            model_graph,
            syndrome_pattern: syndrome_pattern.clone(),
            defect_vertices_hashset: HashSet::new(),
            erasures_hashset: HashSet::new(),
        };
        decoding_graph.set_syndrome(syndrome_pattern);
        decoding_graph
    }

    pub fn set_syndrome(&mut self, syndrome_pattern: Arc<SyndromePattern>) {
        self.defect_vertices_hashset.clear();
        self.erasures_hashset.clear();
        // reserve space for the hashset
        if self.defect_vertices_hashset.capacity() < syndrome_pattern.defect_vertices.len() {
            self.defect_vertices_hashset
                .reserve(syndrome_pattern.defect_vertices.len() - self.defect_vertices_hashset.capacity())
        }
        if self.erasures_hashset.capacity() < syndrome_pattern.erasures.len() {
            self.erasures_hashset
                .reserve(syndrome_pattern.erasures.len() - self.erasures_hashset.capacity())
        }
        // add new syndrome
        for &vertex_index in syndrome_pattern.defect_vertices.iter() {
            self.defect_vertices_hashset.insert(vertex_index);
        }
        for &edge_index in syndrome_pattern.erasures.iter() {
            self.erasures_hashset.insert(edge_index);
        }
    }

    pub fn new_defects(model_graph: Arc<HyperModelGraph>, defect_vertices: Vec<VertexIndex>) -> Self {
        Self::new(model_graph, Arc::new(SyndromePattern::new_vertices(defect_vertices)))
    }

    pub fn find_valid_subgraph(&self, edges: &BTreeSet<EdgeIndex>, vertices: &BTreeSet<VertexIndex>) -> Option<Subgraph> {
        let mut matrix = Echelon::<CompleteMatrix>::new();
        for &edge_index in edges.iter() {
            matrix.add_variable(edge_index);
        }

        for &vertex_index in vertices.iter() {
            let incident_edges = self.get_vertex_neighbors(vertex_index);
            let parity = self.is_vertex_defect(vertex_index);
            matrix.add_constraint(vertex_index, incident_edges, parity);
        }
        matrix.get_solution()
    }

    pub fn find_valid_subgraph_auto_vertices(&self, edges: &BTreeSet<EdgeIndex>) -> Option<Subgraph> {
        self.find_valid_subgraph(edges, &self.get_edges_neighbors(edges))
    }

    pub fn is_valid_cluster(&self, edges: &BTreeSet<EdgeIndex>, vertices: &BTreeSet<VertexIndex>) -> bool {
        self.find_valid_subgraph(edges, vertices).is_some()
    }

    pub fn is_valid_cluster_auto_vertices(&self, edges: &BTreeSet<EdgeIndex>) -> bool {
        self.find_valid_subgraph_auto_vertices(edges).is_some()
    }

    pub fn is_vertex_defect(&self, vertex_index: VertexIndex) -> bool {
        self.defect_vertices_hashset.contains(&vertex_index)
    }

    pub fn get_edge_neighbors(&self, edge_index: EdgeIndex) -> &Vec<VertexIndex> {
        self.model_graph.get_edge_neighbors(edge_index)
    }

    pub fn get_vertex_neighbors(&self, vertex_index: VertexIndex) -> &Vec<EdgeIndex> {
        self.model_graph.get_vertex_neighbors(vertex_index)
    }

    pub fn get_edges_neighbors(&self, edges: &BTreeSet<EdgeIndex>) -> BTreeSet<VertexIndex> {
        self.model_graph.get_edges_neighbors(edges)
    }
}

impl MWPSVisualizer for HyperDecodingGraph {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let mut value = self.model_graph.initializer.snapshot(abbrev);
        let mut vertices = Vec::<serde_json::Value>::new();
        for vertex_index in 0..self.model_graph.initializer.vertex_num {
            vertices.push(json!({
                if abbrev { "s" } else { "is_defect" }: i32::from(self.defect_vertices_hashset.contains(&vertex_index)),
            }));
        }
        snapshot_combine_values(&mut value, json!({ "vertices": vertices }), abbrev);
        value
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::hyper_model_graph::tests::*;

    pub fn color_code_5_decoding_graph(
        defect_vertices: Vec<VertexIndex>,
        visualize_filename: String,
    ) -> (Arc<HyperDecodingGraph>, Visualizer) {
        let (model_graph, mut visualizer) = color_code_5_model_graph(visualize_filename);
        let syndrome_pattern = Arc::new(SyndromePattern::new_vertices(defect_vertices));
        let decoding_graph = Arc::new(HyperDecodingGraph::new(model_graph, syndrome_pattern));
        visualizer
            .snapshot_combined("syndrome".to_string(), vec![decoding_graph.as_ref()])
            .unwrap();
        (decoding_graph, visualizer)
    }

    #[test]
    fn hyper_decoding_graph_basic_1() {
        // cargo test hyper_decoding_graph_basic_1 -- --nocapture
        let visualize_filename = "hyper_decoding_graph_basic_1.json".to_string();
        let defect_vertices = vec![7, 1];
        let (decoding_graph, ..) = color_code_5_decoding_graph(defect_vertices, visualize_filename);
        println!("decoding_graph: {decoding_graph:?}");
    }
}
