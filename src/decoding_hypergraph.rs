// TODO: should adopt full pointer implementation

use crate::dual_module_pq::EdgePtr;
use crate::dual_module_pq::EdgeWeak;
use crate::dual_module_pq::VertexPtr;
use crate::dual_module_pq::VertexWeak;
use crate::matrix::*;
use crate::model_hypergraph::*;
use crate::util::*;
use crate::visualize::*;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DecodingHyperGraph {
    /// model graph
    pub model_graph: Arc<ModelHyperGraph>,
    /// syndrome
    pub syndrome_pattern: Arc<SyndromePattern>,
    /// fast check whether a vertex is defect
    /// TODO: probably should use pointers as well
    pub defect_vertices_hashset: HashSet<VertexIndex>,
    /// fast check whether an edge is erased
    pub erasures_hashset: HashSet<EdgeIndex>,
}

impl DecodingHyperGraph {
    pub fn new(model_graph: Arc<ModelHyperGraph>, syndrome_pattern: Arc<SyndromePattern>) -> Self {
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

    pub fn new_defects(model_graph: Arc<ModelHyperGraph>, defect_vertices: Vec<VertexIndex>) -> Self {
        Self::new(model_graph, Arc::new(SyndromePattern::new_vertices(defect_vertices)))
    }

    pub fn find_valid_subgraph(&self, edges: &FastIterSet<EdgePtr>, vertices: &FastIterSet<VertexWeak>) -> Option<Subgraph> {
        let mut matrix = Echelon::<CompleteMatrix>::new();
        for edge_ptr in edges.iter() {
            matrix.add_variable(edge_ptr.downgrade().clone());
        }

        for vertex_weak in vertices.iter() {
            let vertex_ptr = vertex_weak.upgrade_force();
            let vertex_read = vertex_ptr.read_recursive();
            let vertex_index = vertex_read.vertex_index;
            let incident_edges = &vertex_read.edges;
            let parity = self.is_vertex_defect(vertex_index);
            matrix.add_constraint(vertex_weak.clone(), incident_edges, parity);
        }
        matrix.get_solution().map(|solution| {
            solution
                .into_iter()
                .map(|edge_weak| edge_weak.upgrade_force().read_recursive().edge_index)
                .collect()
        })
    }

    pub fn find_valid_subgraph_auto_vertices(&self, edges: &FastIterSet<EdgePtr>) -> Option<Subgraph> {
        self.find_valid_subgraph(edges, &self.get_edges_neighbors(edges))
    }

    pub fn is_valid_cluster(&self, edges: &FastIterSet<EdgePtr>, vertices: &FastIterSet<VertexWeak>) -> bool {
        self.find_valid_subgraph(edges, vertices).is_some()
    }

    pub fn is_valid_cluster_auto_vertices(&self, edges: &FastIterSet<EdgePtr>) -> bool {
        self.find_valid_subgraph_auto_vertices(edges).is_some()
    }

    pub fn is_vertex_defect(&self, vertex_index: VertexIndex) -> bool {
        self.defect_vertices_hashset.contains(&vertex_index)
    }

    pub fn get_edge_neighbors(&self, edge_ptr: EdgePtr) -> &Vec<VertexIndex> {
        self.model_graph.get_edge_neighbors(edge_ptr.read_recursive().edge_index)
    }

    pub fn get_vertex_neighbors(&self, vertex_index: VertexIndex) -> &Vec<EdgeIndex> {
        self.model_graph.get_vertex_neighbors(vertex_index)
    }

    pub fn get_edges_neighbors(&self, edges: &FastIterSet<EdgePtr>) -> FastIterSet<VertexWeak> {
        edges
            .iter()
            .flat_map(|edge_ptr| edge_ptr.read_recursive().vertices.clone())
            .collect()
    }
}

impl MWPSVisualizer for DecodingHyperGraph {
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
    use crate::model_hypergraph::tests::*;

    pub fn color_code_5_decoding_graph(
        defect_vertices: Vec<VertexIndex>,
        visualize_filename: String,
    ) -> (Arc<DecodingHyperGraph>, Visualizer) {
        let (model_graph, mut visualizer) = color_code_5_model_graph(visualize_filename);
        let syndrome_pattern = Arc::new(SyndromePattern::new_vertices(defect_vertices));
        let decoding_graph = Arc::new(DecodingHyperGraph::new(model_graph, syndrome_pattern));
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
