// cargo run --release --bin paper_figures

use mwpf::dual_module::*;
use mwpf::dual_module_pq::*;
use mwpf::example_codes::*;
use mwpf::invalid_subgraph::*;
use mwpf::model_hypergraph::*;
use mwpf::util::*;
use mwpf::visualize::*;
use num_traits::FromPrimitive;

use std::sync::Arc;

fn hyperedge_example() {
    let visualize_filename = "paper_hyperedge_example.json".to_string();
    // create the example code, but we'll not use the default vertices and edges; rather, we'll construct them manually
    let mut code = CodeCapacityRepetitionCode::new(3, 0.001);
    // manually construct 7 vertices connected to this edge
    code.vertices.clear();
    code.edges = vec![CodeEdge::new(vec![0, 1, 2, 3, 4, 5, 6])];
    for edge in code.edges.iter_mut() {
        edge.weight = Rational::from_float(1.).unwrap();
    }
    code.fill_vertices(7);
    let radius = 2.;
    for i in 0..7 {
        let angle = std::f64::consts::PI * 2.0 / 7.0 * i as f64;
        code.vertices[i].position = VisualizePosition::new(-radius * angle.cos(), radius * angle.sin(), 0.);
    }
    // create dual module
    let initializer = Arc::new(code.get_initializer());
    let model_graph = Arc::new(ModelHyperGraph::new(initializer.clone()));
    let mut dual_module = DualModulePQ::new_empty(&initializer);
    let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());
    // add syndrome
    let syndrome_pattern = Arc::new(SyndromePattern::new_vertices(vec![1, 2, 4, 6]));
    interface_ptr.write().decoding_graph.set_syndrome(syndrome_pattern.clone());
    for vertex_index in syndrome_pattern.defect_vertices.iter() {
        dual_module.vertices[*vertex_index].write().is_defect = true;
    }
    // manually grow the dual variables
    let decoding_graph = interface_ptr.read_recursive().decoding_graph.clone();
    let dual_variables: Vec<(FastIterSet<VertexIndex>, f64)> = vec![
        (fast_iter_set! {5,6}, 0.1),
        (fast_iter_set! {4,5}, 0.1),
        (fast_iter_set! {2}, 0.1),
        (fast_iter_set! {1}, 0.5),
    ];
    for (vertices, dual_variable) in dual_variables.into_iter() {
        let s1 = Arc::new(InvalidSubgraph::new_complete(vertices, fast_iter_set! {}, &decoding_graph));
        let (_, s1_ptr) = interface_ptr.find_or_create_node(&s1, &mut dual_module);
        dual_module.set_grow_rate(&s1_ptr, Rational::from_f64(dual_variable).unwrap());
    }
    dual_module.grow(Rational::from_f64(1.).unwrap());
    // visualize
    let mut visualizer = Visualizer::new(
        Some(visualize_data_folder() + visualize_filename.as_str()),
        code.get_positions(),
        true,
    )
    .unwrap();
    visualizer
        .snapshot_combined("init".to_string(), vec![&interface_ptr, &dual_module])
        .unwrap();
}

fn main() {
    hyperedge_example();
}
