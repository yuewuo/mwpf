// cargo run --release --bin aps2024_demo

use mwpf::dual_module::*;
use mwpf::dual_module_serial::*;
use mwpf::example_codes::*;
use mwpf::invalid_subgraph::InvalidSubgraph;
use mwpf::model_hypergraph::*;
use mwpf::plugin::*;
use mwpf::plugin_single_hair::*;
use mwpf::plugin_union_find::*;
use mwpf::primal_module::*;
use mwpf::primal_module_serial::*;
use mwpf::util::*;
use mwpf::visualize::*;
use num_traits::cast::FromPrimitive;
use std::sync::Arc;
use sugar::*;

fn debug_demo() {
    for is_example in [true, false] {
        let visualize_filename = format!("aps2024_debug_demo{}.json", if is_example { "_ex" } else { "" });
        let mut code = CodeCapacityTailoredCode::new(3, 0., 0.01, 1);
        let initializer = code.get_initializer();
        let model_graph = Arc::new(ModelHyperGraph::new(Arc::new(initializer.clone())));
        let mut dual_module = DualModuleSerial::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());
        code.set_physical_errors(&[4]);
        let syndrome_pattern = Arc::new(code.get_syndrome());
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        print_visualize_link(visualize_filename.clone());
        if is_example {
            visualizer.snapshot_combined("code".to_string(), vec![&code]).unwrap();
            let mut primal_module = PrimalModuleSerial::new_empty(&initializer);
            primal_module.growing_strategy = GrowingStrategy::SingleCluster;
            primal_module.plugins = Arc::new(vec![]);
            primal_module.solve_visualizer(&interface_ptr, syndrome_pattern, &mut dual_module, Some(&mut visualizer));
            let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &subgraph, &weight_range],
                )
                .unwrap();
        } else {
            // manually solve it to have fine control
            interface_ptr.write().decoding_graph.set_syndrome(syndrome_pattern.clone());
            for vertex_index in syndrome_pattern.defect_vertices.iter() {
                dual_module.vertices[*vertex_index].write().is_defect = true;
            }
            visualizer
                .snapshot_combined("begin".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            let decoding_graph = interface_ptr.read_recursive().decoding_graph.clone();
            let s0 = Arc::new(InvalidSubgraph::new_complete(btreeset! {3}, btreeset! {}, &decoding_graph));
            let (_, s0_ptr) = interface_ptr.find_or_create_node(&s0, &mut dual_module);
            dual_module.set_grow_rate(&s0_ptr, Rational::from_usize(1).unwrap());
            for _ in 0..3 {
                dual_module.grow(Rational::new_raw(1.into(), 3.into()));
                visualizer
                    .snapshot_combined("grow 1/3".to_string(), vec![&interface_ptr, &dual_module])
                    .unwrap();
            }
            // create another node
            let s1 = Arc::new(InvalidSubgraph::new_complete(btreeset! {6}, btreeset! {}, &decoding_graph));
            let (_, s1_ptr) = interface_ptr.find_or_create_node(&s1, &mut dual_module);
            dual_module.set_grow_rate(&s0_ptr, -Rational::from_usize(1).unwrap());
            dual_module.set_grow_rate(&s1_ptr, Rational::from_usize(1).unwrap());
            for _ in 0..3 {
                dual_module.grow(Rational::new_raw(1.into(), 3.into()));
                visualizer
                    .snapshot_combined("grow 1/3".to_string(), vec![&interface_ptr, &dual_module])
                    .unwrap();
            }
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &Subgraph::from(vec![4])],
                )
                .unwrap();
        }
    }
}

fn simple_demo() {
    for is_example in [true, false] {
        let visualize_filename = format!("aps2024_simple_demo{}.json", if is_example { "_ex" } else { "" });
        let mut code = CodeCapacityTailoredCode::new(3, 0., 0.01, 1);
        let initializer = code.get_initializer();
        let model_graph = Arc::new(ModelHyperGraph::new(Arc::new(initializer.clone())));
        let mut dual_module = DualModuleSerial::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());
        code.set_physical_errors(&[4]);
        let syndrome_pattern = Arc::new(code.get_syndrome());
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        print_visualize_link(visualize_filename.clone());
        if is_example {
            visualizer.snapshot_combined("code".to_string(), vec![&code]).unwrap();
            let mut primal_module = PrimalModuleSerial::new_empty(&initializer);
            primal_module.growing_strategy = GrowingStrategy::SingleCluster;
            primal_module.plugins = Arc::new(vec![]);
            primal_module.solve_visualizer(&interface_ptr, syndrome_pattern, &mut dual_module, Some(&mut visualizer));
            let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &subgraph, &weight_range],
                )
                .unwrap();
        } else {
            // manually solve it to have fine control
            interface_ptr.write().decoding_graph.set_syndrome(syndrome_pattern.clone());
            for vertex_index in syndrome_pattern.defect_vertices.iter() {
                dual_module.vertices[*vertex_index].write().is_defect = true;
            }
            visualizer
                .snapshot_combined("begin".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            let decoding_graph = interface_ptr.read_recursive().decoding_graph.clone();
            let s0 = Arc::new(InvalidSubgraph::new_complete(btreeset! {3}, btreeset! {}, &decoding_graph));
            let (_, s0_ptr) = interface_ptr.find_or_create_node(&s0, &mut dual_module);
            dual_module.set_grow_rate(&s0_ptr, Rational::from_usize(1).unwrap());
            visualizer
                .snapshot_combined("create s0".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            for _ in 0..1 {
                dual_module.grow(Rational::new_raw(1.into(), 1.into()));
                visualizer
                    .snapshot_combined("grow 1".to_string(), vec![&interface_ptr, &dual_module])
                    .unwrap();
            }
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &Subgraph::from(vec![4])],
                )
                .unwrap();
        }
    }
}

fn challenge_demo() {
    for is_example in [true, false] {
        let visualize_filename = format!("aps2024_challenge_demo{}.json", if is_example { "_ex" } else { "" });
        let mut code = CodeCapacityTailoredCode::new(5, 0., 0.01, 1);
        let initializer = code.get_initializer();
        let model_graph = Arc::new(ModelHyperGraph::new(Arc::new(initializer.clone())));
        let mut dual_module = DualModuleSerial::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());
        let syndrome_pattern = Arc::new(SyndromePattern::new_vertices(vec![10, 15, 16]));
        code.set_syndrome(&syndrome_pattern);
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        print_visualize_link(visualize_filename.clone());
        if is_example {
            visualizer.snapshot_combined("code".to_string(), vec![&code]).unwrap();
            let mut primal_module = PrimalModuleSerial::new_empty(&initializer);
            primal_module.growing_strategy = GrowingStrategy::SingleCluster;
            primal_module.plugins = Arc::new(vec![
                PluginUnionFind::entry(), // to allow timeout using union-find as baseline
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once), // first make all clusters valid single hair
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Multiple {
                    max_repetition: usize::MAX,
                }),
            ]);
            primal_module.solve_visualizer(&interface_ptr, syndrome_pattern, &mut dual_module, Some(&mut visualizer));
            let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &subgraph, &weight_range],
                )
                .unwrap();
        } else {
            // manually solve it to have fine control
            interface_ptr.write().decoding_graph.set_syndrome(syndrome_pattern.clone());
            for vertex_index in syndrome_pattern.defect_vertices.iter() {
                dual_module.vertices[*vertex_index].write().is_defect = true;
            }
            visualizer
                .snapshot_combined("begin".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            let decoding_graph = interface_ptr.read_recursive().decoding_graph.clone();
            let invalid_subgraphs = vec![
                (btreeset! {10}, btreeset! {}),                                                  // s0, y0
                (btreeset! {5, 6, 7, 9, 10, 11, 15, 16, 17}, btreeset! {6, 7, 11, 12}),          // s1, y3
                (btreeset! {15}, btreeset! {}),                                                  // s2, y1
                (btreeset! {}, btreeset! {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18}), // s3, y4
                (btreeset! {}, btreeset! {1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18}), // s4, y5
                (btreeset! {}, btreeset! {0, 4, 9, 10, 11, 14, 15, 16, 19, 20, 21, 22, 23, 24}), // s5, y6
                (btreeset! {}, btreeset! {4, 5, 9, 10, 11, 14, 15, 16, 19, 20, 21, 22, 23, 24}), // s6, y7
            ];
            let mut s_ptr = vec![];
            let set_grow_rate =
                |dual_module: &mut DualModuleSerial, s_ptr: &mut Vec<DualNodePtr>, speeds: Vec<(usize, Rational)>| {
                    for ptr in s_ptr.iter() {
                        dual_module.set_grow_rate(ptr, Rational::from_usize(0).unwrap());
                    }
                    for (index, speed) in speeds.into_iter() {
                        while index >= s_ptr.len() {
                            let (vertices, edges) = invalid_subgraphs[s_ptr.len()].clone();
                            let s = if vertices.is_empty() {
                                Arc::new(InvalidSubgraph::new(edges, &decoding_graph))
                            } else {
                                Arc::new(InvalidSubgraph::new_complete(vertices, edges, &decoding_graph))
                            };
                            let (_, ptr) = interface_ptr.find_or_create_node(&s, dual_module);
                            dual_module.set_grow_rate(&ptr, Rational::from_usize(0).unwrap());
                            s_ptr.push(ptr);
                        }
                        dual_module.set_grow_rate(&s_ptr[index], speed);
                    }
                };
            // 1 -> 2
            set_grow_rate(&mut dual_module, &mut s_ptr, vec![(0, Rational::from_usize(1).unwrap())]);
            dual_module.grow(Rational::new_raw(1.into(), 1.into()));
            visualizer
                .snapshot_combined("".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            // 3 -> 4
            set_grow_rate(&mut dual_module, &mut s_ptr, vec![(1, Rational::from_usize(1).unwrap())]);
            dual_module.grow(Rational::new_raw(1.into(), 1.into()));
            visualizer
                .snapshot_combined("".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            // intermediate result
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &Subgraph::from(vec![0, 5, 11])],
                )
                .unwrap();
            visualizer
                .snapshot_combined("next".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            // 5 -> 6
            set_grow_rate(
                &mut dual_module,
                &mut s_ptr,
                vec![
                    (0, -Rational::from_usize(1).unwrap()),
                    (2, Rational::from_usize(1).unwrap()),
                    (1, -Rational::from_usize(1).unwrap()),
                    (3, Rational::from_usize(1).unwrap()),
                    (4, Rational::from_usize(1).unwrap()),
                ],
            );
            dual_module.grow(Rational::new_raw(1.into(), 2.into()));
            visualizer
                .snapshot_combined("".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            // 7 -> 8
            set_grow_rate(
                &mut dual_module,
                &mut s_ptr,
                vec![
                    (0, -Rational::from_usize(1).unwrap()),
                    (2, Rational::from_usize(1).unwrap()),
                    (1, -Rational::from_usize(1).unwrap()),
                    (5, Rational::from_usize(1).unwrap()),
                    (6, Rational::from_usize(1).unwrap()),
                ],
            );
            dual_module.grow(Rational::new_raw(1.into(), 2.into()));
            visualizer
                .snapshot_combined("".to_string(), vec![&interface_ptr, &dual_module])
                .unwrap();
            // final result
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &Subgraph::from(vec![0, 5, 11])],
                )
                .unwrap();
        }
    }
}

fn main() {
    debug_demo();
    simple_demo();
    challenge_demo();
}
