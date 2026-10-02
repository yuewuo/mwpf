use mwpf::mwpf_solver::{SolverSerialJointSingleHair, SolverTrait};
use mwpf::num_traits::FromPrimitive;
use mwpf::util::{HyperEdge, SolverInitializer, SyndromePattern, Weight};
use std::sync::Arc;

#[test]
fn solves_hypergraph_through_public_rust_api() {
    let initializer = Arc::new(SolverInitializer::new(
        4,
        vec![
            HyperEdge::new(vec![0, 1], Weight::from_i64(100).unwrap()),
            HyperEdge::new(vec![1, 2], Weight::from_i64(100).unwrap()),
            HyperEdge::new(vec![2, 3], Weight::from_i64(100).unwrap()),
            HyperEdge::new(vec![0], Weight::from_i64(100).unwrap()),
            HyperEdge::new(vec![0, 1, 2], Weight::from_i64(60).unwrap()),
        ],
    ));
    let mut solver = SolverSerialJointSingleHair::new(&initializer, "{}".parse().unwrap());
    let defects = vec![0, 1, 3];
    solver.solve(SyndromePattern::new_vertices(defects.clone()));

    let (subgraph, bounds) = solver.subgraph_range();
    assert!(initializer.matches_subgraph_syndrome(&subgraph, &defects));
    assert!(bounds.is_optimal());
    assert_eq!(bounds.lower, Weight::from_i64(160).unwrap());
    assert_eq!(initializer.get_subgraph_total_weight(&subgraph), bounds.upper);
    let mut edges: Vec<_> = subgraph.iter().copied().collect();
    edges.sort_unstable();
    assert_eq!(edges, vec![2, 4]);

    solver.clear();
    solver.solve(SyndromePattern::new_vertices(vec![]));
    let (subgraph, bounds) = solver.subgraph_range();
    assert!(subgraph.iter().next().is_none());
    assert!(bounds.is_optimal());
    assert_eq!(bounds.upper, Weight::from_i64(0).unwrap());
}
