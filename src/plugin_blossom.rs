//! Blossom Plugin
//!
//! This plugin implements the Blossom algorithm for finding augmenting paths and expanding blossoms.

use crate::decoding_hypergraph::*;
use crate::dual_module::*;
use crate::invalid_subgraph::*;
use crate::matrix::*;
use crate::num_traits::One;
use crate::plugin::*;
use crate::relaxer::*;
use crate::util::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct PluginBlossom {}

impl PluginImpl for PluginBlossom {
    fn find_relaxers(
        &self,
        decoding_graph: &DecodingHyperGraph,
        _matrix: &mut EchelonMatrix,
        positive_dual_nodes: &[DualNodePtr],
    ) -> Vec<Relaxer> {
        let mut relaxers = Vec::new();

        let vertex_count = decoding_graph.model_graph.initializer.vertex_num;
        let mut matched = vec![false; vertex_count];
        let mut partners = vec![None; vertex_count];

        for vertex in positive_dual_nodes.iter().map(|node| node.read_recursive().index) {
            if !matched[vertex] {
                if let Some(path) = find_augmenting_path(decoding_graph, vertex, &mut matched, &mut partners) {
                    let (vertices, edges) = construct_subgraph(&path, decoding_graph);
                    let invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(
                        vertices,
                        edges,
                        decoding_graph,
                    ));
                    relaxers.push(Relaxer::new([(invalid_subgraph, Rational::one())].into_iter().collect()));
                }
            }
        }

        for relaxer in &mut relaxers {
            expand_blossoms(decoding_graph, relaxer);
        }

        relaxers
    }
}

fn find_augmenting_path(
    decoding_graph: &DecodingHyperGraph,
    start_vertex: VertexIndex,
    matched: &mut Vec<bool>,
    partners: &mut Vec<Option<VertexIndex>>,
) -> Option<Vec<VertexIndex>> {
    let vertex_count = decoding_graph.model_graph.initializer.vertex_num;
    let mut parent = vec![None; vertex_count];
    let mut base = (0..vertex_count).collect::<Vec<_>>();
    let mut visited = vec![false; vertex_count];
    let mut queue = VecDeque::new();

    queue.push_back(start_vertex);
    visited[start_vertex] = true;

    while let Some(v) = queue.pop_front() {
        for &edge in decoding_graph.get_vertex_neighbors(v) {
            for &u in decoding_graph.get_edge_neighbors(edge) {
                if base[v] == base[u] || (matched[v] && matched[u]) {
                    continue;
                }
                if u == start_vertex || partners[u].is_some() && parent[partners[u].unwrap()].is_some() {
                    let blossom = find_blossom(v, u, &parent, &base);
                    contract_blossom(&mut queue, &mut base, &mut visited, &blossom);
                } else if parent[u].is_none() {
                    parent[u] = Some(v);
                    if let Some(partner) = partners[u] {
                        parent[partner] = Some(u);
                        queue.push_back(partner);
                        visited[partner] = true;
                    } else {
                        // Update matched and partners arrays when a match is found
                        let path = expand_path(start_vertex, u, &parent, partners);
                        for i in 0..path.len() - 1 {
                            matched[path[i]] = !matched[path[i]];
                            matched[path[i + 1]] = !matched[path[i + 1]];
                            partners[path[i]] = Some(path[i + 1]);
                            partners[path[i + 1]] = Some(path[i]);
                        }
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}

fn find_blossom(v: VertexIndex, u: VertexIndex, parent: &[Option<VertexIndex>], base: &[VertexIndex]) -> Vec<VertexIndex> {
    let mut path_v = vec![v];
    let mut path_u = vec![u];

    let mut v_base = base[v];
    let mut u_base = base[u];

    while v_base != u_base {
        if v_base != base[parent[v_base].unwrap()] {
            v_base = base[parent[v_base].unwrap()];
            path_v.push(v_base);
        }
        if u_base != base[parent[u_base].unwrap()] {
            u_base = base[parent[u_base].unwrap()];
            path_u.push(u_base);
        }
    }

    path_v.reverse();
    path_v.append(&mut path_u);
    path_v
}

fn contract_blossom(
    queue: &mut VecDeque<VertexIndex>,
    base: &mut [VertexIndex],
    visited: &mut [bool],
    blossom: &[VertexIndex],
) {
    let base_vertex = blossom[0];
    for &vertex in blossom.iter() {
        base[vertex] = base_vertex;
        if !visited[vertex] {
            queue.push_back(vertex);
            visited[vertex] = true;
        }
    }
}

fn expand_path(start_vertex: VertexIndex, end_vertex: VertexIndex, parent: &[Option<VertexIndex>], partners: &[Option<VertexIndex>]) -> Vec<VertexIndex> {
    let mut path = vec![end_vertex];
    let mut vertex = end_vertex;

    while vertex != start_vertex {
        vertex = parent[vertex].unwrap();
        path.push(vertex);
        if vertex != start_vertex {
            vertex = partners[vertex].unwrap();
            path.push(vertex);
        }
    }

    path.reverse();
    path
}

fn construct_subgraph(path: &[VertexIndex], decoding_graph: &DecodingHyperGraph) -> (BTreeSet<VertexIndex>, BTreeSet<EdgeIndex>) {
    let mut vertices = BTreeSet::new();
    let mut edges = BTreeSet::new();

    for &vertex in path {
        vertices.insert(vertex);
    }

    for edge in &decoding_graph.model_graph.initializer.weighted_edges {
        let edge_vertices: BTreeSet<_> = edge.vertices.iter().copied().collect();
        let path_vertices: BTreeSet<_> = path.iter().copied().collect();
        if edge_vertices.is_subset(&path_vertices) {
            edges.insert(edge.weight as EdgeIndex); // Using weight as a placeholder for edge index
        }
    }

    (vertices, edges)
}

fn expand_blossoms(decoding_graph: &DecodingHyperGraph, relaxer: &mut Relaxer) {
    let mut expanded_direction = BTreeMap::new();

    for (invalid_subgraph, speed) in relaxer.get_direction().iter() {
        if is_blossom(invalid_subgraph, decoding_graph) {
            let blossom_path = get_blossom_path(invalid_subgraph, decoding_graph);
            let (vertices, edges) = construct_subgraph(&blossom_path, decoding_graph);
            let expanded_invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(vertices, edges, decoding_graph));
            expanded_direction.insert(expanded_invalid_subgraph, speed.clone());
        } else {
            expanded_direction.insert(invalid_subgraph.clone(), speed.clone());
        }
    }

    *relaxer = Relaxer::new(expanded_direction);
}

fn is_blossom(invalid_subgraph: &InvalidSubgraph, decoding_graph: &DecodingHyperGraph) -> bool {
    let mut vertex_count = BTreeMap::new();
    for edge in &invalid_subgraph.edges {
        for &vertex in decoding_graph.get_edge_neighbors(*edge) {
            *vertex_count.entry(vertex).or_insert(0) += 1;
        }
    }
    vertex_count.values().all(|&count| count == 2)
}

fn get_blossom_path(invalid_subgraph: &InvalidSubgraph, decoding_graph: &DecodingHyperGraph) -> Vec<VertexIndex> {
    let mut path = Vec::new();
    let mut edges = invalid_subgraph.edges.iter().cloned().collect::<Vec<_>>();
    let mut current_vertex = decoding_graph.get_edge_neighbors(edges[0])[0];

    loop {
        path.push(current_vertex);
        if let Some(next_edge_index) = edges.iter().position(|&edge| {
            let neighbors = decoding_graph.get_edge_neighbors(edge);
            neighbors.contains(&current_vertex)
        }) {
            let edge = edges.remove(next_edge_index);
            let neighbors = decoding_graph.get_edge_neighbors(edge);
            current_vertex = *neighbors
                .iter()
                .find(|&&vertex| vertex != current_vertex)
                .unwrap();
            if current_vertex == path[0] {
                break;
            }
        } else {
            break;
        }
    }

    path
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::example_codes::*;
    use crate::primal_module_serial::{tests::*, GrowingStrategy};
    use test_case::test_case;

    #[test_case("single_cluster", GrowingStrategy::SingleCluster)]
    #[test_case("multiple_cluster", GrowingStrategy::MultipleClusters)]
    fn plugin_blossom_basic_1(suffix: &str, growing_strategy: GrowingStrategy) {
        // cargo test plugin_blossom_basic_1 -- --nocapture
        let visualize_filename = format!("plugin_blossom_basic_1_{suffix}.json");
        let defect_vertices = vec![10, 11, 16, 17];
        let code = CodeCapacityTailoredCode::new(5, 0., 0.01, 1);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            1,
            vec![PluginBlossom::entry()],
            growing_strategy,
        );
    }
}
