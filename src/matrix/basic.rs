use super::interface::*;
use super::row::*;
use super::visualize::*;
use crate::dual_module_pq::EdgeWeak;
use crate::util::*;
use derivative::Derivative;

use crate::dual_module_pq::{VertexPtr, VertexWeak};

#[derive(Clone, Derivative, PartialEq, Eq)]
#[derivative(Default(new = "true"))]
pub struct BasicMatrix {
    /// the vertices already maintained by this parity check
    pub vertices: FastIterSet<VertexWeak>,
    /// the edges maintained by this parity check, mapping to the local indices
    pub edges: FastIterMap<EdgeWeak, VarIndex>,
    /// variable index map to edge index
    pub variables: Vec<EdgeWeak>,
    pub constraints: Vec<ParityRow>,
}

impl MatrixBasic for BasicMatrix {
    fn add_variable(&mut self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        if self.edges.contains_key(&edge_weak) {
            // variable already exists
            return None;
        }
        let var_index = self.variables.len();
        self.edges.insert(edge_weak.clone(), var_index);
        self.variables.push(edge_weak.clone());
        ParityRow::add_one_variable(&mut self.constraints, self.variables.len());
        Some(var_index)
    }

    fn add_constraint(
        &mut self,
        vertex_weak: VertexWeak,
        incident_edges: &[EdgeWeak],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        if self.vertices.contains(&vertex_weak) {
            // no need to add repeat constraint
            return None;
        }
        let mut var_indices = None;
        self.vertices.insert(vertex_weak.clone());
        for edge_weak in incident_edges.iter() {
            if let Some(var_index) = self.add_variable(edge_weak.clone()) {
                // this is a newly added edge
                var_indices.get_or_insert_with(Vec::new).push(var_index);
            }
        }
        let mut row = ParityRow::new_length(self.variables.len());
        for edge_weak in incident_edges.iter() {
            let var_index = self.edges[&edge_weak];
            row.set_left(var_index, true);
        }
        row.set_right(parity);
        self.constraints.push(row);
        var_indices
    }

    /// row operations
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        ParityRow::xor_two_rows(&mut self.constraints, target, source)
    }

    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.constraints.swap(a, b);
    }

    fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool {
        self.constraints[row].get_left(var_index)
    }

    fn get_rhs(&self, row: RowIndex) -> bool {
        self.constraints[row].get_right()
    }

    fn var_to_edge_weak(&self, var_index: VarIndex) -> EdgeWeak {
        self.variables[var_index].clone()
    }

    fn edge_to_var_index(&self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.edges.get(&edge_weak).cloned()
    }

    fn get_vertices(&self) -> FastIterSet<VertexWeak> {
        self.vertices.clone()
    }

    fn get_edges(&self) -> FastIterSet<EdgeWeak> {
        self.edges.keys().cloned().collect()
    }
}

impl MatrixView for BasicMatrix {
    fn columns(&mut self) -> usize {
        self.variables.len()
    }

    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        column
    }

    fn rows(&mut self) -> usize {
        self.constraints.len()
    }
}

impl VizTrait for BasicMatrix {
    fn viz_table(&mut self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dual_module_pq::{Edge, EdgePtr, Vertex, VertexPtr};
    use crate::num_traits::Zero;
    use crate::pointers::*;
    use std::collections::HashSet;

    pub fn initialize_vertex_edges_for_matrix_testing(
        vertex_indices: Vec<VertexIndex>,
        edge_indices: Vec<EdgeIndex>,
    ) -> (Vec<VertexPtr>, Vec<EdgePtr>) {
        // create edges
        let edges: Vec<EdgePtr> = edge_indices
            .into_iter()
            .map(|edge_index| {
                EdgePtr::new_value(
                    Edge {
                        edge_index,
                        weight: Rational::zero(),
                        dual_nodes: vec![],
                        vertices: vec![],
                        last_updated_time: Rational::zero(),
                        growth_at_last_updated_time: Rational::zero(),
                        grow_rate: Rational::zero(),
                        unit_index: Some(0),                 // dummy value
                        connected_to_boundary_vertex: false, // dummy value
                        #[cfg(feature = "incr_lp")]
                        cluster_weights: hashbrown::HashMap::new(),
                    },
                    (0, 0),
                )
            })
            .collect();

        // create vertices
        let vertices: Vec<VertexPtr> = vertex_indices
            .into_iter()
            .map(|vertex_index| {
                VertexPtr::new_value(
                    Vertex {
                        vertex_index,
                        is_defect: false,
                        edges: vec![],
                        mirrored_vertices: vec![], // dummy vlaue
                    },
                    (0, 0),
                )
            })
            .collect();

        (vertices, edges)
    }

    pub fn edge_vec_from_indices(edge_sequences: &[usize], edges: &Vec<EdgePtr>) -> Vec<EdgeWeak> {
        edge_sequences
            .to_vec()
            .iter()
            .map(|&edge_sequence| edges[edge_sequence].downgrade())
            .collect::<Vec<_>>()
    }

    #[test]
    fn basic_matrix_1() {
        // cargo test --features=colorful basic_matrix_1 -- --nocapture
        let mut matrix = BasicMatrix::new();
        matrix.printstd();
        assert_eq!(
            matrix.printstd_str(),
            "\
┌┬───┐
┊┊ = ┊
╞╪═══╡
└┴───┘
"
        );

        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 12, 345];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_variable(edges[0].downgrade());
        matrix.add_variable(edges[1].downgrade());
        matrix.add_variable(edges[2].downgrade());
        matrix.add_variable(edges[3].downgrade());
        matrix.printstd();
        assert_eq!(
            matrix.printstd_str(),
            "\
┌┬─┬─┬─┬─┬───┐
┊┊1┊4┊1┊3┊ = ┊
┊┊ ┊ ┊2┊4┊   ┊
┊┊ ┊ ┊ ┊5┊   ┊
╞╪═╪═╪═╪═╪═══╡
└┴─┴─┴─┴─┴───┘
"
        );
        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.add_constraint(
            vertices[1].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges),
            false,
        );
        matrix.add_constraint(
            vertices[2].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges),
            true,
        );
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊1┊3┊ = ┊
┊ ┊ ┊ ┊2┊4┊   ┊
┊ ┊ ┊ ┊ ┊5┊   ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊1┊ ┊ ┊1┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"
        );
        assert_eq!(
            matrix
                .get_vertices()
                .iter()
                .map(|v| v.upgrade_force().read_recursive().vertex_index)
                .collect::<HashSet<_>>(),
            [0, 1, 2].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            matrix
                .get_view_edges()
                .iter()
                .map(|e| e.upgrade_force().read_recursive().edge_index)
                .collect::<HashSet<_>>(),
            [1, 4, 12, 345].into_iter().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn basic_matrix_should_not_add_repeated_constraint() {
        // cargo test --features=colorful basic_matrix_should_not_add_repeated_constraint -- --nocapture
        let mut matrix = BasicMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 8];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 2], vec![1]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        assert_eq!(
            matrix.add_constraint(
                vertices[0].downgrade(),
                &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
                false
            ),
            Some(vec![0, 1, 2])
        );
        assert_eq!(
            matrix.add_constraint(
                vertices[1].downgrade(),
                &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges),
                true
            ),
            None
        );
        assert_eq!(
            matrix.add_constraint(
                vertices[0].downgrade(),
                &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges),
                true
            ),
            None
        ); // repeated
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬───┐
┊ ┊1┊4┊8┊ = ┊
╞═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊   ┊
├─┼─┼─┼─┼───┤
┊1┊ ┊1┊1┊ 1 ┊
└─┴─┴─┴─┴───┘
"
        );
    }

    #[test]
    fn basic_matrix_row_operations() {
        // cargo test --features=colorful basic_matrix_row_operations -- --nocapture
        let mut matrix = BasicMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.add_constraint(
            vertices[1].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges),
            false,
        );
        matrix.add_constraint(
            vertices[2].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges),
            true,
        );
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊6┊9┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊1┊ ┊ ┊1┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"
        );
        matrix.swap_row(2, 1);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊6┊9┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊1┊ ┊ ┊1┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊2┊ ┊1┊ ┊1┊   ┊
└─┴─┴─┴─┴─┴───┘
"
        );
        matrix.xor_row(0, 1);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊6┊9┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊ ┊1┊1┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊1┊1┊ ┊ ┊1┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊2┊ ┊1┊ ┊1┊   ┊
└─┴─┴─┴─┴─┴───┘
"
        );
    }

    #[test]
    fn basic_matrix_manual_echelon() {
        // cargo test --features=colorful basic_matrix_manual_echelon -- --nocapture
        let mut matrix = BasicMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.add_constraint(
            vertices[1].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges),
            false,
        );
        matrix.add_constraint(
            vertices[2].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges),
            true,
        );
        matrix.xor_row(2, 0);
        matrix.xor_row(0, 1);
        matrix.xor_row(2, 1);
        matrix.xor_row(0, 2);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊6┊9┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊ ┊ ┊1┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊ ┊ ┊1┊ ┊   ┊
└─┴─┴─┴─┴─┴───┘
"
        );
    }
}
