use super::interface::*;
use super::row::*;
use super::visualize::*;
use crate::util::*;
use derivative::Derivative;
use std::collections::{BTreeMap, BTreeSet};

use crate::dual_module_pq::{EdgeWeak, VertexPtr, VertexWeak};
use crate::pointers::UnsafePtr;

/// complete matrix considers a predefined set of edges and won't consider any other edges
#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct CompleteMatrix {
    /// the vertices already maintained by this parity check
    vertices: BTreeSet<VertexWeak>,
    /// the edges maintained by this parity check, mapping to the local indices
    edges: BTreeMap<EdgeWeak, VarIndex>,
    /// variable index map to edge index
    variables: Vec<EdgeWeak>,
    constraints: Vec<ParityRow>,
}

impl MatrixBasic for CompleteMatrix {
    fn add_variable(&mut self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        if self.edges.contains_key(&edge_weak) {
            // variable already exists
            return None;
        }
        if !self.constraints.is_empty() {
            panic!("complete matrix doesn't allow dynamic edges, please insert all edges at the beginning")
        }
        let var_index = self.variables.len();
        self.edges.insert(edge_weak.clone(), var_index);
        self.variables.push(edge_weak);
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
        self.vertices.insert(vertex_weak.clone());
        let mut row = ParityRow::new_length(self.variables.len());
        for edge_weak in incident_edges.iter() {
            if self.exists_edge(edge_weak.clone()) {
                let var_index = self.edges[&edge_weak];
                row.set_left(var_index, true);
            }
        }
        row.set_right(parity);
        self.constraints.push(row);
        // never add new edges
        None
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

    fn get_vertices(&self) -> BTreeSet<VertexWeak> {
        self.vertices.clone()
    }

    fn get_edges(&self) -> BTreeSet<EdgeWeak> {
        self.edges.keys().cloned().collect()
    }
}

impl MatrixView for CompleteMatrix {
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

impl VizTrait for CompleteMatrix {
    fn viz_table(&mut self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use crate::matrix::basic::tests::{edge_vec_from_indices, initialize_vertex_edges_for_matrix_testing};
    use crate::matrix::Echelon;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn complete_matrix_1() {
        // cargo test --features=colorful complete_matrix_1 -- --nocapture
        let mut matrix = CompleteMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 12, 345];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        for edge_ptr in edges.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }
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
    fn complete_matrix_should_not_add_repeated_constraint() {
        // cargo test --features=colorful complete_matrix_should_not_add_repeated_constraint -- --nocapture
        let mut matrix = CompleteMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 8];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 2], vec![1]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        for edge_ptr in edges.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }

        assert_eq!(
            matrix.add_constraint(
                vertices[0].downgrade(),
                &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
                false
            ),
            None
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
    fn complete_matrix_row_operations() {
        // cargo test --features=colorful complete_matrix_row_operations -- --nocapture
        let mut matrix = CompleteMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        for edge_ptr in edges.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }

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
    fn complete_matrix_manual_echelon() {
        // cargo test --features=colorful complete_matrix_manual_echelon -- --nocapture
        let mut matrix = CompleteMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        // add variables [1, 4, 6, 9, 9, 6, 4, 1]
        for edge_ptr in edges.iter() {
            matrix.add_variable(edge_ptr.downgrade());
        }
        for edge_ptr in edges.clone().into_iter().rev() {
            matrix.add_variable(edge_ptr.downgrade());
        }

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

    #[test]
    fn complete_matrix_automatic_echelon() {
        // cargo test --features=colorful complete_matrix_automatic_echelon -- --nocapture
        let mut matrix = Echelon::<CompleteMatrix>::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 11, 12, 23];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2, 4, 5], vec![1, 3, 6, 5], vec![0, 3, 4]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        for edge_index in 0..4 {
            matrix.add_variable(edges[edge_index].downgrade());
        }
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
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊1┊4┊6┊9┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊1┊   ┊4┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊ ┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
    }

    #[test]
    #[should_panic]
    fn complete_matrix_dynamic_variables_forbidden() {
        // cargo test complete_matrix_dynamic_variables_forbidden -- --nocapture
        let mut matrix = Echelon::<CompleteMatrix>::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 2];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        for edge_index in 0..4 {
            matrix.add_variable(edges[edge_index].downgrade());
        }
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
        matrix.add_variable(edges[4].downgrade());
    }
}
