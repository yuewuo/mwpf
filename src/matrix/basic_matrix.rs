use super::matrix_interface::*;
use super::row::*;
use super::table::*;
use crate::util::*;
use derivative::Derivative;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct BasicMatrix {
    /// the vertices already maintained by this parity check
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub vertices: HashSet<VertexIndex>,
    /// the edges maintained by this parity check, mapping to the local indices
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub edges: HashMap<EdgeIndex, VarIndex>,
    /// variable index map to edge index
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub variables: Vec<EdgeIndex>,
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub constraints: Vec<ParityRow>,
}

impl MatrixBasic for BasicMatrix {
    fn add_variable(&mut self, edge_index: EdgeIndex) -> Option<VarIndex> {
        if self.edges.contains_key(&edge_index) {
            // variable already exists
            return None;
        }
        let var_index = self.variables.len();
        self.edges.insert(edge_index, var_index);
        self.variables.push(edge_index);
        ParityRow::add_one_variable(&mut self.constraints, self.variables.len());
        Some(var_index)
    }

    fn add_constraint(
        &mut self,
        vertex_index: VertexIndex,
        incident_edges: &[EdgeIndex],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        if self.vertices.contains(&vertex_index) {
            // no need to add repeat constraint
            return None;
        }
        let mut var_indices = None;
        self.vertices.insert(vertex_index);
        for &edge_index in incident_edges.iter() {
            if let Some(var_index) = self.add_variable(edge_index) {
                // this is a newly added edge
                var_indices.get_or_insert_with(Vec::new).push(var_index);
            }
        }
        let mut row = ParityRow::new_length(self.variables.len());
        for &edge_index in incident_edges.iter() {
            let var_index = self.edges[&edge_index];
            row.set_left(var_index, true);
        }
        row.set_right(parity);
        self.constraints.push(row);
        var_indices
    }

    /// row operations
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        if target < source {
            let (slice_1, slice_2) = self.constraints.split_at_mut(source);
            let source = &slice_2[0];
            let target = &mut slice_1[target];
            target.add(source);
        } else {
            let (slice_1, slice_2) = self.constraints.split_at_mut(target);
            let source = &slice_1[source];
            let target = &mut slice_2[0];
            target.add(source);
        }
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
}

impl MatrixView for BasicMatrix {
    #[inline]
    fn columns(&self) -> usize {
        self.variables.len()
    }

    #[inline]
    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        column
    }

    #[inline]
    fn rows(&self) -> usize {
        self.constraints.len()
    }

    #[inline]
    fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeIndex {
        self.variables[var_index]
    }
}

impl VizTrait for BasicMatrix {
    fn viz_table(&self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

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
        matrix.add_variable(1);
        matrix.add_variable(4);
        matrix.add_variable(12);
        matrix.add_variable(345);
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
        matrix.add_constraint(0, &[1, 4, 12], true);
        matrix.add_constraint(1, &[4, 345], false);
        matrix.add_constraint(2, &[1, 345], true);
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
    }

    #[test]
    fn basic_matrix_should_not_add_repeated_constraint() {
        // cargo test --features=colorful basic_matrix_should_not_add_repeated_constraint -- --nocapture
        let mut matrix = BasicMatrix::new();
        assert_eq!(matrix.add_constraint(0, &[1, 4, 8], false), Some(vec![0, 1, 2]));
        assert_eq!(matrix.add_constraint(1, &[4, 8], true), None);
        assert_eq!(matrix.add_constraint(0, &[4], true), None); // repeated
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
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
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
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
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
