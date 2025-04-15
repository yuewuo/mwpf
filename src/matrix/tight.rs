use super::interface::*;
use super::visualize::*;
use crate::util::*;
use derivative::Derivative;
use std::collections::BTreeSet;

use crate::dual_module_pq::{EdgeWeak, VertexPtr, VertexWeak};

#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct Tight<M: MatrixView> {
    base: M,
    /// the set of tight edges: should be a relatively small set
    tight_edges: BTreeSet<EdgeWeak>,
    /// tight matrix gives a view of only tight edges, with sorted indices
    #[derivative(Default(value = "true"))]
    is_var_indices_outdated: bool,
    /// the internal store of var indices
    var_indices: Vec<VarIndex>,
}

impl<M: MatrixView> Tight<M> {
    pub fn get_base(&self) -> &M {
        &self.base
    }
    pub fn from_base(base: M) -> Self {
        let mut value = Self {
            base,
            tight_edges: BTreeSet::new(),
            is_var_indices_outdated: true,
            var_indices: vec![],
        };
        value.var_indices_lazy_update();
        value
    }
}

impl<M: MatrixView> MatrixTight for Tight<M> {
    fn update_edge_tightness(&mut self, edge_weak: EdgeWeak, is_tight: bool) {
        debug_assert!(self.exists_edge(edge_weak.clone()));
        self.is_var_indices_outdated = true;
        if is_tight {
            self.tight_edges.insert(edge_weak.clone());
        } else {
            self.tight_edges.remove(&edge_weak);
        }
    }

    fn is_tight(&self, edge_weak: EdgeWeak) -> bool {
        debug_assert!(self.exists_edge(edge_weak.clone()));
        self.tight_edges.contains(&edge_weak)
    }

    fn get_tight_edges(&self) -> &BTreeSet<EdgeWeak> {
        &self.tight_edges
    }
}

impl<M: MatrixView> MatrixBasic for Tight<M> {
    fn add_variable(&mut self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.base.add_variable(edge_weak)
    }

    fn add_constraint(
        &mut self,
        vertex_weak: VertexWeak,
        incident_edges: &[EdgeWeak],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        self.base.add_constraint(vertex_weak, incident_edges, parity)
    }

    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        self.base.xor_row(target, source)
    }
    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.base.swap_row(a, b)
    }
    fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool {
        self.get_base().get_lhs(row, var_index)
    }
    fn get_rhs(&self, row: RowIndex) -> bool {
        self.get_base().get_rhs(row)
    }
    fn var_to_edge_weak(&self, var_index: VarIndex) -> EdgeWeak {
        self.get_base().var_to_edge_weak(var_index)
    }
    fn edge_to_var_index(&self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.get_base().edge_to_var_index(edge_weak)
    }
    fn get_vertices(&self) -> BTreeSet<VertexWeak> {
        self.get_base().get_vertices()
    }
    fn get_edges(&self) -> BTreeSet<EdgeWeak> {
        self.get_base().get_edges()
    }
}

impl<M: MatrixView> Tight<M> {
    fn force_update_var_indices(&mut self) {
        self.var_indices.clear();
        for column in 0..self.base.columns() {
            let var_index = self.base.column_to_var_index(column);
            let edge_weak = self.base.var_to_edge_weak(var_index);
            if self.is_tight(edge_weak) {
                self.var_indices.push(var_index);
            }
        }
    }

    fn var_indices_lazy_update(&mut self) {
        if self.is_var_indices_outdated {
            self.force_update_var_indices();
            self.is_var_indices_outdated = false;
        }
    }
}

impl<M: MatrixView> MatrixView for Tight<M> {
    fn columns(&mut self) -> usize {
        self.var_indices_lazy_update();
        self.var_indices.len()
    }

    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        debug_assert!(!self.is_var_indices_outdated, "call `columns` first");
        self.var_indices[column]
    }

    fn rows(&mut self) -> usize {
        self.base.rows()
    }
}

impl<M: MatrixView> VizTrait for Tight<M> {
    fn viz_table(&mut self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::basic::*;
    use super::*;
    use crate::dual_module_pq::{EdgePtr, VertexPtr};
    use crate::matrix::basic::tests::{edge_vec_from_indices, initialize_vertex_edges_for_matrix_testing};
    use std::collections::HashSet;

    type TightMatrix = Tight<BasicMatrix>;

    #[test]
    fn tight_matrix_1() {
        // cargo test --features=colorful tight_matrix_1 -- --nocapture
        let mut matrix = TightMatrix::new();
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
        // this is because by default all edges are not tight
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬───┐
┊ ┊ = ┊
╞═╪═══╡
┊0┊ 1 ┊
├─┼───┤
┊1┊   ┊
├─┼───┤
┊2┊ 1 ┊
└─┴───┘
"
        );
        matrix.update_edge_tightness(edges[1].downgrade(), true);
        matrix.update_edge_tightness(edges[3].downgrade(), true);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬───┐
┊ ┊4┊9┊ = ┊
╞═╪═╪═╪═══╡
┊0┊1┊ ┊ 1 ┊
├─┼─┼─┼───┤
┊1┊1┊1┊   ┊
├─┼─┼─┼───┤
┊2┊ ┊1┊ 1 ┊
└─┴─┴─┴───┘
"
        );
        matrix.update_edge_tightness(edges[3].downgrade(), false);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬───┐
┊ ┊4┊ = ┊
╞═╪═╪═══╡
┊0┊1┊ 1 ┊
├─┼─┼───┤
┊1┊1┊   ┊
├─┼─┼───┤
┊2┊ ┊ 1 ┊
└─┴─┴───┘
"
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn tight_matrix_cannot_set_nonexistent_edge() {
        // cargo test tight_matrix_cannot_set_nonexistent_edge -- --nocapture
        let mut matrix = TightMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 2];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.update_edge_tightness(edges[4].downgrade(), true);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn tight_matrix_cannot_read_nonexistent_edge() {
        // cargo test tight_matrix_cannot_read_nonexistent_edge -- --nocapture
        let mut matrix = TightMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 2];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.is_tight(edges[4].downgrade());
    }

    #[test]
    fn tight_matrix_basic_trait() {
        // cargo test --features=colorful tight_matrix_basic_trait -- --nocapture
        let mut matrix = TightMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 3];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_variable(edges[4].downgrade()); // untight edges will not show
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
        matrix.swap_row(2, 1);
        matrix.xor_row(0, 1);
        for edge_index in 0..4 {
            matrix.update_edge_tightness(edges[edge_index].downgrade(), true);
        }
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
    fn tight_matrix_rebuild_var_indices() {
        // cargo test --features=colorful tight_matrix_rebuild_var_indices -- --nocapture
        let mut matrix = TightMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 3];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        matrix.add_variable(edges[4].downgrade()); // untight edges will not show
        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        assert_eq!(matrix.columns(), 0);
        for edge_index in 0..3 {
            matrix.update_edge_tightness(edges[edge_index].downgrade(), true);
        }
        assert_eq!(matrix.columns(), 3);
        assert_eq!(matrix.columns(), 3); // should only update var_indices_once
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

        matrix.update_edge_tightness(edges[3].downgrade(), true);
        matrix.update_edge_tightness(edges[1].downgrade(), false);
        matrix.update_edge_tightness(edges[2].downgrade(), false);
        assert_eq!(matrix.columns(), 2);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬───┐
┊ ┊1┊9┊ = ┊
╞═╪═╪═╪═══╡
┊0┊1┊ ┊ 1 ┊
├─┼─┼─┼───┤
┊1┊ ┊1┊   ┊
├─┼─┼─┼───┤
┊2┊1┊1┊ 1 ┊
└─┴─┴─┴───┘
"
        );
    }

    #[test]
    #[should_panic]
    fn tight_matrix_cannot_call_dirty_column() {
        // cargo test tight_matrix_cannot_call_dirty_column -- --nocapture
        let mut matrix = TightMatrix::new();
        let vertex_indices = vec![0];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.update_edge_tightness(edges[0].downgrade(), true);
        // even though there is indeed such a column, we forbid such dangerous calls
        // always call `columns()` before accessing any column
        matrix.column_to_var_index(0);
    }
}
