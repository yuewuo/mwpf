use super::interface::*;
use super::visualize::*;
use crate::util::*;
use derivative::Derivative;

#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct Tight<M: MatrixView> {
    base: M,
    /// the set of tight edges: should be a relatively small set
    tight_edges: FastIterSet<EdgeIndex>,
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
            tight_edges: FastIterSet::new(),
            is_var_indices_outdated: true,
            var_indices: vec![],
        };
        value.var_indices_lazy_update();
        value
    }
}

impl<M: MatrixView> MatrixTight for Tight<M> {
    fn update_edge_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool) {
        debug_assert!(self.exists_edge(edge_index));
        self.is_var_indices_outdated = true;
        if is_tight {
            self.tight_edges.insert(edge_index);
        } else {
            self.tight_edges.remove(&edge_index);
        }
    }

    fn is_tight(&self, edge_index: usize) -> bool {
        debug_assert!(self.exists_edge(edge_index));
        self.tight_edges.contains(&edge_index)
    }

    fn get_tight_edges(&self) -> &FastIterSet<EdgeIndex> {
        &self.tight_edges
    }
}

impl<M: MatrixView> MatrixBasic for Tight<M> {
    fn add_variable(&mut self, edge_index: EdgeIndex) -> Option<VarIndex> {
        self.base.add_variable(edge_index)
    }

    fn add_constraint(
        &mut self,
        vertex_index: VertexIndex,
        incident_edges: &[EdgeIndex],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        self.base.add_constraint(vertex_index, incident_edges, parity)
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
    fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeIndex {
        self.get_base().var_to_edge_index(var_index)
    }
    fn edge_to_var_index(&self, edge_index: EdgeIndex) -> Option<VarIndex> {
        self.get_base().edge_to_var_index(edge_index)
    }
    fn get_vertices(&self) -> FastIterSet<VertexIndex> {
        self.get_base().get_vertices()
    }
    fn get_edges(&self) -> FastIterSet<EdgeIndex> {
        self.get_base().get_edges()
    }
}

impl<M: MatrixView> Tight<M> {
    fn force_update_var_indices(&mut self) {
        self.var_indices.clear();
        for column in 0..self.base.columns() {
            let var_index = self.base.column_to_var_index(column);
            let edge_index = self.base.var_to_edge_index(var_index);
            if self.is_tight(edge_index) {
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

    type TightMatrix = Tight<BasicMatrix>;

    #[test]
    fn tight_matrix_1() {
        // cargo test --features=colorful tight_matrix_1 -- --nocapture
        let mut matrix = TightMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
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
        matrix.update_edge_tightness(4, true);
        matrix.update_edge_tightness(9, true);
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
        matrix.update_edge_tightness(9, false);
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
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.update_edge_tightness(2, true);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn tight_matrix_cannot_read_nonexistent_edge() {
        // cargo test tight_matrix_cannot_read_nonexistent_edge -- --nocapture
        let mut matrix = TightMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.is_tight(2);
    }

    #[test]
    fn tight_matrix_basic_trait() {
        // cargo test --features=colorful tight_matrix_basic_trait -- --nocapture
        let mut matrix = TightMatrix::new();
        matrix.add_variable(3); // untight edges will not show
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
        matrix.swap_row(2, 1);
        matrix.xor_row(0, 1);
        for edge_index in [1, 4, 6, 9] {
            matrix.update_edge_tightness(edge_index, true);
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
        matrix.add_variable(3); // untight edges will not show
        matrix.add_constraint(0, &[1, 4, 6], true);
        assert_eq!(matrix.columns(), 0);
        for edge_index in [1, 4, 6] {
            matrix.update_edge_tightness(edge_index, true);
        }
        assert_eq!(matrix.columns(), 3);
        assert_eq!(matrix.columns(), 3); // should only update var_indices_once
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
        matrix.update_edge_tightness(9, true);
        matrix.update_edge_tightness(4, false);
        matrix.update_edge_tightness(6, false);
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
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.update_edge_tightness(1, true);
        // even though there is indeed such a column, we forbid such dangerous calls
        // always call `columns()` before accessing any column
        matrix.column_to_var_index(0);
    }
}
