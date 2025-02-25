//! A hair view is a wrapper on the data that focuses on the hair edges
//!
//! It doesn't introduce new data structure, so the creation is cheap
//!

use super::interface::*;
use super::visualize::*;
use crate::util::*;
use prettytable::*;

pub struct HairView<'a, M: MatrixTail + MatrixEchelon> {
    base: &'a mut M,
    column_bias: ColumnIndex,
    row_bias: RowIndex,
}

impl<'a, M: MatrixTail + MatrixEchelon> HairView<'a, M> {
    pub fn get_base(&self) -> &M {
        self.base
    }
    pub fn get_base_view_edges(&mut self) -> Vec<EdgeIndex> {
        self.base.get_view_edges()
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> HairView<'a, M> {
    pub fn new<EdgeIter>(matrix: &'a mut M, hair: EdgeIter) -> Self
    where
        EdgeIter: Iterator<Item = EdgeIndex>,
    {
        matrix.set_tail_edges(hair);
        let columns = matrix.columns();
        let rows = matrix.rows();
        let mut column_bias = columns;
        let mut row_bias = rows;
        for column in (0..columns).rev() {
            let edge_index = matrix.column_to_edge_index(column);
            if matrix.get_tail_edges().contains(&edge_index) {
                column_bias = column;
            } else {
                break;
            }
        }
        let echelon_info = matrix.get_echelon_info();
        for column in column_bias..columns {
            let column_info = &echelon_info.columns[column];
            if column_info.is_dependent() {
                row_bias = column_info.row;
                break;
            }
        }
        Self {
            base: matrix,
            column_bias,
            row_bias,
        }
    }

    pub fn get_echelon_column_info(&mut self, column: ColumnIndex) -> ColumnInfo {
        self.base.get_echelon_info().columns[column + self.column_bias]
    }

    pub fn get_echelon_row_info(&mut self, row: RowIndex) -> RowInfo {
        self.base.get_echelon_info().rows[row + self.row_bias]
    }

    pub fn get_echelon_satisfiable(&mut self) -> bool {
        self.base.get_echelon_info().satisfiable
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> MatrixTail for HairView<'a, M> {
    fn get_tail_edges(&self) -> &FastIterSet<EdgeIndex> {
        self.get_base().get_tail_edges()
    }
    fn get_tail_edges_mut(&mut self) -> &mut FastIterSet<EdgeIndex> {
        panic!("cannot mutate a hair view");
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> MatrixEchelon for HairView<'a, M> {
    fn get_echelon_info(&mut self) -> &EchelonInfo {
        panic!("cannot get echelon info, please use individual method")
    }
    fn get_echelon_info_immutable(&self) -> &EchelonInfo {
        panic!("cannot get echelon info, please use individual method")
    }
}

impl<'a, M: MatrixTight + MatrixTail + MatrixEchelon> MatrixTight for HairView<'a, M> {
    fn update_edge_tightness(&mut self, _edge_index: EdgeIndex, _is_tight: bool) {
        panic!("cannot mutate a hair view");
    }
    fn is_tight(&self, edge_index: usize) -> bool {
        self.get_base().is_tight(edge_index)
    }
    fn get_tight_edges(&self) -> &FastIterSet<EdgeIndex> {
        self.base.get_tight_edges()
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> MatrixBasic for HairView<'a, M> {
    fn add_variable(&mut self, _edge_index: EdgeIndex) -> Option<VarIndex> {
        panic!("cannot mutate a hair view");
    }

    fn add_constraint(
        &mut self,
        _vertex_index: VertexIndex,
        _incident_edges: &[EdgeIndex],
        _parity: bool,
    ) -> Option<Vec<VarIndex>> {
        panic!("cannot mutate a hair view");
    }

    fn xor_row(&mut self, _target: RowIndex, _source: RowIndex) {
        panic!("cannot mutate a hair view");
    }
    fn swap_row(&mut self, _a: RowIndex, _b: RowIndex) {
        panic!("cannot mutate a hair view");
    }
    fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool {
        self.get_base().get_lhs(row + self.row_bias, var_index)
    }
    fn get_rhs(&self, row: RowIndex) -> bool {
        self.get_base().get_rhs(row + self.row_bias)
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

impl<'a, M: MatrixTail + MatrixEchelon> MatrixView for HairView<'a, M> {
    fn columns(&mut self) -> usize {
        self.base.columns() - self.column_bias
    }

    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        self.base.column_to_var_index(column + self.column_bias)
    }

    fn rows(&mut self) -> usize {
        self.base.rows() - self.row_bias
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> VizTrait for HairView<'a, M> {
    fn viz_table(&mut self) -> VizTable {
        let mut table = VizTable::from(&mut *self);
        // add hair mark
        assert!(table.title.get_cell(0).unwrap().get_content().is_empty());
        if self.base.get_echelon_info().satisfiable {
            table.title.set_cell(Cell::new("H").style_spec("brFg"), 0).unwrap();
        } else {
            table.title.set_cell(Cell::new("X").style_spec("brFr"), 0).unwrap();
        }
        // add row information on the right
        table.title.add_cell(Cell::new("\u{25BC}"));
        let rows = self.rows();
        for row in 0..rows {
            let row_info = self.get_echelon_row_info(row);
            let cell = if row_info.has_leading() {
                Cell::new(
                    self.column_to_edge_index(row_info.column - self.column_bias)
                        .to_string()
                        .as_str(),
                )
                .style_spec("irFm")
            } else {
                Cell::new("*").style_spec("rFr")
            };
            table.rows[row].add_cell(cell);
        }
        // add column information on the bottom
        let mut bottom_row = Row::empty();
        bottom_row.add_cell(Cell::new(" \u{25B6}"));
        let columns = self.columns();
        for column in 0..columns {
            let column_info = self.get_echelon_column_info(column);
            let cell = if column_info.is_dependent() {
                Cell::new(VizTable::force_single_column((column_info.row - self.row_bias).to_string().as_str()).as_str())
                    .style_spec("irFb")
            } else {
                Cell::new("*").style_spec("rFr")
            };
            bottom_row.add_cell(cell);
        }
        bottom_row.add_cell(Cell::new("\u{25C0}  "));
        bottom_row.add_cell(Cell::new("\u{25B2}"));
        table.rows.push(bottom_row);
        table
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::basic::*;
    use super::super::echelon::*;
    use super::super::tail::*;
    use super::super::tight::*;
    use super::*;

    type EchelonMatrix = Echelon<Tail<Tight<BasicMatrix>>>;

    #[test]
    fn hair_view_simple() {
        // cargo test --features=colorful hair_view_simple -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
        assert_eq!(matrix.edge_to_var_index(4), Some(1));
        for edge_index in [1, 4, 6, 9] {
            matrix.update_edge_tightness(edge_index, true);
        }
        matrix.printstd();
        assert_eq!(
            matrix.printstd_str(),
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
        let mut hair_view = HairView::new(&mut matrix, [6, 9].into_iter());
        assert_eq!(hair_view.edge_to_var_index(4), Some(1));
        hair_view.printstd();
        assert_eq!(
            hair_view.printstd_str(),
            "\
┌──┬─┬─┬───┬─┐
┊ H┊6┊9┊ = ┊▼┊
╞══╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊   ┊6┊
├──┼─┼─┼───┼─┤
┊ ▶┊0┊*┊◀  ┊▲┊
└──┴─┴─┴───┴─┘
"
        );
        let mut hair_view = HairView::new(&mut matrix, [1, 6].into_iter());
        hair_view.base.printstd();
        assert_eq!(
            hair_view.base.printstd_str(),
            "\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊4┊9┊1┊6┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊1┊ ┊ 1 ┊4┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊1┊ ┊ 1 ┊9┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊ ┊1┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊*┊2┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
        hair_view.printstd();
        assert_eq!(
            hair_view.printstd_str(),
            "\
┌──┬─┬─┬───┬─┐
┊ H┊1┊6┊ = ┊▼┊
╞══╪═╪═╪═══╪═╡
┊ 0┊ ┊1┊   ┊6┊
├──┼─┼─┼───┼─┤
┊ ▶┊*┊0┊◀  ┊▲┊
└──┴─┴─┴───┴─┘
"
        );
        assert_eq!(hair_view.get_tail_edges_vec(), [1, 6]);
        assert!(hair_view.is_tight(1));
        assert!(hair_view.get_echelon_satisfiable());
        assert_eq!(hair_view.get_vertices(), [0, 1, 2].into());
        assert_eq!(hair_view.get_base_view_edges(), [4, 9, 1, 6]);
    }

    fn generate_demo_matrix() -> EchelonMatrix {
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        for edge_index in [1, 4, 6, 9] {
            matrix.update_edge_tightness(edge_index, true);
        }
        matrix
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_modify_tail_edges() {
        // cargo test hair_view_should_not_modify_tail_edges -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_tail_edges_mut();
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_update_edge_tightness() {
        // cargo test hair_view_should_not_update_edge_tightness -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.update_edge_tightness(1, false);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_add_variable() {
        // cargo test hair_view_should_not_add_variable -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.add_variable(100);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_add_constraint() {
        // cargo test hair_view_should_not_add_constraint -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.add_constraint(5, &[1, 2, 3], false);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_xor_row() {
        // cargo test hair_view_should_not_xor_row -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.xor_row(0, 1);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_swap_row() {
        // cargo test hair_view_should_not_swap_row -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.swap_row(0, 1);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_get_echelon_info() {
        // cargo test hair_view_should_not_get_echelon_info -- --nocapture
        let mut matrix = generate_demo_matrix();
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_echelon_info();
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_get_echelon_info_immutable() {
        // cargo test hair_view_should_not_get_echelon_info_immutable -- --nocapture
        let mut matrix = generate_demo_matrix();
        let hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_echelon_info_immutable();
    }

    #[test]
    fn hair_view_unsatisfiable() {
        // cargo test --features=colorful hair_view_unsatisfiable -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
        matrix.add_constraint(3, &[1, 9], false);
        for edge_index in [1, 4, 6, 9] {
            matrix.update_edge_tightness(edge_index, true);
        }
        matrix.printstd();
        assert_eq!(
            matrix.printstd_str(),
            "\
┌──┬─┬─┬─┬─┬───┬─┐
┊ X┊1┊4┊6┊9┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊1┊   ┊4┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊ ┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 3┊ ┊ ┊ ┊ ┊ 1 ┊*┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
        let mut hair_view = HairView::new(&mut matrix, [6, 9].into_iter());
        hair_view.printstd();
        assert_eq!(
            hair_view.printstd_str(),
            "\
┌──┬─┬─┬───┬─┐
┊ X┊6┊9┊ = ┊▼┊
╞══╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊   ┊6┊
├──┼─┼─┼───┼─┤
┊ 1┊ ┊ ┊ 1 ┊*┊
├──┼─┼─┼───┼─┤
┊ ▶┊0┊*┊◀  ┊▲┊
└──┴─┴─┴───┴─┘
"
        );
        assert!(!hair_view.get_echelon_satisfiable());
    }
}
