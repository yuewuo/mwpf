use super::interface::*;
use super::visualize::*;
use crate::util::*;
use core::panic;
use derivative::Derivative;
use prettytable::*;

#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct Echelon<M: MatrixView> {
    base: M,
    /// echelon form is invalidated on any changes to the underlying matrix
    #[derivative(Default(value = "true"))]
    is_info_outdated: bool,
    /// information about the matrix when it's formatted into the Echelon form;
    info: EchelonInfo,
}

impl<M: MatrixView> Echelon<M> {
    pub fn get_base(&self) -> &M {
        &self.base
    }
    pub fn from_base(base: M) -> Self {
        let mut value = Self {
            base,
            is_info_outdated: true,
            info: Default::default(),
        };
        value.echelon_info_lazy_update();
        value
    }
}

impl<M: MatrixTail + MatrixView> MatrixTail for Echelon<M> {
    fn get_tail_edges(&self) -> &FastIterSet<EdgeIndex> {
        self.base.get_tail_edges()
    }
    fn get_tail_edges_mut(&mut self) -> &mut FastIterSet<EdgeIndex> {
        self.is_info_outdated = true;
        self.base.get_tail_edges_mut()
    }
}

impl<M: MatrixTight> MatrixTight for Echelon<M> {
    fn update_edge_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool) {
        self.is_info_outdated = true;
        self.base.update_edge_tightness(edge_index, is_tight)
    }
    fn is_tight(&self, edge_index: usize) -> bool {
        self.base.is_tight(edge_index)
    }
    fn get_tight_edges(&self) -> &FastIterSet<EdgeIndex> {
        self.base.get_tight_edges()
    }
}

impl<M: MatrixView> MatrixBasic for Echelon<M> {
    fn add_variable(&mut self, edge_index: EdgeIndex) -> Option<VarIndex> {
        self.is_info_outdated = true;
        self.base.add_variable(edge_index)
    }

    fn add_constraint(
        &mut self,
        vertex_index: VertexIndex,
        incident_edges: &[EdgeIndex],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        self.is_info_outdated = true;
        self.base.add_constraint(vertex_index, incident_edges, parity)
    }

    fn xor_row(&mut self, _target: RowIndex, _source: RowIndex) {
        panic!("xor operation on an echelon matrix is forbidden");
    }
    fn swap_row(&mut self, _a: RowIndex, _b: RowIndex) {
        panic!("swap operation on an echelon matrix is forbidden");
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

impl<M: MatrixView> Echelon<M> {
    fn force_update_echelon_info(&mut self) {
        let width = self.base.columns();
        let height = self.base.rows();
        if width != self.info.columns.len() {
            self.info.columns.resize_with(width, Default::default);
        }
        if height != self.info.rows.len() {
            self.info.rows.resize_with(height, Default::default);
        }
        if width == 0 {
            // no variable to satisfy any requirement
            // if any RHS=1, it cannot be satisfied
            for row in 0..height {
                if self.base.get_rhs(row) {
                    self.info.satisfiable = false;
                    self.base.swap_row(0, row); // make it the first row
                    self.info.effective_rows = 1;
                    self.info.rows.truncate(1);
                    self.info.rows[0].set_no_leading();
                    return;
                }
            }
            self.info.satisfiable = true;
            self.info.effective_rows = 0;
            self.info.rows.truncate(0);
            return;
        }
        // prepare info
        self.info.satisfiable = false;
        let mut lead = 0;
        for r in 0..height {
            if lead >= width {
                // no more variables
                self.info.satisfiable = (r..height).all(|row| !self.base.get_rhs(row));
                if self.info.satisfiable {
                    self.info.effective_rows = r;
                    self.info.rows.truncate(r);
                    return;
                }
                // find an unsatisfiable row with rhs=1 and make it the row[r]
                for row in r..height {
                    if self.base.get_rhs(row) {
                        self.base.swap_row(r, row);
                        break;
                    }
                }
                debug_assert!(self.base.get_rhs(r));
                debug_assert!(!self.info.satisfiable);
                self.info.effective_rows = r + 1;
                self.info.rows.truncate(r + 1);
                self.info.rows[r].set_no_leading();
                return;
            }
            let mut i = r;
            // find first non-zero lead and make it the row[r]
            while !self.base.get_lhs(i, self.base.column_to_var_index(lead)) {
                i += 1;
                if i == height {
                    i = r;
                    // couldn't find a leading 1 in this column, indicating this variable is an independent variable
                    self.info.columns[lead].set_not_dependent();
                    lead += 1; // consider the next lead
                    if lead >= width {
                        self.info.satisfiable = (r..height).all(|row| !self.base.get_rhs(row));
                        if self.info.satisfiable {
                            self.info.effective_rows = r;
                            self.info.rows.truncate(r);
                            return;
                        }
                        // find a row with rhs=1 and swap with r row
                        for row in r..height {
                            if self.base.get_rhs(row) {
                                self.base.swap_row(r, row);
                                break;
                            }
                        }
                        debug_assert!(self.base.get_rhs(r));
                        debug_assert!(!self.info.satisfiable);
                        self.info.effective_rows = r + 1;
                        self.info.rows.truncate(r + 1);
                        self.info.rows[r].set_no_leading();
                        return;
                    }
                }
            }
            if i != r {
                // implies r < i
                self.base.swap_row(r, i);
            }
            for j in 0..height {
                if j != r && self.base.get_lhs(j, self.base.column_to_var_index(lead)) {
                    self.base.xor_row(j, r);
                }
            }
            self.info.rows[r].set(lead);
            self.info.columns[lead].set(r);
            self.info.effective_rows = r + 1;
            lead += 1;
        }
        while lead < width {
            self.info.columns[lead].set_not_dependent();
            lead += 1;
        }
        self.info.rows.truncate(self.info.effective_rows);
        self.info.satisfiable = true;
    }

    fn echelon_info_lazy_update(&mut self) {
        if self.is_info_outdated {
            self.force_update_echelon_info();
            self.is_info_outdated = false;
        }
    }
}

impl<M: MatrixView> MatrixEchelon for Echelon<M> {
    fn get_echelon_info(&mut self) -> &EchelonInfo {
        self.echelon_info_lazy_update();
        &self.info
    }
    fn get_echelon_info_immutable(&self) -> &EchelonInfo {
        debug_assert!(!self.is_info_outdated, "call `get_echelon_info` first");
        &self.info
    }
}

impl<M: MatrixView + MatrixTail> MatrixEchelonTail for Echelon<M> {
    fn get_tail_start_index(&mut self) -> Option<ColumnIndex> {
        self.get_tail_edges()
            .clone()
            .into_iter()
            .flat_map(|edge_index| self.edge_to_column_index(edge_index))
            .min()
    }
    /// pass `tail_start_index` by calling `get_tail_start_index`
    fn get_corner_row_index(&mut self, tail_start_index: ColumnIndex) -> RowIndex {
        let echelon_info = self.get_echelon_info();
        let ColumnInfo { row } = echelon_info.columns[tail_start_index];
        row
    }
}

impl<M: MatrixView> MatrixView for Echelon<M> {
    fn columns(&mut self) -> usize {
        self.echelon_info_lazy_update();
        self.base.columns()
    }

    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        debug_assert!(!self.is_info_outdated, "call `columns` first");
        self.base.column_to_var_index(column)
    }

    fn rows(&mut self) -> usize {
        self.echelon_info_lazy_update();
        self.info.effective_rows
    }
}

impl<M: MatrixView> VizTrait for Echelon<M> {
    fn viz_table(&mut self) -> VizTable {
        let mut table = VizTable::from(&mut *self);
        let info: &EchelonInfo = self.get_echelon_info_immutable();
        assert_eq!(info.rows.len(), info.effective_rows);
        // add echelon mark
        assert!(table.title.get_cell(0).unwrap().get_content().is_empty());
        if info.satisfiable {
            table.title.set_cell(Cell::new("E").style_spec("brFg"), 0).unwrap();
        } else {
            table.title.set_cell(Cell::new("X").style_spec("brFr"), 0).unwrap();
        }
        assert_eq!(table.title.len(), info.columns.len() + 2);
        assert_eq!(table.rows.len(), info.effective_rows);
        // add row information on the right
        table.title.add_cell(Cell::new("\u{25BC}"));
        for (row, row_info) in info.rows.iter().enumerate() {
            let cell = if row_info.has_leading() {
                Cell::new(self.column_to_edge_index(row_info.column).to_string().as_str()).style_spec("irFm")
            } else {
                Cell::new("*").style_spec("rFr")
            };
            table.rows[row].add_cell(cell);
        }
        // add column information on the bottom
        let mut bottom_row = Row::empty();
        bottom_row.add_cell(Cell::new(" \u{25B6}"));
        for column_info in info.columns.iter() {
            let cell = if column_info.is_dependent() {
                Cell::new(VizTable::force_single_column(column_info.row.to_string().as_str()).as_str()).style_spec("irFb")
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
    use super::super::tail::*;
    use super::super::tight::*;
    use super::*;
    use crate::rand::{Rng, SeedableRng};

    type EchelonMatrix = Echelon<Tail<Tight<BasicMatrix>>>;

    #[test]
    fn echelon_matrix_simple() {
        // cargo test --features=colorful echelon_matrix_simple -- --nocapture
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
        matrix.set_tail_edges([6, 1].into_iter());
        assert_eq!(matrix.get_tail_edges_vec(), [1, 6]);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
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
        matrix.set_tail_edges([4].into_iter());
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊1┊6┊9┊4┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊ ┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊1┊   ┊9┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
        matrix.update_edge_tightness(6, false);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬─┬─┬─┬───┬─┐
┊ E┊1┊9┊4┊ = ┊▼┊
╞══╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊1┊   ┊9┊
├──┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊*┊◀  ┊▲┊
└──┴─┴─┴─┴───┴─┘
"
        );
        matrix.update_edge_tightness(1, false);
        matrix.update_edge_tightness(9, false);
        matrix.printstd();
    }

    #[test]
    #[should_panic]
    fn echelon_matrix_should_not_xor() {
        // cargo test echelon_matrix_should_not_xor -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.xor_row(0, 1);
    }

    #[test]
    #[should_panic]
    fn echelon_matrix_should_not_swap() {
        // cargo test echelon_matrix_should_not_swap -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.swap_row(0, 1);
    }

    #[test]
    fn echelon_matrix_basic_trait() {
        // cargo test --features=colorful echelon_matrix_basic_trait -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_variable(3); // un-tight edges will not show
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.add_constraint(1, &[4, 9], false);
        matrix.add_constraint(2, &[1, 9], true);
        for edge_index in [1, 4, 6, 9] {
            matrix.update_edge_tightness(edge_index, true);
        }
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
        assert!(matrix.is_tight(1));
        assert_eq!(matrix.edge_to_var_index(4), Some(2));
    }

    #[test]
    #[should_panic]
    fn echelon_matrix_cannot_call_dirty_column() {
        // cargo test echelon_matrix_cannot_call_dirty_column -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.update_edge_tightness(1, true);
        // even though there is indeed such a column, we forbid such dangerous calls
        // always call `columns()` before accessing any column
        matrix.column_to_var_index(0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn echelon_matrix_cannot_call_dirty_echelon_info() {
        // cargo test echelon_matrix_cannot_call_dirty_echelon_info -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.update_edge_tightness(1, true);
        // even though there is indeed such a column, we forbid such dangerous calls
        // always call `columns()` before accessing any column
        matrix.get_echelon_info_immutable();
    }

    #[test]
    fn echelon_matrix_no_constraint() {
        // cargo test --features=colorful echelon_matrix_no_constraint -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬───┬─┐
┊ E┊ = ┊▼┊
╞══╪═══╪═╡
┊ ▶┊◀  ┊▲┊
└──┴───┴─┘
"
        );
        let info = matrix.get_echelon_info();
        assert!(info.satisfiable);
        assert_eq!(info.rows, []);
        assert_eq!(info.columns, []);
        assert_eq!(info.effective_rows, 0);
    }

    #[test]
    fn echelon_matrix_no_variable_satisfiable() {
        // cargo test --features=colorful echelon_matrix_no_variable_satisfiable -- --nocapture
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], false);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬───┬─┐
┊ E┊ = ┊▼┊
╞══╪═══╪═╡
┊ ▶┊◀  ┊▲┊
└──┴───┴─┘
"
        );
        let info = matrix.get_echelon_info();
        assert!(info.satisfiable);
        assert_eq!(info.rows, []);
        assert_eq!(info.columns, []);
        assert_eq!(info.effective_rows, 0);
    }

    #[test]
    fn echelon_matrix_no_variable_unsatisfiable() {
        // cargo test --features=colorful echelon_matrix_no_variable_unsatisfiable -- --nocapture
        let mut matrix: Echelon<Tail<Tight<BasicMatrix>>> = EchelonMatrix::new();
        matrix.add_constraint(0, &[1, 4, 6], true);
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬───┬─┐
┊ X┊ = ┊▼┊
╞══╪═══╪═╡
┊ 0┊ 1 ┊*┊
├──┼───┼─┤
┊ ▶┊◀  ┊▲┊
└──┴───┴─┘
"
        );
        let info = matrix.get_echelon_info();
        assert!(!info.satisfiable);
        assert_eq!(info.rows, vec![RowInfo::no_leading()]);
        assert_eq!(info.columns, []);
        assert_eq!(info.effective_rows, 1);
    }

    #[test]
    fn echelon_matrix_no_more_variable_satisfiable() {
        // cargo test --features=colorful echelon_matrix_no_more_variable_satisfiable -- --nocapture
        let mut matrix: Echelon<Tail<Tight<BasicMatrix>>> = EchelonMatrix::new();
        matrix.add_constraint(0, &[0, 1], true);
        matrix.add_constraint(1, &[1, 2], true);
        matrix.add_constraint(2, &[2, 3], true);
        matrix.add_constraint(3, &[3, 1], false);
        for edge_index in [0, 1, 2, 3] {
            matrix.update_edge_tightness(edge_index, true);
        }
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊0┊1┊2┊3┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊0┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊1┊   ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊1┊ 1 ┊2┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
    }

    #[test]
    fn echelon_matrix_no_more_variable_unsatisfiable() {
        // cargo test --features=colorful echelon_matrix_no_more_variable_satisfiable -- --nocapture
        let mut matrix: Echelon<Tail<Tight<BasicMatrix>>> = EchelonMatrix::new();
        matrix.add_constraint(0, &[0, 1], true);
        matrix.add_constraint(1, &[1, 2], true);
        matrix.add_constraint(2, &[2, 3], true);
        matrix.add_constraint(3, &[3, 1], true);
        for edge_index in [0, 1, 2, 3] {
            matrix.update_edge_tightness(edge_index, true);
        }
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌──┬─┬─┬─┬─┬───┬─┐
┊ X┊0┊1┊2┊3┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊0┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊1┊   ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊1┊ 1 ┊2┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 3┊ ┊ ┊ ┊ ┊ 1 ┊*┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"
        );
    }

    fn generate_random_parity_checks(
        rng: &mut DeterministicRng,
        variable_count: usize,
        constraint_count: usize,
    ) -> Vec<(Vec<EdgeIndex>, bool)> {
        let mut parity_checks = vec![];
        for _ in 0..constraint_count {
            let rhs: bool = rng.gen();
            let lhs = (0..variable_count).filter(|_| rng.gen()).collect();
            parity_checks.push((lhs, rhs))
        }
        parity_checks
    }

    struct YetAnotherRowEchelon {
        matrix: Vec<Vec<bool>>,
    }

    impl YetAnotherRowEchelon {
        fn new(echelon: &EchelonMatrix) -> Self {
            let mut matrix = vec![];
            let base = echelon.get_base().get_base().get_base();
            for row in 0..base.constraints.len() {
                let mut matrix_row = vec![];
                for var_index in 0..base.variables.len() {
                    matrix_row.push(base.get_lhs(row, var_index));
                }
                matrix_row.push(base.get_rhs(row));
                matrix.push(matrix_row);
            }
            Self { matrix }
        }

        // https://rosettacode.org/wiki/Reduced_row_echelon_form#Rust
        fn reduced_row_echelon_form(&mut self) {
            let matrix = &mut self.matrix;
            let mut pivot = 0;
            let row_count = matrix.len();
            if row_count == 0 {
                return;
            }
            let column_count = matrix[0].len();
            'outer: for r in 0..row_count {
                if column_count <= pivot {
                    break;
                }
                let mut i = r;
                while !matrix[i][pivot] {
                    i += 1;
                    if i == row_count {
                        i = r;
                        pivot += 1;
                        if column_count == pivot {
                            break 'outer;
                        }
                    }
                }
                for j in 0..column_count {
                    let temp = matrix[r][j];
                    matrix[r][j] = matrix[i][j];
                    matrix[i][j] = temp;
                }
                for i in 0..row_count {
                    if i != r && matrix[i][pivot] {
                        for k in 0..column_count {
                            matrix[i][k] ^= matrix[r][k];
                        }
                    }
                }
                pivot += 1;
            }
        }

        fn print(&self) {
            for row in self.matrix.iter() {
                for &e in row.iter() {
                    print!("{}", if e { 1 } else { 0 });
                }
                println!();
            }
        }

        fn is_satisfiable(&self) -> bool {
            'outer: for row in self.matrix.iter() {
                if row[row.len() - 1] {
                    for &e in row.iter().take(row.len() - 1) {
                        if e {
                            continue 'outer;
                        }
                    }
                    return false;
                }
            }
            true
        }

        fn effective_rows(&self) -> usize {
            let mut effective_rows = 0;
            for (i, row) in self.matrix.iter().enumerate() {
                for &e in row.iter() {
                    if e {
                        effective_rows = i + 1;
                    }
                }
            }
            effective_rows
        }

        fn assert_eq(&self, echelon: &EchelonMatrix) {
            let satisfiable = self.is_satisfiable();
            assert_eq!(satisfiable, echelon.info.satisfiable);
            if !satisfiable {
                // assert effective_rows is the line where it fails
                let row = echelon.info.effective_rows - 1;
                for column in 0..echelon.get_base().get_base().get_base().variables.len() {
                    assert_eq!(column, echelon.column_to_var_index(column));
                    assert!(!echelon.get_lhs(row, column))
                }
                assert!(echelon.get_rhs(row));
                return;
            }
            let effective_rows = self.effective_rows();
            assert_eq!(echelon.info.effective_rows, effective_rows);
            for (i, row) in self.matrix.iter().enumerate() {
                assert_eq!(echelon.get_base().get_base().get_base().variables.len(), row.len() - 1);
                for (j, &e) in row.iter().enumerate() {
                    if j < row.len() - 1 {
                        assert_eq!(e, echelon.get_lhs(i, j));
                    } else {
                        assert_eq!(e, echelon.get_rhs(i));
                    }
                }
                if i >= echelon.info.effective_rows {
                    // any row below the effective ones are totally zero
                    for j in 0..row.len() - 1 {
                        assert!(!echelon.get_lhs(i, j));
                    }
                    assert!(!echelon.get_rhs(i));
                } else {
                    // an effective row must contain at least one 1
                    let any_one = (0..row.len() - 1).fold(false, |acc, j| acc | echelon.get_lhs(i, j));
                    assert!(any_one | echelon.get_rhs(i));
                }
            }
            // check column and row information
            let mut column_info: Vec<_> = (0..echelon.get_base().get_base().get_base().variables.len())
                .map(|_| ColumnInfo::not_dependent())
                .collect();
            let mut row_info: Vec<_> = (0..echelon.info.effective_rows).map(|_| RowInfo::no_leading()).collect();
            for (i, row) in row_info.iter_mut().enumerate() {
                for (j, column) in column_info.iter_mut().enumerate() {
                    if echelon.get_lhs(i, j) {
                        assert!(!column.is_dependent());
                        column.row = i;
                        row.column = j;
                        break;
                    }
                }
            }
            for (j, &column) in column_info.iter().enumerate() {
                assert_eq!(column, echelon.info.columns[j]);
            }
            for (i, &row) in row_info.iter().enumerate() {
                assert_eq!(row, echelon.info.rows[i]);
            }
            // check row information
        }
    }

    #[test]
    fn echelon_matrix_another_echelon_simple() {
        // cargo test --features=colorful echelon_matrix_another_echelon_simple -- --nocapture
        let mut echelon = EchelonMatrix::new();
        for edge_index in 0..7 {
            echelon.add_tight_variable(edge_index);
        }
        echelon.add_constraint(0, &[0, 1], true);
        echelon.add_constraint(1, &[0, 2], false);
        echelon.add_constraint(2, &[2, 3, 5], false);
        echelon.add_constraint(3, &[1, 3, 4], false);
        echelon.add_constraint(4, &[4, 6], false);
        echelon.add_constraint(5, &[5, 6], true);
        let mut another = YetAnotherRowEchelon::new(&echelon);
        another.print();
        // both go to echelon form
        echelon.printstd();
        another.reduced_row_echelon_form();
        another.print();
        another.assert_eq(&echelon);
    }

    #[test]
    fn echelon_matrix_another_random_tests() {
        // cargo test --features=colorful echelon_matrix_another_random_tests -- --nocapture
        // cargo test --release echelon_matrix_another_random_tests -- --nocapture
        let mut rng = DeterministicRng::seed_from_u64(123);
        let repeat = 50;
        for variable_count in 0..31 {
            for constraint_count in 0..31 {
                for _ in 0..repeat {
                    let mut echelon = EchelonMatrix::new();
                    for edge_index in 0..variable_count {
                        echelon.add_tight_variable(edge_index);
                    }
                    let parity_checks = generate_random_parity_checks(&mut rng, variable_count, constraint_count);
                    // println!("variable_count: {variable_count}, parity_checks: {parity_checks:?}");
                    for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
                        echelon.add_constraint(vertex_index, incident_edges, *parity);
                    }
                    let mut another = YetAnotherRowEchelon::new(&echelon);
                    // echelon.printstd();
                    if variable_count > 0 {
                        another.reduced_row_echelon_form();
                        echelon.echelon_info_lazy_update();
                        // echelon.printstd();
                        // another.print();
                        another.assert_eq(&echelon);
                    }
                }
            }
        }
    }

    fn debug_echelon_matrix_case(variable_count: usize, parity_checks: Vec<(Vec<usize>, bool)>) -> EchelonMatrix {
        let mut echelon = EchelonMatrix::new();
        for edge_index in 0..variable_count {
            echelon.add_tight_variable(edge_index);
        }
        for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
            echelon.add_constraint(vertex_index, incident_edges, *parity);
        }
        echelon
    }

    /// panicked at 'index out of bounds: the len is 0 but the index is 0', src/matrix/echelon_matrix.rs:148:13
    #[test]
    fn echelon_matrix_debug_1() {
        // cargo test --features=colorful echelon_matrix_debug_1 -- --nocapture
        let parity_checks = vec![(vec![0], true), (vec![0, 1], true), (vec![], true)];
        let mut echelon = debug_echelon_matrix_case(2, parity_checks);
        echelon.printstd();
        assert_eq!(
            echelon.printstd_str(),
            "\
┌──┬─┬─┬───┬─┐
┊ X┊0┊1┊ = ┊▼┊
╞══╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ 1 ┊0┊
├──┼─┼─┼───┼─┤
┊ 1┊ ┊1┊   ┊1┊
├──┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊ 1 ┊*┊
├──┼─┼─┼───┼─┤
┊ ▶┊0┊1┊◀  ┊▲┊
└──┴─┴─┴───┴─┘
"
        );
    }

    #[test]
    fn echelon_matrix_debug_2() {
        // cargo test --features=colorful echelon_matrix_debug_2 -- --nocapture
        let parity_checks = vec![];
        let mut echelon = debug_echelon_matrix_case(1, parity_checks);
        echelon.printstd();
        assert_eq!(
            echelon.printstd_str(),
            "\
┌──┬─┬───┬─┐
┊ E┊0┊ = ┊▼┊
╞══╪═╪═══╪═╡
┊ ▶┊*┊◀  ┊▲┊
└──┴─┴───┴─┘
"
        );
    }
}
