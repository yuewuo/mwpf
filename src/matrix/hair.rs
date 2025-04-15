//! A hair view is a wrapper on the data that focuses on the hair edges
//!
//! It doesn't introduce new data structure, so the creation is cheap
//!

use super::interface::*;
use super::visualize::*;
use crate::util::*;
use prettytable::*;
use std::collections::*;

use crate::dual_module_pq::{EdgeWeak, VertexPtr, VertexWeak};
use crate::pointers::UnsafePtr;

pub struct HairView<'a, M: MatrixTail + MatrixEchelon> {
    base: &'a mut M,
    column_bias: ColumnIndex,
    row_bias: RowIndex,
}

impl<'a, M: MatrixTail + MatrixEchelon> HairView<'a, M> {
    pub fn get_base(&self) -> &M {
        self.base
    }
    pub fn get_base_view_edges(&mut self) -> Vec<EdgeWeak> {
        self.base.get_view_edges()
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> HairView<'a, M> {
    pub fn new<EdgeIter>(matrix: &'a mut M, hair: EdgeIter) -> Self
    where
        EdgeIter: Iterator<Item = EdgeWeak>,
    {
        matrix.set_tail_edges(hair);
        let columns = matrix.columns();
        let rows = matrix.rows();
        let mut column_bias = columns;
        let mut row_bias = rows;
        for column in (0..columns).rev() {
            let edge_weak = matrix.column_to_edge_weak(column);
            if matrix.get_tail_edges().contains(&edge_weak) {
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
    fn get_tail_edges(&self) -> &BTreeSet<EdgeWeak> {
        self.get_base().get_tail_edges()
    }
    fn get_tail_edges_mut(&mut self) -> &mut BTreeSet<EdgeWeak> {
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
    fn update_edge_tightness(&mut self, _edge_weak: EdgeWeak, _is_tight: bool) {
        panic!("cannot mutate a hair view");
    }
    fn is_tight(&self, edge_weak: EdgeWeak) -> bool {
        self.get_base().is_tight(edge_weak)
    }
    fn get_tight_edges(&self) -> &BTreeSet<EdgeWeak> {
        self.base.get_tight_edges()
    }
}

impl<'a, M: MatrixTail + MatrixEchelon> MatrixBasic for HairView<'a, M> {
    fn add_variable(&mut self, _edge_weak: EdgeWeak) -> Option<VarIndex> {
        panic!("cannot mutate a hair view");
    }

    fn add_constraint(
        &mut self,
        _vertex_weak: VertexWeak,
        _incident_edges: &[EdgeWeak],
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
                    self.column_to_edge_weak(row_info.column - self.column_bias)
                        .upgrade_force()
                        .read_recursive()
                        .edge_index
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
    use crate::dual_module_pq::{EdgePtr, VertexPtr};
    use crate::matrix::basic::tests::{edge_vec_from_indices, initialize_vertex_edges_for_matrix_testing};
    use std::collections::HashSet;

    type EchelonMatrix = Echelon<Tail<Tight<BasicMatrix>>>;

    #[test]
    fn hair_view_simple() {
        // cargo test --features=colorful hair_view_simple -- --nocapture
        let mut matrix = EchelonMatrix::new();
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
        assert_eq!(matrix.edge_to_var_index(edges[1].downgrade()), Some(1));
        for edge_ptr in edges.iter() {
            matrix.update_edge_tightness(edge_ptr.downgrade(), true);
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
        let mut hair_view = HairView::new(&mut matrix, [edges[2].downgrade(), edges[3].downgrade()].into_iter());
        assert_eq!(hair_view.edge_to_var_index(edges[1].downgrade()), Some(1));
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
        let mut hair_view = HairView::new(&mut matrix, [edges[0].downgrade(), edges[2].downgrade()].into_iter());
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
        assert_eq!(
            hair_view
                .get_tail_edges_vec()
                .iter()
                .map(|e| e.upgrade_force().read_recursive().edge_index)
                .collect::<HashSet<_>>(),
            [1, 6].into_iter().collect::<HashSet<_>>()
        );
        assert!(hair_view.is_tight(edges[0].downgrade()));
        assert!(hair_view.get_echelon_satisfiable());
        assert_eq!(
            hair_view
                .get_vertices()
                .iter()
                .map(|v| v.upgrade_force().read_recursive().vertex_index)
                .collect::<HashSet<_>>(),
            [0, 1, 2].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            hair_view
                .get_base_view_edges()
                .iter()
                .map(|e| e.upgrade_force().read_recursive().edge_index)
                .collect::<HashSet<_>>(),
            [4, 9, 1, 6].into_iter().collect::<HashSet<_>>()
        );
    }

    fn generate_demo_matrix(edges: &Vec<EdgePtr>, vertices: &Vec<VertexPtr>) -> EchelonMatrix {
        let mut matrix = EchelonMatrix::new();
        matrix.add_constraint(
            vertices[0].downgrade(),
            &[edges[0].downgrade(), edges[1].downgrade(), edges[2].downgrade()],
            true,
        );
        matrix.add_constraint(vertices[1].downgrade(), &[edges[1].downgrade(), edges[3].downgrade()], false);
        for edge_index in 0..4 {
            matrix.update_edge_tightness(edges[edge_index].downgrade(), true);
        }
        matrix
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_modify_tail_edges() {
        // cargo test hair_view_should_not_modify_tail_edges -- --nocapture
        let vertex_indices = vec![0, 1];
        let edge_indices = vec![1, 4, 6, 9];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_tail_edges_mut();
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_update_edge_tightness() {
        // cargo test hair_view_should_not_update_edge_tightness -- --nocapture
        let vertex_indices = vec![0, 1];
        let edge_indices = vec![1, 4, 6, 9];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.update_edge_tightness(edges[0].downgrade(), false);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_add_variable() {
        // cargo test hair_view_should_not_add_variable -- --nocapture
        let vertex_indices = vec![0, 1];
        let edge_indices = vec![1, 4, 6, 9, 100];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.add_variable(edges[4].downgrade());
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_add_constraint() {
        // cargo test hair_view_should_not_add_constraint -- --nocapture
        let vertex_indices = vec![0, 1, 5];
        let edge_indices = vec![1, 4, 6, 9, 2, 3];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.add_constraint(
            vertices[2].downgrade(),
            &[edges[0].downgrade(), edges[4].downgrade(), edges[5].downgrade()],
            false,
        );
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_xor_row() {
        // cargo test hair_view_should_not_xor_row -- --nocapture
        let vertex_indices = vec![0, 1, 5];
        let edge_indices = vec![1, 4, 6, 9, 2, 3];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.xor_row(0, 1);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_swap_row() {
        // cargo test hair_view_should_not_swap_row -- --nocapture
        let vertex_indices = vec![0, 1, 5];
        let edge_indices = vec![1, 4, 6, 9, 2, 3];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.swap_row(0, 1);
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_get_echelon_info() {
        // cargo test hair_view_should_not_get_echelon_info -- --nocapture
        let vertex_indices = vec![0, 1, 5];
        let edge_indices = vec![1, 4, 6, 9, 2, 3];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);

        let mut matrix = generate_demo_matrix(&edges, &vertices);
        let mut hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_echelon_info();
    }

    #[test]
    #[should_panic]
    fn hair_view_should_not_get_echelon_info_immutable() {
        // cargo test hair_view_should_not_get_echelon_info_immutable -- --nocapture
        let vertex_indices = vec![0, 1, 5];
        let edge_indices = vec![1, 4, 6, 9, 2, 3];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        let mut matrix = generate_demo_matrix(&edges, &vertices);

        let hair_view = HairView::new(&mut matrix, [].into_iter());
        hair_view.get_echelon_info_immutable();
    }

    #[test]
    fn hair_view_unsatisfiable() {
        // cargo test --features=colorful hair_view_unsatisfiable -- --nocapture
        let mut matrix = EchelonMatrix::new();
        let vertex_indices = vec![0, 1, 2, 3];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3], vec![0, 3]];
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
        matrix.add_constraint(
            vertices[3].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[3], &edges),
            false,
        );

        for edge_ptr in edges.iter() {
            matrix.update_edge_tightness(edge_ptr.downgrade(), true);
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
        let mut hair_view = HairView::new(&mut matrix, [edges[2].downgrade(), edges[3].downgrade()].into_iter());
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
