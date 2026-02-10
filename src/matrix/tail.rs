use super::interface::*;
use super::visualize::*;
use crate::util::*;
use derivative::Derivative;
use crate::dual_module_pq::{EdgeWeak, VertexWeak};

#[derive(Clone, Derivative)]
#[derivative(Default(new = "true"))]
pub struct Tail<M: MatrixView> {
    base: M,
    /// the set of edges that should be placed at the end, if any
    tail_edges: FastIterSet<EdgeWeak>,
    /// var indices are outdated on any changes to the underlying matrix
    #[derivative(Default(value = "true"))]
    is_var_indices_outdated: bool,
    /// the internal store of var indices
    var_indices: Vec<VarIndex>,
    /// internal cache for reducing memory allocation
    tail_var_indices: Vec<VarIndex>,
}

impl<M: MatrixView> Tail<M> {
    pub fn get_base(&self) -> &M {
        &self.base
    }
    pub fn from_base(base: M) -> Self {
        let mut value = Self {
            base,
            tail_edges: FastIterSet::new(),
            is_var_indices_outdated: true,
            var_indices: vec![],
            tail_var_indices: vec![],
        };
        value.var_indices_lazy_update();
        value
    }
}

impl<M: MatrixView> MatrixTail for Tail<M> {
    fn get_tail_edges(&self) -> &FastIterSet<EdgeWeak> {
        &self.tail_edges
    }
    fn get_tail_edges_mut(&mut self) -> &mut FastIterSet<EdgeWeak> {
        self.is_var_indices_outdated = true;
        &mut self.tail_edges
    }
}

impl<M: MatrixTight> MatrixTight for Tail<M> {
    fn update_edge_tightness(&mut self, edge_weak: EdgeWeak, is_tight: bool) {
        self.is_var_indices_outdated = true;
        self.base.update_edge_tightness(edge_weak, is_tight)
    }
    fn is_tight(&self, edge_weak: EdgeWeak) -> bool {
        self.base.is_tight(edge_weak)
    }
    fn get_tight_edges(&self) -> &FastIterSet<EdgeWeak> {
        self.base.get_tight_edges()
    }
}

impl<M: MatrixView> MatrixBasic for Tail<M> {
    fn add_variable(&mut self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.is_var_indices_outdated = true;
        self.base.add_variable(edge_weak.clone())
    }

    fn add_constraint(
        &mut self,
        vertex_weak: VertexWeak,
        incident_edges: &[EdgeWeak],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        self.base.add_constraint(vertex_weak.clone(), incident_edges, parity)
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
    fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeWeak {
        self.get_base().var_to_edge_index(var_index)
    }
    fn edge_to_var_index(&self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.get_base().edge_to_var_index(edge_weak)
    }
    fn get_vertices(&self) -> FastIterSet<VertexWeak> {
        self.get_base().get_vertices()
    }
    fn get_edges(&self) -> FastIterSet<EdgeWeak> {
        self.get_base().get_edges()
    }
}

impl<M: MatrixView> Tail<M> {
    fn force_update_var_indices(&mut self) {
        self.var_indices.clear();
        self.tail_var_indices.clear();
        for column in 0..self.base.columns() {
            let var_index = self.base.column_to_var_index(column);
            let edge_index = self.base.var_to_edge_index(var_index);
            if self.tail_edges.contains(&edge_index) {
                self.tail_var_indices.push(var_index);
            } else {
                self.var_indices.push(var_index);
            }
        }
        self.var_indices.append(&mut self.tail_var_indices)
    }

    fn var_indices_lazy_update(&mut self) {
        if self.is_var_indices_outdated {
            self.force_update_var_indices();
            self.is_var_indices_outdated = false;
        }
    }
}

impl<M: MatrixView> MatrixView for Tail<M> {
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

impl<M: MatrixView> VizTrait for Tail<M> {
    fn viz_table(&mut self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::basic::*;
    use super::super::tight::*;
    use super::*;
    use crate::matrix::basic::tests::{initialize_vertex_edges_for_matrix_testing, edge_vec_from_indices};
    use std::collections::HashSet;
    use crate::dual_module_pq::{EdgePtr, VertexPtr};

    type TailMatrix = Tail<Tight<BasicMatrix>>;

    #[test]
    fn tail_matrix_1() {
        // cargo test --features=colorful tail_matrix_1 -- --nocapture
        let mut matrix = TailMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![
            vec![0, 1, 2],
            vec![1, 3],
            vec![0, 3],
        ];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
       
        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges), true);
        matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges), false);
        matrix.add_constraint(vertices[2].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges), true);
        assert_eq!(matrix.edge_to_var_index(edges[1].downgrade()), Some(1));
        matrix.printstd();
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
        for edge_ptr in edges.iter() {
            matrix.update_edge_tightness(edge_ptr.downgrade(), true);
        }
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
        matrix.set_tail_edges([edges[0].downgrade(), edges[2].downgrade()].into_iter());
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊4┊9┊1┊6┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊ ┊1┊1┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊1┊1┊ ┊ ┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊ ┊1┊1┊ ┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"
        );
        assert_eq!(
            matrix.get_tail_edges_vec().iter().map(|e| e.upgrade_force().read_recursive().edge_index).collect::<HashSet<_>>(), 
            [1, 6].into_iter().collect::<HashSet<_>>());
        assert_eq!(matrix.edge_to_var_index(edges[1].downgrade()), Some(1));
    }

    #[test]
    #[should_panic]
    fn tail_matrix_cannot_call_dirty_column() {
        // cargo test tail_matrix_cannot_call_dirty_column -- --nocapture
        let mut matrix = TailMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9];
        let vertex_incident_edges_vec = vec![
            vec![0, 1, 2],
        ];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges), true);

        matrix.update_edge_tightness(edges[0].downgrade(), true);
        // even though there is indeed such a column, we forbid such dangerous calls
        // always call `columns()` before accessing any column
        matrix.column_to_var_index(0);
    }

    #[test]
    fn tail_matrix_basic_trait() {
        // cargo test --features=colorful tail_matrix_basic_trait -- --nocapture
        let mut matrix = TailMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 6, 9, 3];
        let vertex_incident_edges_vec = vec![
            vec![0, 1, 2],
            vec![1, 3],
            vec![0, 3],
        ];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
       
        matrix.add_variable(edges[4].downgrade()); // untight edges will not show
        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges), true);
        matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges), false);
        matrix.add_constraint(vertices[2].downgrade(), &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges), true);
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
        assert!(matrix.is_tight(edges[0].downgrade()));
        assert_eq!(matrix.edge_to_var_index(edges[1].downgrade()), Some(2));
    }
}
