//! Matrix Definition
//!
//! A matrix has a fixed data layout which consists of multiple rows
//!
//! The only operations to change the basic matrix itself are
//! - add variable (column)
//! - add constraint (row)
//! - xor/swap rows
//!
//! Apart from the matrix itself, we can have a view of the matrix:
//! a view is defined as a list of columns `var_indices: Vec<usize>`
//! and a number of rows `rows_count` representing rows `0..rows_count`.
//!
//! A `var_index` is always referring to the original matrix, to improve performance
//! as well as to avoid confusion.
//!
//! Each variable (column) corresponds to an edge in the hypergraph, so we label the
//! columns in the basic matrix. When it comes to operating the matrix, we'll always
//! use the `var_index` to avoid duplicated translation (at least one translation is necessary).
//!

use crate::util::*;
use derivative::Derivative;
use num_traits::{One, Zero};

pub type VarIndex = usize;
pub type RowIndex = usize;
pub type ColumnIndex = usize;

pub trait MatrixBasic {
    /// add an edge to the basic matrix, return the `var_index` if newly created
    fn add_variable(&mut self, edge_index: EdgeIndex) -> Option<VarIndex>;

    /// add constraint will implicitly call `add_variable` if the edge is not added and return the indices of them
    fn add_constraint(
        &mut self,
        vertex_index: VertexIndex,
        incident_edges: &[EdgeIndex],
        parity: bool,
    ) -> Option<Vec<VarIndex>>;

    /// row operations
    fn xor_row(&mut self, target: RowIndex, source: RowIndex);
    fn swap_row(&mut self, a: RowIndex, b: RowIndex);

    /// view the raw matrix
    fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool;
    fn get_rhs(&self, row: RowIndex) -> bool;

    /// get edge index from the var_index
    fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeIndex;

    fn edge_to_var_index(&self, edge_index: EdgeIndex) -> Option<VarIndex>;

    fn exists_edge(&self, edge_index: EdgeIndex) -> bool {
        self.edge_to_var_index(edge_index).is_some()
    }

    fn get_edges(&self) -> FastIterSet<EdgeIndex>;
    fn get_vertices(&self) -> FastIterSet<VertexIndex>;
}

pub trait MatrixView: MatrixBasic {
    /// the number of columns: to get the `var_index` of each column,
    /// use `column_to_var_index()`; here the mutable reference enables
    /// lazy update of the internal data structure
    fn columns(&mut self) -> usize;

    /// get the `var_index` in the basic matrix
    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex;

    fn column_to_edge_index(&self, column: ColumnIndex) -> EdgeIndex {
        let var_index = self.column_to_var_index(column);
        self.var_to_edge_index(var_index)
    }

    /// the number of rows: rows always have indices 0..rows
    fn rows(&mut self) -> usize;

    fn get_view_edges(&mut self) -> Vec<EdgeIndex> {
        (0..self.columns())
            .map(|column: usize| self.column_to_edge_index(column))
            .collect()
    }

    fn var_to_column_index(&mut self, var_index: VarIndex) -> Option<ColumnIndex> {
        (0..self.columns()).find(|&column| self.column_to_var_index(column) == var_index)
    }

    fn edge_to_column_index(&mut self, edge_index: EdgeIndex) -> Option<ColumnIndex> {
        let var_index = self.edge_to_var_index(edge_index)?;
        self.var_to_column_index(var_index)
    }
}

pub trait MatrixTight: MatrixView {
    fn update_edge_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool);
    fn is_tight(&self, edge_index: usize) -> bool;
    fn get_tight_edges(&self) -> &FastIterSet<EdgeIndex>;

    fn add_variable_with_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool) {
        self.add_variable(edge_index);
        self.update_edge_tightness(edge_index, is_tight);
    }

    fn add_tight_variable(&mut self, edge_index: EdgeIndex) {
        self.add_variable_with_tightness(edge_index, true)
    }
}

pub trait MatrixTail {
    fn get_tail_edges(&self) -> &FastIterSet<EdgeIndex>;
    fn get_tail_edges_mut(&mut self) -> &mut FastIterSet<EdgeIndex>;

    fn set_tail_edges<EdgeIter>(&mut self, edges: EdgeIter)
    where
        EdgeIter: Iterator<Item = EdgeIndex>,
    {
        let tail_edges = self.get_tail_edges_mut();
        tail_edges.clear();
        for edge_index in edges {
            tail_edges.insert(edge_index);
        }
    }

    fn get_tail_edges_vec(&self) -> Vec<EdgeIndex> {
        let mut edges: Vec<EdgeIndex> = self.get_tail_edges().iter().cloned().collect();
        edges.sort();
        edges
    }
}

pub trait MatrixEchelonTail {
    fn get_tail_start_index(&mut self) -> Option<ColumnIndex>;
    fn get_corner_row_index(&mut self, tail_start_index: ColumnIndex) -> RowIndex;
}

pub trait MatrixEchelon: MatrixView {
    fn get_echelon_info(&mut self) -> &EchelonInfo;
    fn get_echelon_info_immutable(&self) -> &EchelonInfo;

    fn get_solution(&mut self) -> Option<Subgraph> {
        self.get_echelon_info(); // make sure it's in echelon form
        let info = self.get_echelon_info_immutable();
        if !info.satisfiable {
            return None; // no solution
        }
        let mut solution = vec![];
        for (row, row_info) in info.rows.iter().enumerate() {
            debug_assert!(row_info.has_leading());
            if self.get_rhs(row) {
                let column = row_info.column;
                let edge_index = self.column_to_edge_index(column);
                solution.push(edge_index);
            }
        }
        Some(solution)
    }

    /// try every independent variables and try to minimize the total weight of the solution
    fn get_solution_local_minimum<F>(&mut self, weight_of: F) -> Option<Subgraph>
    where
        F: Fn(EdgeIndex) -> Weight,
    {
        self.get_echelon_info(); // make sure it's in echelon form
        let info = self.get_echelon_info_immutable();
        if !info.satisfiable {
            return None; // no solution
        }
        let mut solution = FastIterSet::new();
        for (row, row_info) in info.rows.iter().enumerate() {
            debug_assert!(row_info.has_leading());
            if self.get_rhs(row) {
                let column = row_info.column;
                let edge_index = self.column_to_edge_index(column);
                solution.insert(edge_index);
            }
        }
        let mut independent_columns = vec![];
        for (column, column_info) in info.columns.iter().enumerate() {
            if !column_info.is_dependent() {
                independent_columns.push(column);
            }
        }
        let mut total_weight = Rational::zero();
        for &edge_index in solution.iter() {
            total_weight += weight_of(edge_index);
        }
        let mut pending_flip_edge_indices = vec![];
        let mut is_local_minimum = false;
        while !is_local_minimum {
            is_local_minimum = true;
            // try every independent variable and find a local minimum
            for &column in independent_columns.iter() {
                pending_flip_edge_indices.clear();
                let var_index = self.column_to_var_index(column);
                let edge_index = self.var_to_edge_index(var_index);
                let mut primal_delta = (weight_of(edge_index))
                    * if solution.contains(&edge_index) {
                        -Rational::one()
                    } else {
                        Rational::one()
                    };
                pending_flip_edge_indices.push(edge_index);
                for row in 0..info.rows.len() {
                    if self.get_lhs(row, var_index) {
                        debug_assert!(info.rows[row].has_leading());
                        let flip_column = info.rows[row].column;
                        debug_assert!(flip_column < column);
                        let flip_edge_index = self.column_to_edge_index(flip_column);
                        primal_delta += (weight_of(flip_edge_index))
                            * if solution.contains(&flip_edge_index) {
                                -Rational::one()
                            } else {
                                Rational::one()
                            };
                        pending_flip_edge_indices.push(flip_edge_index);
                    }
                }
                // warning: has to be this form (instead of .is_negative) to use the tolerance of OrderedFloat
                if primal_delta < Rational::zero() {
                    total_weight = total_weight + primal_delta;
                    for &edge_index in pending_flip_edge_indices.iter() {
                        if solution.contains(&edge_index) {
                            solution.remove(&edge_index);
                        } else {
                            solution.insert(edge_index);
                        }
                    }
                    is_local_minimum = false;
                    break; // loop over again
                }
            }
        }
        Some(solution.into_iter().collect())
    }
}

#[derive(Clone, Debug, Derivative)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct EchelonInfo {
    /// whether it's a satisfiable matrix, only valid when `is_echelon_form` is true
    pub satisfiable: bool,
    /// (is_dependent, if dependent the only "1" position row)
    pub columns: Vec<ColumnInfo>,
    /// the number of effective rows
    pub effective_rows: usize,
    /// the leading "1" position column
    pub rows: Vec<RowInfo>,
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl EchelonInfo {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[derive(Clone, Copy, Derivative, PartialEq, Eq)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct ColumnInfo {
    pub row: RowIndex,
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl ColumnInfo {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    #[pyo3(name = "is_dependent")]
    fn py_is_dependent(&self) -> bool {
        self.row != RowIndex::MAX
    }
}

impl ColumnInfo {
    pub fn not_dependent() -> Self {
        Self { row: RowIndex::MAX }
    }
    pub fn set(&mut self, row: RowIndex) {
        debug_assert!(row != RowIndex::MAX);
        self.row = row;
    }
    pub fn is_dependent(&self) -> bool {
        self.row != RowIndex::MAX
    }
    pub fn set_not_dependent(&mut self) {
        self.row = RowIndex::MAX;
    }
}

impl std::fmt::Debug for ColumnInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if !self.is_dependent() {
            write!(f, "Row(*)")
        } else {
            write!(f, "Row({})", self.row)
        }
    }
}

#[derive(Clone, Copy, Derivative, PartialEq, Eq)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct RowInfo {
    pub column: ColumnIndex,
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl RowInfo {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl RowInfo {
    pub fn no_leading() -> Self {
        Self {
            column: ColumnIndex::MAX,
        }
    }
    pub fn set(&mut self, column: ColumnIndex) {
        debug_assert!(column != ColumnIndex::MAX);
        self.column = column;
    }
    pub fn has_leading(&self) -> bool {
        self.column != ColumnIndex::MAX
    }
    pub fn set_no_leading(&mut self) {
        self.column = ColumnIndex::MAX;
    }
}

impl std::fmt::Debug for RowInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if !self.has_leading() {
            write!(f, "Col(*)")
        } else {
            write!(f, "Col({})", self.column)
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::*;
    use super::*;

    type TightMatrix = Tight<BasicMatrix>;

    #[test]
    fn matrix_interface_simple() {
        // cargo test --features=colorful matrix_interface_simple -- --nocapture
        let mut matrix = TightMatrix::new();
        matrix.add_tight_variable(233);
        matrix.add_tight_variable(14);
        matrix.add_variable(68);
        matrix.add_tight_variable(75);
        matrix.printstd();
        assert_eq!(matrix.get_view_edges(), [233, 14, 75]);
        assert_eq!(matrix.var_to_column_index(0), Some(0));
        assert_eq!(matrix.var_to_column_index(1), Some(1));
        assert_eq!(matrix.var_to_column_index(2), None);
        assert_eq!(matrix.var_to_column_index(3), Some(2));
        assert_eq!(matrix.edge_to_column_index(233), Some(0));
        assert_eq!(matrix.edge_to_column_index(14), Some(1));
        assert_eq!(matrix.edge_to_column_index(68), None);
        assert_eq!(matrix.edge_to_column_index(75), Some(2));
        assert_eq!(matrix.edge_to_column_index(666), None);
    }

    #[test]
    fn matrix_interface_echelon_info() {
        // cargo test matrix_interface_echelon_info -- --nocapture
        let mut column_info = ColumnInfo::new();
        column_info.set(13);
        assert_eq!(format!("{column_info:?}"), "Row(13)");
        column_info.set_not_dependent();
        assert_eq!(format!("{column_info:?}"), "Row(*)");
        assert_eq!(format!("{:?}", column_info.clone()), "Row(*)");
        let mut row_info = RowInfo::new();
        row_info.set(13);
        assert_eq!(format!("{row_info:?}"), "Col(13)");
        row_info.set_no_leading();
        assert_eq!(format!("{row_info:?}"), "Col(*)");
        assert_eq!(format!("{:?}", row_info.clone()), "Col(*)");
        let echelon_info = EchelonInfo::new();
        println!("echelon_info: {echelon_info:?}");
    }

    #[derive(Default)]
    struct TestEdgeWeights {
        pub weights: FastIterMap<EdgeIndex, Weight>,
    }

    impl TestEdgeWeights {
        fn new(weights: &[(EdgeIndex, Weight)]) -> Self {
            let mut result: TestEdgeWeights = Default::default();
            for (edge_index, weight) in weights {
                result.weights.insert(edge_index.clone(), weight.clone());
            }
            result
        }
        fn get_solution_local_minimum(&self, matrix: &mut Echelon<Tail<BasicMatrix>>) -> Option<Subgraph> {
            matrix.get_solution_local_minimum(|edge_index| {
                if let Some(weight) = self.weights.get(&edge_index) {
                    weight.clone()
                } else {
                    Rational::from_float(1.).unwrap()
                }
            })
        }
    }

    #[test]
    fn matrix_interface_echelon_solution() {
        // cargo test --features=colorful matrix_interface_echelon_solution -- --nocapture
        /* 0,1,2: vertices; (0),(1),(2): edges; !n!: odd vertex
               1 (1) 0
              (2)   (0)
         3 (3) 2 (8)!7!
        (4)   (7)   (9)
        !4!(5) 5 (6) 6
            */
        let mut matrix = Echelon::<Tail<BasicMatrix>>::new();
        let parity_checks = vec![
            (vec![0, 1], false),
            (vec![1, 2], false),
            (vec![2, 3, 7, 8], false),
            (vec![3, 4], false),
            (vec![4, 5], true),
            (vec![5, 6, 7], false),
            (vec![6, 9], false),
            (vec![0, 8, 9], true),
        ];
        for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
            matrix.add_constraint(vertex_index, incident_edges, *parity);
        }
        matrix.printstd();
        assert_eq!(matrix.get_solution(), Some(vec![0, 1, 2, 3, 4]));
        let weights = TestEdgeWeights::new(&[
            (3, Rational::from_float(10.).unwrap()),
            (9, Rational::from_float(10.).unwrap()),
        ]);
        assert_eq!(
            sorted_vec_option(weights.get_solution_local_minimum(&mut matrix)),
            Some(vec![5, 7, 8])
        );
        let weights = TestEdgeWeights::new(&[
            (7, Rational::from_float(10.).unwrap()),
            (9, Rational::from_float(10.).unwrap()),
        ]);
        assert_eq!(
            sorted_vec_option(weights.get_solution_local_minimum(&mut matrix)),
            Some(vec![3, 4, 8])
        );
        let weights = TestEdgeWeights::new(&[
            (3, Rational::from_float(10.).unwrap()),
            (4, Rational::from_float(10.).unwrap()),
            (7, Rational::from_float(10.).unwrap()),
        ]);
        assert_eq!(
            sorted_vec_option(weights.get_solution_local_minimum(&mut matrix)),
            Some(vec![5, 6, 9])
        );
    }

    #[test]
    fn matrix_interface_echelon_no_solution() {
        // cargo test matrix_interface_echelon_no_solution -- --nocapture
        let mut matrix = Echelon::<Tail<BasicMatrix>>::new();
        let parity_checks = [(vec![0, 1], false), (vec![0, 1], true)];
        for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
            matrix.add_constraint(vertex_index, incident_edges, *parity);
        }
        assert_eq!(matrix.get_solution(), None);
        let weights = TestEdgeWeights::new(&[]);
        assert_eq!(weights.get_solution_local_minimum(&mut matrix), None);
    }
}
