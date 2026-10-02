//! row-oriented matrix to build a problem constraint by constraint
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::highs::matrix_col::ColMatrix;
use crate::highs::Problem;

/// Represents a variable
#[derive(Debug, Clone, Copy)]
pub struct Col(pub(crate) usize);

/// A complete optimization problem stored by row
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowMatrix {
    /// column-wise sparse constraints  matrix
    /// Each element in the outer vector represents a column (a variable)
    columns: Vec<(Vec<c_int>, Vec<f64>)>,
}

/// Functions to use when first declaring variables, then constraints.
impl Problem<RowMatrix> {
    /// add a variable to the problem.
    ///  - `col_factor` is the coefficient in front of the variable in the objective function.
    ///  - `bounds` are the maximal and minimal values that the variable can take.
    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) -> Col {
        self.add_column_with_integrality(col_factor, bounds, false)
    }

    /// Same as add_column, but forces the solution to contain an integer value for this variable.
    pub fn add_integer_column<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) -> Col {
        self.add_column_with_integrality(col_factor, bounds, true)
    }

    /// Same as add_column, but lets you define whether the new variable should be integral or continuous.
    #[inline]
    pub fn add_column_with_integrality<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
        is_integer: bool,
    ) -> Col {
        let col = Col(self.num_cols());
        self.add_column_inner(col_factor, bounds, is_integer);
        self.matrix.columns.push((vec![], vec![]));
        col
    }

    /// Add a constraint to the problem.
    ///  - `bounds` are the maximal and minimal allowed values for the linear expression in the constraint
    ///  - `row_factors` are the coefficients in the linear expression expressing the constraint
    ///
    /// ```
    /// use mwpf::highs::*;
    /// let mut pb = RowProblem::new();
    /// // Optimize 3x - 2y with x<=6 and y>=5
    /// let x = pb.add_column(3., ..6);
    /// let y = pb.add_column(-2., 5..);
    /// pb.add_row(2.., &[(x, 3.), (y, 8.)]); // 2 <= x*3 + y*8
    /// ```
    pub fn add_row<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Col, f64)>,
        I: IntoIterator<Item=ITEM>,
    >(
        &mut self,
        bounds: B,
        row_factors: I,
    ) {
        let num_rows: c_int = self.num_rows().try_into().expect("too many rows");
        for r in row_factors {
            let &(col, factor) = r.borrow();
            let c = &mut self.matrix.columns[col.0];
            c.0.push(num_rows);
            c.1.push(factor);
        }
        self.add_row_inner(bounds);
    }
}

impl From<RowMatrix> for ColMatrix {
    fn from(m: RowMatrix) -> Self {
        let mut astart = Vec::with_capacity(m.columns.len());
        astart.push(0);
        let size: usize = m.columns.iter().map(|(v, _)| v.len()).sum();
        let mut aindex = Vec::with_capacity(size);
        let mut avalue = Vec::with_capacity(size);
        for (row_indices, factors) in m.columns {
            aindex.extend_from_slice(&row_indices);
            avalue.extend_from_slice(&factors);
            astart.push(aindex.len().try_into().expect("invalid matrix size"));
        }
        Self {
            astart,
            aindex,
            avalue,
        }
    }
}

#[allow(clippy::float_cmp)]
#[test]
fn test_conversion() {
    use crate::highs::status::HighsModelStatus::Optimal;
    use crate::highs::{ColProblem, Model, RowProblem, Sense};
    let inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    let mut p = RowProblem::default();
    let x: Col = p.add_column(1., -1..2);
    let y: Col = p.add_column(9., 4f64..inf);
    p.add_row(-999f64..inf, &[(x, 666.), (y, 777.)]);
    p.add_row(neg_inf..8880f64, &[(y, 888.)]);
    assert_eq!(
        p,
        RowProblem {
            colcost: vec![1., 9.],
            collower: vec![-1., 4.],
            colupper: vec![2., inf],
            rowlower: vec![-999., neg_inf],
            rowupper: vec![inf, 8880.],
            integrality: None,
            matrix: RowMatrix {
                columns: vec![(vec![0], vec![666.]), (vec![0, 1], vec![777., 888.])],
            },
        }
    );
    let colpb = ColProblem::from(p.clone());
    assert_eq!(
        colpb,
        ColProblem {
            colcost: vec![1., 9.],
            collower: vec![-1., 4.],
            colupper: vec![2., inf],
            rowlower: vec![-999., neg_inf],
            rowupper: vec![inf, 8880.],
            integrality: None,
            matrix: ColMatrix {
                astart: vec![0, 1, 3],
                aindex: vec![0, 0, 1],
                avalue: vec![666., 777., 888.],
            },
        }
    );
    let mut m = Model::new(p);
    m.make_quiet();
    m.set_sense(Sense::Maximise);
    let solved = m.solve();
    assert_eq!(solved.status(), Optimal);
    assert_eq!(solved.get_solution().columns(), &[2., 10.]);
}

impl From<Problem<RowMatrix>> for Problem<ColMatrix> {
    fn from(pb: Problem<RowMatrix>) -> Problem<ColMatrix> {
        Self {
            colcost: pb.colcost,
            collower: pb.collower,
            colupper: pb.colupper,
            rowlower: pb.rowlower,
            rowupper: pb.rowupper,
            integrality: pb.integrality,
            matrix: pb.matrix.into(),
        }
    }
}
