//! col-oriented matrix to build a problem variable by variable
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::highs::Problem;

/// Represents a constraint
#[derive(Debug, Clone, Copy)]
pub struct Row(pub(crate) c_int);

/// A constraint matrix to build column-by-column
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColMatrix {
    // column-wise sparse constraints  matrix
    pub(crate) astart: Vec<c_int>,
    pub(crate) aindex: Vec<c_int>,
    pub(crate) avalue: Vec<f64>,
}

/// To use these functions, you need to first add all your constraints, and then add variables
/// one by one using the [Row] objects.
impl Problem<ColMatrix> {
    /// Add a row (a constraint) to the problem.
    /// The concrete factors are added later, when creating columns.
    pub fn add_row<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        self.add_row_inner(bounds)
    }

    /// Add a continuous variable to the problem.
    ///  - `col_factor` represents the factor in front of the variable in the objective function.
    ///  - `bounds` represents the maximal and minimal allowed values of the variable.
    ///  - `row_factors` defines how much this variable weights in each constraint.
    ///
    /// ```
    /// use mwpf::highs::{ColProblem, Sense};
    /// let mut pb = ColProblem::new();
    /// let constraint = pb.add_row(..=5); // adds a constraint that cannot take a value over 5
    /// // add a variable that has a coefficient 2 in the objective function, is >=0, and has a coefficient
    /// // 2 in the constraint
    /// pb.add_column(2., 0.., &[(constraint, 2.)]);
    /// ```
    pub fn add_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality(col_factor, bounds, row_factors, false);
    }

    /// Same as add_column, but forces the solution to contain an integer value for this variable.
    /// 
    /// ```
    /// use mwpf::highs::{ColProblem, Sense};
    /// let mut pb = ColProblem::new();
    /// let constraint = pb.add_row(..=5); // adds a constraint that cannot take a value over 5
    /// // add an integer variable that has a coefficient 2 in the objective function, is >=0, and has a coefficient
    /// // 2 in the constraint
    /// pb.add_integer_column(2., 0.., &[(constraint, 2.)]);
    /// ```
    pub fn add_integer_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality(col_factor, bounds, row_factors, true);
    }

    /// Same as add_column, but lets you define whether the new variable should be integral or continuous.
    #[inline]
    pub fn add_column_with_integrality<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
        is_integer: bool,
    ) {
        self.matrix
            .astart
            .push(self.matrix.aindex.len().try_into().unwrap());
        let iter = row_factors.into_iter();
        let (size, _) = iter.size_hint();
        self.matrix.aindex.reserve(size);
        self.matrix.avalue.reserve(size);
        for r in iter {
            let &(row, factor) = r.borrow();
            self.matrix.aindex.push(row.0);
            self.matrix.avalue.push(factor);
        }
        self.add_column_inner(col_factor, bounds, is_integer);
    }
}
