#![forbid(missing_docs)]
//! Safe rust binding to the [HiGHS](https://highs.dev) linear programming solver.
//!
//! ## Usage example
//!
//! ### Building a problem constraint by constraint with [RowProblem]
//!
//! Useful for traditional problem modelling where you first declare your variables, then add
//!constraints one by one.
//!
//! ```
//! use mwpf::highs::{Sense, Model, HighsModelStatus, RowProblem};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = RowProblem::default();
//! // Create a variable named x, with a coefficient of 1 in the objective function,
//! // that is bound between 0 and +∞.
//! let x = pb.add_column(1., 0..);
//! let y = pb.add_column(2., 0..);
//! let z = pb.add_column(1., 0..);
//! // constraint c1: x*3 + y*1 is bound to ]-∞; 6]
//! pb.add_row(..=6, &[(x, 3.), (y, 1.)]);
//! // constraint c2: y*1 +  z*2 is bound to ]-∞; 7]
//! pb.add_row(..=7, &[(y, 1.), (z, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
//!
//! ### Building a problem variable by variable with [ColProblem]
//!
//! Useful for resource allocation problems and other problems when you know in advance the number
//! of constraints and their bounds, but dynamically add new variables to the problem.
//!
//! This is slightly more efficient than building the problem constraint by constraint.
//!
//! ```
//! use mwpf::highs::{ColProblem, Sense};
//! let mut pb = ColProblem::new();
//! // We cannot use more then 5 units of sugar in total.
//! let sugar = pb.add_row(..=5);
//! // We cannot use more then 3 units of milk in total.
//! let milk = pb.add_row(..=3);
//! // We have a first cake that we can sell for 2€. Baking it requires 1 unit of milk and 2 of sugar.
//! pb.add_integer_column(2., 0.., &[(sugar, 2.), (milk, 1.)]);
//! // We have a second cake that we can sell for 8€. Baking it requires 2 units of milk and 3 of sugar.
//! pb.add_integer_column(8., 0.., &[(sugar, 3.), (milk, 2.)]);
//! // Find the maximal possible profit
//! let solution = pb.optimise(Sense::Maximise).solve().get_solution();
//! // The solution is to bake 1 cake of each sort
//! assert_eq!(solution.columns(), vec![1., 1.]);
//! ```
//!
//! ```
//! use mwpf::highs::{Sense, Model, HighsModelStatus, ColProblem};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = ColProblem::default();
//! let c1 = pb.add_row(..6.);
//! let c2 = pb.add_row(..7.);
//! // x
//! pb.add_column(1., 0.., &[(c1, 3.)]);
//! // y
//! pb.add_column(2., 0.., &[(c1, 1.), (c2, 1.)]);
//! // z
//! pb.add_column(1., 0.., vec![(c2, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
//!
//! ### Integer variables
//!
//! HiGHS supports mixed integer-linear programming.
//! You can use `add_integer_column` to add an integer variable to the problem,
//! and the solution is then guaranteed to contain a whole number as a value for this variable.
//!
//! ```
//! use mwpf::highs::{Sense, Model, HighsModelStatus, ColProblem};
//! // maximize: x + 2y under constraints x + y <= 3.5 and x - y >= 1
//! let mut pb = ColProblem::default();
//! let c1 = pb.add_row(..3.5);
//! let c2 = pb.add_row(1..);
//! // x (continuous variable)
//! pb.add_column(1., 0.., &[(c1, 1.), (c2, 1.)]);
//! // y (integer variable)
//! pb.add_integer_column(2., 0.., &[(c1, 1.), (c2, -1.)]);
//! let solved = pb.optimise(Sense::Maximise).solve();
//! // The expected solution is x=2.5  y=1
//! assert_eq!(solved.get_solution().columns(), vec![2.5, 1.]);
//! ```

use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void, CString};
use std::num::TryFromIntError;
use std::ops::{Bound, Index, RangeBounds};
use std::os::raw::c_int;

use highs_sys::*;

pub use matrix_col::{ColMatrix, Row};
pub use matrix_row::{Col, RowMatrix};
pub use status::{HighsModelStatus, HighsStatus};

use crate::highs::options::HighsOptionValue;

/// A problem where variables are declared first, and constraints are then added dynamically.
/// See [`Problem<RowMatrix>`](Problem#impl-1).
pub type RowProblem = Problem<RowMatrix>;
/// A problem where constraints are declared first, and variables are then added dynamically.
/// See [`Problem<ColMatrix>`](Problem#impl).
pub type ColProblem = Problem<ColMatrix>;

mod matrix_col;
mod matrix_row;
mod options;
mod status;

/// A complete optimization problem.
/// Depending on the `MATRIX` type parameter, the problem will be built
/// constraint by constraint (with [ColProblem]), or
/// variable by variable (with [RowProblem])
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem<MATRIX = ColMatrix> {
    // columns
    colcost: Vec<f64>,
    collower: Vec<f64>,
    colupper: Vec<f64>,
    // rows
    rowlower: Vec<f64>,
    rowupper: Vec<f64>,
    integrality: Option<Vec<HighsInt>>,
    matrix: MATRIX,
}

impl<MATRIX: Default> Problem<MATRIX>
where
    Problem<ColMatrix>: From<Problem<MATRIX>>,
{
    /// Number of variables in the problem
    pub fn num_cols(&self) -> usize {
        self.colcost.len()
    }

    /// Number of constraints in the problem
    pub fn num_rows(&self) -> usize {
        self.rowlower.len()
    }

    fn add_row_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        let r = Row(self.num_rows().try_into().expect("too many rows"));
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.rowlower.push(low);
        self.rowupper.push(high);
        r
    }

    fn add_column_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, col_factor: f64, bounds: B, is_integral: bool) {
        if is_integral && self.integrality.is_none() {
            self.integrality = Some(vec![0; self.num_cols()]);
        }
        if let Some(integrality) = &mut self.integrality {
            integrality.push(if is_integral { 1 } else { 0 });
        }
        self.colcost.push(col_factor);
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.collower.push(low);
        self.colupper.push(high);
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// If the problem is invalid (according to HiGHS), this function will panic.
    pub fn optimise(self, sense: Sense) -> Model {
        self.try_optimise(sense).expect("invalid problem")
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    pub fn try_optimise(self, sense: Sense) -> Result<Model, HighsStatus> {
        let mut m = Model::try_new(self)?;
        m.set_sense(sense);
        Ok(m)
    }

    /// Create a new problem instance
    pub fn new() -> Self {
        Self::default()
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> HighsInt {
    n.try_into().expect("size too large for HiGHS")
}

macro_rules! highs_call {
    ($function_name:ident ($($param:expr),+)) => {
        try_handle_status(
            $function_name($($param),+),
            stringify!($function_name)
        )
    }
}

/// A model to solve
#[derive(Debug)]
pub struct Model {
    highs: HighsPtr,
}

/// A solved model
#[derive(Debug)]
pub struct SolvedModel {
    highs: HighsPtr,
}

/// Whether to maximize or minimize the objective function
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// max
    Maximise = OBJECTIVE_SENSE_MAXIMIZE as isize,
    /// min
    Minimise = OBJECTIVE_SENSE_MINIMIZE as isize,
}

impl Model {
    /// Set the optimization sense (minimize by default)
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe { Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int) };
        assert_eq!(ret, STATUS_OK, "changeObjectiveSense failed");
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Panics if the problem is incoherent
    pub fn new<P: Into<Problem<ColMatrix>>>(problem: P) -> Self {
        Self::try_new(problem).expect("incoherent problem")
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    /// Returns an error if the problem is incoherent
    pub fn try_new<P: Into<Problem<ColMatrix>>>(problem: P) -> Result<Self, HighsStatus> {
        let mut highs = HighsPtr::default();
        highs.make_quiet();
        let problem = problem.into();
        log::debug!(
            "Adding a problem with {} variables and {} constraints to HiGHS",
            problem.num_cols(),
            problem.num_rows()
        );
        let offset = 0.0;
        unsafe {
            if let Some(integrality) = &problem.integrality {
                highs_call!(Highs_passMip(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr(),
                    integrality.as_ptr()
                ))
            } else {
                highs_call!(Highs_passLp(
                    highs.mut_ptr(),
                    c(problem.num_cols()),
                    c(problem.num_rows()),
                    c(problem.matrix.avalue.len()),
                    MATRIX_FORMAT_COLUMN_WISE,
                    OBJECTIVE_SENSE_MINIMIZE,
                    offset,
                    problem.colcost.as_ptr(),
                    problem.collower.as_ptr(),
                    problem.colupper.as_ptr(),
                    problem.rowlower.as_ptr(),
                    problem.rowupper.as_ptr(),
                    problem.matrix.astart.as_ptr(),
                    problem.matrix.aindex.as_ptr(),
                    problem.matrix.avalue.as_ptr()
                ))
            }
            .map(|_| Self { highs })
        }
    }

    /// Prevents writing anything to the standard output or to files when solving the model
    pub fn make_quiet(&mut self) {
        self.highs.make_quiet()
    }

    /// Set a custom parameter on the model.
    /// For the list of available options and their documentation, see:
    /// <https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html>
    ///
    /// ```
    /// # use mwpf::highs::ColProblem;
    /// # use mwpf::highs::Sense::Maximise;
    /// let mut model = ColProblem::default().optimise(Maximise);
    /// model.set_option("presolve", "off"); // disable the presolver
    /// model.set_option("solver", "ipm"); // use the ipm solver
    /// model.set_option("time_limit", 30.0); // stop after 30 seconds
    /// model.set_option("parallel", "on"); // use multiple cores
    /// model.set_option("threads", 4); // solve on 4 threads
    /// ```
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        self.highs.set_option(option, value)
    }

    /// Find the optimal value for the problem, panic if the problem is incoherent
    pub fn solve(self) -> SolvedModel {
        self.try_solve().expect("HiGHS error: invalid problem")
    }

    /// Find the optimal value for the problem, return an error if the problem is incoherent
    pub fn try_solve(mut self) -> Result<SolvedModel, HighsStatus> {
        unsafe { highs_call!(Highs_run(self.highs.mut_ptr())) }.map(|_| SolvedModel { highs: self.highs })
    }

    /// Changes the bounds of a row.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn change_row_bounds(&mut self, row: Row, bounds: impl RangeBounds<f64>) {
        self.try_change_row_bounds(row, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the bounds of a row in the highs model.
    ///
    /// Returns Ok(()), or the error status value if HIGHS returned an error status.
    pub fn try_change_row_bounds(&mut self, row: Row, bounds: impl RangeBounds<f64>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeRowBounds(
                self.highs.mut_ptr(),
                row.0.try_into().unwrap(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)
            ))?;
        }
        Ok(())
    }

    /// Changes the bounds of a row.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn change_col_bounds(&mut self, col: Col, bounds: impl RangeBounds<f64>) {
        self.try_change_col_bounds(col, bounds)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change the bounds of a row in the highs model.
    ///
    /// Returns Ok(()), or the error status value if HIGHS returned an error status.
    pub fn try_change_col_bounds(&mut self, col: Col, bounds: impl RangeBounds<f64>) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeColBounds(
                self.highs.mut_ptr(),
                col.0.try_into().unwrap(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY)
            ))?;
        }
        Ok(())
    }

    /// Change a coefficient in the constraint matrix.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn change_matrix_coefficient(&mut self, row: Row, col: Col, value: f64) {
        self.try_change_matrix_coefficient(row, col, value)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to change a coefficient in the constraint matrix.
    ///
    /// Returns Ok(()), or the error status value if HIGHS returned an error status.
    pub fn try_change_matrix_coefficient(&mut self, row: Row, col: Col, value: f64) -> Result<(), HighsStatus> {
        unsafe {
            highs_call!(Highs_changeCoeff(
                self.highs.mut_ptr(),
                row.0.try_into().unwrap(),
                col.0.try_into().unwrap(),
                value
            ))?;
        }
        Ok(())
    }

    /// Adds a new constraint to the highs model.
    ///
    /// Returns the added row index.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn add_row(&mut self, bounds: impl RangeBounds<f64>, row_factors: impl IntoIterator<Item = (Col, f64)>) -> Row {
        self.try_add_row(bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new constraint to the highs model.
    ///
    /// Returns the added row index, or the error status value if HIGHS returned an error status.
    pub fn try_add_row(
        &mut self,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Col, f64)>,
    ) -> Result<Row, HighsStatus> {
        let (cols, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();

        unsafe {
            highs_call!(Highs_addRow(
                self.highs.mut_ptr(),
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                cols.len().try_into().unwrap(),
                cols.into_iter().map(|c| c.0.try_into().unwrap()).collect::<Vec<_>>().as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(Row((self.highs.num_rows()? - 1) as c_int))
    }

    /// Adds a new variable to the highs model.
    ///
    /// Returns the added column index.
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value.
    pub fn add_col(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Col {
        self.try_add_column(col_factor, bounds, row_factors)
            .unwrap_or_else(|e| panic!("HiGHS error: {:?}", e))
    }

    /// Tries to add a new variable to the highs model.
    ///
    /// Returns the added column index, or the error status value if HIGHS returned an error status.
    pub fn try_add_column(
        &mut self,
        col_factor: f64,
        bounds: impl RangeBounds<f64>,
        row_factors: impl IntoIterator<Item = (Row, f64)>,
    ) -> Result<Col, HighsStatus> {
        let (rows, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();
        unsafe {
            highs_call!(Highs_addCol(
                self.highs.mut_ptr(),
                col_factor,
                bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                rows.len().try_into().unwrap(),
                rows.into_iter().map(|r| r.0.try_into().unwrap()).collect::<Vec<_>>().as_ptr(),
                factors.as_ptr()
            ))
        }?;

        Ok(Col(self.highs.num_cols()? - 1))
    }
}

impl From<SolvedModel> for Model {
    fn from(solved: SolvedModel) -> Self {
        Self { highs: solved.highs }
    }
}

#[derive(Debug)]
struct HighsPtr(*mut c_void);

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl Default for HighsPtr {
    fn default() -> Self {
        Self(unsafe { Highs_create() })
    }
}

impl HighsPtr {
    // To be used instead of unsafe_mut_ptr wherever possible
    #[allow(dead_code)]
    const fn ptr(&self) -> *const c_void {
        self.0
    }

    // Needed until https://github.com/ERGO-Code/HiGHS/issues/479 is fixed
    unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        // setting log_file seems to cause a double free in Highs.
        // See https://github.com/rust-or/highs/issues/3
        // self.set_option(&b"log_file"[..], "");
        self.set_option(&b"output_flag"[..], false);
        self.set_option(&b"log_to_console"[..], false);
    }

    /// Set a custom parameter on the model
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        let c_str = CString::new(option).expect("invalid option name");
        let status = unsafe { value.apply_to_highs(self.mut_ptr(), c_str.as_ptr()) };
        try_handle_status(status, "Highs_setOptionValue").expect("An error was encountered in HiGHS.");
    }

    /// Number of variables
    fn num_cols(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumCols(self.0) };
        n.try_into()
    }

    /// Number of constraints
    fn num_rows(&self) -> Result<usize, TryFromIntError> {
        let n = unsafe { Highs_getNumRows(self.0) };
        n.try_into()
    }
}

impl SolvedModel {
    /// The status of the solution. Should be Optimal if everything went well
    pub fn status(&self) -> HighsModelStatus {
        let model_status = unsafe { Highs_getModelStatus(self.highs.unsafe_mut_ptr()) };
        HighsModelStatus::try_from(model_status).unwrap()
    }

    /// Get the solution to the problem
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        // Get the primal and dual solution
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Number of variables
    fn num_cols(&self) -> usize {
        self.highs.num_cols().expect("invalid number of columns")
    }

    /// Number of constraints
    fn num_rows(&self) -> usize {
        self.highs.num_rows().expect("invalid number of rows")
    }
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    colvalue: Vec<f64>,
    coldual: Vec<f64>,
    rowvalue: Vec<f64>,
    rowdual: Vec<f64>,
}

impl Solution {
    /// The optimal values for each variables (in the order they were added)
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    /// The optimal values for each variables in the dual problem (in the order they were added)
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    /// The value of the constraint functions
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
    /// The value of the constraint functions in the dual problem
    pub fn dual_rows(&self) -> &[f64] {
        &self.rowdual
    }
}

impl Index<Col> for Solution {
    type Output = f64;
    fn index(&self, col: Col) -> &f64 {
        &self.colvalue[col.0]
    }
}

fn try_handle_status(status: c_int, msg: &str) -> Result<HighsStatus, HighsStatus> {
    let status_enum = HighsStatus::try_from(status).expect(
        "HiGHS returned an unexpected status value. Please report it as a bug to https://github.com/rust-or/highs/issues",
    );
    match status_enum {
        status @ HighsStatus::OK => Ok(status),
        status @ HighsStatus::Warning => {
            log::warn!("HiGHS emitted a warning: {}", msg);
            Ok(status)
        }
        error => Err(error),
    }
}

/// Releases all resources held by the global scheduler instance.
///
/// It is not thread-safe to call this function while calling Highs_run or one of the Highs_XXXcall
/// methods on any other Highs instance in any thread.
///
/// After this function has terminated, it is guaranteed that eventually all previously created scheduler
/// threads will terminate and allocated memory will be released.
///
/// After this function has returned, the option value for the number of threads may be altered to a new
/// value before the next call to Highs_run or one of the Highs_XXXcall methods.
///
/// * param : blocking If the blocking parameter has a nonzero value, then this function will not return
///           until all memory is freed, which might be desirable when debugging heap memory, but it
///           requires the calling thread to wait for all scheduler threads to wake-up which is usually
///           not necessary.
///
/// * return : No status is returned since the function call cannot fail. Calling this function while
///            any Highs instance is in use on any thread is undefined behavior and may cause crashes,
///            but cannot be detected and hence is fully in the callers responsibility.
///
/// note: this should be invoked when using multiple cores/threads to avoid memory leaks
pub unsafe fn highs_release_resources(blocking: bool) {
    Highs_resetGlobalScheduler(match blocking {
        true => 1,
        false => 0,
    });
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_coefs(coefs: [f64; 2]) {
        // See: https://github.com/rust-or/highs/issues/5
        let mut problem = RowProblem::new();
        // Minimize x + y subject to x ≥ 0, y ≥ 0.
        let x = problem.add_column(1., -1..);
        let y = problem.add_column(1., 0..);
        problem.add_row(..1, [x, y].iter().copied().zip(coefs)); // 1 ≥ x + c y.
        let solution = problem.optimise(Sense::Minimise).solve().get_solution();
        assert_eq!([-1., 0.], solution.columns());
    }

    #[test]
    fn test_single_zero_coef() {
        test_coefs([1.0, 0.0]);
        test_coefs([0.0, 1.0]);
    }

    #[test]
    fn test_all_zero_coefs() {
        test_coefs([0.0, 0.0])
    }

    #[test]
    fn test_no_zero_coefs() {
        test_coefs([1.0, 1.0])
    }

    #[test]
    fn test_infeasible_empty_row() {
        let mut problem = RowProblem::new();
        let row_factors: &[(Col, f64)] = &[];
        problem.add_row(2..3, row_factors);
        let _ = problem.optimise(Sense::Minimise).try_solve();
    }

    #[test]
    fn test_add_row_and_col() {
        let mut model = Model::new::<Problem<ColMatrix>>(Problem::default());
        let col = model.add_col(1., 1.0.., vec![]);
        model.add_row(..1.0, vec![(col, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Optimal);
        let solution = solved.get_solution();
        assert_eq!(solution.columns(), vec![1.0]);

        let mut model = Model::from(solved);
        let new_col = model.add_col(1., ..1.0, vec![]);
        model.add_row(2.0.., vec![(new_col, 1.0)]);
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::Infeasible);
    }
}
