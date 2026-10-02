use crate::slp::num_traits::{One, Zero};
use crate::slp::*;

/// Number trait used in this library.
pub trait Number:
    Clone
    + Send
    + Sync
    + One
    + Zero
    + std::str::FromStr
    + std::ops::Neg<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::cmp::PartialOrd
    + std::fmt::Debug
    + std::fmt::Display
{
    /// Returns greatest integer less than or equal to.
    fn floor(&self) -> Self;
    /// Returns least integer greater than or equal to.
    fn ceil(&self) -> Self;
    /// Checks if it is an integer.
    fn is_integer(&self) -> bool;
}

impl Number for f32 {
    fn floor(&self) -> Self {
        f32::floor(*self)
    }
    fn ceil(&self) -> Self {
        f32::ceil(*self)
    }
    fn is_integer(&self) -> bool {
        self.fract().abs() <= std::f32::EPSILON
    }
}

impl Number for f64 {
    fn floor(&self) -> Self {
        f64::floor(*self)
    }
    fn ceil(&self) -> Self {
        f64::ceil(*self)
    }
    fn is_integer(&self) -> bool {
        self.fract().abs() <= std::f64::EPSILON
    }
}

impl Number for Rational32 {
    fn floor(&self) -> Self {
        Rational32::floor(self)
    }
    fn ceil(&self) -> Self {
        Rational32::ceil(self)
    }
    fn is_integer(&self) -> bool {
        Rational32::is_integer(self)
    }
}

impl Number for Rational64 {
    fn floor(&self) -> Self {
        Rational64::floor(self)
    }
    fn ceil(&self) -> Self {
        Rational64::ceil(self)
    }
    fn is_integer(&self) -> bool {
        Rational64::is_integer(self)
    }
}

impl Number for Ratio<BigInt> {
    fn floor(&self) -> Self {
        Self::floor(self)
    }
    fn ceil(&self) -> Self {
        Self::ceil(self)
    }
    fn is_integer(&self) -> bool {
        Self::is_integer(self)
    }
}

/// Solution to an LP instance as returned by
/// the solve method of an LP instance.
#[derive(Debug, PartialEq)]
pub enum Solution<T> {
    /// Represents that LP is infeasible.
    Infeasible,
    /// Represents that LP is unbounded.
    Unbounded,
    /// The first value is the optimal value of the objective and
    /// the second value is the assignment.
    Optimal(T, Vec<T>),
}

/// Solver settings that can be passed to the solver instance.
pub enum SolverSettings {
    /// Enables data parallelism while solving.
    EnableDataParallelism,
}

pub(crate) struct SolverOptions {
    pub parallel: bool,
}
