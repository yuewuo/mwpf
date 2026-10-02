//! `slp` is a Linear Programming Solver.
//!
//! To see the usage docs, visit [here](https://docs.rs/crate/slp/).
//!
//! ## An example
//!
//! ```rust
//! fn main() {
//!     use mwpf::slp::*;
//!     use mwpf::slp::Rational64;
//!     use mwpf::slp::Solution;
//!     let input = "
//!         vars x1>=0, x2>=0
//!         max 2x1+3x2
//!         subject to
//!             2x1 +  x2 <= 18,
//!             6x1 + 5x2 <= 60,
//!             2x1 + 5x2 <= 40
//!         ";
//!     let mut solver = Solver::<Rational64>::new(&input);
//!     let solution = solver.solve();
//!     assert_eq!(solution, Solution::Optimal(Rational64::from_integer(28), vec![
//!         Rational64::from_integer(5),
//!         Rational64::from_integer(6)
//!     ]));
//!     match solution {
//!         Solution::Infeasible => println!("INFEASIBLE"),
//!         Solution::Unbounded => println!("UNBOUNDED"),
//!         Solution::Optimal(obj, model) => {
//!             println!("OPTIMAL {}", obj);
//!             print!("SOLUTION");
//!             for v in model {
//!                 print!(" {}", v);
//!             }
//!             println!();
//!         }
//!     }
//! }
//! ```

#![deny(missing_docs)]

mod common;
pub use common::*;
mod lp;

pub use num_bigint::BigInt;
pub use num_rational::{BigRational, Ratio, Rational32, Rational64};
pub use num_traits;

/// A General Linear Programming Solver.
mod solver;
pub use solver::*;

/// Parser module for Linear Programming Problems.
pub mod parser;
