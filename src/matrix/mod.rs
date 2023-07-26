pub mod basic_matrix;
pub mod echelon_matrix;
pub mod matrix_interface;
pub mod parity_matrix;
pub mod row;
pub mod table;
pub mod tight_matrix;

#[cfg(feature = "python_binding")]
use pyo3::prelude::*;

pub use echelon_matrix::EchelonMatrix;
pub use matrix_interface::{MatrixImpl, VarIndex};
pub use parity_matrix::ParityMatrix;
pub use row::ParityRow;
pub use table::{VizTable, VizTrait};

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ParityMatrix>()?;
    m.add_class::<ParityRow>()?;
    Ok(())
}
