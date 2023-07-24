use super::*;
use crate::dual_module::*;
use crate::hyper_decoding_graph::*;
use crate::parity_matrix_visualize::*;
use crate::prettytable::*;
use crate::util::*;
use derivative::Derivative;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Derivative)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", cfg_eval)]
#[cfg_attr(feature = "python_binding", pyclass)]
pub struct EchelonInfo {
    /// (is_dependent, if dependent the only "1" position row)
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub echelon_column_info: Vec<(bool, usize)>,
    /// the number of effective rows
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub echelon_effective_rows: usize,
    /// whether it's a satisfiable matrix, only valid when `is_echelon_form` is true
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub echelon_satisfiable: bool,
    /// the leading "1" position column
    #[cfg_attr(feature = "python_binding", pyo3(get, set))]
    pub echelon_row_info: Vec<usize>,
}

#[derive(Clone, Debug, Derivative)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", cfg_eval)]
#[cfg_attr(feature = "python_binding", pyclass)]
pub struct ParityMatrix {
    /// matrix itself
    pub matrix: BasicMatrix,
    /// information about the matrix when it's formatted into the Echelon form;
    echelon_info: EchelonInfo,
}

impl ParityMatrix {
    pub fn add_variable(&mut self, edge_index: EdgeIndex) {
        self.matrix.add_variable(edge_index);
        self.echelon_info.echelon_column_info.push((false, 0));
    }

    pub fn add_constraint(&mut self, vertex_index: VertexIndex, incident_edges: &[EdgeIndex], parity: bool) {
        self.matrix.add_constraint(vertex_index, incident_edges, parity);
        self.echelon_info.echelon_row_info.push(0);
        self.echelon_info.echelon_effective_rows = self.constraints.len(); // by default all constraints are taking effect
    }
}

impl std::ops::Deref for ParityMatrix {
    type Target = BasicMatrix;
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl std::ops::DerefMut for ParityMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.matrix
    }
}
