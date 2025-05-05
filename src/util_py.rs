use crate::cluster::*;
use crate::dual_module::*;
use crate::html_export::*;
use crate::matrix::*;
#[cfg(feature = "rational_weight")]
use crate::num_traits::Signed;
use crate::num_traits::ToPrimitive;
use crate::util::*;
use crate::visualize::*;
use num_traits::FromPrimitive;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PySet, PyTuple};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

macro_rules! bind_trait_simple_wrapper {
    ($struct_name:ident, $py_struct_name:ident) => {
        impl From<$struct_name> for $py_struct_name {
            fn from(value: $struct_name) -> Self {
                Self(value)
            }
        }

        impl From<$py_struct_name> for $struct_name {
            fn from(value: $py_struct_name) -> Self {
                value.0
            }
        }
    };
}

#[derive(Clone, Hash)]
#[repr(transparent)]
#[pyclass(module = "mwpf", name = "Rational")]
pub struct PyRational(pub Rational);
bind_trait_simple_wrapper!(Rational, PyRational);

impl From<&Bound<'_, PyAny>> for PyRational {
    fn from(value: &Bound<PyAny>) -> Self {
        if value.is_instance_of::<PyRational>() {
            value.extract::<PyRational>().unwrap()
        } else if value.is_instance_of::<PyFloat>() {
            Self(Rational::from_f64(value.extract().unwrap()).unwrap())
        } else if value.is_instance_of::<PyInt>() {
            cfg_if::cfg_if! {
                if #[cfg(feature="f64_weight")] {
                    Self(Rational::from(value.extract::<f64>().unwrap()))
                } else if #[cfg(feature="rational_weight")] {
                    // python int is unbounded, thus first cast to BigInt to avoid accuracy loss
                    let bigint = value.extract::<num_bigint::BigInt>().unwrap();
                    Self(Rational::from(bigint))
                }
            }
        } else {
            panic!("unsupported type: {}", value.get_type().name().unwrap())
        }
    }
}

#[pymethods]
impl PyRational {
    #[new]
    #[pyo3(signature = (numerator, denominator=None))]
    fn __new__(numerator: &Bound<PyAny>, denominator: Option<&Bound<PyAny>>) -> PyResult<Self> {
        cfg_if::cfg_if! {
            if #[cfg(feature="rational_weight")] {
                use num_bigint::BigInt;
                let denominator: BigInt = denominator.map(|x| x.extract::<BigInt>()).transpose()?.unwrap_or_else(|| BigInt::from(1));
                let numerator: BigInt = numerator.extract()?;
                Ok(Self(Rational::new(numerator, denominator)))
            } else {
                let denominator: f64 = denominator.map(|x| x.extract::<f64>()).transpose()?.unwrap_or(1.);
                let numerator: f64 = numerator.extract()?;
                Ok(Self(Rational::new(numerator / denominator)))
            }
        }
    }
    #[getter]
    fn numer(&self) -> PyObject {
        Python::with_gil(|py| self.0.numer().into_pyobject(py).unwrap().to_owned().into())
    }
    #[getter]
    fn denom(&self) -> PyObject {
        Python::with_gil(|py| self.0.denom().into_pyobject(py).unwrap().to_owned().into())
    }
    fn float(&self) -> f64 {
        self.0.to_f64().unwrap()
    }
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> bool {
        let other = PyRational::from(other);
        op.matches(self.0.cmp(&other.0))
    }
    fn __abs__(&self) -> Self {
        self.0.abs().into()
    }
    fn __mul__(&self, other: &Bound<PyAny>) -> Self {
        let other = PyRational::from(other);
        (self.0.clone() * other.0).into()
    }
    fn __truediv__(&self, other: &Bound<PyAny>) -> Self {
        let other = PyRational::from(other);
        (self.0.clone() / other.0).into()
    }
    fn __add__(&self, other: &Bound<PyAny>) -> Self {
        let other = PyRational::from(other);
        (self.0.clone() + other.0).into()
    }
    fn __sub__(&self, other: &Bound<PyAny>) -> Self {
        let other = PyRational::from(other);
        (self.0.clone() - other.0).into()
    }
    fn __neg__(&self) -> Self {
        (-self.0.clone()).into()
    }
    fn __pos__(&self) -> Self {
        self.clone()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        cfg_if::cfg_if! {
            if #[cfg(feature="rational_weight")] {
                format!("{}/{}", self.0.numer(), self.0.denom())
            } else {
                format!("{}", self.0.to_f64().unwrap())
            }
        }
    }
    fn __hash__(&self) -> u64 {
        // let mut hasher = DefaultHasher::new();
        let mut hasher = DefaultHasher::default();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("numerator", self.numer())?;
        kwargs.set_item("denominator", self.denom())?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
    fn approx_eq(&self, other: &Bound<PyAny>) -> bool {
        let other = PyRational::from(other);
        rational_approx_eq(&self.0, &other.0)
    }
    fn approx_le(&self, other: &Bound<PyAny>) -> bool {
        let other = PyRational::from(other);
        rational_approx_le(&self.0, &other.0)
    }
    fn approx_ge(&self, other: &Bound<PyAny>) -> bool {
        let other = PyRational::from(other);
        rational_approx_ge(&self.0, &other.0)
    }
}

impl std::fmt::Debug for PyRational {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
#[pyclass(module = "mwpf", name = "DualNodePtr")]
pub struct PyDualNodePtr(pub DualNodePtr);
bind_trait_simple_wrapper!(DualNodePtr, PyDualNodePtr);

#[pymethods]
impl PyDualNodePtr {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __eq__(&self, other: &PyDualNodePtr) -> bool {
        self.0 == other.0
    }
    fn __str__(&self) -> String {
        format!("Node({})", self.index())
    }
    fn __hash__(&self) -> u64 {
        self.index() as u64
    }
}

impl PartialOrd for PyDualNodePtr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.index().cmp(&other.index()))
    }
}

impl Ord for PyDualNodePtr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index().cmp(&other.index())
    }
}

impl Hash for PyDualNodePtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index().hash(state);
    }
}

impl std::fmt::Debug for PyDualNodePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

#[pymethods]
impl PyDualNodePtr {
    #[getter]
    fn index(&self) -> NodeIndex {
        self.0.read_recursive().index
    }
    #[getter]
    fn dual_variable(&self) -> PyRational {
        self.0.read_recursive().get_dual_variable().into()
    }
    #[getter]
    fn grow_rate(&self) -> PyRational {
        self.0.read_recursive().grow_rate.clone().into()
    }
    #[getter]
    fn vertices(&self) -> FastIterSet<VertexIndex> {
        self.0.read_recursive().invalid_subgraph.vertices.clone()
    }
    #[getter]
    fn edges(&self) -> FastIterSet<EdgeIndex> {
        self.0.read_recursive().invalid_subgraph.edges.clone()
    }
    #[getter]
    fn hair(&self) -> FastIterSet<EdgeIndex> {
        self.0.read_recursive().invalid_subgraph.hair.clone()
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "mwpf", name = "Obstacle")]
pub enum PyObstacle {
    Conflict { edge_index: EdgeIndex },
    ShrinkToZero { dual_node_ptr: PyDualNodePtr },
}

impl From<Obstacle> for PyObstacle {
    fn from(value: Obstacle) -> Self {
        match value {
            Obstacle::Conflict { edge_index } => Self::Conflict { edge_index },
            Obstacle::ShrinkToZero { dual_node_ptr } => Self::ShrinkToZero {
                dual_node_ptr: dual_node_ptr.ptr.into(),
            },
        }
    }
}

#[pymethods]
impl PyObstacle {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "mwpf", name = "DualReport")]
pub enum PyDualReport {
    Unbounded(),
    ValidGrow(PyRational),
    Obstacles(Vec<PyObstacle>),
}

impl From<DualReport> for PyDualReport {
    fn from(value: DualReport) -> Self {
        match value {
            DualReport::Unbounded => Self::Unbounded(),
            DualReport::ValidGrow(ratio) => Self::ValidGrow(ratio.into()),
            DualReport::Obstacles(obstacles) => Self::Obstacles(obstacles.into_iter().map(|x| x.into()).collect()),
        }
    }
}

#[pymethods]
impl PyDualReport {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python code of `[a, b, c]` or `{a, b, c}` or `{}`
pub fn py_into_set<'py, T: Ord + FromPyObject<'py> + std::hash::Hash>(value: &Bound<'py, PyAny>) -> PyResult<FastIterSet<T>>
where
    T: Clone,
    T: Debug,
{
    let mut result = FastIterSet::<T>::new();
    if value.is_instance_of::<PyList>() {
        let list: &Bound<PyList> = value.downcast()?;
        for element in list.iter() {
            result.insert(element.extract::<T>()?);
        }
    } else if value.is_instance_of::<PySet>() {
        let list: &Bound<PySet> = value.downcast()?;
        for element in list.iter() {
            result.insert(element.extract::<T>()?);
        }
    } else if value.is_instance_of::<PyDict>() {
        let dict: &Bound<PyDict> = value.downcast()?;
        assert!(
            dict.is_empty(),
            "only empty dict is supported; please use set or list instead"
        );
    } else {
        // last resort: try convert the object into a python list
        let result: PyResult<()> = (|| {
            let builtins = PyModule::import(value.py(), "builtins")?;
            let any = builtins.getattr("list")?.call1((value,))?;
            let any_list: &Bound<PyList> = any.downcast()?;
            for element in any_list.iter() {
                result.insert(element.extract::<T>()?);
            }
            Ok(())
        })();
        if result.is_err() {
            let type_name = value.get_type().name()?;
            unimplemented!(
                "unsupported python type, should be set, list, (empty)dict, or anything that can be converted to a list; got {}",
                type_name
            )
        }
    }
    Ok(result)
}

/// Python code of `[a, b, c]` or `{a, b, c}` or `{}`
pub fn py_into_vec<'py, T: Ord + FromPyObject<'py>>(value: &Bound<'py, PyAny>) -> PyResult<Vec<T>> {
    let mut result = Vec::new();
    if value.is_instance_of::<PyList>() {
        let list: &Bound<PyList> = value.downcast()?;
        for element in list.iter() {
            result.push(element.extract::<T>()?);
        }
    } else if value.is_instance_of::<PySet>() {
        let list: &Bound<PySet> = value.downcast()?;
        for element in list.iter() {
            result.push(element.extract::<T>()?);
        }
    } else if value.is_instance_of::<PyDict>() {
        let dict: &Bound<PyDict> = value.downcast()?;
        assert!(
            dict.is_empty(),
            "only empty dict is supported; please use set or list instead"
        );
    } else {
        // last resort: try convert the object into a python list
        let result: PyResult<()> = (|| {
            let builtins = PyModule::import(value.py(), "builtins")?;
            let any = builtins.getattr("list")?.call1((value,))?;
            let any_list: &Bound<PyList> = any.downcast()?;
            for element in any_list.iter() {
                result.push(element.extract::<T>()?);
            }
            Ok(())
        })();
        if result.is_err() {
            let type_name = value.get_type().name()?;
            unimplemented!(
                "unsupported python type, should be set, list, (empty)dict, or anything that can be converted to a list; got {}",
                type_name
            )
        }
    }
    Ok(result)
}

/// Python code of `[(k1, v1), (k2, v2)]` or `{ k1: v1, k2: v2 }` or `dict(k1=v1, k2=v2)`
pub fn py_into_map<'py, K: Ord + Debug + Clone + FromPyObject<'py>, T: FromPyObject<'py>>(
    value: &Bound<'py, PyAny>,
) -> PyResult<hashbrown::HashMap<K, T>>
where
    K: Hash,
{
    let mut result = hashbrown::HashMap::<K, T>::new();
    if value.is_instance_of::<PyList>() {
        let list: &Bound<PyList> = value.downcast()?;
        for element in list.iter() {
            let element: &Bound<PyTuple> = element.downcast()?;
            assert!(element.len() == 2, "each tuple should contain two elements");
            let key = element.get_item(0)?.extract::<K>()?;
            let value = element.get_item(1)?.extract::<T>()?;
            assert!(result.insert(key.clone(), value).is_none(), "duplicate key found: {:?}", key);
        }
    } else if value.is_instance_of::<PyDict>() {
        let dict: &Bound<PyDict> = value.downcast()?;
        for (key, value) in dict.iter() {
            let key = key.extract::<K>()?;
            let value = value.extract::<T>()?;
            assert!(result.insert(key.clone(), value).is_none(), "duplicate key found: {:?}", key);
        }
    } else {
        // last resort: try convert the object into a python dict
        let result: PyResult<()> = (|| {
            let builtins = PyModule::import(value.py(), "builtins")?;
            let any = builtins.getattr("dict")?.call1((value,))?;
            let any_dict: &Bound<PyDict> = any.downcast()?;
            for (key, value) in any_dict.iter() {
                let key = key.extract::<K>()?;
                let value = value.extract::<T>()?;
                assert!(result.insert(key.clone(), value).is_none(), "duplicate key found: {:?}", key);
            }
            Ok(())
        })();
        if result.is_err() {
            let type_name = value.get_type().name()?;
            unimplemented!(
                "unsupported python type, should be list, dict, or anything that can be converted to a dict; got {}",
                type_name
            )
        }
    }
    Ok(result)
}

#[derive(Clone, Debug)]
#[repr(transparent)]
#[pyclass(module = "mwpf", name = "Subgraph")]
pub struct PySubgraph(pub Subgraph);
bind_trait_simple_wrapper!(Subgraph, PySubgraph);

#[pymethods]
impl PySubgraph {
    fn __len__(&self) -> usize {
        self.0.len()
    }
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PySubgraphIter>> {
        let iter = PySubgraphIter {
            inner: slf.0.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("edges", self.0.clone())?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
}

#[pyclass(module = "mwpf")]
struct PySubgraphIter {
    inner: std::vec::IntoIter<EdgeIndex>,
}

#[pymethods]
impl PySubgraphIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<usize> {
        slf.inner.next()
    }
}

impl From<PySubgraph> for OutputSubgraph {
    fn from(value: PySubgraph) -> OutputSubgraph {
        value.0.into()
    }
}

#[pymethods]
impl PySubgraph {
    #[new]
    fn new(subgraph: Subgraph) -> Self {
        Self(subgraph)
    }
    fn __repr__(&self) -> String {
        format!("Subgraph({:?})", self.0)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
    #[pyo3(signature = (abbrev=true))]
    fn snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.0.snapshot(abbrev))
    }
    fn set(&self) -> FastIterSet<EdgeIndex> {
        self.0.iter().cloned().collect()
    }
    fn list(&self) -> Vec<EdgeIndex> {
        self.0.clone()
    }
    fn __eq__(&self, other: &Bound<PyAny>) -> PyResult<bool> {
        let other_set = py_into_set::<EdgeIndex>(other)?;
        let my_set = self.set();
        Ok(other_set == my_set)
    }
}

macro_rules! bind_trait_matrix_basic {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            fn __repr__(&mut self) -> String {
                self.0.printstd_str()
            }
            fn __str__(&mut self) -> String {
                self.__repr__()
            }
            // MatrixBasic trait functions
            fn add_variable(&mut self, edge_index: EdgeIndex) -> Option<VarIndex> {
                self.0.add_variable(edge_index)
            }
            fn add_constraint(
                &mut self,
                vertex_index: VertexIndex,
                incident_edges: &Bound<PyAny>,
                parity: bool,
            ) -> PyResult<Option<Vec<VarIndex>>> {
                let incident_edges: Vec<EdgeIndex> = py_into_vec::<EdgeIndex>(incident_edges)?;
                Ok(self.0.add_constraint(vertex_index, &incident_edges, parity))
            }
            fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool {
                self.0.get_lhs(row, var_index)
            }
            fn get_rhs(&self, row: RowIndex) -> bool {
                self.0.get_rhs(row)
            }
            fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeIndex {
                self.0.var_to_edge_index(var_index)
            }
            fn edge_to_var_index(&self, edge_index: EdgeIndex) -> Option<VarIndex> {
                self.0.edge_to_var_index(edge_index)
            }
            fn exists_edge(&self, edge_index: EdgeIndex) -> bool {
                self.0.exists_edge(edge_index)
            }
            fn get_vertices(&self) -> FastIterSet<VertexIndex> {
                self.0.get_vertices()
            }
            fn get_edges(&self) -> FastIterSet<EdgeIndex> {
                self.0.get_edges()
            }
            // MatrixView trait functions
            #[getter]
            fn get_columns(&mut self) -> usize {
                self.0.columns()
            }
            fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
                self.0.column_to_var_index(column)
            }
            fn column_to_edge_index(&self, column: ColumnIndex) -> EdgeIndex {
                self.0.column_to_edge_index(column)
            }
            fn get_view_edges(&mut self) -> Vec<EdgeIndex> {
                self.0.get_view_edges()
            }
            fn var_to_column_index(&mut self, var_index: VarIndex) -> Option<ColumnIndex> {
                self.0.var_to_column_index(var_index)
            }
            fn edge_to_column_index(&mut self, edge_index: EdgeIndex) -> Option<ColumnIndex> {
                self.0.edge_to_column_index(edge_index)
            }
            #[getter]
            fn get_rows(&mut self) -> usize {
                self.0.rows()
            }
            fn snapshot(&mut self) -> PyObject {
                json_to_pyobject(self.snapshot_json())
            }
            fn show(&mut self) {
                HTMLExport::display_jupyter_matrix_html(self.snapshot_json(), self.__repr__());
            }
            fn clone(&self) -> Self {
                self.0.clone().into()
            }
        }
    };
}

macro_rules! bind_trait_matrix_tight {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            // MatrixTight trait functions
            fn update_edge_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool) {
                self.0.update_edge_tightness(edge_index, is_tight)
            }
            fn is_tight(&self, edge_index: usize) -> bool {
                self.0.is_tight(edge_index)
            }
            fn get_tight_edges(&mut self) -> FastIterSet<EdgeIndex> {
                self.0.get_tight_edges().clone()
            }
            fn add_variable_with_tightness(&mut self, edge_index: EdgeIndex, is_tight: bool) {
                self.0.add_variable_with_tightness(edge_index, is_tight)
            }
            fn add_tight_variable(&mut self, edge_index: EdgeIndex) {
                self.0.add_tight_variable(edge_index)
            }
        }
    };
}

macro_rules! bind_trait_matrix_tail {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            // MatrixTail trait functions
            fn get_tail_edges(&self) -> FastIterSet<EdgeIndex> {
                self.0.get_tail_edges().clone()
            }
            fn set_tail_edges(&mut self, edges: &Bound<PyAny>) -> PyResult<()> {
                let tail_edges = py_into_vec(edges)?;
                self.0.set_tail_edges(tail_edges.into_iter());
                Ok(())
            }
        }
    };
}

macro_rules! bind_trait_matrix_echelon {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            // MatrixEchelon trait functions
            fn get_echelon_info(&mut self) -> EchelonInfo {
                self.0.get_echelon_info().clone()
            }
            fn get_solution(&mut self) -> Option<PySubgraph> {
                self.0.get_solution().map(|x| x.into())
            }
            fn get_solution_local_minimum(&mut self, weight_of: &Bound<PyAny>) -> Option<Subgraph> {
                self.0
                    .get_solution_local_minimum(|x| PyRational::from(&weight_of.call1((x,)).unwrap()).0)
            }
            #[getter]
            fn satisfiable(&mut self) -> bool {
                self.0.get_echelon_info().satisfiable
            }
            fn get_tail_start_index(&mut self) -> Option<ColumnIndex> {
                self.0.get_tail_start_index()
            }
            fn get_corner_row_index(&mut self, tail_start_index: ColumnIndex) -> RowIndex {
                self.0.get_corner_row_index(tail_start_index)
            }
            /// return the number of feasible solutions
            fn __len__(&mut self) -> usize {
                let num_free_variables = self
                    .0
                    .get_echelon_info()
                    .columns
                    .iter()
                    .filter(|column| !column.is_dependent())
                    .count();
                assert!(num_free_variables < 63, "number of free variables is too large");
                1 << num_free_variables
            }
            /// get the proper number of feasible solutions
            fn __getitem__(&mut self, index: usize) -> PyResult<PySubgraph> {
                use crate::itertools::Itertools;
                let echelon_info = self.0.get_echelon_info().clone();
                let free_variables: Vec<usize> = echelon_info
                    .columns
                    .iter()
                    .enumerate()
                    .filter(|(_ci, column)| !column.is_dependent())
                    .map(|(ci, _)| ci)
                    .collect();
                assert!(free_variables.len() < 63, "number of free variables is too large");
                if index >= (1 << free_variables.len()) {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index out of range: {}",
                        index
                    )));
                }
                let mut selected_free_variables = hashbrown::HashSet::new();
                for fi in 0..free_variables.len() {
                    let ci = free_variables[fi];
                    if ((index >> (free_variables.len() - 1 - fi)) & 1) == 1 {
                        selected_free_variables.insert(ci);
                    }
                }
                let mut selected_pivot_variables = vec![];
                for (ci, column) in echelon_info.columns.iter().enumerate() {
                    if !column.is_dependent() {
                        continue;
                    }
                    let ri = column.row;
                    let mut parity = self.0.get_rhs(ri);
                    for ci2 in ci + 1..self.0.columns() {
                        if self.0.get_lhs(ri, self.0.column_to_var_index(ci2)) {
                            parity ^= selected_free_variables.contains(&ci2);
                        }
                    }
                    if parity {
                        selected_pivot_variables.push(ci);
                    }
                }
                let subgraph: Vec<EdgeIndex> = selected_free_variables
                    .into_iter()
                    .chain(selected_pivot_variables.into_iter())
                    .map(|ci| self.0.column_to_edge_index(ci))
                    .sorted()
                    .collect();
                Ok(subgraph.into())
            }
        }
    };
}

type EchelonMatrix = Echelon<Tail<Tight<BasicMatrix>>>;

#[derive(Clone)]
#[pyclass(module = "mwpf", name = "EchelonMatrix")]
pub struct PyEchelonMatrix(pub EchelonMatrix);
bind_trait_simple_wrapper!(EchelonMatrix, PyEchelonMatrix);
bind_trait_matrix_basic!(PyEchelonMatrix);
bind_trait_matrix_tight!(PyEchelonMatrix);
bind_trait_matrix_tail!(PyEchelonMatrix);
bind_trait_matrix_echelon!(PyEchelonMatrix);

impl PyEchelonMatrix {
    fn snapshot_json(&mut self) -> serde_json::Value {
        let mut matrix_json = self.0.viz_table().snapshot();
        let tail_start_index = self.0.get_tail_start_index();
        let matrix_json_obj = matrix_json.as_object_mut().unwrap();
        matrix_json_obj.insert("tail_start_index".to_string(), tail_start_index.into());
        if let Some(tail_start_index) = tail_start_index {
            let row = self.0.get_corner_row_index(tail_start_index);
            matrix_json_obj.insert("corner_row_index".to_string(), row.into());
        }
        matrix_json_obj.insert("is_echelon_form".to_string(), true.into());
        matrix_json
    }
}

#[pymethods]
impl PyEchelonMatrix {
    #[new]
    #[pyo3(signature = (matrix=None, all_tight=false))]
    fn new(matrix: Option<&Bound<PyAny>>, all_tight: bool) -> PyResult<Self> {
        if let Some(matrix) = matrix {
            if let Ok(matrix) = matrix.extract::<PyTailMatrix>() {
                let mut echelon = EchelonMatrix::from_base(matrix.0.clone());
                if all_tight {
                    for edge_index in echelon.get_edges() {
                        echelon.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(Self(echelon));
            }
            if let Ok(matrix) = matrix.extract::<PyTightMatrix>() {
                let mut echelon = EchelonMatrix::from_base(TailMatrix::from_base(matrix.0.clone()));
                if all_tight {
                    for edge_index in echelon.get_edges() {
                        echelon.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(Self(echelon));
            }
            if let Ok(matrix) = matrix.extract::<PyBasicMatrix>() {
                let mut echelon = EchelonMatrix::from_base(TailMatrix::from_base(TightMatrix::from_base(matrix.0.clone())));
                if all_tight {
                    for edge_index in echelon.get_edges() {
                        echelon.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(Self(echelon));
            }
            panic!("unknown input type: {}", matrix.get_type().name()?);
        } else {
            Ok(Self(EchelonMatrix::new()))
        }
    }
    fn get_base(&self) -> PyTailMatrix {
        self.0.get_base().clone().into()
    }
}

type TailMatrix = Tail<Tight<BasicMatrix>>;

/// TailMatrix is a matrix that allows reordering part of the columns to the end.
#[derive(Clone)]
#[pyclass(module = "mwpf", name = "TailMatrix")]
pub struct PyTailMatrix(pub TailMatrix);
bind_trait_simple_wrapper!(TailMatrix, PyTailMatrix);
bind_trait_matrix_basic!(PyTailMatrix);
bind_trait_matrix_tight!(PyTailMatrix);
bind_trait_matrix_tail!(PyTailMatrix);

impl PyTailMatrix {
    fn snapshot_json(&mut self) -> serde_json::Value {
        let mut matrix_json = self.0.viz_table().snapshot();
        let tail_start_index = self
            .get_tail_edges()
            .into_iter()
            .map(|edge_index| self.edge_to_column_index(edge_index))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap())
            .min();
        let matrix_json_obj = matrix_json.as_object_mut().unwrap();
        matrix_json_obj.insert("tail_start_index".to_string(), tail_start_index.into());
        matrix_json
    }
}

#[pymethods]
impl PyTailMatrix {
    #[new]
    #[pyo3(signature = (matrix=None))]
    fn new(matrix: Option<&Bound<PyAny>>) -> PyResult<Self> {
        if let Some(matrix) = matrix {
            if let Ok(matrix) = matrix.extract::<PyTightMatrix>() {
                return Ok(Self(TailMatrix::from_base(matrix.0.clone())));
            }
            if let Ok(matrix) = matrix.extract::<PyBasicMatrix>() {
                return Ok(Self(TailMatrix::from_base(TightMatrix::from_base(matrix.0.clone()))));
            }
            if let Ok(matrix) = matrix.extract::<PyEchelonMatrix>() {
                return Ok(matrix.get_base());
            }
            panic!("unknown input type: {}", matrix.get_type().name()?);
        } else {
            Ok(Self(TailMatrix::new()))
        }
    }
    // MatrixBasic trait functions
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        self.0.xor_row(target, source)
    }
    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.0.swap_row(a, b)
    }
    fn get_base(&self) -> PyTightMatrix {
        self.0.get_base().clone().into()
    }
    // helper functions
    #[getter]
    fn satisfiable(&mut self) -> bool {
        let mut echelon = EchelonMatrix::from_base(self.0.clone());
        echelon.get_echelon_info().satisfiable
    }
}

type TightMatrix = Tight<BasicMatrix>;

/// TightMatrix is a matrix that hides some of the edges that are not tight while still keeping track of them when doing row operations.
#[derive(Clone)]
#[pyclass(module = "mwpf", name = "TightMatrix")]
pub struct PyTightMatrix(pub TightMatrix);
bind_trait_simple_wrapper!(TightMatrix, PyTightMatrix);
bind_trait_matrix_basic!(PyTightMatrix);
bind_trait_matrix_tight!(PyTightMatrix);

impl PyTightMatrix {
    fn snapshot_json(&mut self) -> serde_json::Value {
        self.0.viz_table().snapshot()
    }
}

#[pymethods]
impl PyTightMatrix {
    #[new]
    #[pyo3(signature = (matrix=None, all_tight=false))]
    fn new(matrix: Option<&Bound<PyAny>>, all_tight: bool) -> PyResult<Self> {
        if let Some(matrix) = matrix {
            if let Ok(matrix) = matrix.extract::<PyBasicMatrix>() {
                let mut tight = TightMatrix::from_base(matrix.0.clone());
                if all_tight {
                    for edge_index in tight.get_edges() {
                        tight.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(Self(tight));
            }
            if let Ok(matrix) = matrix.extract::<PyTailMatrix>() {
                let mut tight = matrix.get_base();
                if all_tight {
                    for edge_index in tight.get_edges() {
                        tight.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(tight);
            }
            if let Ok(matrix) = matrix.extract::<PyEchelonMatrix>() {
                let mut tight = matrix.get_base().get_base();
                if all_tight {
                    for edge_index in tight.get_edges() {
                        tight.update_edge_tightness(edge_index, true);
                    }
                }
                return Ok(tight);
            }
            panic!("unknown input type: {}", matrix.get_type().name()?);
        } else {
            Ok(Self(TightMatrix::new()))
        }
    }
    // MatrixBasic trait functions
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        self.0.xor_row(target, source)
    }
    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.0.swap_row(a, b)
    }
    fn get_base(&self) -> PyBasicMatrix {
        self.0.get_base().clone().into()
    }
    // helper functions
    #[getter]
    fn satisfiable(&mut self) -> bool {
        let mut echelon = EchelonMatrix::from_base(TailMatrix::from_base(self.0.clone()));
        echelon.get_echelon_info().satisfiable
    }
}

/// BasicMatrix is a matrix that provides the basic functionality
#[derive(Clone)]
#[pyclass(module = "mwpf", name = "BasicMatrix")]
pub struct PyBasicMatrix(pub BasicMatrix);
bind_trait_simple_wrapper!(BasicMatrix, PyBasicMatrix);
bind_trait_matrix_basic!(PyBasicMatrix);

impl PyBasicMatrix {
    fn snapshot_json(&mut self) -> serde_json::Value {
        self.0.viz_table().snapshot()
    }
}

#[pymethods]
impl PyBasicMatrix {
    #[new]
    #[pyo3(signature = (matrix=None))]
    fn new(matrix: Option<&Bound<PyAny>>) -> PyResult<Self> {
        if let Some(matrix) = matrix {
            if let Ok(matrix) = matrix.extract::<PyTightMatrix>() {
                return Ok(matrix.get_base());
            }
            if let Ok(matrix) = matrix.extract::<PyTailMatrix>() {
                return Ok(matrix.get_base().get_base());
            }
            if let Ok(matrix) = matrix.extract::<PyEchelonMatrix>() {
                return Ok(matrix.get_base().get_base().get_base());
            }
            panic!("unknown input type: {}", matrix.get_type().name()?);
        } else {
            Ok(Self(BasicMatrix::new()))
        }
    }
    // MatrixBasic trait functions
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        self.0.xor_row(target, source)
    }
    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.0.swap_row(a, b)
    }
    // helper functions
    #[getter]
    fn satisfiable(&mut self) -> bool {
        let mut echelon = EchelonMatrix::from_base(TailMatrix::from_base(TightMatrix::from_base(self.0.clone())));
        echelon.get_echelon_info().satisfiable
    }
}

#[derive(Clone, Debug)]
#[pyclass(module = "mwpf", name = "WeightRange")]
pub struct PyWeightRange(pub WeightRange);
bind_trait_simple_wrapper!(WeightRange, PyWeightRange);

#[pymethods]
impl PyWeightRange {
    #[new]
    #[pyo3(signature=(lower, upper))]
    fn py_new(lower: PyRational, upper: PyRational) -> Self {
        WeightRange::new(lower.0, upper.0).into()
    }
    #[getter]
    fn get_lower(&self) -> PyRational {
        self.0.lower.clone().into()
    }
    #[setter]
    fn set_lower(&mut self, value: PyRational) {
        self.0.lower = value.into();
    }
    #[getter]
    fn get_upper(&self) -> PyRational {
        self.0.upper.clone().into()
    }
    #[setter]
    fn set_upper(&mut self, value: PyRational) {
        self.0.upper = value.into();
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    #[pyo3(signature = (abbrev=true))]
    fn snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.0.snapshot(abbrev))
    }
}

#[derive(Clone)]
#[repr(transparent)]
#[pyclass(module = "mwpf", name = "Cluster")]
pub struct PyCluster(pub Cluster);
bind_trait_simple_wrapper!(Cluster, PyCluster);

impl std::fmt::Debug for PyCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.__str__())
    }
}

#[pymethods]
impl PyCluster {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl PyCluster {
    #[getter]
    fn get_vertices(&self) -> FastIterSet<VertexIndex> {
        self.0.vertices.clone()
    }
    #[setter]
    fn set_vertices(&mut self, vertices: &Bound<PyAny>) -> PyResult<()> {
        self.0.vertices = py_into_set(vertices)?;
        Ok(())
    }
    #[getter]
    fn get_edges(&self) -> FastIterSet<EdgeIndex> {
        self.0.edges.clone()
    }
    #[setter]
    fn set_edges(&mut self, edges: &Bound<PyAny>) -> PyResult<()> {
        self.0.edges = py_into_set(edges)?;
        Ok(())
    }
    #[getter]
    fn get_hair(&self) -> FastIterSet<EdgeIndex> {
        self.0.hair.clone()
    }
    #[setter]
    fn set_hair(&mut self, hair: &Bound<PyAny>) -> PyResult<()> {
        self.0.hair = py_into_set(hair)?;
        Ok(())
    }
    #[getter]
    fn get_nodes(&self) -> FastIterSet<PyDualNodePtr> {
        self.0.nodes.iter().map(|x| x.ptr.clone().into()).collect()
    }
    #[setter]
    fn set_nodes(&mut self, nodes: &Bound<PyAny>) -> PyResult<()> {
        let nodes: Vec<PyDualNodePtr> = py_into_vec(nodes)?;
        self.0.nodes = nodes.into_iter().map(|x| x.0.into()).collect();
        Ok(())
    }
    #[getter]
    fn get_parity_matrix(&self) -> PyTightMatrix {
        self.0.parity_matrix.clone().into()
    }
    #[setter]
    fn set_parity_matrix(&mut self, parity_matrix: PyTightMatrix) {
        self.0.parity_matrix = parity_matrix.0.clone();
    }
}

#[pyfunction]
#[pyo3(name = "exclusive_weight_sum")]
pub fn py_exclusive_weight_sum(w1: &Bound<PyAny>, w2: &Bound<PyAny>) -> PyRational {
    PyRational(exclusive_weight_sum(&PyRational::from(w1).0, &PyRational::from(w2).0))
}

#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRational>()?;
    m.add_class::<PyDualNodePtr>()?;
    m.add_class::<PyObstacle>()?;
    m.add_class::<PyDualReport>()?;
    m.add_class::<DualModuleMode>()?;
    m.add_class::<PySubgraph>()?;
    m.add_class::<PyEchelonMatrix>()?;
    m.add_class::<PyTailMatrix>()?;
    m.add_class::<PyTightMatrix>()?;
    m.add_class::<PyBasicMatrix>()?;
    m.add_class::<EchelonInfo>()?;
    m.add_class::<ColumnInfo>()?;
    m.add_class::<RowInfo>()?;
    m.add_class::<PyWeightRange>()?;
    m.add_class::<PyCluster>()?;
    m.add_function(wrap_pyfunction!(py_exclusive_weight_sum, m)?)?;
    Ok(())
}
