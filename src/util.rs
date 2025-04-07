#[cfg(feature = "fast_ds")]
use crate::fast_ds;
use crate::mwpf_solver::*;
#[cfg(not(feature = "float_lp"))]
use crate::num_rational;
use crate::num_traits::{FromPrimitive, ToPrimitive};
use crate::rand_xoshiro;
use crate::rand_xoshiro::rand_core::RngCore;
#[cfg(feature = "python_binding")]
use crate::util_py::*;
use crate::visualize::*;
use itertools::izip;
use lnexp::LnExp;
use num_traits::Zero;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
#[cfg(feature = "python_binding")]
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

cfg_if::cfg_if! {
    if #[cfg(feature="f64_weight")] {
        pub type Rational = crate::ordered_float::OrderedFloat;
        pub fn numer_of(value: &Rational) -> f64 {
            value.numer().to_f64().unwrap()
        }
        pub fn denom_of(value: &Rational) -> i64 {
            value.denom().to_i64().unwrap()
        }
    } else if #[cfg(feature="rational_weight")] {
        use num_bigint::BigInt;
        pub type Rational = num_rational::BigRational;
        pub fn numer_of(value: &Rational) -> BigInt {
            value.numer().clone()
        }
        pub fn denom_of(value: &Rational) -> BigInt {
            value.denom().clone()
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature="fast_ds")] {
        pub type DefaultHasher = gxhash::GxHasher;
    } else {
        pub type DefaultHasher = std::collections::hash_map::DefaultHasher;
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature="python_binding")] {
        pub use crate::python_signal_checker::PYTHON_SIGNAL_CHECKER;
    } else  {
        pub struct NoPythonSignalChecker();
        pub static PYTHON_SIGNAL_CHECKER: NoPythonSignalChecker = NoPythonSignalChecker();
        impl NoPythonSignalChecker {
            #[inline]
            pub fn check(&self) -> Result<(), ()> { Ok(()) }
            #[inline]
            pub fn skip_next(&self) {}
        }
    }
}

pub type Weight = Rational;
pub type EdgeIndex = usize;
pub type VertexIndex = usize;
pub type HeraldIndex = usize;
pub type KnownSafeRefCell<T> = std::cell::RefCell<T>;

pub type NodeIndex = VertexIndex;
pub type DefectIndex = VertexIndex;
pub type VertexNodeIndex = VertexIndex; // must be same as VertexIndex, NodeIndex, DefectIndex
pub type VertexNum = VertexIndex;
pub type NodeNum = VertexIndex;

cfg_if::cfg_if! {
    if #[cfg(feature="fast_ds")] {
        pub type FastIterSet<T> = fast_ds::Set<T>;
        pub type FastIterMap<K, V> = fast_ds::Map<K, V>;
        pub type FastIterEntry<'a, K, V> = fast_ds::Entry<'a, K, V>;
    } else {
        // by default using std BTreeSet and BTreeMap
        pub type FastIterSet<T> = std::collections::BTreeSet<T>;
        pub type FastIterMap<K, V> = std::collections::BTreeMap<K, V>;
        pub type FastIterEntry<'a, K, V> = std::collections::btree_map::Entry<'a, K, V>;
    }
}

#[macro_export]
macro_rules! fast_iter_set {
    ($($key:expr,)+) => (btreeset!($($key),+));

    ( $($key:expr),* ) => {
        {
            let mut _set = FastIterSet::new();
            $(
                _set.insert($key);
            )*
            _set
        }
    };
}

pub use crate::fast_iter_set;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct HyperEdge {
    /// the vertices incident to the hyperedge
    pub vertices: Vec<VertexIndex>,
    /// the weight of the hyperedge
    pub weight: Weight,
}

impl HyperEdge {
    pub fn new(vertices: Vec<VertexIndex>, weight: Weight) -> Self {
        Self { vertices, weight }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl HyperEdge {
    #[new]
    fn py_new(vertices: &Bound<PyAny>, weight: &Bound<PyAny>) -> PyResult<Self> {
        use crate::util_py::py_into_vec;
        let vertices: Vec<VertexIndex> = py_into_vec::<VertexIndex>(vertices)?;
        Ok(Self::new(vertices, PyRational::from(weight).0))
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    #[getter]
    fn get_vertices(&self) -> Vec<VertexIndex> {
        self.vertices.clone()
    }
    #[setter]
    fn set_vertices(&mut self, vertices: Vec<VertexIndex>) {
        self.vertices = vertices;
    }
    #[getter]
    fn get_weight(&self) -> PyRational {
        self.weight.clone().into()
    }
    #[setter]
    fn set_weight(&mut self, weight: &Bound<PyAny>) {
        self.weight = PyRational::from(weight).0;
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("vertices", self.vertices.clone())?;
        kwargs.set_item("weight", self.get_weight())?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
}

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverInitializer {
    /// the number of vertices
    pub vertex_num: VertexNum,
    /// weighted edges, where vertex indices are within the range [0, vertex_num)
    pub weighted_edges: Vec<HyperEdge>,
    /// conditional edge sets; when the heralded detector is one, this specified edges will update
    /// their weight as if these additional errors could be happening (see `compose_weight` function).
    /// note that in case rational number is used, this method only guarantees f64 accuracy
    pub heralds: Vec<Vec<(EdgeIndex, Weight)>>,
}

pub fn exclusive_weight_sum(w1: &Weight, w2: &Weight) -> Weight {
    // w1 = log( (1-p1) / p1 ), weight_2 = log( (1-p2) / p2 )
    // p1 = 1 / (1 + exp(w1)), p2 = 1 / (1 + exp(w2))
    // p' = p1 (1 - p2) + p2 (1 - p1) = (exp(w1) + exp(w2)) / [ (1 + exp(w1)) (1 + exp(w2)) ]
    // 1 - p' = (1 + exp(w1) exp(w2)) / [ (1 + exp(w1)) (1 + exp(w2)) ]
    // w' = log ( (1-p') / p' ) = log((1 + exp(w1) exp(w2)) / (exp(w1) + exp(w2)))
    //    = log(1 + exp(w1) exp(w2)) - log(exp(w1) + exp(w2))
    //    = log(1 + exp(w1+w2)) - w2 - log(1 + exp(w1-w2))
    let (w1, w2) = (w1.to_f64().unwrap(), w2.to_f64().unwrap());
    let (w1, w2) = (w1.max(w2), w1.min(w2)); // make sure w1 >= w2
    let w = (w1 + w2).ln_1p_exp() - w2 - (w1 - w2).ln_1p_exp();
    Weight::from_f64(w).unwrap()
}

impl SolverInitializer {
    pub fn new(vertex_num: VertexNum, weighted_edges: Vec<HyperEdge>) -> Self {
        #[cfg(feature = "f64_weight")]
        assert!(
            weighted_edges.iter().all(|h| h.weight.is_number()),
            "weight must be a number, but (index, weight) = {:?}",
            weighted_edges
                .iter()
                .enumerate()
                .find(|(_, h)| !h.weight.is_number())
                .map(|(i, h)| (i, h.weight.clone()))
                .unwrap()
        );
        Self::new_with_heralds(vertex_num, weighted_edges, vec![])
    }
    pub fn new_with_heralds(
        vertex_num: VertexNum,
        weighted_edges: Vec<HyperEdge>,
        heralds: Vec<Vec<(EdgeIndex, Weight)>>,
    ) -> Self {
        Self {
            vertex_num,
            weighted_edges,
            heralds,
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverInitializer {
    #[new]
    #[pyo3(signature = (vertex_num, weighted_edges, heralds=None))]
    fn py_new(
        py: Python<'_>,
        vertex_num: VertexNum,
        weighted_edges: Vec<HyperEdge>,
        heralds: Option<&Bound<PyList>>,
    ) -> PyResult<Self> {
        let mut heralds_vec: Vec<Vec<(EdgeIndex, Weight)>> = vec![];
        if let Some(heralds) = heralds {
            for herald in heralds.iter() {
                heralds_vec.push(
                    py_into_map::<EdgeIndex, Py<PyAny>>(&herald)?
                        .into_iter()
                        .map(|(k, v)| -> (EdgeIndex, Weight) {
                            let weight: Weight = PyRational::from(v.bind(py)).into();
                            #[cfg(feature = "f64_weight")]
                            assert!(
                                weight.is_number(),
                                "weight must be a number, but at index {:?} got {:?}",
                                k,
                                weight
                            );
                            (k, weight)
                        })
                        .collect(),
                );
            }
        }
        #[cfg(feature = "f64_weight")]
        assert!(
            weighted_edges.iter().all(|h| h.weight.is_number()),
            "weight must be a number, but (index, weight) = {:?}",
            weighted_edges
                .iter()
                .enumerate()
                .find(|(_, h)| !h.weight.is_number())
                .map(|(i, h)| (i, h.weight.clone()))
                .unwrap()
        );
        Ok(Self::new_with_heralds(vertex_num, weighted_edges, heralds_vec))
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    #[getter]
    fn get_vertex_num(&self) -> VertexNum {
        self.vertex_num
    }
    #[setter]
    fn set_vertex_num(&mut self, vertex_num: VertexNum) {
        self.vertex_num = vertex_num;
    }
    #[getter]
    fn get_weighted_edges(&self) -> Vec<HyperEdge> {
        self.weighted_edges.clone()
    }
    #[setter]
    fn set_weighted_edges(&mut self, weighted_edges: Vec<HyperEdge>) {
        self.weighted_edges = weighted_edges;
    }
    #[getter]
    fn get_heralds(&self) -> Vec<std::collections::HashMap<EdgeIndex, PyRational>> {
        self.heralds
            .iter()
            .map(|x| x.iter().map(|(k, v)| (*k, v.clone().into())).collect())
            .collect()
    }
    #[setter]
    fn set_heralds(&mut self, py: Python<'_>, heralds: &Bound<PyList>) -> PyResult<()> {
        self.heralds = vec![];
        for herald in heralds.iter() {
            self.heralds.push(
                py_into_map::<EdgeIndex, Py<PyAny>>(&herald)?
                    .into_iter()
                    .map(|(k, v)| -> (EdgeIndex, Weight) { (k, PyRational::from(v.bind(py)).into()) })
                    .collect(),
            );
        }
        Ok(())
    }
    #[pyo3(name = "snapshot", signature = (abbrev=true))]
    fn py_snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.snapshot(abbrev))
    }
    #[pyo3(name = "get_subgraph_syndrome")]
    fn py_get_subgraph_syndrome(&self, subgraph: PySubgraph) -> FastIterSet<VertexIndex> {
        self.get_subgraph_syndrome(&subgraph.into())
    }
    #[pyo3(name = "matches_subgraph_syndrome")]
    fn py_matches_subgraph_syndrome(&self, subgraph: PySubgraph, defect_vertices: Vec<VertexIndex>) -> bool {
        self.matches_subgraph_syndrome(&subgraph.into(), &defect_vertices)
    }
    #[pyo3(name = "normalize_weights", signature = (avr_weight=None))]
    fn py_normalize_weights<'a>(mut slf: PyRefMut<'a, Self>, avr_weight: Option<&Bound<PyAny>>) -> PyRefMut<'a, Self> {
        let value: &mut Self = &mut *slf;
        use crate::num_traits::One;
        value.normalize_weights(avr_weight.map(|x| PyRational::from(x).0).unwrap_or_else(|| Rational::one()));
        slf
    }
    #[pyo3(name = "uniform_weights", signature = (weight=None))]
    fn py_uniform_weights<'a>(mut slf: PyRefMut<'a, Self>, weight: Option<&Bound<PyAny>>) -> PyRefMut<'a, Self> {
        let value: &mut Self = &mut *slf;
        use crate::num_traits::One;
        value.uniform_weights(weight.map(|x| PyRational::from(x).0).unwrap_or_else(|| Rational::one()));
        slf
    }
    #[pyo3(name = "to_json")]
    fn py_to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(value: &Bound<PyAny>) -> Self {
        serde_json::from_value(pyobject_to_json_locked(value)).unwrap()
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("vertex_num", self.vertex_num)?;
        kwargs.set_item("weighted_edges", self.weighted_edges.clone())?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
}

impl SolverInitializer {
    /// sanity check to avoid duplicate edges that are hard to debug
    pub fn sanity_check(&self) -> Result<(), String> {
        use crate::example_codes::*;
        let code = ErrorPatternReader::from_initializer(self);
        code.sanity_check()
    }

    pub fn matches_subgraph_syndrome(&self, subgraph: &OutputSubgraph, defect_vertices: &[VertexIndex]) -> bool {
        let mut subgraph_defect_vertices: Vec<_> = self.get_subgraph_syndrome(subgraph).into_iter().collect();
        let mut defect_vertices = defect_vertices.to_owned();
        subgraph_defect_vertices.sort();
        defect_vertices.sort();
        subgraph_defect_vertices.sort();
        if defect_vertices.len() != subgraph_defect_vertices.len() {
            println!(
                "defect vertices: {:?}\nsubgraph_defect_vertices: {:?}",
                defect_vertices, subgraph_defect_vertices
            );
            return false;
        }
        for i in 0..defect_vertices.len() {
            if defect_vertices[i] != subgraph_defect_vertices[i] {
                println!(
                    "defect vertices: {:?}\nsubgraph_defect_vertices: {:?}",
                    defect_vertices, subgraph_defect_vertices
                );
                return false;
            }
        }
        true
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_subgraph_total_weight(&self, subgraph: &OutputSubgraph) -> Weight {
        let mut weight = Weight::zero();
        for &edge_index in subgraph.iter() {
            weight += self.weighted_edges[edge_index as usize].weight.clone();
        }
        weight
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_subgraph_syndrome(&self, subgraph: &OutputSubgraph) -> FastIterSet<VertexIndex> {
        let mut defect_vertices = FastIterSet::new();
        for &edge_index in subgraph.iter() {
            let HyperEdge { vertices, .. } = &self.weighted_edges[edge_index as usize];
            for &vertex_index in vertices.iter() {
                if defect_vertices.contains(&vertex_index) {
                    defect_vertices.remove(&vertex_index);
                    // println!("duplicate defect vertex: {}", vertex_index);
                } else {
                    defect_vertices.insert(vertex_index);
                }
            }
        }
        defect_vertices
    }

    pub fn normalize_weights(&mut self, average_weight: Rational) {
        let total_weight = self.weighted_edges.iter().map(|edge| &edge.weight).sum::<Rational>();
        let scale = average_weight / (total_weight / Rational::from_usize(self.weighted_edges.len()).unwrap());
        for edge in self.weighted_edges.iter_mut() {
            edge.weight = edge.weight.clone() * scale.clone();
        }
    }

    pub fn uniform_weights(&mut self, weight: Rational) {
        for edge in self.weighted_edges.iter_mut() {
            edge.weight = weight.clone();
        }
    }
}

impl MWPSVisualizer for SolverInitializer {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let mut vertices = Vec::<serde_json::Value>::new();
        let mut edges = Vec::<serde_json::Value>::new();
        for _ in 0..self.vertex_num {
            vertices.push(json!({}));
        }
        for HyperEdge { vertices, weight } in self.weighted_edges.iter() {
            edges.push(json!({
                if abbrev { "w" } else { "weight" }: weight.to_f64(),
                "wn": numer_of(weight),
                "wd": denom_of(weight),
                if abbrev { "v" } else { "vertices" }: vertices,
            }));
        }
        json!({
            "vertices": vertices,
            "edges": edges,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct SyndromePattern {
    /// the vertices corresponding to defect measurements
    pub defect_vertices: Vec<VertexIndex>,
    /// the edges that experience erasures, i.e. known errors
    pub erasures: Vec<EdgeIndex>,
    /// the heralded weighted edges index
    pub heralds: Vec<HeraldIndex>,
    /// a set of new weights that are mixed with existing weights; this will override
    /// the weight changes of erasures and heralds
    pub override_weights: Option<(Vec<Weight>, Weight, Option<Weight>)>,
}

impl SyndromePattern {
    pub fn new_with_erasure_heralds(
        defect_vertices: Vec<VertexIndex>,
        erasures: Vec<EdgeIndex>,
        heralds: Vec<HeraldIndex>,
    ) -> Self {
        Self {
            defect_vertices,
            erasures,
            heralds,
            override_weights: None,
        }
    }
    pub fn new_with_override_weights(
        defect_vertices: Vec<VertexIndex>,
        weights: Vec<Weight>,
        ratio: Weight,
        floor_weight: Option<Weight>,
    ) -> Self {
        Self {
            defect_vertices,
            erasures: vec![],
            heralds: vec![],
            override_weights: Some((weights, ratio, floor_weight)),
        }
    }
    pub fn new_vertices(defect_vertices: Vec<VertexIndex>) -> Self {
        Self::new_erasure(defect_vertices, vec![])
    }
    pub fn new_erasure(defect_vertices: Vec<VertexIndex>, erasures: Vec<EdgeIndex>) -> Self {
        Self::new_with_erasure_heralds(defect_vertices, erasures, vec![])
    }
    pub fn new_heralds(defect_vertices: Vec<VertexIndex>, heralds: Vec<HeraldIndex>) -> Self {
        Self::new_with_erasure_heralds(defect_vertices, vec![], heralds)
    }
    pub fn new_empty() -> Self {
        Self::new_vertices(vec![])
    }
}

impl MWPSVisualizer for SyndromePattern {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let vertex_num = self.defect_vertices.iter().cloned().max().unwrap_or_default() + 1;
        let mut vertices = vec![json!(null); vertex_num];
        for &vertex_index in self.defect_vertices.iter() {
            vertices[vertex_index] = json!({
                if abbrev { "s" } else { "is_defect" }: 1,
            })
        }
        assert!(self.erasures.is_empty(), "erasures are not supported in the snapshot");
        json!({
            "hint_no_vertices_check": true,
            "vertices": vertices,
        })
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SyndromePattern {
    #[new]
    #[pyo3(signature = (defect_vertices=None, erasures=None, heralds=None, override_weights=None, override_ratio=None, floor_weight=None))]
    fn py_new(
        defect_vertices: Option<&Bound<PyAny>>,
        erasures: Option<&Bound<PyAny>>,
        heralds: Option<&Bound<PyAny>>,
        override_weights: Option<&Bound<PyList>>,
        override_ratio: Option<&Bound<PyAny>>,
        floor_weight: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        use crate::util_py::py_into_vec;
        let defect_vertices: Vec<VertexIndex> = if let Some(defect_vertices) = defect_vertices {
            py_into_vec(defect_vertices)?
        } else {
            vec![]
        };
        if let Some(override_weights) = override_weights {
            assert!(
                erasures.is_none() && heralds.is_none(),
                "do not set erasures or heralds when override weights are provided"
            );
            let ratio = override_ratio
                .map(|x| PyRational::from(x).into())
                .unwrap_or_else(|| Rational::from_f64(1.0).unwrap());
            let floor_weight = floor_weight.map(|x| PyRational::from(x).into());
            Ok(Self::new_with_override_weights(
                defect_vertices,
                override_weights.iter().map(|x| PyRational::from(&x).into()).collect(),
                ratio,
                floor_weight,
            ))
        } else {
            let erasures: Vec<EdgeIndex> = if let Some(erasures) = erasures {
                py_into_vec(erasures)?
            } else {
                vec![]
            };
            let heralds: Vec<HeraldIndex> = if let Some(heralds) = heralds {
                py_into_vec(heralds)?
            } else {
                vec![]
            };
            Ok(Self::new_with_erasure_heralds(defect_vertices, erasures, heralds))
        }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    #[getter]
    fn get_defect_vertices(&self) -> Vec<VertexIndex> {
        self.defect_vertices.clone()
    }
    #[setter]
    fn set_defect_vertices(&mut self, defect_vertices: Vec<VertexIndex>) {
        self.defect_vertices = defect_vertices;
    }
    #[getter]
    fn get_erasures(&self) -> Vec<EdgeIndex> {
        self.erasures.clone()
    }
    #[setter]
    fn set_erasures(&mut self, erasures: Vec<EdgeIndex>) {
        self.erasures = erasures;
    }
    #[getter]
    fn get_heralds(&self) -> Vec<HeraldIndex> {
        self.heralds.clone()
    }
    #[setter]
    fn set_heralds(&mut self, heralds: Vec<HeraldIndex>) {
        self.heralds = heralds;
    }
    #[getter]
    fn get_override_weights(&self) -> Option<(Vec<PyRational>, PyRational, Option<PyRational>)> {
        if let Some((weights, ratio, floor_weight)) = self.override_weights.as_ref() {
            return Some((
                weights.iter().map(|x| x.clone().into()).collect(),
                ratio.clone().into(),
                floor_weight.clone().map(|x| x.into()),
            ));
        }
        None
    }
    #[setter]
    fn set_override_weights(&mut self, override_weights: Option<(Vec<PyRational>, PyRational, Option<PyRational>)>) {
        if let Some((weights, ratio, floor_weight)) = override_weights {
            self.override_weights = Some((
                weights.iter().map(|x| x.0.clone()).collect(),
                ratio.0.clone(),
                floor_weight.map(|x| x.0.clone()),
            ));
        } else {
            self.override_weights = None;
        }
    }
    #[pyo3(name="snapshot", signature = (abbrev=true))]
    fn py_snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.snapshot(abbrev))
    }
    #[pyo3(name = "to_json")]
    fn py_to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(value: &Bound<PyAny>) -> Self {
        serde_json::from_value(pyobject_to_json_locked(value)).unwrap()
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("defect_vertices", self.defect_vertices.clone())?;
        kwargs.set_item("erasures", self.erasures.clone())?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
}

#[allow(dead_code)]
/// use Xoshiro256StarStar for deterministic random number generator
pub type DeterministicRng = rand_xoshiro::Xoshiro256StarStar;

pub trait F64Rng {
    fn next_f64(&mut self) -> f64;
}

impl F64Rng for DeterministicRng {
    fn next_f64(&mut self) -> f64 {
        f64::from_bits(0x3FF << 52 | self.next_u64() >> 12) - 1.
    }
}

/// the result of MWPF algorithm: a parity subgraph (defined by some edges that,
/// if are selected, will generate the parity result in the syndrome)
pub type Subgraph = Vec<EdgeIndex>;

pub struct OutputSubgraph {
    pub subgraph: Subgraph,
    pub flip_edge_indices: hashbrown::HashSet<EdgeIndex>,
}

impl OutputSubgraph {
    pub fn new(subgraph: Subgraph, flip_edge_indices: hashbrown::HashSet<EdgeIndex>) -> Self {
        Self {
            subgraph,
            flip_edge_indices,
        }
    }

    pub fn iter(&self) -> OutputSubgraphIter {
        OutputSubgraphIter {
            subgraph_iter: self.subgraph.iter(),
            flip_edge_indices: &self.flip_edge_indices,
            remaining_indices: self.flip_edge_indices.clone(),
        }
    }

    // Mutable iterator with updates to `subgraph` during iteration
    pub fn iter_mut(&mut self) -> OutputSubgraphIterMut {
        OutputSubgraphIterMut {
            subgraph: &mut self.subgraph,
            subgraph_iter: 0, // Start iterating from the beginning of `subgraph`
            flip_edge_indices: &mut self.flip_edge_indices,
        }
    }
}

impl From<Subgraph> for OutputSubgraph {
    fn from(value: Subgraph) -> Self {
        Self::new(value, hashbrown::HashSet::new())
    }
}

// consuming iterators
// Implementing `IntoIterator` for `&OutputSubgraph` (for `iter`)
impl<'a> IntoIterator for &'a OutputSubgraph {
    type Item = &'a usize;
    type IntoIter = OutputSubgraphIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        OutputSubgraphIter {
            subgraph_iter: self.subgraph.iter(),
            flip_edge_indices: &self.flip_edge_indices,
            remaining_indices: self.flip_edge_indices.clone(),
        }
    }
}

// Implementing `IntoIterator` for `&mut OutputSubgraph` (for `iter_mut`)
impl<'a> IntoIterator for &'a mut OutputSubgraph {
    type Item = &'a mut usize;
    type IntoIter = OutputSubgraphIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        OutputSubgraphIterMut {
            subgraph: &mut self.subgraph,
            subgraph_iter: 0, // Start at the beginning of `subgraph`
            flip_edge_indices: &mut self.flip_edge_indices,
        }
    }
}

// Implementing `IntoIterator` for `OutputSubgraph` (for `into_iter`)
impl IntoIterator for OutputSubgraph {
    type Item = usize;
    type IntoIter = OutputSubgraphIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        OutputSubgraphIntoIter {
            subgraph_iter: self.subgraph.into_iter(),
            flip_edge_indices: self.flip_edge_indices.clone(),
            remaining_indices: self.flip_edge_indices,
        }
    }
}

pub struct OutputSubgraphIter<'a> {
    subgraph_iter: std::slice::Iter<'a, usize>,
    flip_edge_indices: &'a hashbrown::HashSet<EdgeIndex>,
    remaining_indices: hashbrown::HashSet<EdgeIndex>,
}

impl<'a> Iterator for OutputSubgraphIter<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        // note: optional short circuiting
        if self.flip_edge_indices.is_empty() {
            return self.subgraph_iter.next();
        }

        // Iterate over the `subgraph` elements
        while let Some(index) = self.subgraph_iter.next() {
            if self.flip_edge_indices.contains(index) {
                // Record this index as seen and skip it in output
                self.remaining_indices.remove(index);
                continue;
            } else {
                return Some(index);
            }
        }

        // After finishing subgraph, yield elements from `flip_edge_indices` that were not seen
        if let Some(&remaining_index) = self.remaining_indices.iter().next() {
            self.remaining_indices.remove(&remaining_index);
            return Some(self.flip_edge_indices.get(&remaining_index).unwrap());
        }

        // No more elements to yield
        None
    }
}

// Mutable iterator
pub struct OutputSubgraphIterMut<'a> {
    subgraph: &'a mut Subgraph,
    subgraph_iter: usize, // Index within `subgraph`
    flip_edge_indices: &'a mut hashbrown::HashSet<EdgeIndex>,
}

// note: use of unsafe
impl<'a> Iterator for OutputSubgraphIterMut<'a> {
    type Item = &'a mut usize;

    fn next(&mut self) -> Option<Self::Item> {
        // Iterate over the `subgraph` elements first

        let len = self.subgraph.len();
        while self.subgraph_iter < len {
            let index = self.subgraph_iter;
            self.subgraph_iter += 1;
            let elem = self.subgraph[index];

            // Skip elements in `flip_edge_indices`
            if self.flip_edge_indices.contains(&elem) {
                self.flip_edge_indices.remove(&elem);
                self.subgraph_iter -= 1;
                self.subgraph.remove(self.subgraph_iter);
                continue;
            } else {
                // Using `unsafe` to circumvent borrowing rules safely
                return Some(unsafe { &mut *(&mut self.subgraph[index] as *mut usize) });
            }
        }

        // After `subgraph` elements, add remaining `flip_edge_indices` to `subgraph`
        if let Some(&remaining_index) = self.flip_edge_indices.iter().next() {
            self.flip_edge_indices.remove(&remaining_index);
            self.subgraph.push(remaining_index);
            self.subgraph_iter += 1; // Update to point to the newly added element

            // Using `unsafe` to return a mutable reference to the last element, guaranteed to exist
            return Some(unsafe { &mut *(self.subgraph.last_mut().unwrap() as *mut usize) });
        }

        // No more elements to yield
        None
    }
}

// Consuming iterator
pub struct OutputSubgraphIntoIter {
    subgraph_iter: std::vec::IntoIter<usize>,
    flip_edge_indices: hashbrown::HashSet<EdgeIndex>,
    remaining_indices: hashbrown::HashSet<EdgeIndex>,
}

impl Iterator for OutputSubgraphIntoIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.flip_edge_indices.is_empty() {
            return self.subgraph_iter.next();
        }

        while let Some(index) = self.subgraph_iter.next() {
            if self.flip_edge_indices.contains(&index) {
                self.remaining_indices.remove(&index);
                continue;
            } else {
                return Some(index);
            }
        }

        if let Some(&remaining_index) = self.remaining_indices.iter().next() {
            self.remaining_indices.remove(&remaining_index);
            Some(remaining_index)
        } else {
            None
        }
    }
}

impl MWPSVisualizer for OutputSubgraph {
    fn snapshot(&self, _abbrev: bool) -> serde_json::Value {
        let mut adjusted_subgraph_set = self.subgraph.iter().collect::<hashbrown::HashSet<_>>();
        for to_flip in self.flip_edge_indices.iter() {
            if adjusted_subgraph_set.contains(to_flip) {
                adjusted_subgraph_set.remove(to_flip);
            } else {
                adjusted_subgraph_set.insert(to_flip);
            }
        }
        let adjusted_subgraph = adjusted_subgraph_set.into_iter().collect::<Vec<_>>();
        json!({
            "subgraph": self.subgraph,
            "flip_edge_indices": self.flip_edge_indices.iter().collect::<Vec<_>>(),
            "adjusted_subgraph_for_negative_weight": adjusted_subgraph
        })
    }
}

#[allow(clippy::to_string_in_format_args)]
impl std::fmt::Debug for OutputSubgraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Create adjusted subgraph set
        let mut adjusted_subgraph_set = self.subgraph.iter().copied().collect::<hashbrown::HashSet<_>>();
        for &to_flip in &self.flip_edge_indices {
            if adjusted_subgraph_set.contains(&to_flip) {
                adjusted_subgraph_set.remove(&to_flip);
            } else {
                adjusted_subgraph_set.insert(to_flip);
            }
        }
        let adjusted_subgraph = adjusted_subgraph_set.into_iter().collect::<Vec<_>>();

        // Output debug information in similar format to snapshot
        write!(
            f,
            "{}",
            json!({
                "subgraph": self.subgraph,
                "flip_edge_indices": self.flip_edge_indices.iter().collect::<Vec<_>>(),
                "adjusted_subgraph_for_negative_weight": adjusted_subgraph
            })
            .to_string()
        )
    }
}

impl MWPSVisualizer for Subgraph {
    fn snapshot(&self, _abbrev: bool) -> serde_json::Value {
        json!({
            "subgraph": self,
        })
    }
}

/// the range of the optimal MWPF solution's weight
#[derive(Clone, Debug)]
pub struct WeightRange {
    pub lower: Rational,
    pub upper: Rational,
}

impl WeightRange {
    pub fn new(lower: Rational, upper: Rational) -> Self {
        Self { lower, upper }
    }
    /// a solution is optimal only if the range is a single point
    pub fn is_optimal(&self) -> bool {
        self.lower == self.upper
    }
}

impl MWPSVisualizer for WeightRange {
    fn snapshot(&self, _abbrev: bool) -> serde_json::Value {
        json!({
            "weight_range": {
                "lower": self.lower.to_f64(),
                "upper": self.upper.to_f64(),
                "ln": numer_of(&self.lower),
                "ld": denom_of(&self.lower),
                "un": numer_of(&self.upper),
                "ud": denom_of(&self.upper),
            },
        })
    }
}

/// record the decoding time of multiple syndrome patterns
pub struct BenchmarkProfiler {
    /// each record corresponds to a different syndrome pattern
    pub records: Vec<BenchmarkProfilerEntry>,
    /// summation of all decoding time
    pub sum_round_time: f64,
    /// syndrome count
    pub sum_syndrome: usize,
    /// error count
    pub sum_error: usize,
    /// noisy measurement round
    pub noisy_measurements: VertexNum,
    /// the file to output the profiler results
    pub benchmark_profiler_output: Option<File>,

    /// summation of all tuning time
    pub sum_tuning_time: f64,
}

impl BenchmarkProfiler {
    pub fn new(noisy_measurements: VertexNum, detail_log_file: Option<String>) -> Self {
        let benchmark_profiler_output = detail_log_file.map(|filename| {
            let mut file = File::create(filename).unwrap();
            file.write_all(
                serde_json::to_string(&json!({
                    "noisy_measurements": noisy_measurements,
                }))
                .unwrap()
                .as_bytes(),
            )
            .unwrap();
            file.write_all(b"\n").unwrap();
            file
        });
        Self {
            records: vec![],
            sum_round_time: 0.,
            sum_syndrome: 0,
            sum_error: 0,
            noisy_measurements,
            benchmark_profiler_output,
            sum_tuning_time: 0.,
        }
    }
    /// record the beginning of a decoding procedure
    pub fn begin(&mut self, syndrome_pattern: &SyndromePattern, error_pattern: &Subgraph) {
        // sanity check last entry, if exists, is complete
        if let Some(last_entry) = self.records.last() {
            assert!(
                last_entry.is_complete(),
                "the last benchmark profiler entry is not complete, make sure to call `begin` and `end` in pairs"
            );
        }
        let entry = BenchmarkProfilerEntry::new(syndrome_pattern, error_pattern);
        self.records.push(entry);
        self.records.last_mut().unwrap().record_begin();
    }
    pub fn event(&mut self, event_name: String) {
        let last_entry = self
            .records
            .last_mut()
            .expect("last entry not exists, call `begin` before `end`");
        last_entry.record_event(event_name);
    }
    /// record the ending of a decoding procedure
    pub fn end(&mut self, solver: Option<&dyn SolverTrait>) {
        let last_entry = self
            .records
            .last_mut()
            .expect("last entry not exists, call `begin` before `end`");
        last_entry.record_end();
        self.sum_round_time += last_entry.round_time.unwrap();
        self.sum_syndrome += last_entry.syndrome_pattern.defect_vertices.len();
        self.sum_error += last_entry.error_pattern.len();
        if let Some(file) = self.benchmark_profiler_output.as_mut() {
            let mut events = serde_json::Map::new();
            for (event_name, time) in last_entry.events.iter() {
                events.insert(event_name.clone(), json!(time));
            }
            let mut value = json!({
                "round_time": last_entry.round_time.unwrap(),
                "defect_num": last_entry.syndrome_pattern.defect_vertices.len(),
                "error_num": last_entry.error_pattern.len(),
                "events": events,
            });
            if let Some(solver) = solver {
                let solver_profile = solver.generate_profiler_report();
                let value_mut = value.as_object_mut().unwrap();
                value_mut.insert("solver_profile".to_string(), solver_profile);
                if let Some(tuning_time) = solver.get_tuning_time() {
                    value_mut.insert("tuning_time".to_string(), tuning_time.into());
                    self.sum_tuning_time += tuning_time;
                }
            }
            file.write_all(serde_json::to_string(&value).unwrap().as_bytes()).unwrap();
            file.write_all(b"\n").unwrap();
        } else if let Some(solver) = solver {
            if let Some(tuning_time) = solver.get_tuning_time() {
                self.sum_tuning_time += tuning_time;
            }
        }
    }
    /// print out a brief one-line statistics
    pub fn brief(&self) -> String {
        let total = self.sum_round_time / (self.records.len() as f64);
        let per_round = total / (1. + self.noisy_measurements as f64);
        let per_defect = self.sum_round_time / (self.sum_syndrome as f64);
        format!("total: {total:.3e}, round: {per_round:.3e}, syndrome: {per_defect:.3e},")
    }
}

pub struct BenchmarkProfilerEntry {
    /// the syndrome pattern of this decoding problem
    pub syndrome_pattern: SyndromePattern,
    /// the error pattern
    pub error_pattern: Subgraph,
    /// the time of beginning a decoding procedure
    begin_time: Option<Instant>,
    /// record additional events
    pub events: Vec<(String, f64)>,
    /// interval between calling [`Self::record_begin`] to calling [`Self::record_end`]
    pub round_time: Option<f64>,
}

impl BenchmarkProfilerEntry {
    pub fn new(syndrome_pattern: &SyndromePattern, error_pattern: &Subgraph) -> Self {
        Self {
            syndrome_pattern: syndrome_pattern.clone(),
            error_pattern: error_pattern.clone(),
            begin_time: None,
            events: vec![],
            round_time: None,
        }
    }
    /// record the beginning of a decoding procedure
    pub fn record_begin(&mut self) {
        assert_eq!(self.begin_time, None, "do not call `record_begin` twice on the same entry");
        self.begin_time = Some(Instant::now());
    }
    /// record the ending of a decoding procedure
    pub fn record_end(&mut self) {
        let begin_time = self
            .begin_time
            .as_ref()
            .expect("make sure to call `record_begin` before calling `record_end`");
        self.round_time = Some(begin_time.elapsed().as_secs_f64());
    }
    pub fn record_event(&mut self, event_name: String) {
        let begin_time = self
            .begin_time
            .as_ref()
            .expect("make sure to call `record_begin` before calling `record_end`");
        self.events.push((event_name, begin_time.elapsed().as_secs_f64()));
    }
    pub fn is_complete(&self) -> bool {
        self.round_time.is_some()
    }
}

#[cfg(feature = "python_binding")]
pub fn json_to_pyobject_locked(value: serde_json::Value, py: Python) -> PyObject {
    match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(value) => value.into_pyobject(py).unwrap().to_owned().into(),
        serde_json::Value::Number(value) => {
            if value.is_i64() {
                value.as_i64().into_pyobject(py).unwrap().to_owned().into()
            } else {
                value.as_f64().into_pyobject(py).unwrap().to_owned().into()
            }
        }
        serde_json::Value::String(value) => value.into_pyobject(py).unwrap().to_owned().into(),
        serde_json::Value::Array(array) => {
            let elements: Vec<PyObject> = array.into_iter().map(|value| json_to_pyobject_locked(value, py)).collect();
            PyList::new(py, elements).unwrap().into()
        }
        serde_json::Value::Object(map) => {
            let pydict = PyDict::new(py);
            for (key, value) in map.into_iter() {
                let pyobject = json_to_pyobject_locked(value, py);
                pydict.set_item(key, pyobject).unwrap();
            }
            pydict.into()
        }
    }
}

#[cfg(feature = "python_binding")]
pub fn json_to_pyobject(value: serde_json::Value) -> PyObject {
    Python::with_gil(|py| json_to_pyobject_locked(value, py))
}

#[cfg(feature = "python_binding")]
pub fn pyobject_to_json_locked(value: &Bound<PyAny>) -> serde_json::Value {
    if value.is_none() {
        serde_json::Value::Null
    } else if value.is_instance_of::<pyo3::types::PyBool>() {
        json!(value.extract::<bool>().unwrap())
    } else if value.is_instance_of::<pyo3::types::PyInt>() {
        json!(value.extract::<i64>().unwrap())
    } else if value.is_instance_of::<PyFloat>() {
        json!(value.extract::<f64>().unwrap())
    } else if value.is_instance_of::<pyo3::types::PyString>() {
        json!(value.extract::<String>().unwrap())
    } else if value.is_instance_of::<PyList>() {
        let elements: Vec<serde_json::Value> = value
            .downcast::<PyList>()
            .unwrap()
            .into_iter()
            .map(|object| pyobject_to_json_locked(&object))
            .collect();
        json!(elements)
    } else if value.is_instance_of::<PyDict>() {
        let map: &Bound<PyDict> = value.downcast().unwrap();
        let mut json_map = serde_json::Map::new();
        for (key, value) in map.iter() {
            json_map.insert(key.extract::<String>().unwrap(), pyobject_to_json_locked(&value));
        }
        serde_json::Value::Object(json_map)
    } else {
        unimplemented!("unsupported python type, should be (cascaded) dict, list and basic numerical types")
    }
}

#[cfg(feature = "python_binding")]
pub fn pyobject_to_json(value: PyObject) -> serde_json::Value {
    Python::with_gil(|py| pyobject_to_json_locked(value.bind(py)))
}

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SolverInitializer>()?;
    m.add_class::<SyndromePattern>()?;
    m.add_class::<HyperEdge>()?;
    m.add_class::<BenchmarkSuite>()?;
    Ok(())
}

pub fn rational_approx_eq(a: &Rational, b: &Rational) -> bool {
    #[cfg(feature = "rational_weight")]
    use crate::num_traits::Signed;
    if a == b {
        return true;
    }
    (a - b).abs() / b < Rational::from_float(1e-6).unwrap()
}

pub fn rational_approx_le(a: &Rational, b: &Rational) -> bool {
    if a < b {
        return true;
    }
    (b - a) / b < Rational::from_float(1e-6).unwrap()
}

pub fn rational_approx_ge(a: &Rational, b: &Rational) -> bool {
    if a > b {
        return true;
    }
    (b - a) / b < Rational::from_float(1e-6).unwrap()
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct BenchmarkSuite {
    pub initializer: SolverInitializer,
    pub syndrome_patterns: Vec<SyndromePattern>,
}

impl BenchmarkSuite {
    pub fn new(initializer: SolverInitializer, syndrome_patterns: Vec<SyndromePattern>) -> Self {
        Self {
            initializer,
            syndrome_patterns,
        }
    }
    pub fn save_cbor(&self, filename: &str) {
        let file = File::create(filename).expect("Failed to create file");
        let writer = BufWriter::new(file);
        ciborium::ser::into_writer(&CompressedBenchmarkSuite::from(self), writer).expect("Failed to serialize data");
    }
    pub fn from_cbor(filename: &str) -> Self {
        let file = File::open(filename).expect("Failed to open file");
        let reader = BufReader::new(file);
        let compressed: CompressedBenchmarkSuite = ciborium::de::from_reader(reader).expect("Failed to deserialize data");
        (&compressed).into()
    }
    pub fn append(&mut self, syndrome: SyndromePattern) {
        self.syndrome_patterns.push(syndrome);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBenchmarkSuite {
    // initializer
    pub vertex_num: VertexNum,
    pub weighted_edges: Vec<(Vec<VertexIndex>, Weight)>,
    pub heralds: Vec<Vec<(EdgeIndex, Weight)>>,
    // syndrome patterns
    pub syndrome_defect_vertices: Vec<Vec<VertexIndex>>,
    pub syndrome_erasures: Vec<Vec<EdgeIndex>>,
    pub syndrome_heralds: Vec<Vec<HeraldIndex>>,
    pub syndrome_override_weights: Vec<Option<(Vec<Weight>, Weight, Option<Weight>)>>,
}

impl From<&BenchmarkSuite> for CompressedBenchmarkSuite {
    fn from(benchmark_suite: &BenchmarkSuite) -> Self {
        Self {
            vertex_num: benchmark_suite.initializer.vertex_num,
            weighted_edges: benchmark_suite
                .initializer
                .weighted_edges
                .iter()
                .map(|hyperedge| (hyperedge.vertices.clone(), hyperedge.weight.clone()))
                .collect(),
            heralds: benchmark_suite.initializer.heralds.clone(),
            syndrome_defect_vertices: benchmark_suite
                .syndrome_patterns
                .iter()
                .map(|syndrome| syndrome.defect_vertices.clone())
                .collect(),
            syndrome_erasures: benchmark_suite
                .syndrome_patterns
                .iter()
                .map(|syndrome| syndrome.erasures.clone())
                .collect(),
            syndrome_heralds: benchmark_suite
                .syndrome_patterns
                .iter()
                .map(|syndrome| syndrome.heralds.clone())
                .collect(),
            syndrome_override_weights: benchmark_suite
                .syndrome_patterns
                .iter()
                .map(|syndrome| syndrome.override_weights.clone())
                .collect(),
        }
    }
}

impl From<&CompressedBenchmarkSuite> for BenchmarkSuite {
    fn from(compressed_benchmark_suite: &CompressedBenchmarkSuite) -> Self {
        let initializer = SolverInitializer {
            vertex_num: compressed_benchmark_suite.vertex_num,
            weighted_edges: compressed_benchmark_suite
                .weighted_edges
                .iter()
                .map(|(vertices, weight)| HyperEdge {
                    vertices: vertices.clone(),
                    weight: weight.clone(),
                })
                .collect(),
            heralds: compressed_benchmark_suite.heralds.clone(),
        };
        let syndrome_patterns = izip!(
            compressed_benchmark_suite.syndrome_defect_vertices.iter(),
            compressed_benchmark_suite.syndrome_erasures.iter(),
            compressed_benchmark_suite.syndrome_heralds.iter(),
            compressed_benchmark_suite.syndrome_override_weights.iter()
        )
        .map(|(defect_vertices, erasures, heralds, override_weights)| SyndromePattern {
            defect_vertices: defect_vertices.clone(),
            erasures: erasures.clone(),
            heralds: heralds.clone(),
            override_weights: override_weights.clone(),
        })
        .collect();
        Self::new(initializer, syndrome_patterns)
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl BenchmarkSuite {
    #[new]
    #[pyo3(signature = (initializer, syndrome_patterns=vec![]))]
    fn py_new(initializer: SolverInitializer, syndrome_patterns: Vec<SyndromePattern>) -> PyResult<Self> {
        Ok(Self::new(initializer, syndrome_patterns))
    }
    fn __repr__(&self) -> String {
        format!(
            "BenchmarkSuite {{ vertex_num: {}, edge_num: {}, shots: {} }}",
            self.initializer.vertex_num,
            self.initializer.weighted_edges.len(),
            self.syndrome_patterns.len()
        )
    }
    #[pyo3(name = "save_cbor")]
    fn py_save_cbor(&self, filename: String) {
        self.save_cbor(&filename)
    }
    #[staticmethod]
    #[pyo3(name = "from_cbor")]
    fn py_from_cbor(filename: String) -> Self {
        Self::from_cbor(&filename)
    }
    #[pyo3(name = "append")]
    fn py_append(&mut self, syndrome: SyndromePattern) {
        self.append(syndrome)
    }
}

pub fn sorted_vec<T: Ord>(vec: Vec<T>) -> Vec<T> {
    let mut vec = vec;
    vec.sort();
    vec
}

pub fn sorted_vec_option<T: Ord>(vec: Option<Vec<T>>) -> Option<Vec<T>> {
    vec.map(|vec| sorted_vec(vec))
}

#[cfg(test)]
pub mod tests {
    use crate::example_codes::ExampleCode;

    use super::*;
    use bytebuffer::ByteBuffer;
    use hashbrown::HashSet;
    use num_bigint::BigInt;
    use std::str::FromStr;

    #[test]
    fn util_py_json_bigint() {
        // cargo test util_py_json_bigint -- --nocapture
        let small_int = BigInt::from(123);
        let big_int = BigInt::from_str("123456789012345678901234567890123").unwrap();
        println!("small_int: {:?}, json: {}", small_int, json!(small_int));
        println!("positive big_int: {:?}, json: {}", big_int, json!(big_int));
        println!("negative big_int: {:?}, json: {}", -big_int.clone(), json!(-big_int));
        let zero_int = BigInt::from(0);
        println!("zero_int: {:?}, json: {}", zero_int, json!(zero_int));
    }

    #[test]
    fn test_iter() {
        let subgraph = vec![1, 2, 3, 4];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(5);

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Expected behavior: `2` is skipped, and `5` is added at the end.
        let result: Vec<_> = output_subgraph.iter().cloned().collect();
        assert_eq!(result, vec![1, 3, 4, 5]);
    }

    #[test]
    fn test_iter_empty_flip_edge_indices() {
        let subgraph = vec![1, 2, 3];
        let flip_edge_indices = HashSet::new();

        let output_subgraph = OutputSubgraph::new(subgraph.clone(), flip_edge_indices);

        // With empty `flip_edge_indices`, should just return all elements in `subgraph`.
        let result: Vec<_> = output_subgraph.iter().cloned().collect();
        assert_eq!(result, subgraph);
    }

    #[test]
    fn test_iter_all_elements_flipped() {
        let subgraph = vec![1, 2, 3];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(1);
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(3);
        flip_edge_indices.insert(4);

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Expected behavior: all elements in `subgraph` are skipped, and `4` is added at the end.
        let result: Vec<_> = output_subgraph.iter().cloned().collect();
        assert_eq!(result, vec![4]);
    }

    #[test]
    fn test_iter_mut() {
        let subgraph = vec![1, 2, 3, 4];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(5);

        let mut output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Modify elements during mutable iteration
        for elem in output_subgraph.iter_mut() {
            *elem *= 2;
        }

        // Verify that `2` was skipped and `5` was added and modified at the end
        assert_eq!(output_subgraph.subgraph, vec![2, 6, 8, 10]);
        assert!(output_subgraph.flip_edge_indices.is_empty());
    }

    #[test]
    fn test_iter_mut_no_modifications() {
        let subgraph = vec![10, 20, 30];
        let flip_edge_indices = HashSet::new(); // Empty flip edge indices

        let mut output_subgraph = OutputSubgraph::new(subgraph.clone(), flip_edge_indices);

        // Expected to iterate through all without any modifications to flip_edge_indices
        for elem in output_subgraph.iter_mut() {
            *elem += 1;
        }

        // Verify that all elements were incremented and no `flip_edge_indices` remains
        assert_eq!(output_subgraph.subgraph, vec![11, 21, 31]);
        assert!(output_subgraph.flip_edge_indices.is_empty());
    }

    #[test]
    fn test_into_iter() {
        let subgraph = vec![1, 2, 3, 4];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(5);

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Consuming iterator, so `output_subgraph` cannot be used afterward
        let result: Vec<_> = output_subgraph.into_iter().collect();

        // Expected behavior: `2` is skipped, and `5` is added at the end.
        assert_eq!(result, vec![1, 3, 4, 5]);
    }

    #[test]
    fn test_into_iter_all_elements_flipped() {
        let subgraph = vec![1, 2, 3];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(1);
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(3);
        flip_edge_indices.insert(4);

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Consuming iterator, expected to yield only `4` at the end since all `subgraph` elements are flipped.
        let result: Vec<_> = output_subgraph.into_iter().collect();
        assert_eq!(result, vec![4]);
    }

    #[test]
    fn test_iter_empty_subgraph() {
        let subgraph = vec![];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(1);
        flip_edge_indices.insert(2);

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // With empty `subgraph`, should only yield elements in `flip_edge_indices`
        let mut result: Vec<_> = output_subgraph.iter().cloned().collect();
        result.sort();
        assert_eq!(result, vec![1, 2]); // order here doesn't matter
    }

    #[test]
    fn test_iter_mut_update_subgraph() {
        let subgraph = vec![1, 2, 3, 4];
        let mut flip_edge_indices = HashSet::new();
        flip_edge_indices.insert(2);
        flip_edge_indices.insert(5);

        let mut output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices);

        // Expected behavior: `2` is skipped, and `5` is added at the end.
        let result: Vec<_> = output_subgraph.iter_mut().map(|x| *x).collect();
        assert_eq!(result, vec![1, 3, 4, 5]);

        assert_eq!(output_subgraph.subgraph, vec![1, 3, 4, 5]);
        assert!(output_subgraph.flip_edge_indices.is_empty());
    }

    #[test]
    fn test_initializer_normalize_weight() {
        // cargo test test_initializer_normalize_weight -- --nocapture
        use crate::example_codes::CodeCapacityRepetitionCode;
        use crate::num_traits::One;
        let code = CodeCapacityRepetitionCode::new(7, 0.2);
        let mut initializer = code.get_initializer();
        initializer.normalize_weights(Rational::one());
        println!("initializer: {:?}", initializer);
        for HyperEdge { weight, .. } in initializer.weighted_edges.iter() {
            assert_eq!(weight, &Rational::one());
        }
    }

    #[test]
    fn test_exclusive_weight_sum() {
        // cargo test test_exclusive_weight_sum -- --nocapture
        // cargo test test_exclusive_weight_sum --no-default-features --features rational_weight -- --nocapture
        use crate::num_traits::One;
        let one = Weight::one();
        let zero = Weight::zero();
        assert!(rational_approx_eq(&exclusive_weight_sum(&one, &zero), &zero));
        assert!(rational_approx_eq(&exclusive_weight_sum(&zero, &one), &zero));
        assert!(rational_approx_eq(&exclusive_weight_sum(&zero, &zero), &zero));
        assert!(rational_approx_eq(
            &exclusive_weight_sum(&one, &one),
            &Weight::from_f64(0.4337808304830274).unwrap()
        ));
        let million = Weight::from_f64(1e6).unwrap();
        assert!(rational_approx_eq(&exclusive_weight_sum(&million, &zero), &zero));
        assert!(rational_approx_eq(&exclusive_weight_sum(&zero, &million), &zero));
        assert!(rational_approx_eq(
            &exclusive_weight_sum(&million, &million),
            &Weight::from_f64(1e6 - (2f64).ln()).unwrap()
        ));
    }

    fn cbor_length_of(data: impl Serialize) -> usize {
        let mut buffer = ByteBuffer::new();
        ciborium::ser::into_writer(&data, &mut buffer).unwrap();
        let length = buffer.len();
        // println!("buffer: {:?}", buffer.into_vec());
        length
    }

    #[test]
    fn test_cbor_serialization() {
        // cargo test test_cbor_serialization -- --nocapture
        assert_eq!(cbor_length_of(vec![1usize; 100]), 102); // 1 bytes each
        assert_eq!(cbor_length_of(vec![23usize; 100]), 102); // 1 bytes each
        assert_eq!(cbor_length_of(vec![24usize; 100]), 202); // 2 bytes each, not sure why 24 is the boundary
        assert_eq!(cbor_length_of(vec![255usize; 100]), 202); // 2 bytes each
        assert_eq!(cbor_length_of(vec![256usize; 100]), 302); // 3 bytes each
        assert_eq!(cbor_length_of(vec![65535usize; 100]), 302); // 3 bytes each
        assert_eq!(cbor_length_of(vec![65536usize; 100]), 502); // 5 bytes each

        // also test Vec<Vec<usize>>
        assert_eq!(cbor_length_of(vec![vec![1usize; 100]; 100]), 10202); // 1 bytes each
        assert_eq!(cbor_length_of(vec![vec![23usize; 100]; 100]), 10202); // 1 bytes each
        assert_eq!(cbor_length_of(vec![vec![24usize; 100]; 100]), 20202); // 2 bytes each, not sure why 24 is the boundary
        assert_eq!(cbor_length_of(vec![vec![255usize; 100]; 100]), 20202); // 2 bytes each
        assert_eq!(cbor_length_of(vec![vec![256usize; 100]; 100]), 30202); // 3 bytes each
        assert_eq!(cbor_length_of(vec![vec![65535usize; 100]; 100]), 30202); // 3 bytes each
        assert_eq!(cbor_length_of(vec![vec![65536usize; 100]; 100]), 50202); // 5 bytes each

        // test Vec<float>
        assert_eq!(cbor_length_of(vec![1f64; 100]), 302); // 3 bytes each
        assert_eq!(cbor_length_of(vec![3.14159001001f64; 100]), 902); // 9 bytes each

        // also test Vec<Vec<float>>
        assert_eq!(cbor_length_of(vec![vec![1f64; 100]; 100]), 30202); // 3 bytes each
        assert_eq!(cbor_length_of(vec![vec![3.14159001001f64; 100]; 100]), 90202);

        // test Vec<(usize, usize)>
        assert_eq!(cbor_length_of(vec![(1usize, 2usize); 100]), 302); // 3 bytes each
        assert_eq!(cbor_length_of(vec![(1usize, 2usize, 3usize, 4usize); 100]), 502);
        assert_eq!(cbor_length_of(vec![(1usize, 65535usize, 3usize, 65536usize); 100]), 1102);

        // test Vec<vec![]>
        assert_eq!(cbor_length_of(vec![Vec::<usize>::new(); 100]), 102); // 1 bytes each for empty vec
        assert_eq!(cbor_length_of(vec![None::<usize>; 100]), 102); // 1 bytes each for null vec
    }
}
