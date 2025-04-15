use crate::dual_module_pq::EdgeWeak;
use crate::mwpf_solver::*;
#[cfg(not(feature = "float_lp"))]
use crate::num_rational;
use crate::num_traits::{FromPrimitive, ToPrimitive};
use crate::pointers::UnsafePtr;
use crate::rand_xoshiro;
use crate::rand_xoshiro::rand_core::RngCore;
#[cfg(feature = "python_binding")]
use crate::util_py::*;
use crate::visualize::*;
use num_traits::Zero;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
#[cfg(feature = "python_binding")]
use pyo3::types::{PyDict, PyFloat, PyList};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{create_dir_all, File};
use std::io::prelude::*;
use std::time::Instant;

use hashbrown::{HashMap, HashSet};
use petgraph::{Graph, Undirected};
use std::hash::Hash;

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

pub type Weight = Rational;
pub type EdgeIndex = usize;
pub type VertexIndex = usize;
cfg_if::cfg_if! {
    if #[cfg(feature="unsafe_pointer")] {
        pub type KnownSafeRefCell<T> = std::cell::UnsafeCell<T>;
    } else {
        pub type KnownSafeRefCell<T> = std::cell::RefCell<T>;
    }
}

pub type NodeIndex = VertexIndex;
pub type DefectIndex = VertexIndex;
pub type VertexNodeIndex = VertexIndex; // must be same as VertexIndex, NodeIndex, DefectIndex
pub type VertexNum = VertexIndex;
pub type NodeNum = VertexIndex;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python_binding", pyclass)]
pub struct HyperEdge {
    /// the vertices incident to the hyperedge
    pub vertices: Vec<VertexIndex>,
    /// the weight of the hyperedge
    pub weight: Weight,
    /// whether this hyperedge is connected to any boundary vertex, used for parallel implementation
    pub connected_to_boundary_vertex: bool,
}

impl HyperEdge {
    pub fn new(vertices: Vec<VertexIndex>, weight: Weight) -> Self {
        Self {
            vertices,
            weight,
            connected_to_boundary_vertex: false,
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl HyperEdge {
    #[new]
    fn py_new(vertices: &Bound<PyAny>, weight: &Bound<PyAny>) -> PyResult<Self> {
        use crate::util_py::py_into_btree_set;
        let vertices: Vec<VertexIndex> = py_into_btree_set::<VertexIndex>(vertices)?.into_iter().collect();
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
}

#[cfg_attr(feature = "python_binding", pyclass(get_all, set_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverInitializer {
    /// the number of vertices
    pub vertex_num: VertexNum,
    /// weighted edges, where vertex indices are within the range [0, vertex_num)
    pub weighted_edges: Vec<HyperEdge>,
}

impl SolverInitializer {
    pub fn new(vertex_num: VertexNum, weighted_edges: Vec<HyperEdge>) -> Self {
        Self {
            vertex_num,
            weighted_edges,
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverInitializer {
    #[new]
    fn py_new(vertex_num: VertexNum, weighted_edges: Vec<HyperEdge>) -> Self {
        Self::new(vertex_num, weighted_edges)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    #[pyo3(name = "snapshot", signature = (abbrev=true))]
    fn py_snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.snapshot(abbrev))
    }
    #[pyo3(name = "get_subgraph_syndrome")]
    fn py_get_subgraph_syndrome(&self, subgraph: PySubgraph) -> BTreeSet<VertexIndex> {
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
}

impl SolverInitializer {
    /// sanity check to avoid duplicate edges that are hard to debug
    pub fn sanity_check(&self) -> Result<(), String> {
        use crate::example_codes::*;
        let code = ErrorPatternReader::from_initializer(self);
        code.sanity_check()
    }

    pub fn matches_subgraph_syndrome(&self, subgraph: &OutputSubgraph, defect_vertices: &[VertexIndex]) -> bool {
        let subgraph_defect_vertices: Vec<_> = self.get_subgraph_syndrome(subgraph).into_iter().collect();
        let mut defect_vertices = defect_vertices.to_owned();
        defect_vertices.sort();
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
        let internal_subgraph = OutputSubgraph::get_internal_subgraph(&subgraph);
        if internal_subgraph.is_empty() {
            let mut weight = Weight::zero();
            for &edge_index in subgraph.iter() {
                weight += self.weighted_edges[edge_index as usize].weight.clone();
            }
            return weight;
        } else {
            let mut weight = Weight::zero();
            for edge_weak in internal_subgraph.iter() {
                weight += edge_weak.upgrade_force().read_recursive().weight.clone();
            }
            return weight;
        }
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_subgraph_syndrome(&self, subgraph: &OutputSubgraph) -> BTreeSet<VertexIndex> {
        let internal_subgraph = OutputSubgraph::get_internal_subgraph(&subgraph);
        let mut defect_vertices = BTreeSet::new();
        for edge_weak in internal_subgraph.iter() {
            // let HyperEdge { vertices, .. } = &self.weighted_edges[edge_index as usize];
            let edge_ptr = edge_weak.upgrade_force();
            let edge = edge_ptr.read_recursive();
            let vertices = &edge.vertices;
            let unique_vertices = vertices
                .into_iter()
                .map(|v| v.upgrade_force().read_recursive().vertex_index)
                .collect::<Vec<_>>();
            for &vertex_index in unique_vertices.iter() {
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
        for HyperEdge {
            vertices,
            weight,
            connected_to_boundary_vertex: _,
        } in self.weighted_edges.iter()
        {
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
#[cfg_attr(feature = "python_binding", pyclass(get_all, set_all))]
pub struct SyndromePattern {
    /// the vertices corresponding to defect measurements
    pub defect_vertices: Vec<VertexIndex>,
    /// the edges that experience erasures, i.e. known errors
    pub erasures: Vec<EdgeIndex>,
}

impl SyndromePattern {
    pub fn new(defect_vertices: Vec<VertexIndex>, erasures: Vec<EdgeIndex>) -> Self {
        Self {
            defect_vertices,
            erasures,
        }
    }
}

impl SyndromePattern {
    pub fn new_vertices(defect_vertices: Vec<VertexIndex>) -> Self {
        Self::new(defect_vertices, vec![])
    }
    pub fn new_empty() -> Self {
        Self::new(vec![], vec![])
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
    #[pyo3(signature = (defect_vertices=None, erasures=None))]
    fn py_new(defect_vertices: Option<&Bound<PyAny>>, erasures: Option<&Bound<PyAny>>) -> PyResult<Self> {
        use crate::util_py::py_into_btree_set;
        let defect_vertices: Vec<VertexIndex> = if let Some(defect_vertices) = defect_vertices {
            py_into_btree_set(defect_vertices)?.into_iter().collect()
        } else {
            vec![]
        };
        let erasures: Vec<EdgeIndex> = if let Some(erasures) = erasures {
            py_into_btree_set(erasures)?.into_iter().collect()
        } else {
            vec![]
        };
        Ok(Self::new(defect_vertices, erasures))
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
    #[pyo3(name="snapshot", signature = (abbrev=true))]
    fn py_snapshot(&mut self, abbrev: bool) -> PyObject {
        json_to_pyobject(self.snapshot(abbrev))
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
pub type InternalSubgraph = Vec<EdgeWeak>;

pub struct OutputSubgraph {
    pub subgraph: Subgraph,
    pub flip_edge_indices: hashbrown::HashSet<EdgeIndex>,
    internal_subgraph: InternalSubgraph,
    // pub defect_vertices: hashbrown::HashSet<VertexIndex>,
}

// impl InternalSubgraph {
//     pub fn new(internal_subgraph: Vec<EdgeWeak>) -> Self {
//         Self {
//             internal_subgraph,
//         }
//     }

//     // pub fn to_output_subgraph(&self) -> OutputSubgraph {
//     //     let subgraph = self.internal_subgraph.iter().map(|edge| edge.upgrade_force().read_recursive().edge_index).collect();
//     //     // let mut defect_vertices = hashbrown::HashSet::new();
//     //     // for edge_weak in self.internal_subgraph.iter() {
//     //     //     let edge_ptr = edge_weak.upgrade_force();
//     //     //     let edge = edge_ptr.read_recursive();
//     //     //     let vertices = &edge.vertices;
//     //     //     let unique_vertices = vertices.into_iter().map(|v| v.upgrade_force().read_recursive().vertex_index).collect::<Vec<_>>();
//     //     //     for &vertex_index in unique_vertices.iter() {
//     //     //         if defect_vertices.contains(&vertex_index) {
//     //     //             defect_vertices.remove(&vertex_index);
//     //     //             // println!("duplicate defect vertex: {}", vertex_index);
//     //     //         } else {
//     //     //             defect_vertices.insert(vertex_index);
//     //     //         }
//     //     //     }
//     //     // }
//     //     OutputSubgraph::new(subgraph, self.flip_edge_indices.clone())
//     // }

//     pub fn get_internal_subgraph(&self) -> &Vec<EdgeWeak> {
//         &self.internal_subgraph
//     }

//     // pub fn get_flip_edge_indices(&self) -> &hashbrown::HashSet<EdgeIndex> {
//     //     &self.flip_edge_indices
//     // }
// }

impl OutputSubgraph {
    pub fn new(
        subgraph: Subgraph,
        flip_edge_indices: hashbrown::HashSet<EdgeIndex>,
        internal_subgraph: InternalSubgraph,
    ) -> Self {
        Self {
            subgraph,
            flip_edge_indices,
            internal_subgraph,
            // defect_vertices,
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

    pub fn get_internal_subgraph(&self) -> &InternalSubgraph {
        &self.internal_subgraph
    }
}

// impl From<Subgraph> for OutputSubgraph {
//     fn from(value: Subgraph) -> Self {
//         Self::new(value, hashbrown::HashSet::new())
//     }
// }

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
    pub fn new(noisy_measurements: VertexNum, detail_log_file: Option<String>, partition_info: &PartitionInfo) -> Self {
        let benchmark_profiler_output = detail_log_file.map(|filename| {
            let parent_dir = std::path::Path::new(&filename).parent();
            if let Some(dir) = parent_dir {
                create_dir_all(dir).unwrap(); // Ensure all parent directories exist
            }
            let mut file = File::create(filename).unwrap();
            file.write_all(serde_json::to_string(&partition_info.config).unwrap().as_bytes())
                .unwrap();
            file.write_all(b"\n").unwrap();
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
        serde_json::Value::Bool(value) => value.to_object(py),
        serde_json::Value::Number(value) => {
            if value.is_i64() {
                value.as_i64().to_object(py)
            } else {
                value.as_f64().to_object(py)
            }
        }
        serde_json::Value::String(value) => value.to_object(py),
        serde_json::Value::Array(array) => {
            let elements: Vec<PyObject> = array.into_iter().map(|value| json_to_pyobject_locked(value, py)).collect();
            PyList::new_bound(py, elements).into()
        }
        serde_json::Value::Object(map) => {
            let pydict = PyDict::new_bound(py);
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
    Ok(())
}

/// for parallel implementation
///
/// an efficient representation of partitioned vertices and erasures when they're ordered
#[derive(Debug, Clone, Serialize)]

pub struct PartitionedSyndromePattern<'a> {
    /// the original syndrome pattern to be partitioned
    pub syndrome_pattern: &'a SyndromePattern,
    /// the defect range of this partition: it must be continuous if the defect vertices are ordered
    pub owned_defect_range: DefectRange,
}

impl<'a> PartitionedSyndromePattern<'a> {
    pub fn new(syndrome_pattern: &'a SyndromePattern) -> Self {
        assert!(
            syndrome_pattern.erasures.is_empty(),
            "erasure partition not supported yet;
        even if the edges in the erasure is well ordered, they may not be able to be represented as
        a single range simply because the partition is vertex-based. need more consideration"
        );
        Self {
            syndrome_pattern,
            owned_defect_range: DefectRange::new(0, syndrome_pattern.defect_vertices.len() as DefectIndex),
        }
    }
}

// ////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////
// /////////////// We implement the HashSet to specify vertices in set ////////////////////

// #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
// pub struct IndexSet {
//     // spaced-out individual index
//     pub individual_indices: BTreeSet<VertexNodeIndex>,
//     // indices that can be described using range, we assume that there is only one big range among all vertex indices
//     pub range: [VertexNodeIndex; 2],
// }

// // just to distinguish them in code, essentially nothing different
// pub type VertexSet = IndexSet;
// pub type DefectSet = IndexSet;
// pub type NodeSet = IndexSet;

// impl IndexSet {
//     // initialize a IndexSet that only has a continuous range of indices but no spaced out individual indices
//     fn new_range(start: VertexNodeIndex, end: VertexNodeIndex) -> Self {
//         debug_assert!(end > start, "invalid range [{}, {})", start, end);
//         Self {
//             individual_indices: BTreeSet::<VertexNodeIndex>::new(),
//             range: [start, end],
//         }
//     }

//     // initialize a IndexSet that only has spaced out individual indicies
//     fn new_individual_indices(indices: Vec<VertexNodeIndex>) -> Self {
//         let mut new_set = BTreeSet::<VertexNodeIndex>::new();
//         for index in indices {
//             new_set.insert(index);
//         }
//         Self {
//             individual_indices: new_set,
//             range: [0, 0],
//         }
//     }

//     // initialize a IndexSet that has both continuous range of indices and individual spaced out indices
//     pub fn new(start: VertexNodeIndex, end: VertexNodeIndex, indices: Vec<VertexNodeIndex>) -> Self {
//         debug_assert!(end > start, "invalid range [{}, {})", start, end);
//         if start == end && indices.len() == 0{
//             // range is invalid, we check whether indices are empty
//             // indices are empty too
//             panic!("both the input range and individual indices are invalid");
//         } else if start == end {
//             return Self::new_individual_indices(indices);
//         } else if indices.len() == 0{
//             return Self::new_range(start, end);
//         } else {
//             let mut new_set = BTreeSet::<VertexNodeIndex>::new();
//             for index in indices {
//                 new_set.insert(index);
//             }

//             return Self {
//                 individual_indices: new_set,
//                 range: [start, end],
//             }
//         }
//     }

//     // add more individual index to the already created IndexSet
//     pub fn add_individual_index(&mut self, index: VertexNodeIndex) {
//         self.individual_indices.insert(index);
//     }

//     pub fn new_range_by_length(start: VertexNodeIndex, length: VertexNodeIndex) -> Self {
//         Self::new_range(start, start + length)
//     }

//     pub fn is_empty(&self) -> bool {
//         self.range[1] == self.range[0] && self.individual_indices.is_empty()
//     }

//     #[allow(clippy::unnecessary_cast)]
//     pub fn len(&self) -> usize {
//         (self.range[1] - self.range[0] + self.individual_indices.len()) as usize
//     }
//     pub fn range_start(&self) -> VertexNodeIndex {
//         self.range[0]
//     }
//     pub fn range_end(&self) -> VertexNodeIndex {
//         self.range[1]
//     }
//     pub fn extend_range_by(&mut self, append_count: VertexNodeIndex) {
//         self.range[1] += append_count;
//     }
//     pub fn bias_by(&mut self, bias: VertexNodeIndex) {
//         self.range[0] += bias;
//         self.range[1] += bias;

//         let set = std::mem::replace(&mut self.individual_indices, BTreeSet::new());
//         self.individual_indices = set.into_iter()
//             .map(|p| p + bias)
//             .collect();
//     }
//     pub fn sanity_check(&self) {
//         assert!(self.range_start() <= self.range_end(), "invalid vertex range {:?}", self);
//     }
//     pub fn contains(&self, vertex_index: VertexNodeIndex) -> bool {
//         (vertex_index >= self.range_start() && vertex_index < self.range_end()) || self.individual_indices.contains(&vertex_index)
//     }
//     // /// fuse two ranges together, returning (the whole range, the interfacing range)
//     // pub fn fuse(&self, other: &Self) -> (Self, Self) {
//     //     self.sanity_check();
//     //     other.sanity_check();
//     //     assert!(self.range[1] <= other.range[0], "only lower range can fuse higher range");
//     //     (
//     //         Self::new(self.range[0], other.range[1]),
//     //         Self::new(self.range[1], other.range[0]),
//     //     )
//     // }
// }

// we leave the code here just in case we need to describe the vertices in continuos range
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct IndexRange {
    pub range: [VertexNodeIndex; 2],
}

// just to distinguish them in code, essentially nothing different
pub type VertexRange = IndexRange;
pub type DefectRange = IndexRange;
pub type NodeRange = IndexRange;
pub type EdgeRange = IndexRange;

impl IndexRange {
    pub fn new(start: VertexNodeIndex, end: VertexNodeIndex) -> Self {
        debug_assert!(end >= start, "invalid range [{}, {})", start, end);
        Self { range: [start, end] }
    }
    pub fn new_length(start: VertexNodeIndex, length: VertexNodeIndex) -> Self {
        Self::new(start, start + length)
    }
    pub fn is_empty(&self) -> bool {
        self.range[1] == self.range[0]
    }
    #[allow(clippy::unnecessary_cast)]
    pub fn len(&self) -> usize {
        (self.range[1] - self.range[0]) as usize
    }
    pub fn start(&self) -> VertexNodeIndex {
        self.range[0]
    }
    pub fn end(&self) -> VertexNodeIndex {
        self.range[1]
    }
    pub fn append_by(&mut self, append_count: VertexNodeIndex) {
        self.range[1] += append_count;
    }
    pub fn bias_by(&mut self, bias: VertexNodeIndex) {
        self.range[0] += bias;
        self.range[1] += bias;
    }
    pub fn sanity_check(&self) {
        assert!(self.start() <= self.end(), "invalid vertex range {:?}", self);
    }
    pub fn contains(&self, vertex_index: VertexNodeIndex) -> bool {
        vertex_index >= self.start() && vertex_index < self.end()
    }
    /// fuse two ranges together, returning (the whole range, the interfacing range)
    pub fn fuse(&self, other: &Self) -> (Self, Self) {
        self.sanity_check();
        other.sanity_check();
        assert!(self.range[1] <= other.range[0], "only lower range can fuse higher range");
        (
            Self::new(self.range[0], other.range[1]),
            Self::new(self.range[1], other.range[0]),
        )
    }
}

impl IndexRange {
    pub fn iter(&self) -> std::ops::Range<VertexNodeIndex> {
        self.range[0]..self.range[1]
    }
    pub fn contains_any(&self, vertex_indices: &[VertexNodeIndex]) -> bool {
        for vertex_index in vertex_indices.iter() {
            if self.contains(*vertex_index) {
                return true;
            }
        }
        false
    }
}

impl std::hash::Hash for IndexRange {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.range[0].hash(state);
        self.range[1].hash(state);
    }
}

// /// a general partition unit that could contain mirrored vertices
// #[derive(Debug, Clone)]
// pub struct PartitionUnit {
//     /// unit index
//     pub unit_index: usize,
// }

// pub type PartitionUnitPtr = ArcRwLock<PartitionUnit>;
// pub type PartitionUnitWeak = WeakRwLock<PartitionUnit>;

// impl std::fmt::Debug for PartitionUnitPtr {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         let partition_unit = self.read_recursive();
//         write!(
//             f,
//             "{}",
//             partition_unit.unit_index
//         )
//     }
// }

// impl std::fmt::Debug for PartitionUnitWeak {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         self.upgrade_force().fmt(f)
//     }
// }

/// user input partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionConfig {
    /// the number of vertices
    pub vertex_num: VertexNum,
    /// detailed plan of partitioning serial modules: each serial module possesses a list of vertices, including all interface vertices
    pub partitions: Vec<VertexRange>,
    /// detailed plan of interfacing vertices
    pub fusions: Vec<(usize, usize)>,
    /// undirected acyclic graph (DAG) to keep track of the relationship between different partition units
    pub dag_partition_units: Graph<(), bool, Undirected>,
    /// defect vertices (global index)
    pub defect_vertices: BTreeSet<usize>,
}

impl PartitionConfig {
    pub fn new(vertex_num: VertexNum) -> Self {
        Self {
            vertex_num,
            partitions: vec![VertexRange::new(0, vertex_num as VertexIndex)],
            fusions: vec![],
            dag_partition_units: Graph::new_undirected(),
            defect_vertices: BTreeSet::new(),
        }
    }

    pub fn new_separate(vertex_num: VertexNum, fusions: Vec<(usize, usize)>, defect_vertices: Vec<usize>) -> Self {
        Self {
            vertex_num,
            partitions: vec![], // we do not partition this newly (additionally) added unit
            fusions,
            dag_partition_units: Graph::new_undirected(), // we do not use this for the newly (additionally) added unit
            defect_vertices: BTreeSet::from_iter(defect_vertices),
        }
    }

    /// the partition below relies on the fact that the vertices' indices are continuous
    #[allow(clippy::unnecessary_cast)]
    pub fn info(&self) -> PartitionInfo {
        assert!(!self.partitions.is_empty(), "at least one partition must exist");
        if self.fusions.is_empty() {
            PartitionInfo {
                config: self.clone(),
                units: vec![],
                vertex_to_owning_unit: HashMap::new(),
            }
        } else {
            let mut owning_ranges = vec![];
            let unit_count = self.partitions.len() + self.fusions.len();
            let partitions_len = self.partitions.len();

            for &partition in self.partitions.iter() {
                partition.sanity_check();
                assert!(
                    partition.end() <= self.vertex_num as VertexIndex,
                    "invalid vertex index {} in partitions",
                    partition.end()
                );
                owning_ranges.push(partition);
            }

            // find boundary vertices
            let mut interface_ranges = vec![];
            let mut unit_index_to_adjacent_indices: HashMap<usize, Vec<usize>> = HashMap::new();

            for (boundary_unit_index, (left_index, right_index)) in self.fusions.iter().enumerate() {
                let boundary_unit_index = boundary_unit_index + partitions_len;
                // find the interface_range
                let (_whole_range, interface_range) = self.partitions[*left_index].fuse(&self.partitions[*right_index]);
                interface_ranges.push(interface_range);
                owning_ranges.push(interface_range);
                if let Some(adjacent_indices) = unit_index_to_adjacent_indices.get_mut(left_index) {
                    adjacent_indices.push(boundary_unit_index);
                } else {
                    let mut adjacent_indices = vec![];
                    adjacent_indices.push(boundary_unit_index);
                    unit_index_to_adjacent_indices.insert(*left_index, adjacent_indices.clone());
                }

                if let Some(adjacent_indices) = unit_index_to_adjacent_indices.get_mut(right_index) {
                    adjacent_indices.push(boundary_unit_index);
                } else {
                    let mut adjacent_indices = vec![];
                    adjacent_indices.push(boundary_unit_index);
                    unit_index_to_adjacent_indices.insert(*right_index, adjacent_indices.clone());
                }

                // now we insert the key-value pair for boundary_unit_index and its adjacent
                if let Some(adjacent_indices) = unit_index_to_adjacent_indices.get_mut(&boundary_unit_index) {
                    adjacent_indices.push(*left_index);
                    adjacent_indices.push(*right_index);
                } else {
                    let mut adjacent_indices = vec![];
                    adjacent_indices.push(*left_index);
                    adjacent_indices.push(*right_index);
                    unit_index_to_adjacent_indices.insert(boundary_unit_index, adjacent_indices.clone());
                }
            }

            let mut boundary_vertices: HashMap<usize, Vec<IndexRange>> = HashMap::new();
            for (unit_index, adjacent_unit_indices) in unit_index_to_adjacent_indices.iter() {
                if let Some(adjacent_vertices) = boundary_vertices.get_mut(unit_index) {
                    for adjacent_unit_index in adjacent_unit_indices {
                        adjacent_vertices.push(owning_ranges[*adjacent_unit_index]);
                    }
                } else {
                    let mut adjacent_vertices = vec![];
                    for adjacent_unit_index in adjacent_unit_indices {
                        adjacent_vertices.push(owning_ranges[*adjacent_unit_index]);
                    }
                    boundary_vertices.insert(*unit_index, adjacent_vertices.clone());
                }
            }

            // construct partition info, assuming partition along the time axis
            let partition_unit_info: Vec<_> = (0..unit_count)
                .map(|i| PartitionUnitInfo {
                    // owning_range: if i == self.partitions.len() - 1 {
                    //     owning_ranges[i]
                    // }else {
                    //     IndexRange::new(owning_ranges[i].start(), interface_ranges[i].end())  // owning_ranges[i],
                    // },
                    owning_range: owning_ranges[i],
                    unit_index: i,
                    is_boundary_unit: if i < partitions_len { false } else { true },
                    adjacent_parallel_units: unit_index_to_adjacent_indices.get(&i).unwrap().clone(),
                    boundary_vertices: boundary_vertices.get(&i).unwrap().clone(),
                })
                .collect();

            // create vertex_to_owning_unit for owning_ranges
            let mut vertex_to_owning_unit = HashMap::new();
            for partition_unit in partition_unit_info.iter() {
                // create vertex_to_owning_unit for owning_ranges
                for vertex_index in partition_unit.owning_range.iter() {
                    vertex_to_owning_unit.insert(vertex_index, partition_unit.unit_index);
                }
            }

            PartitionInfo {
                config: self.clone(),
                units: partition_unit_info,
                vertex_to_owning_unit,
            }
        }
    }

    pub fn info_seperate(&self, unit_index: usize, boundary_vertics: VertexRange) -> PartitionInfo {
        // the self.partitins, dag should be empty
        let partition_unit_info = PartitionUnitInfo {
            owning_range: VertexRange::new(0, self.vertex_num),
            unit_index,
            is_boundary_unit: false, // not useful for this newly (additionally) added unit
            adjacent_parallel_units: vec![self.fusions[0].1], // note that self.fusions = (self, other)
            boundary_vertices: vec![boundary_vertics],
        };
        PartitionInfo {
            config: self.clone(),
            units: vec![partition_unit_info],
            vertex_to_owning_unit: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    /// the initial configuration that creates this info
    pub config: PartitionConfig,
    /// individual info of each unit
    pub units: Vec<PartitionUnitInfo>,
    /// the mapping from vertices to the owning unit: serial unit (holding real vertices) as well as parallel units (holding interfacing vertices);
    /// used for loading syndrome to the holding units
    pub vertex_to_owning_unit: HashMap<VertexIndex, usize>,
}

// impl PartitionInfo {
/// split a sequence of syndrome into multiple parts, each corresponds to a unit;
/// this is a slow method and should only be used when the syndrome pattern is not well-ordered
// #[allow(clippy::unnecessary_cast)]
// pub fn partition_syndrome_unordered(&self, syndrome_pattern: &SyndromePattern) -> Vec<SyndromePattern> {
//     let mut partitioned_syndrome: Vec<_> = (0..self.units.len()).map(|_| SyndromePattern::new_empty()).collect();
//     for defect_vertex in syndrome_pattern.defect_vertices.iter() {
//         let unit_index = self.vertex_to_owning_unit.get(defect_vertex);
//         match unit_index {
//             Some(unit_index) => partitioned_syndrome[*unit_index].defect_vertices.push(*defect_vertex),
//             None => // the syndrome is on the boudnary vertices

//         }
//     }
//     // TODO: partition edges
//     partitioned_syndrome
// }
// }

// for primal module parallel
impl<'a> PartitionedSyndromePattern<'a> {
    /// partition the syndrome pattern into 2 partitioned syndrome pattern and my whole range
    #[allow(clippy::unnecessary_cast)]
    pub fn partition(&self, partition_unit_info: &PartitionUnitInfo) -> Self {
        // first binary search the start of owning defect vertices
        let owning_start_index = {
            let mut left_index = self.owned_defect_range.start(); // since owned_defect_range is initialized to the length of all defect vertices
            let mut right_index = self.owned_defect_range.end();
            while left_index != right_index {
                let mid_index = (left_index + right_index) / 2;
                let mid_defect_vertex = self.syndrome_pattern.defect_vertices[mid_index as usize];
                if mid_defect_vertex < partition_unit_info.owning_range.start() {
                    left_index = mid_index + 1;
                } else {
                    right_index = mid_index;
                }
            }
            left_index
        };
        // println!("start of owning defect vertice: {owning_start_index:?}");
        // second binary search the end of owning defect vertices
        let owning_end_index = {
            let mut left_index = self.owned_defect_range.start();
            let mut right_index = self.owned_defect_range.end();
            while left_index != right_index {
                let mid_index = (left_index + right_index) / 2;
                let mid_defect_vertex = self.syndrome_pattern.defect_vertices[mid_index as usize];
                if mid_defect_vertex < partition_unit_info.owning_range.end() {
                    left_index = mid_index + 1;
                } else {
                    right_index = mid_index;
                }
            }
            left_index
        };
        // println!("end of owning defect vertice: {owning_end_index:?}");

        Self {
            syndrome_pattern: self.syndrome_pattern,
            owned_defect_range: DefectRange::new(owning_start_index, owning_end_index),
        }
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn expand(&self) -> SyndromePattern {
        let mut defect_vertices = Vec::with_capacity(self.owned_defect_range.len());
        for defect_index in self.owned_defect_range.iter() {
            defect_vertices.push(self.syndrome_pattern.defect_vertices[defect_index as usize]);
        }
        SyndromePattern::new(defect_vertices, vec![])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionUnitInfo {
    /// the owning range of units, the vertices exlusive to this unit
    pub owning_range: VertexRange,
    /// partition unit index
    pub unit_index: usize,
    /// if this unit is boundary unit
    pub is_boundary_unit: bool,

    pub adjacent_parallel_units: Vec<usize>,

    /// the boundary vertices near to this unit
    pub boundary_vertices: Vec<IndexRange>,
    // /// boundary vertices, following the global vertex index
    // /// key: indexrange of the boundary vertices. value: (unit_index, unit_index), the pair of unit_index of the two partition units adjacent to the boundary
    // pub boundary_vertices: Option<HashMap<IndexRange, (usize, usize)>>,
    // /// adjacent PartitionUnits, vector of partition unit_index
    // pub adjacent_partition_units: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PartitionedSolverInitializer {
    /// unit index
    pub unit_index: usize,
    /// the number of all vertices (including those partitioned into other serial modules)
    pub vertex_num: VertexNum,
    /// the number of all edges (including those partitioned into other serial modules)
    pub edge_num: usize,
    /// vertices exclusively owned by this partition; this part must be a continuous range
    pub owning_range: VertexRange,
    /// weighted edges, where the first vertex index is within the range [vertex_index_bias, vertex_index_bias + vertex_num) and
    /// the second is either in [vertex_index_bias, vertex_index_bias + vertex_num) or inside
    /// the second element in the tuple is the global edge index of the respective hyper_edge
    pub weighted_edges: Vec<(HyperEdge, usize)>,
    /// (not sure whether we need it, just in case)
    pub boundary_vertices: Vec<IndexRange>,
    /// whether this unit is boundary-unit
    pub is_boundary_unit: bool,
    /// all defect vertices (global index), not just for this unit
    pub defect_vertices: BTreeSet<usize>,
    // /// (not sure whether we need it, just in case)
    // pub adjacent_partition_units: Vec<usize>,
    // /// applicable when all the owning vertices are partitioned (i.e. this belongs to a fusion unit)
    // pub owning_interface: Option<PartitionUnitWeak>,
}

/// perform index transformation
#[allow(clippy::unnecessary_cast)]
pub fn build_old_to_new(reordered_vertices: &Vec<VertexIndex>) -> Vec<Option<VertexIndex>> {
    let mut old_to_new: Vec<Option<VertexIndex>> = (0..reordered_vertices.len()).map(|_| None).collect();
    for (new_index, old_index) in reordered_vertices.iter().enumerate() {
        assert_eq!(old_to_new[*old_index as usize], None, "duplicate vertex found {}", old_index);
        old_to_new[*old_index as usize] = Some(new_index as VertexIndex);
    }
    old_to_new
}

/// translate defect vertices into the current new index given reordered_vertices
#[allow(clippy::unnecessary_cast)]
pub fn translated_defect_to_reordered(
    reordered_vertices: &Vec<VertexIndex>,
    old_defect_vertices: &[VertexIndex],
) -> Vec<VertexIndex> {
    let old_to_new = build_old_to_new(reordered_vertices);
    old_defect_vertices
        .iter()
        .map(|old_index| old_to_new[*old_index as usize].unwrap())
        .collect()
}

#[cfg(test)]
pub mod tests {
    use crate::example_codes::ExampleCode;

    use super::*;
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

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

        // Expected behavior: `2` is skipped, and `5` is added at the end.
        let result: Vec<_> = output_subgraph.iter().cloned().collect();
        assert_eq!(result, vec![1, 3, 4, 5]);
    }

    #[test]
    fn test_iter_empty_flip_edge_indices() {
        let subgraph = vec![1, 2, 3];
        let flip_edge_indices = HashSet::new();

        let output_subgraph = OutputSubgraph::new(subgraph.clone(), flip_edge_indices, vec![]);

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

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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

        let mut output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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

        let mut output_subgraph = OutputSubgraph::new(subgraph.clone(), flip_edge_indices, vec![]);

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

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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

        let output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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

        let mut output_subgraph = OutputSubgraph::new(subgraph, flip_edge_indices, vec![]);

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
}
