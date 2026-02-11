//! Example Decoding
//!
//! This module contains several abstract decoding hypergraph and it's randomized simulator utilities.
//! This helps to debug, but it doesn't corresponds to real noise model, nor it's capable of simulating circuit-level noise model.
//! For complex noise model and simulator functionality, please see <https://github.com/yuewuo/QEC-Playground>
//!
//! Note that these examples are not optimized for cache for simplicity.
//! To maximize code efficiency, user should design how to group vertices such that memory speed is constant for arbitrary large code distance.
//!

use crate::derivative::Derivative;
use crate::model_hypergraph::*;
use crate::num_traits::{FromPrimitive, ToPrimitive, Zero};
use crate::rand_xoshiro::rand_core::SeedableRng;
use crate::serde_json;
use crate::util::*;
#[cfg(feature = "python_binding")]
use crate::util_py::*;
use crate::visualize::*;
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;

use std::fs::File;
use std::io::{self, BufRead};
use std::sync::Arc;

/// Vertex corresponds to a stabilizer measurement bit
#[derive(Derivative, Clone)]
#[derivative(Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeVertex {
    /// position helps to visualize
    pub position: VisualizePosition,
    /// neighbor edges helps to set find individual edge
    pub neighbor_edges: Vec<EdgeIndex>,
    /// whether it's a defect
    pub is_defect: bool,
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeVertex {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

/// Edge flips the measurement result of two vertices
#[derive(Derivative, Clone)]
#[derivative(Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct CodeEdge {
    /// the two vertices incident to this edge; in quantum LDPC codes this should be only a handful of vertices
    pub vertices: Vec<VertexIndex>,
    /// probability of flipping the results of these vertices; do not set p to 0 to remove edge: if desired, create a new code type without those edges
    pub p: f64,
    /// probability of having a reported event of error on this edge (aka erasure errors)
    pub pe: f64,
    /// the integer weight of this edge
    pub weight: Weight,
    /// whether this edge is erased
    pub is_erasure: bool,
}

impl CodeEdge {
    pub fn new(vertices: Vec<VertexIndex>) -> Self {
        Self {
            vertices,
            p: 0.,
            pe: 0.,
            weight: Rational::zero(),
            is_erasure: false,
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeEdge {
    #[new]
    fn py_new(vertices: Vec<VertexIndex>) -> Self {
        Self::new(vertices)
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
    fn get_p(&self) -> f64 {
        self.p.clone()
    }
    #[setter]
    fn set_p(&mut self, p: f64) {
        self.p = p;
    }
    #[getter]
    fn get_pe(&self) -> f64 {
        self.pe.clone()
    }
    #[setter]
    fn set_pe(&mut self, pe: f64) {
        self.pe = pe;
    }
    #[getter]
    fn get_weight(&self) -> PyRational {
        self.weight.clone().into()
    }
    #[setter]
    fn set_weight(&mut self, weight: &Bound<PyAny>) {
        self.weight = PyRational::from(weight).0;
    }
    #[getter]
    fn get_is_erasure(&self) -> bool {
        self.is_erasure.clone()
    }
    #[setter]
    fn set_is_erasure(&mut self, is_erasure: bool) {
        self.is_erasure = is_erasure;
    }
}

/// default function for computing (pre-scaled) weight from probability
#[cfg_attr(feature = "python_binding", pyfunction)]
pub fn weight_of_p(p: f64) -> f64 {
    // note: allowed negative weight handling
    // assert!((0. ..0.5).contains(&p), "p must be a reasonable value between 0 and 50%");
    ((1. - p) / p).ln()
}

pub trait ExampleCode {
    /// get mutable references to vertices and edges
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>);
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>);

    /// get the number of vertices
    fn vertex_num(&self) -> VertexNum {
        self.immutable_vertices_edges().0.len() as VertexNum
    }

    /// get the number of edges
    fn edge_num(&self) -> usize {
        self.immutable_vertices_edges().1.len()
    }

    /// get edges for iteration
    fn edges(&self) -> &Vec<CodeEdge> {
        self.immutable_vertices_edges().1
    }

    /// get mutable edges for iteration
    fn edges_mut(&mut self) -> &mut Vec<CodeEdge> {
        self.vertices_edges().1
    }

    /// generic method that automatically computes integer weights from probabilities,
    /// scales such that the maximum integer weight is 10000 and the minimum is 1
    fn compute_weights(&mut self) {
        let (_vertices, edges) = self.vertices_edges();

        for edge in edges.iter_mut() {
            let weight = weight_of_p(edge.p);
            edge.weight = Rational::from_f64(weight).unwrap();
        }
    }

    /// get weights of dual module
    fn get_weights(&self) -> Vec<Weight> {
        let (_vertices, edges) = self.immutable_vertices_edges();
        let mut weights = Vec::with_capacity(edges.len());
        for edge in edges.iter() {
            weights.push(edge.weight.clone());
        }
        weights
    }

    /// remove duplicate edges by keeping one with largest probability
    #[allow(clippy::unnecessary_cast)]
    fn remove_duplicate_edges(&mut self) {
        let (_vertices, edges) = self.vertices_edges();
        let mut remove_edges = HashSet::new();
        let mut existing_edges = HashMap::<Vec<VertexIndex>, EdgeIndex>::with_capacity(edges.len() * 2);
        for (edge_idx, edge) in edges.iter().enumerate() {
            let mut vertices = edge.vertices.clone();
            vertices.sort();
            if existing_edges.contains_key(&vertices) {
                let previous_idx = existing_edges[&vertices];
                if edge.p > edges[previous_idx as usize].p {
                    remove_edges.insert(previous_idx);
                } else {
                    remove_edges.insert(edge_idx as EdgeIndex);
                }
            }
            existing_edges.insert(vertices, edge_idx as EdgeIndex);
        }
        let mut dedup_edges = Vec::with_capacity(edges.len());
        for (edge_idx, edge) in edges.drain(..).enumerate() {
            if !remove_edges.contains(&(edge_idx as EdgeIndex)) {
                dedup_edges.push(edge);
            }
        }
        *edges = dedup_edges;
    }

    /// sanity check to avoid duplicate edges that are hard to debug
    fn sanity_check(&self) -> Result<(), String> {
        let (vertices, edges) = self.immutable_vertices_edges();
        // check the graph is reasonable
        if vertices.is_empty() || edges.is_empty() {
            return Err("empty graph".to_string());
        }
        // check duplicated edges
        let mut existing_edges = HashMap::<Vec<VertexIndex>, EdgeIndex>::with_capacity(edges.len() * 2);
        for (edge_idx, edge) in edges.iter().enumerate() {
            let mut vertices = edge.vertices.clone();
            if vertices.is_empty() {
                return Err(format!("empty hyperedge {}", edge_idx));
            }
            vertices.sort();
            let length_before_dedup = vertices.len();
            vertices.dedup();
            if vertices.len() != length_before_dedup {
                return Err(format!(
                    "edge {} contains duplicate vertices, after dedup it's {:?}",
                    edge_idx, vertices
                ));
            }
            if existing_edges.contains_key(&vertices) {
                let previous_idx = existing_edges[&vertices];
                return Err(format!(
                    "duplicate edge {} and {} with incident vertices {:?}",
                    previous_idx, edge_idx, vertices
                ));
            }
            existing_edges.insert(vertices, edge_idx as EdgeIndex);
        }
        // check duplicated referenced edge from each vertex
        for (vertex_idx, vertex) in vertices.iter().enumerate() {
            let mut existing_edges = HashMap::<EdgeIndex, ()>::new();
            if vertex.neighbor_edges.is_empty() {
                return Err(format!("vertex {} do not have any neighbor edges", vertex_idx));
            }
            for edge_idx in vertex.neighbor_edges.iter() {
                if existing_edges.contains_key(edge_idx) {
                    return Err(format!("duplicate referred edge {} from vertex {}", edge_idx, vertex_idx));
                }
                existing_edges.insert(*edge_idx, ());
            }
        }
        Ok(())
    }

    /// set probability of all edges; user can set individual probabilities
    fn set_probability(&mut self, p: f64) {
        let (_vertices, edges) = self.vertices_edges();
        for edge in edges.iter_mut() {
            edge.p = p;
        }
    }

    /// set erasure probability of all edges; user can set individual probabilities
    fn set_erasure_probability(&mut self, pe: f64) {
        let (_vertices, edges) = self.vertices_edges();
        for edge in edges.iter_mut() {
            edge.pe = pe;
        }
    }

    /// automatically create vertices given edges
    #[allow(clippy::unnecessary_cast)]
    fn fill_vertices(&mut self, vertex_num: VertexNum) {
        let (vertices, edges) = self.vertices_edges();
        vertices.clear();
        vertices.reserve(vertex_num as usize);
        for _ in 0..vertex_num {
            vertices.push(CodeVertex {
                position: VisualizePosition::new(0., 0., 0.),
                neighbor_edges: Vec::new(),
                is_defect: false,
            });
        }
        for (edge_idx, edge) in edges.iter().enumerate() {
            for vertex_index in edge.vertices.iter() {
                let vertex = &mut vertices[*vertex_index as usize];
                vertex.neighbor_edges.push(edge_idx as EdgeIndex);
            }
        }
    }

    /// gather all positions of vertices
    fn get_positions(&self) -> Vec<VisualizePosition> {
        let (vertices, _edges) = self.immutable_vertices_edges();
        let mut positions = Vec::with_capacity(vertices.len());
        for vertex in vertices.iter() {
            positions.push(vertex.position.clone());
        }
        positions
    }

    /// generate standard interface to instantiate MWPF solver
    fn get_initializer(&self) -> SolverInitializer {
        let (vertices, edges) = self.immutable_vertices_edges();
        let vertex_num = vertices.len() as VertexIndex;
        let mut weighted_edges = Vec::with_capacity(edges.len());
        for edge in edges.iter() {
            weighted_edges.push(HyperEdge::new(edge.vertices.clone(), edge.weight.clone()));
        }
        SolverInitializer::new(vertex_num, weighted_edges)
    }

    fn get_model_graph(&self) -> Arc<ModelHyperGraph> {
        let initializer = Arc::new(self.get_initializer());
        Arc::new(ModelHyperGraph::new(initializer))
    }

    /// set defect vertices (non-trivial measurement result in case of single round of measurement,
    /// or different result from the previous round in case of multiple rounds of measurement)
    #[allow(clippy::unnecessary_cast)]
    fn set_defect_vertices(&mut self, defect_vertices: &[VertexIndex]) {
        let (vertices, _edges) = self.vertices_edges();
        for vertex in vertices.iter_mut() {
            vertex.is_defect = false;
        }
        for vertex_idx in defect_vertices.iter() {
            let vertex = &mut vertices[*vertex_idx as usize];
            vertex.is_defect = true;
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn set_physical_errors(&mut self, physical_errors: &[EdgeIndex]) {
        // clear existing errors
        self.set_defect_vertices(&[]);
        let (vertices, edges) = self.vertices_edges();
        for edge_idx in physical_errors {
            let edge = &edges[*edge_idx as usize];
            for vertex_idx in edge.vertices.iter() {
                vertices[*vertex_idx as usize].is_defect = !vertices[*vertex_idx as usize].is_defect;
            }
        }
    }

    /// check if the correction is valid, i.e., has the same syndrome with the input
    fn validate_correction(&mut self, correction: &OutputSubgraph) {
        // first check if the correction is valid, i.e., has the same defect
        let original_defect_vertices = self.get_defect_vertices();
        let correction_edges: Vec<EdgeIndex> = correction.iter().cloned().collect();
        self.set_physical_errors(&correction_edges);
        let new_defect_vertices = self.get_defect_vertices();
        assert_eq!(
            original_defect_vertices, new_defect_vertices,
            "invalid correction: parity check does not match input"
        );
        self.set_defect_vertices(&original_defect_vertices);
    }

    /// set erasure edges
    #[allow(clippy::unnecessary_cast)]
    fn set_erasures(&mut self, erasures: &[EdgeIndex]) {
        let (_vertices, edges) = self.vertices_edges();
        for edge in edges.iter_mut() {
            edge.is_erasure = false;
        }
        for edge_idx in erasures.iter() {
            let edge = &mut edges[*edge_idx as usize];
            edge.is_erasure = true;
        }
    }

    /// set syndrome
    fn set_syndrome(&mut self, syndrome_pattern: &SyndromePattern) {
        self.set_defect_vertices(&syndrome_pattern.defect_vertices);
        self.set_erasures(&syndrome_pattern.erasures);
    }

    /// get current defect vertices
    fn get_defect_vertices(&self) -> Vec<VertexIndex> {
        let (vertices, _edges) = self.immutable_vertices_edges();
        let mut syndrome = Vec::new();
        for (vertex_idx, vertex) in vertices.iter().enumerate() {
            if vertex.is_defect {
                syndrome.push(vertex_idx as VertexIndex);
            }
        }
        syndrome
    }

    /// get current erasure edges
    fn get_erasures(&self) -> Vec<EdgeIndex> {
        let (_vertices, edges) = self.immutable_vertices_edges();
        let mut erasures = Vec::new();
        for (edge_idx, edge) in edges.iter().enumerate() {
            if edge.is_erasure {
                erasures.push(edge_idx as EdgeIndex);
            }
        }
        erasures
    }

    /// get current syndrome
    fn get_syndrome(&self) -> SyndromePattern {
        SyndromePattern::new_erasure(self.get_defect_vertices(), self.get_erasures())
    }

    /// apply an error by flipping the vertices incident to it
    #[allow(clippy::unnecessary_cast)]
    fn apply_error(&mut self, edge_index: EdgeIndex) {
        let (vertices, edges) = self.vertices_edges();
        let edge = &edges[edge_index as usize];
        for vertex_index in edge.vertices.iter() {
            let vertex = &mut vertices[*vertex_index as usize];
            vertex.is_defect = !vertex.is_defect;
        }
    }

    fn apply_errors(&mut self, edge_indices: &[EdgeIndex]) {
        for &edge_index in edge_indices.iter() {
            self.apply_error(edge_index);
        }
    }

    /// generate random errors based on the edge probabilities and a seed for pseudo number generator
    #[allow(clippy::unnecessary_cast)]
    fn generate_random_errors(&mut self, seed: u64) -> (SyndromePattern, Subgraph) {
        let mut rng = DeterministicRng::seed_from_u64(seed);
        let (vertices, edges) = self.vertices_edges();
        for vertex in vertices.iter_mut() {
            vertex.is_defect = false;
        }
        let mut error_pattern = vec![];
        for (edge_index, edge) in edges.iter_mut().enumerate() {
            let p = if rng.next_f64() < edge.pe {
                edge.is_erasure = true;
                0.5 // when erasure happens, there are 50% chance of error
            } else {
                edge.is_erasure = false;
                edge.p
            };
            if rng.next_f64() < p {
                for vertex_index in edge.vertices.iter() {
                    let vertex = &mut vertices[*vertex_index as usize];
                    vertex.is_defect = !vertex.is_defect;
                }
                error_pattern.push(edge_index as EdgeIndex)
            }
        }
        (self.get_syndrome(), error_pattern)
    }

    fn is_defect(&self, vertex_idx: usize) -> bool {
        let (vertices, _edges) = self.immutable_vertices_edges();
        vertices[vertex_idx].is_defect
    }
}

#[cfg(feature = "python_binding")]
use rand::{thread_rng, Rng};

#[cfg(feature = "python_binding")]
macro_rules! bind_trait_example_code {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            fn __repr__(&self) -> String {
                format!("{:?}", self)
            }
            #[pyo3(name = "vertex_num")]
            fn trait_vertex_num(&self) -> VertexNum {
                self.vertex_num()
            }
            #[pyo3(name = "compute_weights")]
            fn trait_compute_weights(&mut self) {
                self.compute_weights()
            }
            #[pyo3(name = "sanity_check")]
            fn trait_sanity_check(&self) -> Option<String> {
                self.sanity_check().err()
            }
            #[pyo3(name = "set_probability")]
            fn trait_set_probability(&mut self, p: f64) {
                self.set_probability(p)
            }
            #[pyo3(name = "set_erasure_probability")]
            fn trait_set_erasure_probability(&mut self, p: f64) {
                self.set_erasure_probability(p)
            }
            #[pyo3(name = "fill_vertices")]
            fn trait_fill_vertices(&mut self, vertex_num: VertexNum) {
                self.fill_vertices(vertex_num)
            }
            #[pyo3(name = "get_positions")]
            fn trait_get_positions(&self) -> Vec<VisualizePosition> {
                self.get_positions()
            }
            #[pyo3(name = "get_initializer")]
            fn trait_get_initializer(&self) -> SolverInitializer {
                self.get_initializer()
            }
            #[pyo3(name = "set_defect_vertices")]
            fn trait_set_defect_vertices(&mut self, defect_vertices: Vec<VertexIndex>) {
                self.set_defect_vertices(&defect_vertices)
            }
            #[pyo3(name = "set_physical_errors")]
            fn trait_set_physical_errors(&mut self, physical_errors: Vec<EdgeIndex>) {
                self.set_physical_errors(&physical_errors)
            }
            #[pyo3(name = "set_erasures")]
            fn trait_set_erasures(&mut self, erasures: Vec<EdgeIndex>) {
                self.set_erasures(&erasures)
            }
            #[pyo3(name = "set_syndrome")]
            fn trait_set_syndrome(&mut self, syndrome_pattern: &SyndromePattern) {
                self.set_syndrome(syndrome_pattern)
            }
            #[pyo3(name = "get_defect_vertices")]
            fn trait_get_defect_vertices(&self) -> Vec<VertexIndex> {
                self.get_defect_vertices()
            }
            #[pyo3(name = "validate_correction")]
            fn trait_validate_correction(&mut self, correction: Vec<EdgeIndex>) {
                self.validate_correction(&OutputSubgraph::from(correction))
            }
            #[pyo3(name = "get_erasures")]
            fn trait_get_erasures(&self) -> Vec<EdgeIndex> {
                self.get_erasures()
            }
            #[pyo3(name = "get_syndrome")]
            fn trait_get_syndrome(&self) -> SyndromePattern {
                self.get_syndrome()
            }
            #[pyo3(name = "generate_random_errors", signature = (seed=thread_rng().gen()))]
            fn trait_generate_random_errors(&mut self, seed: u64) -> (SyndromePattern, Subgraph) {
                self.generate_random_errors(seed)
            }
            #[pyo3(name = "is_defect")]
            fn trait_is_defect(&mut self, vertex_idx: usize) -> bool {
                self.is_defect(vertex_idx)
            }
            #[pyo3(name = "snapshot", signature = (abbrev=true))]
            fn trait_snapshot(&mut self, abbrev: bool) -> PyObject {
                json_to_pyobject(self.snapshot(abbrev))
            }
        }
    };
}

impl<T> MWPSVisualizer for T
where
    T: ExampleCode,
{
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let (self_vertices, self_edges) = self.immutable_vertices_edges();
        let mut vertices = Vec::<serde_json::Value>::new();
        for vertex in self_vertices.iter() {
            vertices.push(json!({
                if abbrev { "s" } else { "is_defect" }: i32::from(vertex.is_defect),
            }));
        }
        let mut edges = Vec::<serde_json::Value>::new();
        for edge in self_edges.iter() {
            edges.push(json!({
                if abbrev { "w" } else { "weight" }: edge.weight.to_f64(),
                "wn": numer_of(&edge.weight),
                "wd": denom_of(&edge.weight),
                if abbrev { "v" } else { "vertices" }: edge.vertices,
            }));
        }
        json!({
            "vertices": vertices,
            "edges": edges,
        })
    }
}

/// perfect quantum repetition code
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeCapacityRepetitionCode {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// unscaled weights for BP
    pub unscaled_weights: Vec<f64>,
}

impl ExampleCode for CodeCapacityRepetitionCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
}

#[cfg(feature = "python_binding")]
bind_trait_example_code! {CodeCapacityRepetitionCode}

impl CodeCapacityRepetitionCode {
    pub fn new(d: VertexNum, p: f64) -> Self {
        let mut code = Self::create_code(d);
        code.set_probability(p);
        code.compute_weights();
        code
    }

    pub fn create_code(d: VertexNum) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd integer >= 3");
        let vertex_num = d - 1;
        // create edges
        let mut edges = Vec::new();
        for i in 0..d - 2 {
            edges.push(CodeEdge::new(vec![i, i + 1]));
        }
        edges.push(CodeEdge::new(vec![0])); // the left-most edge
        edges.push(CodeEdge::new(vec![d - 2])); // the right-most edge
        let mut code = Self {
            vertices: Vec::new(),
            edges,
            unscaled_weights: Vec::new(),
        };
        // create vertices
        code.fill_vertices(vertex_num);
        let mut positions = Vec::new();
        for i in 0..d - 1 {
            positions.push(VisualizePosition::new(0., i as f64, 0.));
        }
        for (i, position) in positions.into_iter().enumerate() {
            code.vertices[i].position = position;
        }
        code
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeCapacityRepetitionCode {
    #[new]
    #[pyo3(signature = (d, p))]
    fn py_new(d: VertexNum, p: f64) -> Self {
        Self::new(d, p)
    }

    #[staticmethod]
    #[pyo3(name = "create_code")]
    fn py_create_code(d: VertexNum) -> Self {
        Self::create_code(d)
    }
}

/// code capacity noise model is a single measurement round with perfect stabilizer measurements;
/// e.g. this is the decoding graph of a CSS surface code (standard one, not rotated one) with X-type stabilizers
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeCapacityPlanarCode {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// unscaled weights for BP
    pub unscaled_weights: Vec<f64>,
}

impl ExampleCode for CodeCapacityPlanarCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
}

#[cfg(feature = "python_binding")]
bind_trait_example_code! {CodeCapacityPlanarCode}

impl CodeCapacityPlanarCode {
    pub fn new(d: VertexNum, p: f64) -> Self {
        let mut code = Self::create_code(d);
        code.set_probability(p);
        code.compute_weights();
        code
    }

    pub fn create_code(d: VertexNum) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd integer >= 3");
        let row_vertex_num = d - 1;
        let vertex_num = row_vertex_num * d; // `d` rows
                                             // create edges
        let mut edges = Vec::new();
        for row in 0..d {
            let bias = row * row_vertex_num;
            for i in 0..d - 2 {
                edges.push(CodeEdge::new(vec![bias + i, bias + i + 1]));
            }
            edges.push(CodeEdge::new(vec![bias])); // the left-most edge
            edges.push(CodeEdge::new(vec![bias + d - 2])); // the right-most edge
            if row + 1 < d {
                for i in 0..d - 1 {
                    edges.push(CodeEdge::new(vec![bias + i, bias + i + row_vertex_num]));
                }
            }
        }
        let mut code = Self {
            vertices: Vec::new(),
            edges,
            unscaled_weights: Vec::new(),
        };
        // create vertices
        code.fill_vertices(vertex_num);
        let mut positions = Vec::new();
        for row in 0..d {
            for i in 0..row_vertex_num {
                positions.push(VisualizePosition::new(row as f64, i as f64, 0.));
            }
        }
        for (i, position) in positions.into_iter().enumerate() {
            code.vertices[i].position = position;
        }
        code
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeCapacityPlanarCode {
    #[new]
    #[pyo3(signature = (d, p))]
    fn py_new(d: VertexNum, p: f64) -> Self {
        Self::new(d, p)
    }

    #[staticmethod]
    #[pyo3(name = "create_code")]
    fn py_create_code(d: VertexNum) -> Self {
        Self::create_code(d)
    }
}

/// code capacity noise model is a single measurement round with perfect stabilizer measurements;
/// e.g. this is the decoding graph of a CSS surface code (standard one, not rotated one) with both stabilizers and
/// depolarizing noise model (X, Y, Z)
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeCapacityDepolarizePlanarCode {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// unscaled weights for BP
    pub unscaled_weights: Vec<f64>,
}

impl ExampleCode for CodeCapacityDepolarizePlanarCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
}

#[cfg(feature = "python_binding")]
bind_trait_example_code! {CodeCapacityDepolarizePlanarCode}

impl CodeCapacityDepolarizePlanarCode {
    pub fn new(d: VertexNum, p: f64) -> Self {
        let mut code = Self::create_code(d, true);
        code.set_probability(p);
        code.compute_weights();
        code
    }

    pub fn new_no_y(d: VertexNum, p: f64) -> Self {
        let mut code = Self::create_code(d, false);
        code.set_probability(p);
        code.compute_weights();
        code
    }

    pub fn create_code(d: VertexNum, with_y: bool) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd integer >= 3");
        let row_vertex_num = d - 1;
        // `d` rows
        let vertex_num = 2 * row_vertex_num * d;
        // first iterate all vertices
        let mut positions = Vec::new();
        let mut vertices: FastIterMap<(isize, isize), usize> = FastIterMap::new();
        // X and Z stabilizer vertices
        for is_z in [false, true] {
            for row in 0..d {
                for i in 0..row_vertex_num {
                    let vertex_index = vertices.len();
                    let a = row as isize * 2 - (d - 1) as isize;
                    let b = i as isize * 2 - (row_vertex_num - 1) as isize;
                    let vertex_position = if is_z { (a, b) } else { (b, a) };
                    vertices.insert(vertex_position, vertex_index);
                    positions.push(VisualizePosition::new(
                        vertex_position.0 as f64 / 1.6,
                        vertex_position.1 as f64 / 1.6,
                        0.,
                    ));
                }
            }
        }
        // create edges
        let mut edges = Vec::new();
        let is_in_range = |i: isize, j: isize| -> bool {
            for v in [i, j] {
                if v < -((d - 1) as isize) || v > (d - 1) as isize {
                    return false;
                }
            }
            true
        };
        let mut add_edge = |pos_vec: &[(isize, isize)]| {
            let mut edge_vertices = vec![];
            for &(i, j) in pos_vec {
                if is_in_range(i, j) {
                    edge_vertices.push(*vertices.get(&(i, j)).unwrap());
                }
            }
            edges.push(CodeEdge::new(edge_vertices));
        };
        let mut add_depolarize = |a: isize, b: isize| {
            add_edge(&[(a + 1, b), (a - 1, b)]);
            add_edge(&[(a, b + 1), (a, b - 1)]);
            if with_y {
                add_edge(&[(a, b + 1), (a, b - 1), (a + 1, b), (a - 1, b)]);
            }
        };
        for i in 0..d {
            for j in 0..d {
                let a = 2 * i as isize - (d - 1) as isize;
                let b = 2 * j as isize - (d - 1) as isize;
                add_depolarize(a, b)
            }
        }
        for i in 0..(d - 1) {
            for j in 0..(d - 1) {
                let a = 2 * i as isize - (d - 1) as isize + 1;
                let b = 2 * j as isize - (d - 1) as isize + 1;
                add_depolarize(a, b)
            }
        }
        let mut code = Self {
            vertices: Vec::new(),
            edges,
            unscaled_weights: Vec::new(),
        };
        // create vertices
        code.fill_vertices(vertex_num);
        for (i, position) in positions.into_iter().enumerate() {
            code.vertices[i].position = position;
        }
        code
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeCapacityDepolarizePlanarCode {
    #[new]
    #[pyo3(signature = (d, p, ))]
    fn py_new(d: VertexNum, p: f64) -> Self {
        Self::new(d, p)
    }

    #[staticmethod]
    #[pyo3(name = "new_no_y", signature = (d, p))]
    fn py_new_no_y(d: VertexNum, p: f64) -> Self {
        Self::new_no_y(d, p)
    }

    #[staticmethod]
    #[pyo3(name = "create_code")]
    fn py_create_code(d: VertexNum, with_y: bool) -> Self {
        Self::create_code(d, with_y)
    }
}

/// code capacity noise model is a single measurement round with perfect stabilizer measurements;
/// e.g. this is the decoding hypergraph of a rotated tailored surface code that have all the stabilizers and including degree-4 hyperedges;
/// the noise is biased to Z errors, with X and Y-typed stabilizers
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeCapacityTailoredCode {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// unscaled weights for BP
    pub unscaled_weights: Vec<f64>,
}

impl ExampleCode for CodeCapacityTailoredCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
}

#[cfg(feature = "python_binding")]
bind_trait_example_code! {CodeCapacityTailoredCode}

impl CodeCapacityTailoredCode {
    pub fn new(d: VertexNum, pxy: f64, pz: f64) -> Self {
        let mut code = Self::create_code(d, pxy, pz);
        code.compute_weights();
        code
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn create_code(d: VertexNum, pxy: f64, pz: f64) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd integer >= 3");
        // generate all the existing stabilizers
        let boundary_stab_num = (d - 1) / 2;
        let vertex_num = (d - 1) * (d - 1) + 4 * boundary_stab_num; // `d` rows
        let mut positions = Vec::new();
        let mut stabilizers = HashMap::<(usize, usize), VertexIndex>::new();
        for i in 0..boundary_stab_num as usize {
            stabilizers.insert((0, 4 + 4 * i), positions.len() as VertexIndex);
            positions.push(VisualizePosition::new(0., (2 + 2 * i) as f64, 0.))
        }
        for i in 0..boundary_stab_num as usize {
            stabilizers.insert((2 * d as usize, 2 + 4 * i), positions.len() as VertexIndex);
            positions.push(VisualizePosition::new(d as f64, (1 + 2 * i) as f64, 0.))
        }
        for row in 0..d as usize - 1 {
            for idx in 0..d as usize {
                let i = 2 + 2 * row;
                let j = 2 * idx + (if row % 2 == 0 { 0 } else { 2 });
                stabilizers.insert((i, j), positions.len() as VertexIndex);
                positions.push(VisualizePosition::new((i / 2) as f64, (j / 2) as f64, 0.))
            }
        }
        assert_eq!(positions.len(), vertex_num as usize);
        let mut edges = Vec::new();
        // first add Z errors
        if pz > 0. {
            for di in (1..2 * d as usize).step_by(2) {
                for dj in (1..2 * d as usize).step_by(2) {
                    let mut vertices = vec![];
                    for (si, sj) in [(di - 1, dj - 1), (di - 1, dj + 1), (di + 1, dj - 1), (di + 1, dj + 1)] {
                        if stabilizers.contains_key(&(si, sj)) {
                            vertices.push(stabilizers[&(si, sj)]);
                        }
                    }
                    let mut edge = CodeEdge::new(vertices);
                    edge.p = pz;
                    edges.push(edge);
                }
            }
        }
        // then add X and Y errors
        fn is_x_stab(si: usize, sj: usize) -> bool {
            (si + sj) % 4 == 2
        }
        if pxy > 0. {
            for di in (1..2 * d as usize).step_by(2) {
                for dj in (1..2 * d as usize).step_by(2) {
                    let mut x_error_vertices = vec![];
                    let mut y_error_vertices = vec![];
                    for (si, sj) in [(di - 1, dj - 1), (di - 1, dj + 1), (di + 1, dj - 1), (di + 1, dj + 1)] {
                        if stabilizers.contains_key(&(si, sj)) {
                            if !is_x_stab(si, sj) {
                                // X error is only detectable by Y stabilizers
                                x_error_vertices.push(stabilizers[&(si, sj)]);
                            }
                            if is_x_stab(si, sj) {
                                // Y error is only detectable by X stabilizers
                                y_error_vertices.push(stabilizers[&(si, sj)]);
                            }
                        }
                    }
                    for mut edge in [CodeEdge::new(x_error_vertices), CodeEdge::new(y_error_vertices)] {
                        edge.p = pxy;
                        edges.push(edge);
                    }
                }
            }
        }
        let mut code = Self {
            vertices: Vec::new(),
            edges,
            unscaled_weights: Vec::new(),
        };
        // there might be duplicate edges; select a larger probability one
        code.remove_duplicate_edges();
        // create vertices
        code.fill_vertices(vertex_num);
        for (i, position) in positions.into_iter().enumerate() {
            code.vertices[i].position = position;
        }
        code
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeCapacityTailoredCode {
    #[new]
    #[pyo3(signature = (d, pxy, pz,))]
    fn py_new(d: VertexNum, pxy: f64, pz: f64) -> Self {
        Self::new(d, pxy, pz)
    }

    #[staticmethod]
    #[pyo3(name = "create_code")]
    fn py_create_code(d: VertexNum, pxy: f64, pz: f64) -> Self {
        Self::create_code(d, pxy, pz)
    }
}

/// code capacity noise model is a single measurement round with perfect stabilizer measurements;
/// e.g. this is the decoding hypergraph of a color code that have all only the Z stabilizers
/// (because X and Z have the same location, for simplicity and better visual)
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct CodeCapacityColorCode {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// unscaled weights for BP
    pub unscaled_weights: Vec<f64>,
}

impl ExampleCode for CodeCapacityColorCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
}

#[cfg(feature = "python_binding")]
bind_trait_example_code! {CodeCapacityColorCode}

impl CodeCapacityColorCode {
    pub fn new(d: VertexNum, p: f64) -> Self {
        let mut code = Self::create_code(d);
        code.set_probability(p);
        code.compute_weights();
        code
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn create_code(d: VertexNum) -> Self {
        assert!(d >= 3 && d % 2 == 1, "d must be odd integer >= 3");
        // generate all the existing stabilizers
        let row_num = (d - 1) / 2 * 3 + 1;
        let vertex_num = (d - 1) * (d + 1) / 8 * 3;
        let mut positions = Vec::new();
        let mut stabilizers = HashMap::<(usize, usize), VertexIndex>::new();
        fn exists(d: VertexNum, i: isize, j: isize) -> bool {
            i >= 0 && j >= 0 && i + j <= (d as isize - 1) * 3 / 2
        }
        for row in 0..(d as usize - 1) / 2 {
            for column in 0..(d as usize - 1) / 2 - row {
                let gi = 1 + row * 3;
                let gj = column * 3;
                for (i, j) in [(gi, gj), (gi - 1, gj + 2), (gi + 1, gj + 1)] {
                    assert!(exists(d, i as isize, j as isize));
                    stabilizers.insert((i, j), positions.len() as VertexIndex);
                    let ratio = 0.7;
                    let x = (i as f64 + j as f64) * ratio;
                    let y = (j as f64 - i as f64) / 3f64.sqrt() * ratio;
                    positions.push(VisualizePosition::new(x, y, 0.))
                }
            }
        }
        assert_eq!(positions.len(), vertex_num as usize);
        let mut edges = Vec::new();
        for di in 0..row_num as isize {
            for dj in 0..row_num as isize - di {
                assert!(exists(d, di, dj));
                if (di + 2 * dj) % 3 != 1 {
                    // is data qubit
                    let mut vertices = vec![];
                    let directions = if (di + 2 * dj) % 3 == 0 {
                        [(0, -1), (1, 0), (-1, 1)]
                    } else {
                        [(1, -1), (-1, 0), (0, 1)]
                    };
                    for (dsi, dsj) in directions {
                        let (si, sj) = (di + dsi, dj + dsj);
                        if exists(d, si, sj) {
                            vertices.push(stabilizers[&(si as usize, sj as usize)]);
                        }
                    }
                    edges.push(CodeEdge::new(vertices));
                }
            }
        }
        let mut code = Self {
            vertices: Vec::new(),
            edges,
            unscaled_weights: Vec::new(),
        };
        // create vertices
        code.fill_vertices(vertex_num);
        for (i, position) in positions.into_iter().enumerate() {
            code.vertices[i].position = position;
        }
        code
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl CodeCapacityColorCode {
    #[new]
    #[pyo3(signature = (d, p))]
    fn py_new(d: VertexNum, p: f64) -> Self {
        Self::new(d, p)
    }

    #[staticmethod]
    #[pyo3(name = "create_code")]
    fn py_create_code(d: VertexNum) -> Self {
        Self::create_code(d)
    }
}

/// example code with QEC-Playground as simulator
#[cfg(feature = "qecp_integrate")]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
#[derive(Debug, Clone)]
pub struct QECPlaygroundCode {
    simulator: qecp::simulator::Simulator,
    noise_model: std::sync::Arc<qecp::noise_model::NoiseModel>,
    edge_index_map: std::sync::Arc<HashMap<usize, EdgeIndex>>,
    model_hypergraph: Arc<qecp::model_hypergraph::ModelHypergraph>,
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
}

#[cfg(all(feature = "python_binding", feature = "qecp_integrate"))]
#[pymethods]
impl QECPlaygroundCode {
    #[getter]
    fn get_vertices(&self) -> Vec<CodeVertex> {
        self.vertices.clone()
    }
    #[setter]
    fn set_vertices(&mut self, vertices: Vec<CodeVertex>) {
        self.vertices = vertices;
    }
    #[getter]
    fn get_edges(&self) -> Vec<CodeEdge> {
        self.edges.clone()
    }
    #[setter]
    fn set_edges(&mut self, edges: Vec<CodeEdge>) {
        self.edges = edges;
    }
    #[new]
    #[pyo3(signature = (d, p, config))]
    fn py_new(d: VertexNum, p: f64, config: &Bound<PyAny>) -> Self {
        Self::new(d, p, serde_json::from_value(pyobject_to_json_locked(config)).unwrap())
    }
}

#[cfg(feature = "qecp_integrate")]
impl ExampleCode for QECPlaygroundCode {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
    // override simulation function
    #[allow(clippy::unnecessary_cast)]
    fn generate_random_errors(&mut self, seed: u64) -> (SyndromePattern, Subgraph) {
        use qecp::simulator::SimulatorGenerics;
        let rng = qecp::reproducible_rand::Xoroshiro128StarStar::seed_from_u64(seed);
        self.simulator.set_rng(rng);
        let (error_count, erasure_count) = self.simulator.generate_random_errors(&self.noise_model);
        assert!(erasure_count == 0, "not implemented");
        // let sparse_detected_erasures = if erasure_count != 0 {
        //     self.simulator.generate_sparse_detected_erasures()
        // } else {
        //     qecp::simulator::SparseErasures::new()
        // };
        let sparse_measurement = if error_count != 0 {
            self.simulator.generate_sparse_measurement()
        } else {
            qecp::simulator::SparseMeasurement::new()
        };
        let defects: Vec<_> = sparse_measurement
            .defects
            .iter()
            .map(|defect| self.model_hypergraph.vertex_indices[defect])
            .collect();
        let syndrome_pattern = SyndromePattern::new_vertices(defects);
        for vertex in self.vertices.iter_mut() {
            vertex.is_defect = false;
        }
        for &vertex_index in syndrome_pattern.defect_vertices.iter() {
            self.vertices[vertex_index].is_defect = true;
        }
        for edge in self.edges.iter_mut() {
            edge.is_erasure = false;
        }
        for &edge_index in syndrome_pattern.erasures.iter() {
            if let Some(new_index) = self.edge_index_map.get(&edge_index) {
                self.edges[*new_index as usize].is_erasure = true;
            }
        }
        // TODO: generate the real error pattern
        (self.get_syndrome(), vec![])
    }
}

#[cfg(feature = "qecp_integrate")]
impl QECPlaygroundCode {
    #[allow(clippy::unnecessary_cast)]
    pub fn new(d: usize, p: f64, config: serde_json::Value) -> Self {
        let config: QECPlaygroundCodeConfig = serde_json::from_value(config).unwrap();
        let di = config.di.unwrap_or(d);
        let dj = config.dj.unwrap_or(d);
        let nm = config.nm.unwrap_or(d);
        let mut simulator = qecp::simulator::Simulator::new(config.code_type, qecp::code_builder::CodeSize::new(nm, di, dj));
        let mut noise_model = qecp::noise_model::NoiseModel::new(&simulator);
        let px = p / (1. + config.bias_eta) / 2.;
        let py = px;
        let pz = p - 2. * px;
        simulator.set_error_rates(&mut noise_model, px, py, pz, config.pe);
        // apply customized noise model
        if let Some(noise_model_builder) = &config.noise_model {
            noise_model_builder.apply(
                &mut simulator,
                &mut noise_model,
                &config.noise_model_configuration,
                p,
                config.bias_eta,
                config.pe,
            );
        }
        simulator.compress_error_rates(&mut noise_model); // by default compress all error rates
        let noise_model = std::sync::Arc::new(noise_model);
        // construct vertices and edges
        let hyperion_config: HyperionDecoderConfig = serde_json::from_value(json!({})).unwrap();
        let mut model_hypergraph = qecp::model_hypergraph::ModelHypergraph::new(&simulator);
        model_hypergraph.build(
            &mut simulator,
            Arc::clone(&noise_model),
            &hyperion_config.weight_function,
            config.parallel_init,
            hyperion_config.use_combined_probability,
            config.use_brief_edge,
        );
        let model_hypergraph = Arc::new(model_hypergraph);
        // implementing: model_hypergraph.generate_mwpf_hypergraph(config.max_weight);

        let mut weighted_edges = Vec::with_capacity(model_hypergraph.weighted_edges.len());
        for (defect_vertices, hyperedge_group) in model_hypergraph.weighted_edges.iter() {
            if hyperedge_group.hyperedge.probability > 0. {
                // only add those possible edges; for erasures, handle later
                let weight = hyperedge_group.hyperedge.weight;
                assert!(weight.is_finite(), "weight must be normal");
                // assert!(weight >= 0., "weight must be non-negative");
                // assert!(weight <= config.max_weight as f64, "weight must be smaller than max weight");
                let vertex_indices: Vec<_> = defect_vertices.0.iter().map(|x| model_hypergraph.vertex_indices[x]).collect();
                weighted_edges.push(HyperEdge::new(vertex_indices, Rational::from_f64(weight).unwrap()));
            }
        }
        let vertex_num = model_hypergraph.vertex_positions.len();
        let initializer = Arc::new(SolverInitializer::new(vertex_num, weighted_edges));
        let positions = &model_hypergraph.vertex_positions;
        let mut code = Self {
            simulator,
            noise_model,
            model_hypergraph: model_hypergraph.clone(),
            edge_index_map: std::sync::Arc::new(HashMap::new()), // overwrite later
            vertices: Vec::with_capacity(initializer.vertex_num),
            edges: Vec::with_capacity(initializer.weighted_edges.len()),
        };
        let mut edge_index_map = HashMap::new();
        for (edge_index, hyperedge) in initializer.weighted_edges.iter().cloned().enumerate() {
            let new_index = edge_index_map.len() as EdgeIndex;
            edge_index_map.insert(edge_index, new_index);
            code.edges.push(CodeEdge {
                vertices: hyperedge.vertices,
                p: 0.,  // doesn't matter
                pe: 0., // doesn't matter
                weight: hyperedge.weight,
                is_erasure: false, // doesn't matter
            });
        }
        code.edge_index_map = std::sync::Arc::new(edge_index_map);
        // automatically create the vertices and nearest-neighbor connection
        code.fill_vertices(code.model_hypergraph.vertex_positions.len());
        // set virtual vertices and positions
        for (vertex_index, position) in positions.iter().cloned().enumerate() {
            code.vertices[vertex_index].position =
                VisualizePosition::new(position.i as f64, position.j as f64, position.t as f64 / 3.0);
        }
        code
    }
}

#[cfg(all(feature = "python_binding", feature = "qecp_integrate"))]
bind_trait_example_code! {QECPlaygroundCode}

#[cfg(feature = "qecp_integrate")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QECPlaygroundCodeConfig {
    // default to d
    pub di: Option<usize>,
    pub dj: Option<usize>,
    pub nm: Option<usize>,
    #[serde(default = "qec_playground_default_configs::pe")]
    pub pe: f64,
    pub noise_model_modifier: Option<serde_json::Value>,
    #[serde(default = "qec_playground_default_configs::code_type")]
    pub code_type: qecp::code_builder::CodeType,
    #[serde(default = "qec_playground_default_configs::bias_eta")]
    pub bias_eta: f64,
    pub noise_model: Option<qecp::noise_model_builder::NoiseModelBuilder>,
    #[serde(default = "qec_playground_default_configs::noise_model_configuration")]
    pub noise_model_configuration: serde_json::Value,
    #[serde(default = "qec_playground_default_configs::parallel_init")]
    pub parallel_init: usize,
    #[serde(default = "qec_playground_default_configs::use_brief_edge")]
    pub use_brief_edge: bool,
    // specify the target qubit type
    pub qubit_type: Option<qecp::types::QubitType>,
}

#[cfg(feature = "qecp_integrate")]
pub mod qec_playground_default_configs {
    pub fn pe() -> f64 {
        0.
    }
    pub fn bias_eta() -> f64 {
        0.5
    }
    pub fn noise_model_configuration() -> serde_json::Value {
        json!({})
    }
    pub fn code_type() -> qecp::code_builder::CodeType {
        qecp::code_builder::CodeType::StandardPlanarCode
    }
    pub fn parallel_init() -> usize {
        1
    }
    pub fn use_brief_edge() -> bool {
        false
    }
}

#[cfg(feature = "qecp_integrate")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionDecoderConfig {
    /// weight function, by default using [`WeightFunction::AutotuneImproved`]
    #[serde(alias = "wf")] // abbreviation
    #[serde(default = "hyperion_default_configs::weight_function")]
    pub weight_function: qecp::model_graph::WeightFunction,
    /// combined probability can improve accuracy, but will cause probabilities differ a lot even in the case of i.i.d. noise model
    #[serde(alias = "ucp")] // abbreviation
    #[serde(default = "hyperion_default_configs::use_combined_probability")]
    pub use_combined_probability: bool,
    #[serde(default = "hyperion_default_configs::default_hyperion_config")]
    pub hyperion_config: serde_json::Value,
}

#[cfg(feature = "qecp_integrate")]
pub mod hyperion_default_configs {
    use super::*;
    pub fn default_hyperion_config() -> serde_json::Value {
        json!({})
    }
    pub fn weight_function() -> qecp::model_graph::WeightFunction {
        qecp::model_graph::WeightFunction::AutotuneImproved
    }
    pub fn use_combined_probability() -> bool {
        true
    } // default use combined probability for better accuracy
}

/// read from file, including the error patterns;
/// the point is to avoid bad cache performance, because generating random error requires iterating over a large memory space,
/// invalidating all cache. also, this can reduce the time of decoding by prepare the data before hand and could be shared between
/// different partition configurations
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct ErrorPatternReader {
    /// vertices in the code
    pub vertices: Vec<CodeVertex>,
    /// nearest-neighbor edges in the decoding graph
    pub edges: Vec<CodeEdge>,
    /// pre-generated syndrome patterns
    pub syndrome_patterns: Vec<SyndromePattern>,
    /// cursor of current syndrome
    pub syndrome_index: usize,
}

impl ExampleCode for ErrorPatternReader {
    fn vertices_edges(&mut self) -> (&mut Vec<CodeVertex>, &mut Vec<CodeEdge>) {
        (&mut self.vertices, &mut self.edges)
    }
    fn immutable_vertices_edges(&self) -> (&Vec<CodeVertex>, &Vec<CodeEdge>) {
        (&self.vertices, &self.edges)
    }
    fn generate_random_errors(&mut self, _seed: u64) -> (SyndromePattern, Subgraph) {
        assert!(
            self.syndrome_index < self.syndrome_patterns.len(),
            "reading syndrome pattern more than in the file, consider generate the file with more data points"
        );
        let syndrome_pattern = self.syndrome_patterns[self.syndrome_index].clone();
        self.syndrome_index += 1;
        (syndrome_pattern, vec![])
    }
}

impl ErrorPatternReader {
    #[allow(clippy::unnecessary_cast)]
    pub fn new(mut config: serde_json::Value) -> Self {
        let mut filename = "tmp/syndrome_patterns.txt".to_string();
        let config = config.as_object_mut().expect("config must be JSON object");
        if let Some(value) = config.remove("filename") {
            filename = value.as_str().expect("filename string").to_string();
        }
        if !config.is_empty() {
            panic!("unknown config keys: {:?}", config.keys().collect::<Vec<&String>>());
        }
        let file = File::open(filename).unwrap();
        let mut syndrome_patterns = vec![];
        let mut initializer: Option<SolverInitializer> = None;
        let mut positions: Option<Vec<VisualizePosition>> = None;
        for (line_index, line) in io::BufReader::new(file).lines().enumerate() {
            if let Ok(value) = line {
                match line_index {
                    0 => {
                        assert!(value.starts_with("Syndrome Pattern v1.0 "), "incompatible file version");
                    }
                    1 => {
                        initializer = Some(serde_json::from_str(&value).unwrap());
                    }
                    2 => {
                        positions = Some(serde_json::from_str(&value).unwrap());
                    }
                    _ => {
                        let syndrome_pattern: SyndromePattern = serde_json::from_str(&value).unwrap();
                        syndrome_patterns.push(syndrome_pattern);
                    }
                }
            }
        }
        let initializer = initializer.expect("initializer not present in file");
        let positions = positions.expect("positions not present in file");
        assert_eq!(positions.len(), initializer.vertex_num as usize);
        let mut code = Self::from_initializer(&initializer);
        code.syndrome_patterns = syndrome_patterns;
        // set virtual vertices and positions
        for (vertex_index, position) in positions.into_iter().enumerate() {
            code.vertices[vertex_index].position = position;
        }
        code
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn from_initializer(initializer: &SolverInitializer) -> Self {
        let mut code = Self {
            vertices: Vec::with_capacity(initializer.vertex_num as usize),
            edges: Vec::with_capacity(initializer.weighted_edges.len()),
            syndrome_patterns: vec![],
            syndrome_index: 0,
        };
        for hyperedge in initializer.weighted_edges.iter() {
            code.edges.push(CodeEdge {
                vertices: hyperedge.vertices.clone(),
                p: 0.,  // doesn't matter
                pe: 0., // doesn't matter
                weight: hyperedge.weight.clone(),
                is_erasure: false, // doesn't matter
            });
        }
        // automatically create the vertices and nearest-neighbor connection
        code.fill_vertices(initializer.vertex_num);
        code
    }
}

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CodeVertex>()?;
    m.add_class::<CodeEdge>()?;
    m.add_function(wrap_pyfunction!(weight_of_p, m)?)?;
    m.add_class::<CodeCapacityRepetitionCode>()?;
    m.add_class::<CodeCapacityPlanarCode>()?;
    m.add_class::<CodeCapacityTailoredCode>()?;
    m.add_class::<CodeCapacityColorCode>()?;
    m.add_class::<CodeCapacityDepolarizePlanarCode>()?;
    m.add_class::<QECPlaygroundCode>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::*;
    use rand::{thread_rng, Rng};

    fn visualize_code(code: &mut impl ExampleCode, visualize_filename: String) {
        let visualizer_path = visualize_data_folder() + visualize_filename.as_str();
        let mut visualizer = Visualizer::new(Some(visualizer_path.clone()), code.get_positions(), true).unwrap();
        visualizer.snapshot("code".to_string(), code).unwrap();
        for round in 0..3 {
            code.generate_random_errors(round);
            visualizer.snapshot(format!("syndrome {}", round + 1), code).unwrap();
        }
        if cfg!(feature = "embed_visualizer") {
            let html = visualizer.generate_html(json!({}));
            assert!(visualizer_path.ends_with(".json"));
            let html_path = format!("{}.html", &visualizer_path.as_str()[..visualizer_path.len() - 5]);
            std::fs::write(&html_path, html).expect("Unable to write file");
            println!("visualizer path: {}", &html_path);
        }
    }

    #[test]
    fn example_code_capacity_repetition_code() {
        // cargo test example_code_capacity_repetition_code -- --nocapture
        let mut code = CodeCapacityRepetitionCode::new(7, 0.2);
        code.sanity_check().unwrap();
        visualize_code(&mut code, "example_code_capacity_repetition_code.json".to_string());
    }

    #[test]
    fn example_code_capacity_planar_code() {
        // cargo test example_code_capacity_planar_code -- --nocapture
        let mut code = CodeCapacityPlanarCode::new(7, 0.1);
        code.sanity_check().unwrap();
        visualize_code(&mut code, "example_code_capacity_planar_code.json".to_string());
    }

    #[test]
    fn example_code_capacity_depolarize_planar_code() {
        // cargo test example_code_capacity_depolarize_planar_code -- --nocapture
        let mut code = CodeCapacityDepolarizePlanarCode::new(5, 0.1);
        code.sanity_check().unwrap();
        visualize_code(&mut code, "example_code_capacity_depolarize_planar_code.json".to_string());
        let mut code_no_y = CodeCapacityDepolarizePlanarCode::new_no_y(5, 0.1);
        code_no_y.sanity_check().unwrap();
        visualize_code(
            &mut code_no_y,
            "example_code_capacity_depolarize_planar_code_no_y.json".to_string(),
        );
    }

    #[test]
    fn example_code_capacity_tailored_code() {
        // cargo test example_code_capacity_tailored_code -- --nocapture
        let mut code = CodeCapacityTailoredCode::new(5, 0.001, 0.1);
        code.sanity_check().unwrap();
        visualize_code(&mut code, "example_code_capacity_tailored_code.json".to_string());
    }

    #[test]
    fn example_code_capacity_color_code() {
        // cargo test example_code_capacity_color_code -- --nocapture
        let mut code = CodeCapacityColorCode::new(7, 0.1);
        code.sanity_check().unwrap();
        visualize_code(&mut code, "example_code_capacity_color_code.json".to_string());
    }

    #[test]
    fn example_code_correction_validity_code_capacity_repetition_code() {
        // cargo test --release example_code_correction_validity_code_capacity_repetition_code -- --nocapture
        let d_vec = [3, 5, 7, 9, 11];
        let p_vec = [0.1, 0.01];
        let repeat = 10000;
        for d in d_vec {
            for p in p_vec {
                println!("d={d}, p={p}");
                let mut code = CodeCapacityRepetitionCode::new(d, p);
                code.sanity_check().unwrap();
                let initializer = Arc::new(code.get_initializer());
                let mut solver = SolverType::JointSingleHair.build(&initializer, &code, json!({ "cluster_node_limit": 50 }));
                for _ in 0..repeat {
                    let (syndrome, _) = code.generate_random_errors(thread_rng().gen::<u64>());
                    solver.solve(syndrome);
                    let (subgraph, _weight_range) = solver.subgraph_range();
                    code.validate_correction(&subgraph);
                    solver.clear();
                }
            }
        }
    }

    #[cfg(feature = "f64_weight")] // too slow, skip
    #[test]
    fn example_code_correction_validity_code_capacity_depolarize_planar_code() {
        // cargo test --release example_code_correction_validity_code_capacity_depolarize_planar_code -- --nocapture
        let d_vec = [3, 5, 7];
        let p_vec = [0.03, 0.01];
        let repeat = 10000;
        for d in d_vec {
            for p in p_vec {
                println!("d={d}, p={p}");
                let mut code = CodeCapacityDepolarizePlanarCode::new(d, p);
                code.sanity_check().unwrap();
                let initializer = Arc::new(code.get_initializer());
                let mut solver = SolverType::JointSingleHair.build(&initializer, &code, json!({ "cluster_node_limit": 50 }));
                for _ in 0..repeat {
                    let (syndrome, _) = code.generate_random_errors(thread_rng().gen::<u64>());
                    solver.solve(syndrome);
                    let (subgraph, _weight_range) = solver.subgraph_range();
                    code.validate_correction(&subgraph);
                    solver.clear();
                }
            }
        }
    }

    #[cfg(feature = "f64_weight")] // too slow, skip
    #[test]
    fn example_code_correction_validity_code_capacity_color_code() {
        // cargo test --release example_code_correction_validity_code_capacity_color_code -- --nocapture
        let d_vec = [3, 5, 7, 9];
        let p_vec = [0.1, 0.01];
        let repeat = 10000;
        for d in d_vec {
            for p in p_vec {
                println!("d={d}, p={p}");
                let mut code = CodeCapacityColorCode::new(d, p);
                code.sanity_check().unwrap();
                let initializer = Arc::new(code.get_initializer());
                let mut solver = SolverType::JointSingleHair.build(&initializer, &code, json!({ "cluster_node_limit": 50 }));
                for _ in 0..repeat {
                    let (syndrome, _) = code.generate_random_errors(thread_rng().gen::<u64>());
                    solver.solve(syndrome);
                    let (subgraph, _weight_range) = solver.subgraph_range();
                    code.validate_correction(&subgraph);
                    solver.clear();
                }
            }
        }
    }

    #[cfg(feature = "f64_weight")] // too slow, skip
    #[test]
    fn example_code_optimality_code_capacity_tailored_code() {
        // cargo test --release example_code_optimality_code_capacity_tailored_code -- --nocapture
        let d_vec = [3, 5, 7];
        let p_vec = [0.1, 0.01];
        let repeat = 10000;
        for d in d_vec {
            for p in p_vec {
                println!("d={d}, p={p}");
                let mut code = CodeCapacityTailoredCode::new(d, 0., p);
                code.sanity_check().unwrap();
                let initializer = Arc::new(code.get_initializer());
                let mut solver = SolverType::JointSingleHair.build(&initializer, &code, json!({})); // "cluster_node_limit": 50
                for _ in 0..repeat {
                    let (syndrome, _) = code.generate_random_errors(thread_rng().gen::<u64>());
                    solver.solve(syndrome.clone());
                    let (subgraph, weight_range) = solver.subgraph_range();
                    code.validate_correction(&subgraph);
                    if weight_range.lower != weight_range.upper {
                        println!("weight range: {:?}, syndrome = {:?}", weight_range, syndrome);
                    }
                    assert!(
                        rational_approx_eq(&weight_range.lower, &weight_range.upper),
                        "must be optimal"
                    );
                    solver.clear();
                }
            }
        }
    }
}
