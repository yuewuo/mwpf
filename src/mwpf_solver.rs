//! Minimum-Weight Parity Factor Solver
//!
//! This module includes some common usage of primal and dual modules to solve MWPF problems.
//! Note that you can call different primal and dual modules, even interchangeably, by following the examples in this file
//!

use crate::cluster::*;
use crate::dual_module::*;
use crate::dual_module_pq::*;
use crate::example_codes::*;
use crate::matrix::*;
use crate::model_hypergraph::*;
use crate::plugin::*;
use crate::plugin_single_hair::*;
use crate::plugin_union_find::PluginUnionFind;
use crate::primal_module::*;
use crate::primal_module_serial::*;
use crate::util::*;
use crate::visualize::*;

use bp::bp::BpDecoder;

use core::panic;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use std::fs::File;
use std::io::{prelude::*, BufWriter};
use std::sync::Arc;

cfg_if::cfg_if! {
    if #[cfg(feature="python_binding")] {
        use crate::invalid_subgraph::*;
        use crate::util_py::*;

        use bp::bp::BpSparse;

        use pyo3::prelude::*;
        use pyo3::types::{PyTuple, PyDict};
    }
}

pub trait SolverTrait {
    fn debug_print(&self) {
        unimplemented!();
    }
    fn clear(&mut self);
    fn solve_visualizer(&mut self, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>);
    fn solve(&mut self, syndrome_pattern: SyndromePattern) {
        self.solve_visualizer(syndrome_pattern, None)
    }
    fn subgraph_range_visualizer(&mut self, visualizer: Option<&mut Visualizer>) -> (OutputSubgraph, WeightRange);
    fn subgraph_range(&mut self) -> (OutputSubgraph, WeightRange) {
        self.subgraph_range_visualizer(None)
    }
    fn subgraph(&mut self) -> OutputSubgraph {
        self.subgraph_range().0
    }
    fn sum_dual_variables(&self) -> Rational;
    fn generate_profiler_report(&self) -> serde_json::Value;

    fn get_tuning_time(&self) -> Option<f64>;
    fn clear_tuning_time(&mut self);
    fn print_clusters(&self) {
        panic!();
    }
    fn update_weights(&mut self, new_weights: Vec<Weight>, mix_ratio: Weight);
    fn get_model_graph(&self) -> Arc<ModelHyperGraph>;
    fn solver_base(&self) -> SolverBase;
}

#[cfg(feature = "python_binding")]
macro_rules! bind_trait_to_python {
    ($struct_name:ident) => {
        #[pymethods]
        impl $struct_name {
            #[pyo3(name = "clear")]
            fn py_clear(&mut self, py: Python<'_>) {
                py.allow_threads(move || self.clear())
            }
            #[pyo3(name = "solve", signature = (syndrome_pattern, visualizer=None))] // in Python, `solve` and `solve_visualizer` is the same because it can take optional parameter
            fn py_solve(&mut self, py: Python<'_>, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>) {
                py.allow_threads(move || self.solve_visualizer(syndrome_pattern, visualizer));
            }
            #[pyo3(name = "subgraph_range", signature = (visualizer=None))] // in Python, `subgraph_range` and `subgraph_range_visualizer` is the same
            fn py_subgraph_range(
                &mut self,
                py: Python<'_>,
                visualizer: Option<&mut Visualizer>,
            ) -> (PySubgraph, PyWeightRange) {
                let (subgraph, range) = py.allow_threads(move || self.subgraph_range_visualizer(visualizer));
                let mut complete_subgraph = subgraph.into_iter().collect::<Vec<EdgeIndex>>();
                complete_subgraph.sort();
                (complete_subgraph.into(), range.into())
            }
            #[pyo3(name = "subgraph", signature = (visualizer=None))]
            fn py_subgraph(&mut self, py: Python<'_>, visualizer: Option<&mut Visualizer>) -> Subgraph {
                py.allow_threads(move || self.subgraph_range_visualizer(visualizer).0.into_iter().collect())
            }
            #[pyo3(name = "sum_dual_variables")]
            fn py_sum_dual_variables(&self) -> PyRational {
                self.sum_dual_variables().clone().into()
            }
            #[pyo3(name = "load_syndrome", signature = (syndrome_pattern, visualizer=None, skip_initial_duals=false))]
            pub fn py_load_syndrome(
                &mut self,
                py: Python<'_>,
                syndrome_pattern: &SyndromePattern,
                visualizer: Option<&mut Visualizer>,
                skip_initial_duals: bool,
            ) {
                py.allow_threads(move || self.0.load_syndrome(syndrome_pattern, visualizer, skip_initial_duals))
            }
            #[pyo3(name = "get_node", signature = (node_index))]
            pub fn py_get_node(&mut self, node_index: NodeIndex) -> Option<PyDualNodePtr> {
                self.0.interface_ptr.get_node(node_index).map(|x| x.into())
            }
            #[pyo3(name = "find_node", signature = (vertices=None, edges=None))]
            pub fn py_find_node(
                &self,
                vertices: Option<&Bound<PyAny>>,
                edges: Option<&Bound<PyAny>>,
            ) -> PyResult<Option<PyDualNodePtr>> {
                let invalid_subgraph = Arc::new(self.py_construct_invalid_subgraph(vertices, edges)?);
                Ok(self.0.interface_ptr.find_node(&invalid_subgraph).map(|x| x.into()))
            }
            #[pyo3(name = "sanity_check_node", signature = (node))]
            pub fn py_sanity_check_node(&self, node: PyDualNodePtr) {
                let interface = self.0.interface_ptr.read_recursive();
                node.0
                    .read_recursive()
                    .invalid_subgraph
                    .sanity_check(&interface.decoding_graph)
                    .unwrap();
            }
            #[pyo3(name = "create_node", signature = (vertices=None, edges=None, find_existing_node=true))]
            pub fn py_create_node(
                &mut self,
                vertices: Option<&Bound<PyAny>>,
                edges: Option<&Bound<PyAny>>,
                find_existing_node: bool,
            ) -> PyResult<PyDualNodePtr> {
                let invalid_subgraph = Arc::new(self.py_construct_invalid_subgraph(vertices, edges)?);
                if find_existing_node {
                    if let Some(node) = self.py_find_node(vertices, edges)? {
                        return Ok(node);
                    }
                }
                let interface_ptr = self.0.interface_ptr.clone();
                Ok(match self.0.dual_module.mode() {
                    DualModuleMode::Search => interface_ptr.create_node(invalid_subgraph, &mut self.0.dual_module),
                    DualModuleMode::Tune => interface_ptr.create_node_tune(invalid_subgraph, &mut self.0.dual_module),
                }
                .into())
            }
            /// create a node without providing any information about the invalid cluster itself, hence no safety checks
            #[pyo3(name = "create_node_hair_unchecked", signature = (hair, vertices=None, edges=None))]
            pub fn py_create_node_hair_unchecked(
                &mut self,
                hair: &Bound<PyAny>,
                vertices: Option<&Bound<PyAny>>,
                edges: Option<&Bound<PyAny>>,
            ) -> PyResult<PyDualNodePtr> {
                let hair = py_into_set(hair)?;
                assert!(!hair.is_empty(), "hair must not be empty");
                let vertices = if let Some(vertices) = vertices {
                    py_into_set(vertices)?
                } else {
                    FastIterSet::new()
                };
                let edges = if let Some(edges) = edges {
                    py_into_set(edges)?
                } else {
                    FastIterSet::new()
                };
                let invalid_subgraph = Arc::new(InvalidSubgraph::new_raw(vertices, edges, hair));
                let interface_ptr = self.0.interface_ptr.clone();
                Ok(match self.0.dual_module.mode() {
                    DualModuleMode::Search => interface_ptr.create_node(invalid_subgraph, &mut self.0.dual_module),
                    DualModuleMode::Tune => interface_ptr.create_node_tune(invalid_subgraph, &mut self.0.dual_module),
                }
                .into())
            }
            #[pyo3(name = "grow", signature = (length))]
            fn py_grow(&mut self, py: Python<'_>, length: PyRational) {
                let length: Rational = length.into();
                py.allow_threads(move || {
                    if let Some(max_valid_grow) = self.0.dual_module.compute_max_valid_grow() {
                        assert!(
                            length <= max_valid_grow,
                            "growth overflow: attempting to grow {} but can only grow {} maximum",
                            length,
                            max_valid_grow
                        );
                    };
                    self.0.dual_module.grow(length);
                });
            }
            #[pyo3(name = "snapshot", signature = (abbrev=true))]
            fn py_snapshot(&mut self, py: Python<'_>, abbrev: bool) -> PyObject {
                let value = py.allow_threads(move || self.0.snapshot(abbrev));
                json_to_pyobject(value)
            }
            #[pyo3(name = "dual_report")]
            fn py_dual_report(&mut self) -> PyDualReport {
                self.0.dual_module.report().into()
            }
            #[pyo3(name = "get_edge_nodes")]
            fn py_get_edge_nodes(&self, edge_index: EdgeIndex) -> Vec<PyDualNodePtr> {
                self.0
                    .dual_module
                    .get_edge_nodes(edge_index)
                    .into_iter()
                    .map(|x| x.into())
                    .collect()
            }
            #[pyo3(name = "set_grow_rate")]
            fn py_set_grow_rate(&mut self, dual_node_ptr: PyDualNodePtr, grow_rate: PyRational) {
                self.0.dual_module.set_grow_rate(&dual_node_ptr.0, grow_rate.into())
            }
            #[pyo3(name = "stop_all")]
            fn py_stop_all(&mut self) {
                let mut node_index = 0;
                use crate::num_traits::Zero;
                while let Some(node_ptr) = self.0.interface_ptr.get_node(node_index) {
                    self.0.dual_module.set_grow_rate(&node_ptr, Rational::zero());
                    node_index += 1;
                }
            }
            #[pyo3(name = "get_cluster")]
            fn py_get_cluster(&self, vertex_index: VertexIndex) -> PyCluster {
                self.get_cluster(vertex_index).into()
            }
            /// a shortcut for creating a visualizer to display the current state of the solver
            #[pyo3(name = "show")]
            fn py_show(&self, py: Python<'_>, positions: Vec<VisualizePosition>) {
                py.allow_threads(move || {
                    let mut visualizer = Visualizer::new(Some(String::new()), positions, true).unwrap();
                    visualizer.snapshot("show".to_string(), &self.0).unwrap();
                    visualizer.show_py(None, None);
                });
            }
            #[pyo3(name = "get_initializer")]
            fn py_get_initializer(&self) -> SolverInitializer {
                self.0.model_graph.initializer.as_ref().clone()
            }
            fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("initializer", self.py_get_initializer())?;
                kwargs.set_item("config", json_to_pyobject(json!(self.0.config.clone())))?;
                let args = PyTuple::empty(py);
                Ok((args, kwargs).into_pyobject(py)?.unbind())
            }
            #[getter]
            fn get_config(&self) -> PyObject {
                json_to_pyobject(json!(self.0.config.clone()))
            }
            #[pyo3(name = "get_solver_base")]
            fn py_get_solver_base(&self) -> SolverBase {
                self.solver_base()
            }
        }
        impl $struct_name {
            pub fn py_construct_invalid_subgraph(
                &self,
                vertices: Option<&Bound<PyAny>>,
                edges: Option<&Bound<PyAny>>,
            ) -> PyResult<InvalidSubgraph> {
                // edges default to empty set
                let edges = if let Some(edges) = edges {
                    py_into_set(edges)?
                } else {
                    FastIterSet::new()
                };
                // vertices must be superset of the union of all edges
                let interface = self.0.interface_ptr.read_recursive();
                Ok(if let Some(vertices) = vertices {
                    let vertices = py_into_set(vertices)?;
                    InvalidSubgraph::new_complete(vertices, edges, &interface.decoding_graph)
                } else {
                    InvalidSubgraph::new(edges, &interface.decoding_graph)
                })
            }
        }
    };
}

macro_rules! inherit_solver_plugin_methods {
    ($struct_name:ident) => {
        impl $struct_name {
            pub fn get_cluster(&self, vertex_index: VertexIndex) -> Cluster {
                self.0.get_cluster(vertex_index)
            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverSerialPluginsConfig {
    #[serde(flatten)]
    flatten_primal: PrimalModuleSerialConfig,
    /// legacy config
    primal: Option<PrimalModuleSerialConfig>,
}

#[derive(Clone)]
pub struct SolverSerialPlugins {
    dual_module: DualModulePQ,
    primal_module: PrimalModuleSerial,
    interface_ptr: DualModuleInterfacePtr,
    model_graph: Arc<ModelHyperGraph>,
    pub config: SolverSerialPluginsConfig,
    syndrome_loaded: bool,
}

impl MWPSVisualizer for SolverSerialPlugins {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let mut value = self.primal_module.snapshot(abbrev);
        snapshot_combine_values(&mut value, self.dual_module.snapshot(abbrev), abbrev);
        snapshot_combine_values(&mut value, self.interface_ptr.snapshot(abbrev), abbrev);
        value
    }
}

impl SolverSerialPlugins {
    pub fn new(initializer: &Arc<SolverInitializer>, plugins: Arc<Vec<PluginEntry>>, config: serde_json::Value) -> Self {
        let model_graph = Arc::new(ModelHyperGraph::new(initializer.clone()));
        let mut primal_module = PrimalModuleSerial::new_empty(initializer); // question: why does this need initializer?
        let config: SolverSerialPluginsConfig = serde_json::from_value(config).unwrap();
        primal_module.plugins = plugins;
        primal_module.config = config.primal.as_ref().unwrap_or(&config.flatten_primal).clone();
        Self {
            dual_module: DualModulePQ::new_empty(initializer),
            primal_module,
            interface_ptr: DualModuleInterfacePtr::new(model_graph.clone()),
            model_graph,
            config,
            syndrome_loaded: false,
        }
    }

    /// APIs for step-by-step solving in Python
    pub fn load_syndrome(
        &mut self,
        syndrome_pattern: &SyndromePattern,
        visualizer: Option<&mut Visualizer>,
        skip_initial_duals: bool,
    ) {
        if self.syndrome_loaded {
            self.clear(); // automatic clear before loading new syndrome in case user forgets to call `clear`
        }
        self.syndrome_loaded = true;

        if !skip_initial_duals {
            self.interface_ptr
                .load(Arc::new(syndrome_pattern.clone()), &mut self.dual_module);
            self.primal_module.load(&self.interface_ptr, &mut self.dual_module);
        } else {
            self.interface_ptr
                .write()
                .decoding_graph
                .set_syndrome(Arc::new(syndrome_pattern.clone()));
            // also manually set the defect flag in the dual module
            for &vertex_index in syndrome_pattern.defect_vertices.iter() {
                self.dual_module.vertices[vertex_index].write().is_defect = true;
            }
        }
        if let Some(visualizer) = visualizer {
            visualizer
                .snapshot_combined(
                    "syndrome loaded".to_string(),
                    vec![&self.interface_ptr, &self.dual_module, &self.primal_module],
                )
                .unwrap();
        }
    }

    /// get the cluster information of a vertex
    pub fn get_cluster(&self, vertex_index: VertexIndex) -> Cluster {
        let mut cluster = Cluster::new();
        // visit the graph via tight edges
        let mut current_vertices = FastIterSet::new();
        current_vertices.insert(vertex_index);
        while !current_vertices.is_empty() {
            let mut next_vertices = FastIterSet::new();
            for &vertex_index in current_vertices.iter() {
                cluster.add_vertex(vertex_index);
                for &edge_index in self.model_graph.get_vertex_neighbors(vertex_index).iter() {
                    if self.dual_module.is_edge_tight(edge_index) {
                        cluster.add_edge(edge_index);
                        cluster.parity_matrix.add_tight_variable(edge_index);
                        for &next_vertex_index in self.model_graph.get_edge_neighbors(edge_index).iter() {
                            if !cluster.vertices.contains(&next_vertex_index) {
                                next_vertices.insert(next_vertex_index);
                            }
                        }
                    } else {
                        cluster.add_hair(edge_index);
                    }
                }
            }
            current_vertices = next_vertices;
        }
        // add dual variables
        for &edge_index in cluster.edges.iter() {
            for node_ptr in self.dual_module.get_edge_nodes(edge_index).iter() {
                cluster.nodes.insert(node_ptr.clone().into());
            }
        }
        // construct the parity matrix
        let interface = self.interface_ptr.read();
        for &vertex_index in cluster.vertices.iter() {
            let incident_edges = self.model_graph.get_vertex_neighbors(vertex_index);
            let parity = interface.decoding_graph.is_vertex_defect(vertex_index);
            cluster.parity_matrix.add_constraint(vertex_index, incident_edges, parity);
        }
        cluster
    }
}

impl SolverTrait for SolverSerialPlugins {
    fn clear(&mut self) {
        self.primal_module.clear();
        self.dual_module.clear();
        self.interface_ptr.clear();
        self.syndrome_loaded = false;
    }
    fn solve_visualizer(&mut self, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>) {
        if self.syndrome_loaded {
            self.clear(); // automatic clear before loading new syndrome in case user forgets to call `clear`
        }
        self.syndrome_loaded = true;

        let original_syndrome_pattern = Arc::new(syndrome_pattern);

        let syndrome_pattern = self.primal_module.weight_preprocessing(
            original_syndrome_pattern.clone(),
            &mut self.dual_module,
            &self.model_graph.initializer,
        );
        self.primal_module.solve_visualizer(
            &self.interface_ptr,
            syndrome_pattern.clone(),
            &mut self.dual_module,
            visualizer,
        );
        debug_assert!(
            {
                let subgraph = self.subgraph();
                self.model_graph
                    .matches_subgraph_syndrome(&subgraph, &original_syndrome_pattern.defect_vertices)
            },
            "the subgraph does not generate the syndrome"
        );
    }
    fn subgraph_range_visualizer(&mut self, visualizer: Option<&mut Visualizer>) -> (OutputSubgraph, WeightRange) {
        let (subgraph, weight_range) = self.primal_module.subgraph_range(&self.interface_ptr, &mut self.dual_module);
        if let Some(visualizer) = visualizer {
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&self.interface_ptr, &self.dual_module, &subgraph, &weight_range],
                )
                .unwrap();
        }
        (subgraph, weight_range)
    }
    fn sum_dual_variables(&self) -> Rational {
        self.interface_ptr.sum_dual_variables()
    }
    fn generate_profiler_report(&self) -> serde_json::Value {
        json!({
            // "dual": self.dual_module.generate_profiler_report(),
            "primal": self.primal_module.generate_profiler_report(),
        })
    }
    fn get_tuning_time(&self) -> Option<f64> {
        self.dual_module.get_total_tuning_time()
    }
    fn clear_tuning_time(&mut self) {
        self.dual_module.clear_tuning_time()
    }
    fn print_clusters(&self) {
        self.primal_module.print_clusters();
    }
    fn update_weights(&mut self, new_weights: Vec<Weight>, mix_ratio: Weight) {
        self.dual_module.update_weights(new_weights, mix_ratio);
    }
    fn get_model_graph(&self) -> Arc<ModelHyperGraph> {
        self.model_graph.clone()
    }
    fn debug_print(&self) {
        self.dual_module.debug_print();
    }
    fn solver_base(&self) -> SolverBase {
        panic!("solver_base is not implemented for SolverSerialPlugins")
    }
}

macro_rules! bind_solver_trait {
    ($struct_name:ident) => {
        impl SolverTrait for $struct_name {
            fn clear(&mut self) {
                self.0.clear()
            }
            fn solve_visualizer(&mut self, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>) {
                self.0.solve_visualizer(syndrome_pattern, visualizer)
            }
            fn subgraph_range_visualizer(&mut self, visualizer: Option<&mut Visualizer>) -> (OutputSubgraph, WeightRange) {
                self.0.subgraph_range_visualizer(visualizer)
            }
            fn sum_dual_variables(&self) -> Weight {
                self.0.sum_dual_variables()
            }
            fn generate_profiler_report(&self) -> serde_json::Value {
                self.0.generate_profiler_report()
            }
            fn get_tuning_time(&self) -> Option<f64> {
                self.0.get_tuning_time()
            }
            fn clear_tuning_time(&mut self) {
                self.0.clear_tuning_time()
            }
            fn print_clusters(&self) {
                self.0.print_clusters()
            }
            fn update_weights(&mut self, new_weights: Vec<Weight>, mix_ratio: Weight) {
                self.0.update_weights(new_weights, mix_ratio)
            }
            fn get_model_graph(&self) -> Arc<ModelHyperGraph> {
                self.0.model_graph.clone()
            }
            fn debug_print(&self) {
                self.0.debug_print()
            }
            fn solver_base(&self) -> SolverBase {
                SolverBase {
                    inner: SolverEnum::$struct_name(self.clone()),
                }
            }
        }
    };
}

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
#[derive(Clone)]
pub struct SolverSerialUnionFind(SolverSerialPlugins);

impl SolverSerialUnionFind {
    pub fn new(initializer: &Arc<SolverInitializer>, config: serde_json::Value) -> Self {
        Self(SolverSerialPlugins::new(initializer, Arc::new(vec![]), config))
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverSerialUnionFind {
    #[new]
    #[pyo3(signature = (initializer, config=None))]
    pub fn new_python(py: Python<'_>, initializer: &SolverInitializer, config: Option<PyObject>) -> Self {
        let config = config.map(|x| pyobject_to_json(x)).unwrap_or(json!({}));
        py.allow_threads(move || Self::new(&Arc::new(initializer.clone()), config))
    }
}

bind_solver_trait!(SolverSerialUnionFind);

#[cfg(feature = "python_binding")]
bind_trait_to_python!(SolverSerialUnionFind);
inherit_solver_plugin_methods!(SolverSerialUnionFind);

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
#[derive(Clone)]
pub struct SolverSerialSingleHair(SolverSerialPlugins);

impl SolverSerialSingleHair {
    pub fn new(initializer: &Arc<SolverInitializer>, config: serde_json::Value) -> Self {
        Self(SolverSerialPlugins::new(
            initializer,
            Arc::new(vec![
                PluginUnionFind::entry(), // to allow timeout using union-find as baseline
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once),
            ]),
            config,
        ))
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverSerialSingleHair {
    #[new]
    #[pyo3(signature = (initializer, config=None))]
    pub fn new_python(py: Python<'_>, initializer: &SolverInitializer, config: Option<PyObject>) -> Self {
        let config = config.map(|x| pyobject_to_json(x)).unwrap_or(json!({}));
        py.allow_threads(move || Self::new(&Arc::new(initializer.clone()), config))
    }
}

bind_solver_trait!(SolverSerialSingleHair);

#[cfg(feature = "python_binding")]
bind_trait_to_python!(SolverSerialSingleHair);
inherit_solver_plugin_methods!(SolverSerialSingleHair);

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
#[derive(Clone)]
pub struct SolverSerialJointSingleHair(SolverSerialPlugins);

impl SolverSerialJointSingleHair {
    pub fn new(initializer: &Arc<SolverInitializer>, config: serde_json::Value) -> Self {
        Self(SolverSerialPlugins::new(
            initializer,
            Arc::new(vec![
                PluginUnionFind::entry(), // to allow timeout using union-find as baseline
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once), // first make all clusters valid single hair
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Multiple {
                    max_repetition: usize::MAX,
                }),
            ]),
            config,
        ))
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverSerialJointSingleHair {
    #[new]
    #[pyo3(signature = (initializer, config=None))]
    pub fn new_python(py: Python<'_>, initializer: &SolverInitializer, config: Option<PyObject>) -> Self {
        let config = config.map(pyobject_to_json).unwrap_or(json!({}));
        py.allow_threads(move || Self::new(&Arc::new(initializer.clone()), config))
    }
}

bind_solver_trait!(SolverSerialJointSingleHair);

#[cfg(feature = "python_binding")]
bind_trait_to_python!(SolverSerialJointSingleHair);
inherit_solver_plugin_methods!(SolverSerialJointSingleHair);

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct SolverErrorPatternLogger {
    file: BufWriter<File>,
}

impl SolverErrorPatternLogger {
    pub fn new(initializer: &SolverInitializer, code: &dyn ExampleCode, mut config: serde_json::Value) -> Self {
        let mut filename = "tmp/syndrome_patterns.txt".to_string();
        let config = config.as_object_mut().expect("config must be JSON object");
        if let Some(value) = config.remove("filename") {
            filename = value.as_str().expect("filename string").to_string();
        }
        if !config.is_empty() {
            panic!("unknown config keys: {:?}", config.keys().collect::<Vec<&String>>());
        }
        let file = File::create(filename).unwrap();
        let mut file = BufWriter::new(file);
        file.write_all(b"Syndrome Pattern v1.0   <initializer> <positions> <syndrome_pattern>*\n")
            .unwrap();
        serde_json::to_writer(&mut file, &initializer).unwrap(); // large object write to file directly
        file.write_all(b"\n").unwrap();
        serde_json::to_writer(&mut file, &code.get_positions()).unwrap();
        file.write_all(b"\n").unwrap();
        Self { file }
    }
}

impl SolverTrait for SolverErrorPatternLogger {
    fn clear(&mut self) {}
    fn solve_visualizer(&mut self, syndrome_pattern: SyndromePattern, _visualizer: Option<&mut Visualizer>) {
        self.file
            .write_all(
                serde_json::to_string(&serde_json::json!(syndrome_pattern))
                    .unwrap()
                    .as_bytes(),
            )
            .unwrap();
        self.file.write_all(b"\n").unwrap();
    }
    fn subgraph_range_visualizer(&mut self, _visualizer: Option<&mut Visualizer>) -> (OutputSubgraph, WeightRange) {
        panic!("error pattern logger do not actually solve the problem, please use Verifier::None by `--verifier none`")
    }
    fn sum_dual_variables(&self) -> Rational {
        panic!("error pattern logger do not actually solve the problem")
    }
    fn generate_profiler_report(&self) -> serde_json::Value {
        json!({})
    }
    fn get_tuning_time(&self) -> Option<f64> {
        None
    }
    fn clear_tuning_time(&mut self) {}
    fn get_model_graph(&self) -> Arc<ModelHyperGraph> {
        panic!("error pattern logger do not actually solve the problem")
    }
    fn update_weights(&mut self, _new_weights: Vec<Weight>, _mix_ratio: Weight) {
        panic!("error pattern logger do not actually solve the problem")
    }
    fn solver_base(&self) -> SolverBase {
        panic!("error pattern logger does not construct solver base")
    }
}

pub enum SolverEnum {
    SolverSerialUnionFind(SolverSerialUnionFind),
    SolverSerialSingleHair(SolverSerialSingleHair),
    SolverSerialJointSingleHair(SolverSerialJointSingleHair),
    SolverErrorPatternLogger(SolverErrorPatternLogger),
}

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", name = "BPSolverBase"))]
// base of all solvers
pub struct SolverBase {
    pub inner: SolverEnum,
}

impl Clone for SolverBase {
    fn clone(&self) -> Self {
        SolverBase {
            inner: match &self.inner {
                SolverEnum::SolverSerialUnionFind(x) => SolverEnum::SolverSerialUnionFind(x.clone()),
                SolverEnum::SolverSerialSingleHair(x) => SolverEnum::SolverSerialSingleHair(x.clone()),
                SolverEnum::SolverSerialJointSingleHair(x) => SolverEnum::SolverSerialJointSingleHair(x.clone()),
                SolverEnum::SolverErrorPatternLogger(_) => panic!("cannot clone error pattern logger"),
            },
        }
    }
}

// Note: for the potential slowdown for unsendable, it should really be "unsyncable", but does not have the option
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", unsendable, name = "BP"))]
#[derive(Clone)]
pub struct SolverBPWrapper {
    pub solver: SolverBase,
    pub bp_decoder: BpDecoder,
    pub bp_application_ratio: f64,
    pub initial_log_ratios: Vec<f64>,
}

impl SolverBPWrapper {
    pub fn new(solver: SolverBase, max_iter: usize, bp_application_ratio: f64) -> Self {
        let model_graph = match &solver.inner {
            SolverEnum::SolverSerialUnionFind(x) => x.get_model_graph(),
            SolverEnum::SolverSerialSingleHair(x) => x.get_model_graph(),
            SolverEnum::SolverSerialJointSingleHair(x) => x.get_model_graph(),
            SolverEnum::SolverErrorPatternLogger(_) => panic!("cannot create BP solver from error pattern logger"),
        };
        let vertex_num = model_graph.initializer.vertex_num;
        let check_size = model_graph.initializer.weighted_edges.len();

        let mut pcm = bp::bp::BpSparse::new(vertex_num, check_size, 0);
        let mut initial_log_ratios = Vec::with_capacity(check_size);
        let mut channel_probabilities = Vec::with_capacity(check_size);

        for (col_index, HyperEdge { weight, vertices }) in model_graph.initializer.weighted_edges.iter().enumerate() {
            channel_probabilities.push(p_of_weight(weight.to_f64().unwrap()));
            for row_index in vertices.iter() {
                pcm.insert_entry(*row_index, col_index);
            }
            initial_log_ratios.push(weight.to_f64().unwrap())
        }

        let bp_decoder = BpDecoder::new_3(pcm, channel_probabilities, max_iter).unwrap();
        Self {
            solver,
            bp_decoder,
            bp_application_ratio,
            initial_log_ratios,
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverBPWrapper {
    #[new]
    pub fn py_new(solver: SolverBase, max_iter: usize, bp_application_ratio: f64) -> Self {
        let model_graph = match &solver.inner {
            SolverEnum::SolverSerialUnionFind(x) => x.get_model_graph(),
            SolverEnum::SolverSerialSingleHair(x) => x.get_model_graph(),
            SolverEnum::SolverSerialJointSingleHair(x) => x.get_model_graph(),
            SolverEnum::SolverErrorPatternLogger(_) => panic!("cannot create BP solver from error pattern logger"),
        };
        let vertex_num = model_graph.initializer.vertex_num;
        let check_size = model_graph.initializer.weighted_edges.len();

        let mut pcm = BpSparse::new(vertex_num, check_size, 0);
        let mut initial_log_ratios = Vec::with_capacity(check_size);
        let mut channel_probabilities = Vec::with_capacity(check_size);

        for (col_index, HyperEdge { weight, vertices }) in model_graph.initializer.weighted_edges.iter().enumerate() {
            channel_probabilities.push(p_of_weight(weight.to_f64().unwrap()));
            for row_index in vertices.iter() {
                pcm.insert_entry(*row_index, col_index);
            }
            initial_log_ratios.push(weight.to_f64().unwrap())
        }

        let bp_decoder = BpDecoder::new_3(pcm, channel_probabilities, max_iter).unwrap();
        Self {
            solver,
            bp_decoder,
            bp_application_ratio,
            initial_log_ratios,
        }
    }
}

#[cfg(feature = "python_binding")]
// SolverBase macros
macro_rules! SolverBase_delegate_solver_method {
    // immutable
    ($fn_name:ident(&self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        fn $fn_name(&self, $($arg: $arg_ty),*) -> $ret {
            match &self.inner {
                SolverEnum::SolverSerialUnionFind(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverSerialSingleHair(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverSerialJointSingleHair(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverErrorPatternLogger(_) => panic!(
                    concat!("cannot call ", stringify!($fn_name), " for error pattern logger")
                ),
            }
        }
    };
    // mutable
    ($fn_name:ident(&mut self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        fn $fn_name(&mut self, $($arg: $arg_ty),*) -> $ret {
            match &mut self.inner {
                SolverEnum::SolverSerialUnionFind(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverSerialSingleHair(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverSerialJointSingleHair(x) => x.0.$fn_name($($arg),*),
                SolverEnum::SolverErrorPatternLogger(_) => panic!(
                    concat!("cannot call ", stringify!($fn_name), " for error pattern logger")
                ),
            }
        }
    };
}
#[cfg(feature = "python_binding")]
macro_rules! SolverBase_delegate_solver_field {
    // immutable
    ($field_name:ident -> $ret:ty) => {
        fn $field_name(&self) -> $ret {
            match &self.inner {
                SolverEnum::SolverSerialUnionFind(x) => &x.0.$field_name,
                SolverEnum::SolverSerialSingleHair(x) => &x.0.$field_name,
                SolverEnum::SolverSerialJointSingleHair(x) => &x.0.$field_name,
                SolverEnum::SolverErrorPatternLogger(_) => panic!(concat!(
                    "cannot access ",
                    stringify!($field_name),
                    " for error pattern logger"
                )),
            }
        }
    };
}

#[cfg(feature = "python_binding")]
impl SolverBase {
    // retrieving methods
    SolverBase_delegate_solver_method!(load_syndrome(&mut self, syndrome_pattern: &SyndromePattern, visualizer: Option<&mut Visualizer>, skip_initial_duals: bool) -> ());
    SolverBase_delegate_solver_method!(snapshot(&self, abbrev: bool) -> serde_json::Value);
    SolverBase_delegate_solver_method!(get_cluster(&self, vertex_index: VertexIndex) -> Cluster);

    // retrieving fields
    SolverBase_delegate_solver_field!(interface_ptr -> &DualModuleInterfacePtr);
    SolverBase_delegate_solver_field!(dual_module -> &DualModulePQ);
    SolverBase_delegate_solver_field!(model_graph -> &Arc<ModelHyperGraph>);
    SolverBase_delegate_solver_field!(config -> &SolverSerialPluginsConfig);

    // mutable field
    fn dual_module_mut(&mut self) -> &mut DualModulePQ {
        match &mut self.inner {
            SolverEnum::SolverSerialUnionFind(x) => &mut x.0.dual_module,
            SolverEnum::SolverSerialSingleHair(x) => &mut x.0.dual_module,
            SolverEnum::SolverSerialJointSingleHair(x) => &mut x.0.dual_module,
            SolverEnum::SolverErrorPatternLogger(_) => panic!("cannot get dual module for error pattern logger"),
        }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl SolverBPWrapper {
    #[pyo3(name = "clear")]
    fn py_clear(&mut self, py: Python<'_>) {
        py.allow_threads(move || self.clear())
    }
    #[pyo3(name = "solve", signature = (syndrome_pattern, visualizer=None))] // in Python, `solve` and `solve_visualizer` is the same because it can take optional parameter
    fn py_solve(&mut self, py: Python<'_>, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>) {
        py.allow_threads(move || self.solve_visualizer(syndrome_pattern, visualizer));
    }
    #[pyo3(name = "subgraph_range", signature = (visualizer=None))] // in Python, `subgraph_range` and `subgraph_range_visualizer` is the same
    fn py_subgraph_range(&mut self, py: Python<'_>, visualizer: Option<&mut Visualizer>) -> (PySubgraph, PyWeightRange) {
        let (subgraph, range) = py.allow_threads(move || self.subgraph_range_visualizer(visualizer));
        let mut complete_subgraph = subgraph.into_iter().collect::<Vec<EdgeIndex>>();
        complete_subgraph.sort();
        (complete_subgraph.into(), range.into())
    }
    #[pyo3(name = "subgraph", signature = (visualizer=None))]
    fn py_subgraph(&mut self, py: Python<'_>, visualizer: Option<&mut Visualizer>) -> Subgraph {
        py.allow_threads(move || self.subgraph_range_visualizer(visualizer).0.into_iter().collect())
    }
    #[pyo3(name = "sum_dual_variables")]
    fn py_sum_dual_variables(&self) -> PyRational {
        self.sum_dual_variables().clone().into()
    }
    #[pyo3(name = "load_syndrome", signature = (syndrome_pattern, visualizer=None, skip_initial_duals=false))]
    pub fn py_load_syndrome(
        &mut self,
        py: Python<'_>,
        syndrome_pattern: &SyndromePattern,
        visualizer: Option<&mut Visualizer>,
        skip_initial_duals: bool,
    ) {
        py.allow_threads(move || self.solver.load_syndrome(syndrome_pattern, visualizer, skip_initial_duals))
    }
    #[pyo3(name = "get_node", signature = (node_index))]
    pub fn py_get_node(&mut self, node_index: NodeIndex) -> Option<PyDualNodePtr> {
        self.solver.interface_ptr().get_node(node_index).map(|x| x.into())
    }
    #[pyo3(name = "find_node", signature = (vertices=None, edges=None))]
    pub fn py_find_node(
        &self,
        vertices: Option<&Bound<PyAny>>,
        edges: Option<&Bound<PyAny>>,
    ) -> PyResult<Option<PyDualNodePtr>> {
        let invalid_subgraph = Arc::new(self.py_construct_invalid_subgraph(vertices, edges)?);
        Ok(self.solver.interface_ptr().find_node(&invalid_subgraph).map(|x| x.into()))
    }
    #[pyo3(name = "create_node", signature = (vertices=None, edges=None, find_existing_node=true))]
    pub fn py_create_node(
        &mut self,
        vertices: Option<&Bound<PyAny>>,
        edges: Option<&Bound<PyAny>>,
        find_existing_node: bool,
    ) -> PyResult<PyDualNodePtr> {
        let invalid_subgraph = Arc::new(self.py_construct_invalid_subgraph(vertices, edges)?);
        if find_existing_node {
            if let Some(node) = self.py_find_node(vertices, edges)? {
                return Ok(node);
            }
        }
        let interface_ptr = self.solver.interface_ptr().clone();
        Ok(match self.solver.dual_module().mode() {
            DualModuleMode::Search => interface_ptr.create_node(invalid_subgraph, self.solver.dual_module_mut()),
            DualModuleMode::Tune => interface_ptr.create_node_tune(invalid_subgraph, self.solver.dual_module_mut()),
        }
        .into())
    }
    /// create a node without providing any information about the invalid cluster itself, hence no safety checks
    #[pyo3(name = "create_node_hair_unchecked", signature = (hair, vertices=None, edges=None))]
    pub fn py_create_node_hair_unchecked(
        &mut self,
        hair: &Bound<PyAny>,
        vertices: Option<&Bound<PyAny>>,
        edges: Option<&Bound<PyAny>>,
    ) -> PyResult<PyDualNodePtr> {
        let hair = py_into_set(hair)?;
        assert!(!hair.is_empty(), "hair must not be empty");
        let vertices = if let Some(vertices) = vertices {
            py_into_set(vertices)?
        } else {
            FastIterSet::new()
        };
        let edges = if let Some(edges) = edges {
            py_into_set(edges)?
        } else {
            FastIterSet::new()
        };
        let invalid_subgraph = Arc::new(InvalidSubgraph::new_raw(vertices, edges, hair));
        let interface_ptr = self.solver.interface_ptr().clone();
        Ok(match self.solver.dual_module_mut().mode() {
            DualModuleMode::Search => interface_ptr.create_node(invalid_subgraph, self.solver.dual_module_mut()),
            DualModuleMode::Tune => interface_ptr.create_node_tune(invalid_subgraph, self.solver.dual_module_mut()),
        }
        .into())
    }
    #[pyo3(name = "grow", signature = (length))]
    fn py_grow(&mut self, py: Python<'_>, length: PyRational) {
        let length: Rational = length.into();
        py.allow_threads(move || {
            if let Some(max_valid_grow) = self.solver.dual_module_mut().compute_max_valid_grow() {
                assert!(
                    length <= max_valid_grow,
                    "growth overflow: attempting to grow {} but can only grow {} maximum",
                    length,
                    max_valid_grow
                );
            };
            self.solver.dual_module_mut().grow(length);
        });
    }
    #[pyo3(name = "snapshot", signature = (abbrev=true))]
    fn py_snapshot(&mut self, py: Python<'_>, abbrev: bool) -> PyObject {
        let value = py.allow_threads(move || self.solver.snapshot(abbrev));
        json_to_pyobject(value)
    }
    #[pyo3(name = "dual_report")]
    fn py_dual_report(&mut self) -> PyDualReport {
        self.solver.dual_module_mut().report().into()
    }
    #[pyo3(name = "get_edge_nodes")]
    fn py_get_edge_nodes(&self, edge_index: EdgeIndex) -> Vec<PyDualNodePtr> {
        self.solver
            .dual_module()
            .get_edge_nodes(edge_index)
            .into_iter()
            .map(|x| x.into())
            .collect()
    }
    #[pyo3(name = "set_grow_rate")]
    fn py_set_grow_rate(&mut self, dual_node_ptr: PyDualNodePtr, grow_rate: PyRational) {
        self.solver
            .dual_module_mut()
            .set_grow_rate(&dual_node_ptr.0, grow_rate.into())
    }
    #[pyo3(name = "stop_all")]
    fn py_stop_all(&mut self) {
        let mut node_index = 0;
        use crate::num_traits::Zero;
        while let Some(node_ptr) = self.solver.interface_ptr().get_node(node_index) {
            self.solver.dual_module_mut().set_grow_rate(&node_ptr, Rational::zero());
            node_index += 1;
        }
    }
    #[pyo3(name = "get_cluster")]
    fn py_get_cluster(&self, vertex_index: VertexIndex) -> PyCluster {
        self.solver.get_cluster(vertex_index).into()
    }
    /// a shortcut for creating a visualizer to display the current state of the solver
    #[pyo3(name = "show")]
    fn py_show(&self, positions: Vec<VisualizePosition>) {
        // NOTE: removed multiple threads for showing, since bp is `unsendable`, but should not be performance hit during actual runs
        let mut visualizer = Visualizer::new(Some(String::new()), positions, true).unwrap();
        visualizer
            .snapshot(
                "show".to_string(),
                match &self.solver.inner {
                    SolverEnum::SolverSerialUnionFind(x) => &x.0,
                    SolverEnum::SolverSerialSingleHair(x) => &x.0,
                    SolverEnum::SolverSerialJointSingleHair(x) => &x.0,
                    SolverEnum::SolverErrorPatternLogger(_) => {
                        panic!("cannot create visualizer for error pattern logger")
                    }
                },
            )
            .unwrap();
        visualizer.show_py(None, None);
    }
    #[pyo3(name = "get_initializer")]
    fn py_get_initializer(&self) -> SolverInitializer {
        self.solver.model_graph().initializer.as_ref().clone()
    }
    fn __getnewargs_ex__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("initializer", self.py_get_initializer())?;
        kwargs.set_item("config", json_to_pyobject(json!(self.solver.config().clone())))?;
        let args = PyTuple::empty(py);
        Ok((args, kwargs).into_pyobject(py)?.unbind())
    }
    #[getter]
    fn get_config(&self) -> PyObject {
        json_to_pyobject(json!(self.solver.config().clone()))
    }
    #[pyo3(name = "get_solver_base")]
    fn py_get_solver_base(&self) -> SolverBase {
        self.solver.clone()
    }
}
#[cfg(feature = "python_binding")]
impl SolverBPWrapper {
    pub fn py_construct_invalid_subgraph(
        &self,
        vertices: Option<&Bound<PyAny>>,
        edges: Option<&Bound<PyAny>>,
    ) -> PyResult<InvalidSubgraph> {
        // edges default to empty set
        let edges = if let Some(edges) = edges {
            py_into_set(edges)?
        } else {
            FastIterSet::new()
        };
        // vertices must be superset of the union of all edges
        let interface = self.solver.interface_ptr().read_recursive();
        Ok(if let Some(vertices) = vertices {
            let vertices = py_into_set(vertices)?;
            InvalidSubgraph::new_complete(vertices, edges, &interface.decoding_graph)
        } else {
            InvalidSubgraph::new(edges, &interface.decoding_graph)
        })
    }
}

macro_rules! SolverTrait_delegate_solver_call {
    ($self:ident . $method:ident ( $($arg:expr),* ) as mut) => {
        match &mut $self.solver.inner {
            SolverEnum::SolverSerialUnionFind(x) => x.$method($($arg),*),
            SolverEnum::SolverSerialSingleHair(x) => x.$method($($arg),*),
            SolverEnum::SolverSerialJointSingleHair(x) => x.$method($($arg),*),
            SolverEnum::SolverErrorPatternLogger(x) => x.$method($($arg),*),
        }
    };
    ($self:ident . $method:ident ( $($arg:expr),* )) => {
        match &$self.solver.inner {
            SolverEnum::SolverSerialUnionFind(x) => x.$method($($arg),*),
            SolverEnum::SolverSerialSingleHair(x) => x.$method($($arg),*),
            SolverEnum::SolverSerialJointSingleHair(x) => x.$method($($arg),*),
            SolverEnum::SolverErrorPatternLogger(x) => x.$method($($arg),*),
        }
    };
}

macro_rules! SolverTrait_solve_with_bp {
    ($self:ident, $solver:ident, $syndrome_pattern:ident, $visualizer:ident) => {{
        let mut syndrome_array = vec![0u8; $solver.get_model_graph().vertices.len()];
        $syndrome_pattern.defect_vertices.iter().for_each(|&x| syndrome_array[x] = 1);

        $self.bp_decoder.set_log_domain_bp(&$self.initial_log_ratios);

        // Solve the BP and update weights
        $self.bp_decoder.decode(&syndrome_array);
        let llrs: Vec<Weight> = $self
            .bp_decoder
            .log_prob_ratios
            .iter()
            .map(|v| Weight::from_float(*v).unwrap())
            .collect();

        #[cfg(feature = "f64_weight")]
        debug_assert!(
            llrs.iter().all(|x| x.is_number()),
            "llrs must be all numbers, but got: {:?}",
            llrs
        );

        $solver.update_weights(llrs, Weight::from_float($self.bp_application_ratio).unwrap());
        $solver.solve_visualizer($syndrome_pattern, $visualizer);
    }};
}

impl SolverTrait for SolverBPWrapper {
    fn clear(&mut self) {
        SolverTrait_delegate_solver_call!(self.clear() as mut)
    }
    fn solve_visualizer(&mut self, syndrome_pattern: SyndromePattern, visualizer: Option<&mut Visualizer>) {
        match &mut self.solver.inner {
            SolverEnum::SolverSerialUnionFind(x) => SolverTrait_solve_with_bp!(self, x, syndrome_pattern, visualizer),
            SolverEnum::SolverSerialSingleHair(x) => SolverTrait_solve_with_bp!(self, x, syndrome_pattern, visualizer),
            SolverEnum::SolverSerialJointSingleHair(x) => SolverTrait_solve_with_bp!(self, x, syndrome_pattern, visualizer),
            SolverEnum::SolverErrorPatternLogger(_) => panic!("error pattern logger does not solve"),
        }
    }
    fn subgraph_range_visualizer(&mut self, visualizer: Option<&mut Visualizer>) -> (OutputSubgraph, WeightRange) {
        SolverTrait_delegate_solver_call!(self.subgraph_range_visualizer(visualizer) as mut)
    }
    fn sum_dual_variables(&self) -> Rational {
        SolverTrait_delegate_solver_call!(self.sum_dual_variables())
    }
    fn generate_profiler_report(&self) -> serde_json::Value {
        SolverTrait_delegate_solver_call!(self.generate_profiler_report())
    }
    fn get_tuning_time(&self) -> Option<f64> {
        SolverTrait_delegate_solver_call!(self.get_tuning_time())
    }
    fn clear_tuning_time(&mut self) {
        SolverTrait_delegate_solver_call!(self.clear_tuning_time() as mut)
    }
    fn print_clusters(&self) {
        SolverTrait_delegate_solver_call!(self.print_clusters())
    }
    fn update_weights(&mut self, new_weights: Vec<Weight>, mix_ratio: Weight) {
        SolverTrait_delegate_solver_call!(self.update_weights(new_weights, mix_ratio) as mut)
    }
    fn get_model_graph(&self) -> Arc<ModelHyperGraph> {
        SolverTrait_delegate_solver_call!(self.get_model_graph())
    }
    fn solver_base(&self) -> SolverBase {
        self.solver.clone()
    }
}

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SolverSerialUnionFind>()?;
    m.add_class::<SolverSerialSingleHair>()?;
    m.add_class::<SolverSerialJointSingleHair>()?;
    m.add_class::<SolverErrorPatternLogger>()?;
    m.add_class::<SolverBPWrapper>()?;
    // add Solver as default class
    m.add("Solver", m.getattr("SolverSerialJointSingleHair")?)?;
    Ok(())
}

fn p_of_weight(w: f64) -> f64 {
    1.0 / (w.exp() + 1.0)
}
