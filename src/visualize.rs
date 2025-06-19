//! Visualizer
//!
//! This module helps visualize the progress of a mwpf module
//!

use crate::html_export::*;
use crate::serde::{Deserialize, Serialize};
use crate::serde_json;
#[cfg(feature = "python_binding")]
use crate::util::*;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
#[cfg(feature = "python_binding")]
use pyo3::types::PyTuple;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use tempfile::SpooledTempFile;

pub trait MWPSVisualizer {
    /// take a snapshot, set `abbrev` to true to save space
    fn snapshot(&self, abbrev: bool) -> serde_json::Value;
}

#[macro_export]
macro_rules! bind_trait_mwpf_visualizer {
    ($struct_name:ident) => {
        #[cfg(feature = "python_binding")]
        #[pymethods]
        impl $struct_name {
            #[pyo3(name = "snapshot")]
            #[args(abbrev = "true")]
            fn trait_snapshot(&self, abbrev: bool) -> PyObject {
                json_to_pyobject(self.snapshot(abbrev))
            }
        }
    };
}
#[allow(unused_imports)]
pub use bind_trait_mwpf_visualizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct VisualizePosition {
    /// vertical axis, -i is up, +i is down (left-up corner is smallest i,j)
    #[serde(default = "Default::default")]
    pub i: f64,
    /// horizontal axis, -j is left, +j is right (left-up corner is smallest i,j)
    #[serde(default = "Default::default")]
    pub j: f64,
    /// time axis, top and bottom (orthogonal to the initial view, which looks at -t direction)
    #[serde(default = "Default::default")]
    pub t: f64,
}

impl VisualizePosition {
    /// create a visualization position
    pub fn new(i: f64, j: f64, t: f64) -> Self {
        Self { i, j, t }
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl VisualizePosition {
    #[new]
    fn py_new(i: f64, j: f64, t: f64) -> Self {
        Self::new(i, j, t)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

trait VisualizerFileTrait: std::io::Write + std::io::Read + std::io::Seek + std::fmt::Debug + Send {
    fn set_len(&mut self, len: u64) -> std::io::Result<()>;
    fn sync_all(&mut self) -> std::io::Result<()>;
}

impl VisualizerFileTrait for File {
    fn set_len(&mut self, len: u64) -> std::io::Result<()> {
        File::set_len(self, len)
    }
    fn sync_all(&mut self) -> std::io::Result<()> {
        File::sync_all(self)
    }
}

impl VisualizerFileTrait for SpooledTempFile {
    fn set_len(&mut self, len: u64) -> std::io::Result<()> {
        self.set_len(len)
    }
    fn sync_all(&mut self) -> std::io::Result<()> {
        // doesn't matter whether it's written to file, because it's temporary anyway
        Ok(())
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct Visualizer {
    /// original filepath
    pub filepath: Option<String>,
    /// save to file if applicable
    file: Option<Box<dyn VisualizerFileTrait + Sync>>,
    /// if waiting for the first snapshot
    empty_snapshot: bool,
    /// names of the snapshots
    pub snapshots: Vec<String>,
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl Visualizer {
    #[getter]
    fn filepath(&self) -> Option<String> {
        self.filepath.clone()
    }
    #[getter]
    fn snapshots(&self) -> Vec<String> {
        self.snapshots.clone()
    }
}

pub fn snapshot_fix_missing_fields(value: &mut serde_json::Value, abbrev: bool) {
    let value = value.as_object_mut().expect("snapshot must be an object");
    // fix vertices missing fields
    let vertices = value
        .get_mut("vertices")
        .expect("missing unrecoverable field")
        .as_array_mut()
        .expect("vertices must be an array");
    for vertex in vertices {
        if vertex.is_null() {
            continue;
        } // vertex not present, probably currently don't care
        let vertex = vertex.as_object_mut().expect("each vertex must be an object");
        let key_is_defect = if abbrev { "s" } else { "is_defect" };
        // recover
        if !vertex.contains_key(key_is_defect) {
            vertex.insert(key_is_defect.to_string(), json!(0)); // by default no syndrome
        }
    }
    // fix edges missing fields
    let edges = value
        .get_mut("edges")
        .expect("missing unrecoverable field")
        .as_array_mut()
        .expect("edges must be an array");
    for edge in edges {
        if edge.is_null() {
            continue;
        } // edge not present, probably currently don't care
        let edge = edge.as_object_mut().expect("each edge must be an object");
        let key_weight = if abbrev { "w" } else { "weight" };
        let key_vertices = if abbrev { "v" } else { "vertices" };
        let _key_growth = if abbrev { "g" } else { "growth" };
        // recover
        assert!(edge.contains_key(key_weight), "missing unrecoverable field");
        assert!(edge.contains_key(key_vertices), "missing unrecoverable field");
    }
}

pub type ObjectMap = serde_json::Map<String, serde_json::Value>;
pub fn snapshot_combine_object_known_key(obj: &mut ObjectMap, obj_2: &mut ObjectMap, key: &str) {
    match (obj.contains_key(key), obj_2.contains_key(key)) {
        (_, false) => {} // do nothing
        (false, true) => {
            obj.insert(key.to_string(), obj_2.remove(key).unwrap());
        }
        (true, true) => {
            // println!("[snapshot_combine_object_known_key] {}: {:?} == {:?}", key, obj[key], obj_2[key]);
            assert_eq!(
                obj[key], obj_2[key],
                "cannot combine different values: please make sure values don't conflict"
            );
            obj_2.remove(key).unwrap();
        }
    }
}

pub fn snapshot_copy_remaining_fields(obj: &mut ObjectMap, obj_2: &mut ObjectMap) {
    let mut keys = Vec::<String>::new();
    for key in obj_2.keys() {
        keys.push(key.clone());
    }
    for key in keys.iter() {
        match obj.contains_key(key) {
            false => {
                obj.insert(key.to_string(), obj_2.remove(key).unwrap());
            }
            true => {
                // println!("[snapshot_copy_remaining_fields] {}: {:?} == {:?}", key, obj[key], obj_2[key]);
                // println!("obj: {obj:?}");
                // println!("obj_2: {obj_2:?}");
                assert_eq!(
                    obj[key], obj_2[key],
                    "cannot combine unknown fields: don't know what to do, please modify `snapshot_combine_values` function"
                );
                obj_2.remove(key).unwrap();
            }
        }
    }
}

pub fn snapshot_combine_values(value: &mut serde_json::Value, mut value_2: serde_json::Value, abbrev: bool) {
    let value = value.as_object_mut().expect("snapshot must be an object");
    let value_2 = value_2.as_object_mut().expect("snapshot must be an object");
    let hint_no_vertices_check =
        value.contains_key("hint_no_vertices_check") || value_2.contains_key("hint_no_vertices_check");
    match (value.contains_key("vertices"), value_2.contains_key("vertices")) {
        (_, false) => {} // do nothing
        (false, true) => {
            value.insert("vertices".to_string(), value_2.remove("vertices").unwrap());
        }
        (true, true) => {
            // combine
            let vertices = value
                .get_mut("vertices")
                .unwrap()
                .as_array_mut()
                .expect("vertices must be an array");
            let vertices_2 = value_2
                .get_mut("vertices")
                .unwrap()
                .as_array_mut()
                .expect("vertices must be an array");
            if !hint_no_vertices_check {
                assert!(vertices.len() == vertices_2.len(), "vertices must be compatible");
            }
            let vertex_length = std::cmp::max(vertices.len(), vertices_2.len());
            for vertex_index in 0..vertex_length {
                let vertex = if vertex_index < vertices.len() {
                    vertices.get_mut(vertex_index).unwrap()
                } else {
                    vertices.push(json!(null));
                    vertices.last_mut().unwrap()
                };
                let vertex_2 = if vertex_index < vertices_2.len() {
                    vertices_2.get_mut(vertex_index).unwrap()
                } else {
                    vertices_2.push(json!(null));
                    vertices_2.last_mut().unwrap()
                };
                if vertex_2.is_null() {
                    continue;
                }
                if vertex.is_null() {
                    *vertex = vertex_2.clone();
                    continue;
                }
                let vertex = vertex.as_object_mut().expect("each vertex must be an object");
                let vertex_2 = vertex_2.as_object_mut().expect("each vertex must be an object");
                // list known keys
                let key_is_virtual = if abbrev { "v" } else { "is_virtual" };
                let key_is_defect = if abbrev { "s" } else { "is_defect" };
                let known_keys = [key_is_virtual, key_is_defect];
                for key in known_keys {
                    snapshot_combine_object_known_key(vertex, vertex_2, key);
                }
                snapshot_copy_remaining_fields(vertex, vertex_2);
                assert_eq!(vertex_2.len(), 0, "there should be nothing left");
            }
            value_2.remove("vertices").unwrap();
        }
    }
    match (value.contains_key("edges"), value_2.contains_key("edges")) {
        (_, false) => {} // do nothing
        (false, true) => {
            value.insert("edges".to_string(), value_2.remove("edges").unwrap());
        }
        (true, true) => {
            // combine
            let edges = value
                .get_mut("edges")
                .unwrap()
                .as_array_mut()
                .expect("edges must be an array");
            let edges_2 = value_2
                .get_mut("edges")
                .unwrap()
                .as_array_mut()
                .expect("edges must be an array");
            assert!(edges.len() == edges_2.len(), "edges must be compatible");
            for (edge_idx, edge) in edges.iter_mut().enumerate() {
                let edge_2 = &mut edges_2[edge_idx];
                if edge_2.is_null() {
                    continue;
                }
                if edge.is_null() {
                    *edge = edge_2.clone();
                    continue;
                }
                let edge = edge.as_object_mut().expect("each edge must be an object");
                let edge_2 = edge_2.as_object_mut().expect("each edge must be an object");
                // list known keys
                let key_weight = if abbrev { "w" } else { "weight" };
                let key_left = if abbrev { "l" } else { "left" };
                let key_right = if abbrev { "r" } else { "right" };
                let key_growth = if abbrev { "g" } else { "growth" };
                let known_keys = [key_weight, key_left, key_right, key_growth];
                for key in known_keys {
                    snapshot_combine_object_known_key(edge, edge_2, key);
                }
                snapshot_copy_remaining_fields(edge, edge_2);
                assert_eq!(edge_2.len(), 0, "there should be nothing left");
            }
            value_2.remove("edges").unwrap();
        }
    }
    snapshot_copy_remaining_fields(value, value_2);
}

#[cfg_attr(feature = "python_binding", pyfunction)]
pub fn center_positions(mut positions: Vec<VisualizePosition>) -> Vec<VisualizePosition> {
    if !positions.is_empty() {
        let mut max_i = positions[0].i;
        let mut min_i = positions[0].i;
        let mut max_j = positions[0].j;
        let mut min_j = positions[0].j;
        let mut max_t = positions[0].t;
        let mut min_t = positions[0].t;
        for position in positions.iter_mut() {
            if position.i > max_i {
                max_i = position.i;
            }
            if position.j > max_j {
                max_j = position.j;
            }
            if position.t > max_t {
                max_t = position.t;
            }
            if position.i < min_i {
                min_i = position.i;
            }
            if position.j < min_j {
                min_j = position.j;
            }
            if position.t < min_t {
                min_t = position.t;
            }
        }
        let (ci, cj, ct) = ((max_i + min_i) / 2., (max_j + min_j) / 2., (max_t + min_t) / 2.);
        for position in positions.iter_mut() {
            position.i -= ci;
            position.j -= cj;
            position.t -= ct;
        }
    }
    positions
}

impl Visualizer {
    /// create a new visualizer with target filename and node layout
    pub fn new(filepath: Option<String>, mut positions: Vec<VisualizePosition>, center: bool) -> std::io::Result<Self> {
        if center {
            positions = center_positions(positions);
        }
        let mut file: Option<Box<dyn VisualizerFileTrait + Sync>> = match filepath {
            Some(ref filepath) => Some(if filepath.is_empty() {
                // 256MB max memory (uncompressed JSON can be very large, no need to write to file)
                Box::new(SpooledTempFile::new(256 * 1024 * 1024))
            } else {
                Box::new(
                    // manually enable read, see
                    // https://doc.rust-lang.org/std/fs/struct.File.html#method.create
                    OpenOptions::new()
                        .write(true)
                        .read(true)
                        .create(true)
                        .truncate(true)
                        .open(filepath)?,
                )
            }),
            None => None,
        };
        if let Some(file) = file.as_mut() {
            file.set_len(0)?; // truncate the file
            file.seek(SeekFrom::Start(0))?; // move the cursor to the front
            file.write_all(format!("{{\"format\":\"mwpf\",\"version\":\"{}\"", env!("CARGO_PKG_VERSION")).as_bytes())?;
            file.write_all(b",\"positions\":")?;
            file.write_all(json!(positions).to_string().as_bytes())?;
            file.write_all(b",\"snapshots\":[]}")?;
            file.sync_all()?;
        }
        Ok(Self {
            filepath,
            file,
            empty_snapshot: true,
            snapshots: vec![],
        })
    }

    pub fn incremental_save(&mut self, name: String, value: serde_json::Value) -> std::io::Result<()> {
        if let Some(file) = self.file.as_mut() {
            self.snapshots.push(name.clone());
            file.seek(SeekFrom::End(-2))?; // move the cursor before the ending ]}
            if !self.empty_snapshot {
                file.write_all(b",")?;
            }
            self.empty_snapshot = false;
            file.write_all(json!((name, value)).to_string().as_bytes())?;
            file.write_all(b"]}")?;
            file.sync_all()?;
        }
        Ok(())
    }

    /// append another snapshot of the mwpf modules, and also update the file in case
    pub fn snapshot_combined(&mut self, name: String, mwpf_algorithms: Vec<&dyn MWPSVisualizer>) -> std::io::Result<()> {
        let abbrev = true;
        let mut value = json!({});
        for mwpf_algorithm in mwpf_algorithms.iter() {
            let value_2 = mwpf_algorithm.snapshot(abbrev);
            snapshot_combine_values(&mut value, value_2, abbrev);
        }
        snapshot_fix_missing_fields(&mut value, abbrev);
        self.incremental_save(name, value)?;
        Ok(())
    }

    /// append another snapshot of the mwpf modules, and also update the file in case
    pub fn snapshot(&mut self, name: String, mwpf_algorithm: &impl MWPSVisualizer) -> std::io::Result<()> {
        let abbrev = true;
        let mut value = mwpf_algorithm.snapshot(abbrev);
        snapshot_fix_missing_fields(&mut value, abbrev);
        self.incremental_save(name, value)?;
        Ok(())
    }

    pub fn snapshot_combined_value(&mut self, name: String, values: Vec<serde_json::Value>) -> std::io::Result<()> {
        let abbrev = true;
        let mut value = json!({});
        for value_2 in values.into_iter() {
            snapshot_combine_values(&mut value, value_2, abbrev);
        }
        snapshot_fix_missing_fields(&mut value, abbrev);
        self.incremental_save(name, value)?;
        Ok(())
    }

    pub fn snapshot_value(&mut self, name: String, mut value: serde_json::Value) -> std::io::Result<()> {
        let abbrev = true;
        snapshot_fix_missing_fields(&mut value, abbrev);
        self.incremental_save(name, value)?;
        Ok(())
    }

    pub fn get_visualizer_data(&mut self) -> serde_json::Value {
        // read JSON data from the file
        let file = self.file.as_mut().expect("visualizer file is not opened, please provide filename (could be empty string for temporary file) when constructing the visualizer");
        file.seek(SeekFrom::Start(0))
            .expect("cannot seek to the beginning of the file");
        serde_json::from_reader(file).expect("cannot read JSON from visualizer file")
    }

    pub fn generate_html(&mut self, override_config: serde_json::Value) -> String {
        HTMLExport::generate_html(self.get_visualizer_data(), override_config)
    }

    pub fn save_html(&mut self, path: &str) {
        let html = self.generate_html(json!({}));
        let mut file = File::create(path).expect("cannot create HTML file");
        file.write_all(html.as_bytes()).expect("cannot write to HTML file");
    }

    pub fn html_along_json_path(&self) -> String {
        let path = self
            .filepath
            .clone()
            .expect("unknown filepath, please provide a proper json file path when constructing the Visualizer");
        assert!(path.ends_with(".json"));
        format!("{}.html", &path.as_str()[..path.len() - 5])
    }

    pub fn save_html_along_json(&mut self) {
        let html_path = self.html_along_json_path();
        self.save_html(&html_path);
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl Visualizer {
    #[new]
    #[pyo3(signature = (*, filepath="".to_string(), positions=None, center=true))]
    fn py_new(filepath: Option<String>, positions: Option<Vec<VisualizePosition>>, center: bool) -> std::io::Result<Self> {
        Self::new(
            filepath,
            positions.expect("vertex positions must be provided, e.g. `positions=[...]`"),
            center,
        )
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    #[pyo3(name = "snapshot", signature=(name, object, *py_args))]
    pub fn snapshot_py(&mut self, name: String, object: &Bound<PyAny>, py_args: &Bound<PyTuple>) -> PyResult<()> {
        let mut values = Vec::<serde_json::Value>::with_capacity(py_args.len() + 1);
        values.push(pyobject_to_json(object.call_method0("snapshot")?.extract::<PyObject>()?));
        for i in 0..py_args.len() {
            let object_arg = py_args.get_item(i)?;
            values.push(pyobject_to_json(object_arg.call_method0("snapshot")?.extract::<PyObject>()?));
        }
        self.snapshot_combined_value(name, values)?;
        Ok(())
    }

    #[pyo3(name = "snapshot_value")]
    pub fn snapshot_value_py(&mut self, name: String, value_py: PyObject) -> std::io::Result<()> {
        let value = pyobject_to_json(value_py);
        self.snapshot_value(name, value)
    }

    #[pyo3(name = "show", signature = (override_config = None, *, snapshot_index=None))]
    pub fn show_py(&mut self, override_config: Option<PyObject>, snapshot_index: Option<usize>) {
        let mut override_config = if let Some(override_config) = override_config {
            pyobject_to_json(override_config)
        } else {
            json!({})
        };
        if let Some(snapshot_index) = snapshot_index {
            override_config
                .as_object_mut()
                .unwrap()
                .insert("snapshot_index".to_string(), json!(snapshot_index));
        }
        HTMLExport::display_jupyter_html(self.get_visualizer_data(), override_config);
    }

    #[pyo3(name = "get_visualizer_data")]
    pub fn py_get_visualizer_data(&mut self) -> PyObject {
        json_to_pyobject(self.get_visualizer_data())
    }

    #[staticmethod]
    #[pyo3(name = "embed", signature = (force=true))]
    pub fn embed_py(force: bool) {
        if force || !HTMLExport::library_injected() {
            HTMLExport::force_inject_library();
        }
    }

    #[pyo3(name = "generate_html", signature = (override_config = None))]
    pub fn generate_html_py(&mut self, override_config: Option<PyObject>) -> String {
        let override_config = if let Some(override_config) = override_config {
            pyobject_to_json(override_config)
        } else {
            json!({})
        };
        self.generate_html(override_config)
    }

    #[pyo3(name = "save_html", signature = (path, override_config = None))]
    pub fn save_html_py(&mut self, path: String, override_config: Option<PyObject>) {
        let html = self.generate_html_py(override_config);
        let mut file = File::create(path).expect("cannot create HTML file");
        file.write_all(html.as_bytes()).expect("cannot write to HTML file");
    }
}

const DEFAULT_VISUALIZE_DATA_FOLDER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/visualize/data/");

// only used locally, because this is compile time directory
pub fn visualize_data_folder() -> String {
    DEFAULT_VISUALIZE_DATA_FOLDER.to_string()
}

pub fn static_visualize_data_filename() -> String {
    "visualizer.json".to_string()
}

pub fn static_visualize_html_filename() -> String {
    "visualizer.html".to_string()
}

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VisualizePosition>()?;
    m.add_class::<Visualizer>()?;
    m.add_function(wrap_pyfunction!(center_positions, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {}
