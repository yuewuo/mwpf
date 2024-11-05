//! HTML Export
//!
//! This module helps generate standalone HTML files for visualization.
//!

#[cfg(feature = "python_binding")]
use crate::rand::Rng;
#[cfg(feature = "python_binding")]
use crate::util::*;
use base64::prelude::*;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
use std::io::prelude::*;
use std::sync::Mutex;

lazy_static! {
    static ref HYPERION_VISUAL_JUPYTER_LOADED: Mutex<bool> = Mutex::new(false);
}

#[cfg(feature = "embed_visualizer")]
lazy_static! {
    static ref HYPERION_VISUAL_LIBRARY_BODY: &'static str = {
        let template_html = include_str!("../visualize/dist/standalone.html");
        let library_flag = "HYPERION_VISUAL_MODULE_LOADER";
        let (_, library_body, _) = HTMLExport::slice_content(template_html, library_flag);
        library_body
    };
}

#[cfg_attr(feature = "python_binding", cfg_eval)]
#[cfg_attr(feature = "python_binding", pyclass)]
pub struct HTMLExport {}

impl HTMLExport {
    fn begin(name: &str) -> String {
        format!("/* {name}_BEGIN */")
    }

    fn end(name: &str) -> String {
        format!("/* {name}_END */")
    }

    fn slice_content<'a>(content: &'a str, name: &str) -> (&'a str, &'a str, &'a str) {
        let begin = Self::begin(name);
        let begin_flag = begin.as_str();
        let end = Self::end(name);
        let end_flag = end.as_str();
        let start_index = content
            .find(begin_flag)
            .unwrap_or_else(|| panic!("begin flag {} not found in content", begin_flag));
        let end_index = content
            .find(end_flag)
            .unwrap_or_else(|| panic!("end flag {} not found in content", end_flag));
        assert!(
            start_index + begin.len() < end_index,
            "start and end flag misplaced in index.html"
        );
        (
            &content[0..start_index],
            &content[start_index + begin.len()..end_index].trim(),
            &content[end_index + end.len()..].trim(),
        )
    }

    pub fn generate_html(visualizer_data: serde_json::Value, mut override_config: serde_json::Value) -> String {
        let template_html =
            Self::get_template_html().expect("template html not available, please rebuild with `embed_visualizer` feature");
        // force full screen because we're generating standalone html
        override_config
            .as_object_mut()
            .expect("config must be an object")
            .insert("full_screen".to_string(), json!(true));
        let override_str = serde_json::to_string(&override_config).expect("override config must be serializable");
        // compress visualizer data; user can then use the webGUI to export uncompressed JSON or HTML
        let visualizer_json = serde_json::to_string(&visualizer_data).expect("data must be serializable");
        let javascript_data = HTMLExport::compress_content(visualizer_json.as_str());
        // process the frontend code
        let data_flag = "HYPERION_VISUAL_DATA";
        let (vis_data_head, _, vis_data_tail) = Self::slice_content(template_html, data_flag);
        let override_config_flag = "HYPERION_VISUAL_OVERRIDE_CONFIG";
        let (override_head, _, override_tail) = Self::slice_content(vis_data_tail, override_config_flag);
        // construct standalone html
        let new_vis_data_tail = format!(
            "{}\n{}\n{}\n{}\n{}",
            override_head,
            Self::begin(override_config_flag),
            override_str,
            Self::end(override_config_flag),
            override_tail
        );
        let new_html = format!(
            "{}\n{}\n'{}'\n{}\n{}",
            vis_data_head,
            Self::begin(data_flag),
            javascript_data,
            Self::end(data_flag),
            new_vis_data_tail
        );
        new_html
    }

    #[cfg(feature = "python_binding")]
    pub fn force_inject_library() {
        let script_body = Self::get_library_body().unwrap();
        let script_block = format!(
            r#"<div><span style="color: white; font-size: 8px; padding: 4px; background-color: rgba(36, 110, 36); border-radius: 4px;">MWPF visualization library embedded</span></div><script type="module" id='hyperion_visual_compressed_js_caller'>
/* HYPERION_VISUAL_MODULE_CODE_BEGIN */
{script_body}
/* HYPERION_VISUAL_MODULE_CODE_END */
</script>"#
        );
        Python::with_gil(|py| -> PyResult<()> {
            let display = PyModule::import_bound(py, "IPython.display")?;
            display.call_method1("display", (display.call_method1("HTML", (script_block,))?,))?;
            Ok(())
        })
        .unwrap();
    }

    #[cfg(feature = "python_binding")]
    pub fn display_jupyter_html(visualizer_data: serde_json::Value, mut override_config: serde_json::Value) {
        let template_html =
            Self::get_template_html().expect("template html not available, please rebuild with `embed_visualizer` feature");
        // if the hyperion_visual library is not loaded yet, load it
        if !*HYPERION_VISUAL_JUPYTER_LOADED.lock().unwrap() {
            *HYPERION_VISUAL_JUPYTER_LOADED.lock().unwrap() = true;
            Self::force_inject_library();
        }
        // create a div block
        let div_id: String = {
            const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let mut rng = rand::thread_rng();
            let one_char = || CHARSET[rng.gen_range(0..CHARSET.len())] as char;
            std::iter::repeat_with(one_char).take(16).collect()
        };
        let div_block = format!(r#"<div id="{div_id}" style="width: auto; height: min(max(60vh, 400px), 100vw);"></div>"#);
        Python::with_gil(|py| -> PyResult<()> {
            let display = PyModule::import_bound(py, "IPython.display")?;
            display.call_method1("display", (display.call_method1("HTML", (div_block,))?,))?;
            Ok(())
        })
        .unwrap();
        // force none full screen because we're displaying in jupyter
        override_config
            .as_object_mut()
            .expect("config must be an object")
            .insert("full_screen".to_string(), json!(false));
        let override_str = serde_json::to_string(&override_config).expect("override config must be serializable");
        // compress visualizer data; user can then use the webGUI to export uncompressed JSON or HTML
        let visualizer_json = serde_json::to_string(&visualizer_data).expect("data must be serializable");
        let javascript_data = HTMLExport::compress_content(visualizer_json.as_str());
        // load the visualizer to the div block
        let bootstrap_flag = "HYPERION_VISUAL_BOOTSTRAP_CODE";
        let (_, bootstrap_code, _) = Self::slice_content(template_html, bootstrap_flag);
        // generate the javascript code
        let js_code = format!(
            r###"{bootstrap_code}
            async function main() {{
                const visualizer_data = '{javascript_data}';
                const override_config = {override_str};
                // get the current height and width of the div block
                const div = document.getElementById("{div_id}");
                const initial_aspect_ratio = div.clientWidth / div.clientHeight;
                if (override_config.initial_aspect_ratio == undefined) {{
                    override_config.initial_aspect_ratio = initial_aspect_ratio;
                }}
                // bind the visualizer to the div block
                const app = await window.hyperion_visual.bind_to_div("#{div_id}", visualizer_data, {{ ...window.hyperion_visual.default_config(), ...override_config }});
                // observe the div block for removal
                const script_dom = document.getElementById('{div_id}');
                new MutationObserver(function(mutations) {{
                    if(!document.body.contains(script_dom)) {{
                        app.unmount()
                        this.disconnect()
                    }}
                }}).observe(script_dom.parentElement.parentElement.parentElement.parentElement.parentElement, {{ childList: true, subtree: true }});
            }}
            on_hyperion_library_ready(main)
        "###
        );
        Python::with_gil(|py| -> PyResult<()> {
            let display = PyModule::import_bound(py, "IPython.display")?;
            display.call_method1("display", (display.call_method1("Javascript", (js_code,))?,))?;
            Ok(())
        })
        .unwrap();
    }
}

#[cfg_attr(feature = "python_binding", cfg_eval)]
#[cfg_attr(feature = "python_binding", pymethods)]
impl HTMLExport {
    #[cfg_attr(feature = "python_binding", staticmethod)]
    pub fn get_template_html() -> Option<&'static str> {
        cfg_if::cfg_if! {
            if #[cfg(feature="embed_visualizer")] {
                Some(include_str!("../visualize/dist/standalone.html"))
            } else {
                None
            }
        }
    }

    #[cfg_attr(feature = "python_binding", staticmethod)]
    pub fn get_library_body() -> Option<&'static str> {
        cfg_if::cfg_if! {
            if #[cfg(feature="embed_visualizer")] {
                Some(*HYPERION_VISUAL_LIBRARY_BODY)
            } else {
                None
            }
        }
    }

    #[cfg_attr(feature = "python_binding", staticmethod)]
    pub fn compress_content(data: &str) -> String {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        BASE64_STANDARD.encode(compressed).to_string()
    }

    #[cfg_attr(feature = "python_binding", staticmethod)]
    pub fn decompress_content(base64_str: &str) -> String {
        let compressed = BASE64_STANDARD.decode(base64_str.as_bytes()).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut uncompressed = String::new();
        decoder.read_to_string(&mut uncompressed).unwrap();
        uncompressed
    }

    #[cfg(feature = "python_binding")]
    #[staticmethod]
    #[pyo3(name = "generate_html", signature = (visualizer_data, override_config = None))]
    pub fn generate_html_py(visualizer_data: PyObject, override_config: Option<PyObject>) -> std::io::Result<String> {
        let visualizer_data = pyobject_to_json(visualizer_data);
        let override_config = if let Some(override_config) = override_config {
            pyobject_to_json(override_config)
        } else {
            json!({})
        };
        Ok(Self::generate_html(visualizer_data, override_config))
    }

    #[cfg(feature = "python_binding")]
    #[staticmethod]
    #[pyo3(name = "display_jupyter_html", signature = (visualizer_data, override_config = None))]
    pub fn display_jupyter_html_py(visualizer_data: PyObject, override_config: Option<PyObject>) -> std::io::Result<()> {
        let visualizer_data = pyobject_to_json(visualizer_data);
        let override_config = if let Some(override_config) = override_config {
            pyobject_to_json(override_config)
        } else {
            json!({})
        };
        cfg_if::cfg_if! {
            if #[cfg(feature="embed_visualizer")] {
                Self::display_jupyter_html(visualizer_data, override_config);
                Ok(())
            } else {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "feature `embed_visualizer` is not enabled"))
            }
        }
    }
}

#[cfg(feature = "python_binding")]
#[pyfunction]
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HTMLExport>()?;
    Ok(())
}

#[cfg(all(test, feature = "embed_visualizer"))]
mod tests {
    use super::*;

    #[test]
    fn html_export_compress_js() {
        // cargo test html_export_compress_js -- --nocapture
        let data = "hello world".to_string();
        let compressed = HTMLExport::compress_content(data.as_str());
        println!("compressed: {compressed}");
        let decompressed = HTMLExport::decompress_content(compressed.as_str());
        println!("decompressed: {decompressed}");
        assert_eq!(data, decompressed);
    }
}
