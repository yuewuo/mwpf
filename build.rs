use chrono::Local;

fn main() {
    // fix highs build error on MacOS
    println!("cargo:rustc-link-search=all=/opt/homebrew/opt/libomp/lib");

    // export timestamp of compile time
    let now = Local::now();
    let formatted_time = now.format("%Y_%m_%d_%H_%M_%S").to_string();
    println!("cargo:rustc-env=MWPF_BUILD_RS_TIMESTAMP={formatted_time}");

    println!("cargo:rerun-if-env-changed=SKIP_FRONTEND_BUILD");

    if cfg!(feature = "embed_visualizer") {
        let template = std::path::Path::new("visualize/dist/standalone.html");
        println!("cargo:rerun-if-changed={}", template.display());

        // Published packages contain this asset and never need npm.
        if !template.is_file() {
            assert!(
                std::env::var_os("SKIP_FRONTEND_BUILD").is_none(),
                "visualizer asset is missing; run `make frontend` before enabling embed_visualizer"
            );
            assert!(std::process::Command::new("npm")
                .current_dir("./visualize")
                .arg("ci")
                .arg("--include=dev")
                .status()
                .expect("npm install failed")
                .success());

            assert!(std::process::Command::new("npm")
                .current_dir("./visualize")
                .arg("run")
                .arg("build")
                .status()
                .expect("npm build failed")
                .success());
            assert!(template.is_file(), "frontend build did not produce {}", template.display());
        }
    }
}
