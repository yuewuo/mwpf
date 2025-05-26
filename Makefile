all: test check build python wasm

fmt:
	cargo fmt --check

# A collection of lints to catch common mistakes and improve your Rust code.
clippy:
	cargo clippy -- -Dwarnings
	cargo clippy --all-targets --features=python_binding -- -D warnings

clean:
	cargo clean
# the following subfolder targets only appears when running `maturin develop --release`
	cd src/heapz && cargo clean
	cd src/highs/fuzz && cargo clean
	cd src/highs && cargo clean
	cd src/pheap && cargo clean
	cd src/slp && cargo clean


clean-env: clean fmt

test: clean-env
	cargo test
	cargo test --release
	cargo test --no-default-features --features rational_weight,embed_visualizer,qecp_integrate,progress_bar
	cargo test -r --no-default-features --features rational_weight,embed_visualizer,qecp_integrate,progress_bar

ci_rust_test:
	cargo test --release
	cargo test -r --no-default-features --features rational_weight,embed_visualizer,qecp_integrate,progress_bar

build: clean-env
	cargo build
	cargo build --release

# build test binary
	cargo test --no-run
	cargo test --no-run --release
	cargo test --no-run --features python_binding
	cargo test --no-run --features python_binding --release

check: clean-env
	cargo check
	# cargo check --lib --no-default-features --features wasm_binding,rational_weight,embed_visualizer
	cargo check --release

python: clean-env
	maturin develop
	# pytest tests/python

wasm: clean-env
	wasm-pack build --no-default-features --features wasm_binding,rational_weight,embed_visualizer

# test code coverage: see https://lib.rs/crates/cargo-llvm-cov
coverage:
	cargo llvm-cov --html
	# open target/llvm-cov/html/index.html

install-py:
	# first check the project is in a clean state
	python3 pyproject-patch.py rational dry
	python3 pyproject-patch.py incr dry
	python3 pyproject-patch.py fast dry

	RUSTFLAGS="-C target-cpu=native" pip install .

	python3 pyproject-patch.py fast apply
	RUSTFLAGS="-C target-cpu=native" pip install .
	python3 pyproject-patch.py fast revert

	python3 pyproject-patch.py rational apply
	RUSTFLAGS="-C target-cpu=native" pip install .
	python3 pyproject-patch.py rational revert

	python3 pyproject-patch.py incr apply
	RUSTFLAGS="-C target-cpu=native" pip install .
	python3 pyproject-patch.py incr revert


uninstall-py:
	pip uninstall mwpf mwpf_rational mwpf_incr mwpf_fast -y
