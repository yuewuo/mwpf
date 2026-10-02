# Bundled SLP solver

This modified copy of slp 0.2.0 is built as `mwpf::slp`, not as a separate crate.
Original package metadata and Prateek Kumar's MIT notice are retained in
`Cargo.toml.orig` and `LICENSE`. The grammar is packaged with mwpf.

Run the rational solver tests from the repository root:

```sh
cargo test --no-default-features --features rational_weight --lib relaxer_optimizer::
cargo test --doc --no-default-features --features rational_weight
```