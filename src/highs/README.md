# highs

This modified copy is built as `mwpf::highs`, not as a separate crate. Original
package metadata and the MIT notice are retained in `Cargo.toml.orig` and `LICENSE`.
Run its tests from the repository root with `cargo test --lib highs::`.

[![highs docs badge](https://docs.rs/highs/badge.svg)](https://docs.rs/highs)

Safe rust bindings to the Highs MILP Solver. Best used from the [**good_lp**](https://crates.io/crates/good_lp) linear
programming modeler.

## Usage examples

#### Building a problem variable by variable

```rust
use mwpf::highs::{ColProblem, Sense};

fn main() {
    let mut pb = ColProblem::new();
    // We cannot use more then 5 units of sugar in total.
    let sugar = pb.add_row(..=5);
    // We cannot use more then 3 units of milk in total.
    let milk = pb.add_row(..=3);
    // We have a first cake that we can sell for 2€. Baking it requires 1 unit of milk and 2 of sugar.
    pb.add_integer_column(2., 0.., &[(sugar, 2.), (milk, 1.)]);
    // We have a second cake that we can sell for 8€. Baking it requires 2 units of milk and 3 of sugar.
    pb.add_integer_column(8., 0.., &[(sugar, 3.), (milk, 2.)]);
    // Find the maximal possible profit
    let solution = pb.optimise(Sense::Maximise).solve().get_solution();
    // The solution is to bake one cake of each sort
    assert_eq!(solution.columns(), vec![1., 1.]);
}
```

#### Building a problem constraint by constraint

```rust
use mwpf::highs::*;

fn main() {
    let mut pb = RowProblem::new();
    // Optimize 3x - 2y with x<=6 and y>=5
    let x = pb.add_column(3., ..6);
    let y = pb.add_column(-2., 5..);
    pb.add_row(2.., &[(x, 3.), (y, 8.)]); // 2 <= x*3 + y*8
}
```
