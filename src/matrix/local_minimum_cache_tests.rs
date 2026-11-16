use super::*;
use crate::matrix::{BasicMatrix, Echelon, Tail};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
struct Decision {
    column: usize,
    delta_bits: u64,
    accepted: bool,
    flip_edges: Vec<EdgeIndex>,
}

thread_local! {
    static TRACE: RefCell<Option<Vec<Decision>>> = const { RefCell::new(None) };
}

pub(super) fn record(column: usize, delta: &Rational, flips: &[EdgeIndex]) {
    TRACE.with(|trace| {
        if let Some(trace) = trace.borrow_mut().as_mut() {
            trace.push(Decision {
                column,
                delta_bits: delta.numer().to_bits(),
                accepted: *delta < Rational::zero(),
                flip_edges: flips.to_vec(),
            });
        }
    });
}

fn traced<T>(run: impl FnOnce() -> T) -> (T, Vec<Decision>) {
    TRACE.with(|trace| *trace.borrow_mut() = Some(vec![]));
    let result = run();
    let trace = TRACE.with(|trace| trace.borrow_mut().take().unwrap());
    (result, trace)
}

// The pre-cache implementation, retaining its arithmetic and restart order.
fn reference<M: MatrixEchelon, F: Fn(EdgeIndex) -> Weight>(matrix: &mut M, weight_of: F) -> Option<Subgraph> {
    matrix.get_echelon_info();
    let info = matrix.get_echelon_info_immutable();
    if !info.satisfiable {
        return None;
    }
    let mut solution = FastIterSet::new();
    for (row, row_info) in info.rows.iter().enumerate() {
        debug_assert!(row_info.has_leading());
        if matrix.get_rhs(row) {
            let column = row_info.column;
            let edge_index = matrix.column_to_edge_index(column);
            solution.insert(edge_index);
        }
    }
    let mut independent_columns = vec![];
    for (column, column_info) in info.columns.iter().enumerate() {
        if !column_info.is_dependent() {
            independent_columns.push(column);
        }
    }
    let mut total_weight = Rational::zero();
    for &edge_index in solution.iter() {
        total_weight += weight_of(edge_index);
    }
    let mut pending_flip_edge_indices = vec![];
    let mut is_local_minimum = false;
    while !is_local_minimum {
        is_local_minimum = true;
        for &column in independent_columns.iter() {
            pending_flip_edge_indices.clear();
            let var_index = matrix.column_to_var_index(column);
            let edge_index = matrix.var_to_edge_index(var_index);
            let mut primal_delta = (weight_of(edge_index))
                * if solution.contains(&edge_index) {
                    -Rational::one()
                } else {
                    Rational::one()
                };
            pending_flip_edge_indices.push(edge_index);
            for row in 0..info.rows.len() {
                if matrix.get_lhs(row, var_index) {
                    debug_assert!(info.rows[row].has_leading());
                    let flip_column = info.rows[row].column;
                    debug_assert!(flip_column < column);
                    let flip_edge_index = matrix.column_to_edge_index(flip_column);
                    primal_delta += (weight_of(flip_edge_index))
                        * if solution.contains(&flip_edge_index) {
                            -Rational::one()
                        } else {
                            Rational::one()
                        };
                    pending_flip_edge_indices.push(flip_edge_index);
                }
            }
            record(column, &primal_delta, &pending_flip_edge_indices);
            if primal_delta < Rational::zero() {
                total_weight = total_weight + primal_delta;
                for &edge_index in pending_flip_edge_indices.iter() {
                    if solution.contains(&edge_index) {
                        solution.remove(&edge_index);
                    } else {
                        solution.insert(edge_index);
                    }
                }
                is_local_minimum = false;
                break;
            }
        }
    }
    Some(solution.into_iter().collect())
}

fn compare<M: MatrixEchelon + Clone>(matrix: M, weights: &BTreeMap<EdgeIndex, f64>) -> Vec<Decision> {
    let mut candidate = matrix.clone();
    let mut original = matrix;
    let candidate_calls = RefCell::new(vec![]);
    let original_calls = RefCell::new(vec![]);
    let mut actual = traced(|| {
        candidate.get_solution_local_minimum(|edge| {
            let weight = weights[&edge];
            candidate_calls.borrow_mut().push((edge, weight.to_bits()));
            Rational::new(weight)
        })
    });
    let mut expected = traced(|| {
        reference(&mut original, |edge| {
            let weight = weights[&edge];
            original_calls.borrow_mut().push((edge, weight.to_bits()));
            Rational::new(weight)
        })
    });
    let mut candidate_calls = candidate_calls.into_inner();
    let mut original_calls = original_calls.into_inner();
    if cfg!(feature = "fast_ds") {
        // Only the initial weight sum and returned correction iterate a randomized set.
        actual.0 = sorted_vec_option(actual.0);
        expected.0 = sorted_vec_option(expected.0);
        let initial_weight_count = candidate.get_solution().map_or(0, |solution| solution.len());
        candidate_calls[..initial_weight_count].sort_unstable();
        original_calls[..initial_weight_count].sort_unstable();
    }
    assert_eq!(actual.0, expected.0, "correction changed");
    assert_eq!(actual.1, expected.1, "a trial, delta bit pattern, flip, or restart changed");
    assert_eq!(candidate_calls, original_calls, "weight call order changed");
    actual.1
}

#[test]
fn exact_decisions_preserve_ties_tolerance_and_nonassociative_sums() {
    let mut base = BasicMatrix::new();
    for edge in [100, 1, 42] {
        base.add_variable(edge);
    }
    base.add_constraint(0, &[100, 42], true);
    base.add_constraint(1, &[1, 42], true);
    for delta in [0.0, -5e-11, 5e-11, -2e-10, 2e-10] {
        let weights = BTreeMap::from([(100, 1.0), (1, 1.0), (42, 2.0 + delta)]);
        compare(Echelon::from_base(base.clone()), &weights);
    }
    let trace = compare(Echelon::from_base(base), &BTreeMap::from([(100, 1e16), (1, 1.0), (42, 1e16)]));
    assert_eq!(trace[0].delta_bits, (-1.0f64).to_bits());
    assert!(trace[0].accepted);
}

#[test]
fn exact_decisions_match_random_basic_and_reordered_tail_matrices() {
    let mut state = 0x32ab_917c_0504_a12du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for width in [0, 1, 7, 63, 64, 65, 127, 130] {
        for iteration in 0..12 {
            let ids: Vec<_> = (0..width).map(|i| 1000 + (width - 1 - i) * 13).collect();
            let hidden: Vec<_> = (0..width).map(|_| next() & 1 != 0).collect();
            let weights: BTreeMap<_, _> = ids
                .iter()
                .enumerate()
                .map(|(i, &edge)| (edge, [0.0, 0.25, 1.0, 1.0 + 5e-11, 2.0][i % 5]))
                .collect();
            let mut base = BasicMatrix::new();
            for &edge in &ids {
                base.add_variable(edge);
            }
            for row in 0..(width / 2 + 3) {
                let columns: Vec<_> = (0..width).filter(|_| next() & 7 == 0).collect();
                let parity = columns.iter().fold(false, |value, &i| value ^ hidden[i]);
                let incident: Vec<_> = columns.iter().map(|&i| ids[i]).collect();
                base.add_constraint(row, &incident, parity);
            }
            if iteration % 4 == 0 {
                base.add_constraint(width + 20, &[], true);
            }
            compare(Echelon::from_base(base.clone()), &weights);
            let mut tail = Tail::from_base(base);
            tail.set_tail_edges(ids.iter().step_by(2).copied());
            compare(Echelon::from_base(tail), &weights);
        }
    }
}

#[test]
fn supports_are_call_local_across_weight_and_matrix_updates() {
    let mut matrix = Echelon::<BasicMatrix>::new();
    for edge in [8, 3, 14] {
        matrix.add_variable(edge);
    }
    matrix.add_constraint(0, &[8, 14], true);
    matrix.add_constraint(1, &[3, 14], false);
    let first = BTreeMap::from([(8, 2.0), (3, 1.0), (14, 0.25)]);
    compare(matrix.clone(), &first);
    let _ = matrix.get_solution_local_minimum(|edge| Rational::new(first[&edge]));
    compare(matrix.clone(), &BTreeMap::from([(8, 0.25), (3, 1.0), (14, 2.0)]));
    matrix.add_variable(55);
    matrix.add_constraint(2, &[3, 55], true);
    compare(matrix, &BTreeMap::from([(8, 0.25), (3, 1.0), (14, 2.0), (55, -0.0)]));
}
