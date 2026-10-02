//! sparse_matrix_util.rs
//!
//! Utility functions for sparse matrix

use crate::bp::sparse_matrix_base::SparseMatrixBase;
use std::fmt::Write;

pub fn print_sparse_matrix<M: Clone + Default>(matrix: &SparseMatrixBase<M>, silent: bool) -> String {
    let mut ss = String::new();
    let m = matrix.m;
    let n = matrix.n;
    for j in 0..m {
        for i in 0..n {
            let e = matrix.get_entry(j, i);
            println!("e: {:?}", unsafe { (*e).row_index });
            if unsafe { (*e).at_end() } {
                ss.push('0');
            } else {
                write!(&mut ss, "{}", unsafe { (*e).str() }).unwrap();
            }
            if i != (n - 1) {
                ss.push(' ');
            }
        }
        if j != m - 1 {
            ss.push('\n');
        }
    }
    if !silent {
        println!("{}", ss);
    }
    ss
}

pub fn print_vector<T: ToString>(input: &[T]) {
    let length = input.len();
    print!("[");
    for (i, item) in input.iter().enumerate() {
        print!("{}", item.to_string());
        if i != length - 1 {
            print!(" ");
        }
    }
    println!("]");
}

pub fn print_array<T: ToString>(array: &[T]) {
    for (i, item) in array.iter().enumerate() {
        print!("{}", item.to_string());
        if i != array.len() - 1 {
            print!(" ");
        }
    }
    println!();
}
