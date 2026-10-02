//! gf2codes.rs
//!
//! This module contains functions to generate parity check matrices for various binary codes.

use crate::bp::gf2sparse::{GF2Entry, GF2Sparse};

/// Creates the parity check matrix of a repetition code of length `n`.
///
/// # Parameters
/// - `n`: The length of the repetition code.
///
/// # Returns
/// A boxed `GF2Sparse<T>` matrix representing the parity check matrix.
pub fn rep_code(n: usize) -> Box<GF2Sparse<GF2Entry>> {
    let mut pcm = GF2Sparse::new(n - 1, n, 0);
    for i in 0..n - 1 {
        pcm.insert_entry(i, i); // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, i + 1); // Insert a 1 in the position to the right of the diagonal.
    }
    Box::new(pcm)
}

/// Creates the parity check matrix of a cyclic repetition code of length `n`.
///
/// # Parameters
/// - `n`: The length of the cyclic repetition code.
///
/// # Returns
/// A boxed `GF2Sparse<T>` matrix representing the parity check matrix.
pub fn ring_code(n: usize) -> Box<GF2Sparse<GF2Entry>> {
    let mut pcm = GF2Sparse::new(n, n, 0);
    for i in 0..n {
        pcm.insert_entry(i, i); // Insert a 1 in the diagonal position.
        pcm.insert_entry(i, (i + 1) % n); // Insert a 1 with wraparound.
    }
    Box::new(pcm)
}

/// Creates the parity check matrix of a Hamming code with given rank.
///
/// # Parameters
/// - `r`: The rank of the Hamming code, where the block length is 2^r - 1.
///
/// # Returns
/// A boxed `GF2Sparse<T>` matrix representing the parity check matrix.
pub fn hamming_code(r: usize) -> Box<GF2Sparse<GF2Entry>> {
    let n = (1 << r) - 1; // block length
    let mut pcm = GF2Sparse::new(r, n, 0);
    for i in 0..n {
        let binary = i + 1;
        for j in 0..r {
            if binary & (1 << j) != 0 {
                pcm.insert_entry(j, i);
            }
        }
    }
    Box::new(pcm)
}
