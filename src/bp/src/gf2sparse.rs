//! gf2sparse.rs
//!
//! gf2 implementation of the base matrix

use crate::bp::sparse_matrix_base::SparseMatrixBase;
use std::collections::HashSet;
use std::ptr;

/// An entry in a sparse matrix over GF(2)
#[derive(Clone)]
pub struct GF2Entry {
    pub row_index: isize,
    pub col_index: isize,
    pub left: *mut GF2Entry,
    pub right: *mut GF2Entry,
    pub up: *mut GF2Entry,
    pub down: *mut GF2Entry,
}

impl Default for GF2Entry {
    fn default() -> Self {
        GF2Entry::new()
    }
}

impl GF2Entry {
    /// Creates a new `GF2Entry` with default values.
    pub fn new() -> Self {
        let mut entry = GF2Entry {
            row_index: -100,
            col_index: -100,
            left: ptr::null_mut(),
            right: ptr::null_mut(),
            up: ptr::null_mut(),
            down: ptr::null_mut(),
        };
        let entry_ptr: *mut GF2Entry = &mut entry;
        entry.left = entry_ptr;
        entry.right = entry_ptr;
        entry.up = entry_ptr;
        entry.down = entry_ptr;
        entry
    }

    /// Resets the values of the entry to their default values.
    pub fn reset(&mut self) {
        self.row_index = -100;
        self.col_index = -100;
        let self_ptr: *mut GF2Entry = self;
        self.left = self_ptr;
        self.right = self_ptr;
        self.up = self_ptr;
        self.down = self_ptr;
    }

    /// Checks if the entry is at the end of the list.
    pub fn at_end(&self) -> bool {
        self.row_index == -100
    }
}

/// A sparse matrix over GF(2)
#[derive(Default, Debug)]
pub struct GF2Sparse<EntryObj = GF2Entry>
where
    EntryObj: Default + Clone,
{
    pub base: SparseMatrixBase<EntryObj>,
}

impl<T: Default + Clone> Clone for GF2Sparse<T> {
    fn clone(&self) -> Self {
        Self { base: self.base.clone() }
    }
}

impl<EntryObj> GF2Sparse<EntryObj>
where
    EntryObj: Default + Clone,
{
    /// Constructor for creating a new GF2Sparse object with the given dimensions
    pub fn new(rows: usize, cols: usize, entry_count: usize) -> Self {
        let mut base = SparseMatrixBase::<EntryObj>::new();
        base.initialise_sparse_matrix(rows, cols, entry_count);
        GF2Sparse { base }
    }

    /// Default constructor for creating a new GF2Sparse object.
    /// Creates an empty GF2Sparse matrix with zero rows and columns.
    pub fn default() -> Self {
        GF2Sparse {
            base: SparseMatrixBase::<EntryObj>::new(),
        }
    }

    /// Inserts a row of entries in compressed sparse row (CSR) format
    pub fn csr_row_insert(&mut self, row_index: usize, column_indices: &[usize]) {
        for &col_index in column_indices {
            self.insert_entry(row_index, col_index);
        }
    }

    /// Inserts a matrix in CSR format
    pub fn csr_insert(&mut self, csr_matrix: &[Vec<usize>]) {
        for (i, row) in csr_matrix.iter().enumerate() {
            self.csr_row_insert(i, row);
        }
    }

    /// Inserts a matrix in CSC format
    pub fn csc_insert(&mut self, csc_matrix: &[Vec<usize>]) {
        for (i, col) in csc_matrix.iter().enumerate() {
            for &row_index in col {
                self.insert_entry(row_index, i);
            }
        }
    }

    /// Inserts an entry into the matrix
    pub fn insert_entry(&mut self, row_index: usize, col_index: usize) {
        self.base.insert_entry(row_index, col_index);
    }

    /// Multiplies the matrix by a vector and stores the result in another vector
    pub fn mulvec_inplace(&self, input_vector: &[u8], output_vector: &mut [u8]) {
        assert_eq!(input_vector.len(), self.base.n);
        assert_eq!(output_vector.len(), self.base.m);

        for i in 0..self.base.m {
            output_vector[i] = 0;
        }

        for i in 0..self.base.m {
            for e_ptr in self.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    output_vector[i] ^= input_vector[e.col_index as usize];
                }
            }
        }
    }

    /// Multiplies the matrix by a vector and returns the result as a new vector
    pub fn mulvec(&self, input_vector: &[u8]) -> Vec<u8> {
        assert_eq!(input_vector.len(), self.base.n);
        let mut output_vector = vec![0u8; self.base.m];

        for i in 0..self.base.m {
            for e_ptr in self.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    output_vector[i] ^= input_vector[e.col_index as usize];
                }
            }
        }
        output_vector
    }

    /// Multiplies the matrix by another matrix and returns the result as a new matrix
    pub fn matmul<EntryObj2>(&self, mat_right: &GF2Sparse<EntryObj2>) -> GF2Sparse<EntryObj>
    where
        EntryObj2: Default + Clone,
    {
        if self.base.n != mat_right.base.m {
            panic!("Input matrices have invalid dimensions!");
        }

        let mut output_mat = GF2Sparse::<EntryObj>::new(self.base.m, mat_right.base.n, 0);

        for i in 0..self.base.m {
            let mut row_entries = HashSet::new();
            for e_ptr in self.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    row_entries.insert(e.col_index as usize);
                }
            }
            for j in 0..mat_right.base.n {
                let mut sum = 0u8;
                for e_ptr in mat_right.base.iterate_column(j) {
                    unsafe {
                        let e = &*e_ptr;
                        if row_entries.contains(&(e.row_index as usize)) {
                            sum ^= 1;
                        }
                    }
                }
                if sum != 0 {
                    output_mat.insert_entry(i, j);
                }
            }
        }

        output_mat
    }

    /// Adds a row to another row
    pub fn add_rows(&mut self, i: usize, j: usize) {
        let mut entries_to_remove = Vec::new();
        let mut entries_to_add = Vec::new();

        let row_i_cols: HashSet<usize> = self
            .base
            .iterate_row(i)
            .map(|e_ptr| unsafe { (*e_ptr).col_index as usize })
            .collect();

        for e_ptr in self.base.iterate_row(j) {
            let col_index = unsafe { (*e_ptr).col_index as usize };
            if row_i_cols.contains(&col_index) {
                // Mark for removal
                entries_to_remove.push((i, col_index));
            } else {
                entries_to_add.push((i, col_index));
            }
        }

        // Resolve all modifications after collecting necessary info
        for (row, col) in entries_to_remove {
            self.base.remove_entry(row, col);
        }

        for (row, col) in entries_to_add {
            self.base.insert_entry(row, col);
        }
    }

    /// Transposes the matrix
    pub fn transpose(&self) -> GF2Sparse<EntryObj> {
        let mut transposed = GF2Sparse::<EntryObj>::new(self.base.n, self.base.m, 0);

        for i in 0..self.base.m {
            for e_ptr in self.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    transposed.insert_entry(e.col_index as usize, e.row_index as usize);
                }
            }
        }

        transposed
    }

    /// Compares two matrices for equality
    pub fn gf2_equal<EntryObj2>(&self, matrix2: &GF2Sparse<EntryObj2>) -> bool
    where
        EntryObj2: Default + Clone,
    {
        if self.base.m != matrix2.base.m || self.base.n != matrix2.base.n {
            return false;
        }

        for i in 0..self.base.m {
            let mut entries_self = HashSet::new();
            for e_ptr in self.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    entries_self.insert(e.col_index as usize);
                }
            }

            let mut entries_other = HashSet::new();
            for e_ptr in matrix2.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    entries_other.insert(e.col_index as usize);
                }
            }

            if entries_self != entries_other {
                return false;
            }
        }

        true
    }

    /// Allocates memory for the sparse matrix
    pub fn allocate(&mut self, m: usize, n: usize, entry_count: usize) {
        self.base.initialise_sparse_matrix(m, n, entry_count);
    }

    /// Copies selected columns from the matrix
    pub fn copy_cols(&self, cols: &[usize]) -> GF2Sparse<EntryObj> {
        let m = self.base.m;
        let n = cols.len();
        let mut copy_mat = GF2Sparse::<EntryObj>::new(m, n, 0);

        for (new_col_index, &col_index) in cols.iter().enumerate() {
            for e_ptr in self.base.iterate_column(col_index) {
                unsafe {
                    let e = &*e_ptr;
                    copy_mat.insert_entry(e.row_index as usize, new_col_index);
                }
            }
        }

        copy_mat
    }

    /// Vertically stacks multiple matrices
    pub fn vstack(mats: &[GF2Sparse<EntryObj>]) -> GF2Sparse<EntryObj> {
        let n = mats[0].base.n;
        let mut m = 0;
        for mat in mats {
            if mat.base.n != n {
                panic!("All matrices must have the same number of columns");
            }
            m += mat.base.m;
        }

        let mut stacked_mat = GF2Sparse::<EntryObj>::new(m, n, 0);

        let mut row_offset = 0;
        for mat in mats {
            for i in 0..mat.base.m {
                for e_ptr in mat.base.iterate_row(i) {
                    unsafe {
                        let e = &*e_ptr;
                        stacked_mat.insert_entry(row_offset + e.row_index as usize, e.col_index as usize);
                    }
                }
            }
            row_offset += mat.base.m;
        }

        stacked_mat
    }

    /// Horizontally stacks multiple matrices
    pub fn hstack(mats: &[GF2Sparse<EntryObj>]) -> GF2Sparse<EntryObj> {
        let m = mats[0].base.m;
        let mut n = 0;
        for mat in mats {
            if mat.base.m != m {
                panic!("All matrices must have the same number of rows");
            }
            n += mat.base.n;
        }

        let mut stacked_mat = GF2Sparse::<EntryObj>::new(m, n, 0);

        let mut col_offset = 0;
        for mat in mats {
            for i in 0..mat.base.m {
                for e_ptr in mat.base.iterate_row(i) {
                    unsafe {
                        let e = &*e_ptr;
                        stacked_mat.insert_entry(e.row_index as usize, col_offset + e.col_index as usize);
                    }
                }
            }
            col_offset += mat.base.n;
        }

        stacked_mat
    }

    /// Kronecker product of two matrices
    pub fn kron(mat1: &GF2Sparse<EntryObj>, mat2: &GF2Sparse<EntryObj>) -> GF2Sparse<EntryObj> {
        let m1 = mat1.base.m;
        let n1 = mat1.base.n;
        let m2 = mat2.base.m;
        let n2 = mat2.base.n;

        let mut kron_mat = GF2Sparse::<EntryObj>::new(m1 * m2, n1 * n2, 0);

        for i1 in 0..m1 {
            for e_ptr1 in mat1.base.iterate_row(i1) {
                unsafe {
                    let e1 = &*e_ptr1;
                    let row_offset = e1.row_index as usize * m2;
                    let col_offset = e1.col_index as usize * n2;

                    for i2 in 0..m2 {
                        for e_ptr2 in mat2.base.iterate_row(i2) {
                            let e2 = &*e_ptr2;
                            kron_mat.insert_entry(row_offset + e2.row_index as usize, col_offset + e2.col_index as usize);
                        }
                    }
                }
            }
        }

        kron_mat
    }

    /// Adds two matrices
    pub fn add(mat1: &GF2Sparse<EntryObj>, mat2: &GF2Sparse<EntryObj>) -> GF2Sparse<EntryObj> {
        if mat1.base.m != mat2.base.m || mat1.base.n != mat2.base.n {
            panic!("Matrices must have the same dimensions");
        }

        let mut sum_mat = GF2Sparse::<EntryObj>::new(mat1.base.m, mat1.base.n, 0);

        // Copy mat1 entries
        for i in 0..mat1.base.m {
            for e_ptr in mat1.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    sum_mat.insert_entry(e.row_index as usize, e.col_index as usize);
                }
            }
        }

        // Add mat2 entries
        for i in 0..mat2.base.m {
            for e_ptr in mat2.base.iterate_row(i) {
                unsafe {
                    let e = &*e_ptr;
                    let entry = sum_mat.base.get_entry_mut(e.row_index as usize, e.col_index as usize);
                    if !(*entry).at_end() {
                        sum_mat.base.remove(entry);
                    } else {
                        sum_mat.insert_entry(e.row_index as usize, e.col_index as usize);
                    }
                }
            }
        }

        sum_mat
    }

    /// Converts a CSR matrix to GF2Sparse
    pub fn csr_to_gf2sparse(csr_matrix: &[Vec<usize>]) -> GF2Sparse<EntryObj> {
        let row_count = csr_matrix.len();
        let mut col_count = 0;
        for row in csr_matrix {
            for &col in row {
                if col > col_count {
                    col_count = col;
                }
            }
        }
        col_count += 1; // Adjust for 0-based indexing

        let mut gf2sparse_mat = GF2Sparse::<EntryObj>::new(row_count, col_count, 0);

        for (row_index, row) in csr_matrix.iter().enumerate() {
            for &col_index in row {
                gf2sparse_mat.insert_entry(row_index, col_index);
            }
        }

        gf2sparse_mat
    }

    /// Converts a CSC matrix to GF2Sparse
    pub fn csc_to_gf2sparse(csc_matrix: &[Vec<usize>]) -> GF2Sparse<EntryObj> {
        let col_count = csc_matrix.len();
        let mut row_count = 0;
        for col in csc_matrix {
            for &row in col {
                if row > row_count {
                    row_count = row;
                }
            }
        }
        row_count += 1; // Adjust for 0-based indexing

        let mut gf2sparse_mat = GF2Sparse::<EntryObj>::new(row_count, col_count, 0);

        for (col_index, col) in csc_matrix.iter().enumerate() {
            for &row_index in col {
                gf2sparse_mat.insert_entry(row_index, col_index);
            }
        }

        gf2sparse_mat
    }
}
