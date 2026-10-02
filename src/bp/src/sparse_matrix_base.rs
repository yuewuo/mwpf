//! sparse_matrix_base.rs
//!
//! Base Sparse Matrix Implementation

use std::{
    collections::HashMap,
    fmt,
    ops::{Deref, DerefMut},
};

/// Wrapper for a mutable pointer to `EntryBase` that implements `Send`.
#[derive(Clone, Debug)]
pub struct EntryBasePtr<T: Clone + Default> {
    ptr: *mut EntryBase<T>,
}

unsafe impl<T: Clone + Default> Send for EntryBasePtr<T> {}

impl<T: Clone + Default> Deref for EntryBasePtr<T> {
    type Target = EntryBase<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T: Clone + Default> DerefMut for EntryBasePtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

impl<T: Clone + Default> Copy for EntryBasePtr<T> {}

unsafe impl<T: Clone + Default + Send> Send for EntryBase<T> {}

impl<T: Clone + Default> EntryBasePtr<T> {
    pub fn null_mut() -> Self {
        EntryBasePtr {
            ptr: std::ptr::null_mut(),
        }
    }

    pub fn from_raw(ptr: *mut EntryBase<T>) -> Self {
        EntryBasePtr { ptr }
    }

    pub fn as_ref(&self) -> Option<&EntryBase<T>> {
        if self.ptr.is_null() {
            None
        } else {
            Some(unsafe { &*self.ptr })
        }
    }

    pub fn as_mut(&mut self) -> Option<&mut EntryBase<T>> {
        if self.ptr.is_null() {
            None
        } else {
            Some(unsafe { &mut *self.ptr })
        }
    }
}

impl<T: Clone + Default> Default for EntryBasePtr<T> {
    fn default() -> Self {
        EntryBasePtr::null_mut()
    }
}

/// Base class for defining the node types for Sparse Matrices.
///
/// This struct defines the basic properties of a node in a sparse matrix such as its row index,
/// column index, and pointers to its neighboring nodes in the same row and column.
/// Each node struct that derives from this base struct will inherit these properties and add any
/// additional properties as required by the specific sparse matrix implementation.
pub struct EntryBase<T: Clone + Default> {
    pub row_index: isize,
    pub col_index: isize,
    pub left: *mut EntryBase<T>,
    pub right: *mut EntryBase<T>,
    pub up: *mut EntryBase<T>,
    pub down: *mut EntryBase<T>,
    pub inner: T,
}

impl<T: Clone + Default> Default for EntryBase<T> {
    fn default() -> Self {
        EntryBase::new()
    }
}

impl<T: Clone + Default> std::clone::Clone for EntryBase<T> {
    fn clone(&self) -> Self {
        EntryBase {
            row_index: self.row_index,
            col_index: self.col_index,
            left: self.left,
            right: self.right,
            up: self.up,
            down: self.down,
            inner: self.inner.clone(),
        }
    }
}

impl<T: Clone + Default> EntryBase<T> {
    /// Creates a new `EntryBase` with default values.
    pub fn new() -> Self {
        let mut entry = EntryBase {
            row_index: -100,
            col_index: -100,
            left: std::ptr::null_mut(),
            right: std::ptr::null_mut(),
            up: std::ptr::null_mut(),
            down: std::ptr::null_mut(),
            inner: T::default(),
        };
        let entry_ptr: *mut EntryBase<T> = &mut entry;
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
        let self_ptr: *mut EntryBase<T> = self;
        self.left = self_ptr;
        self.right = self_ptr;
        self.up = self_ptr;
        self.down = self_ptr;
    }

    /// Checks if the entry is at the end of the list.
    pub fn at_end(&self) -> bool {
        self.row_index == -100
    }

    /// Returns a string representation of the entry.
    ///     only returns 1 because other wise an entry doesn't exsit, and will be 0
    pub fn str(&self) -> &str {
        "1"
    }
}

impl<T: Clone + Default> fmt::Debug for EntryBase<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EntryBase(row_index: {}, col_index: {})", self.row_index, self.col_index)
    }
}

pub struct CsrMatrix {
    pub m: usize,
    pub n: usize,
    pub entry_count: usize,
    pub row_adjacency_list: Vec<Vec<usize>>,
}

/// Template base class for implementing sparse matrices in a doubly linked list format.
///
/// This struct allows for the construction of sparse matrices with custom data types by
/// passing node objects via the `EntryObj` generic parameter. The matrix is stored as a
/// doubly linked list, where each row and column is represented by a linked list of entries.
/// Each entry contains a reference to the next and previous entries in its row and column,
/// respectively.
#[derive(Default, Debug)]
pub struct SparseMatrixBase<T: Clone + Default, E = EntryBase<T>, P = EntryBasePtr<T>> {
    pub m: usize, // Number of rows (checks)
    pub n: usize, // Number of columns (bits)
    pub node_count: usize,
    pub entry_block_size: usize,
    pub allocated_entry_count: usize,
    pub released_entry_count: usize,
    pub block_position: usize,
    pub block_idx: isize,
    pub entries: Vec<Vec<E>>,
    pub removed_entries: Vec<P>,
    pub row_heads: Vec<P>,
    pub column_heads: Vec<P>,
    pub memory_allocated: bool,
    pub _marker: std::marker::PhantomData<T>,
}

impl<T: Clone + Default> Clone for SparseMatrixBase<T> {
    fn clone(&self) -> Self {
        // Create a new SparseMatrixBase<T>
        let mut new_matrix = SparseMatrixBase::new();

        // Copy over the primitive fields
        new_matrix.m = self.m;
        new_matrix.n = self.n;
        new_matrix.node_count = self.node_count;
        new_matrix.entry_block_size = self.entry_block_size;
        new_matrix.allocated_entry_count = self.allocated_entry_count;
        new_matrix.released_entry_count = self.released_entry_count;
        new_matrix.block_position = self.block_position;
        new_matrix.block_idx = self.block_idx;
        new_matrix.memory_allocated = self.memory_allocated;
        new_matrix._marker = std::marker::PhantomData;

        // Map to track old pointers to new pointers
        let mut entry_map = HashMap::new();

        // Clone the entries and build the mapping
        for block in &self.entries {
            let mut new_block = Vec::with_capacity(block.capacity());
            for entry in block {
                new_block.push(entry.clone());
            }
            new_matrix.entries.push(new_block);
        }

        // build mapping
        for (old_block, new_block) in self.entries.iter().zip(new_matrix.entries.iter_mut()) {
            for (old_entry, new_entry) in old_block.iter().zip(new_block.iter_mut()) {
                let old_ptr = old_entry as *const EntryBase<T> as *mut EntryBase<T>;
                let new_ptr = new_entry as *mut EntryBase<T>;
                entry_map.insert(old_ptr, new_ptr);
            }
        }

        // set the pointers to point to new entries
        for block in new_matrix.entries.iter_mut() {
            for entry in block.iter_mut() {
                if entry.row_index == -100 && entry.col_index == -100 {
                    // set all pointers to point to self, if is an invalid pointer
                    entry.left = entry as *mut EntryBase<T>;
                    entry.right = entry as *mut EntryBase<T>;
                    entry.up = entry as *mut EntryBase<T>;
                    entry.down = entry as *mut EntryBase<T>;
                } else {
                    entry.left = *entry_map
                        .get(&entry.left)
                        .expect(format!("left: {:p}, entry: {:?}", entry.left, unsafe { &*(entry.left) }).as_str());
                    entry.right = *entry_map.get(&entry.right).unwrap();
                    entry.up = *entry_map.get(&entry.up).unwrap();
                    entry.down = *entry_map.get(&entry.down).unwrap();
                }
            }
        }
        new_matrix.row_heads = Vec::with_capacity(self.row_heads.capacity());
        for old_head in self.row_heads.iter() {
            let old_head_ptr = old_head.ptr;
            let new_head = EntryBasePtr {
                ptr: *entry_map.get(&old_head_ptr).unwrap(),
            };
            new_matrix.row_heads.push(new_head);
        }
        new_matrix.column_heads = Vec::with_capacity(self.column_heads.capacity());
        for old_head in self.column_heads.iter() {
            let old_head_ptr = old_head.ptr;
            let new_head = EntryBasePtr {
                ptr: *entry_map.get(&old_head_ptr).unwrap(),
            };
            new_matrix.column_heads.push(new_head);
        }
        new_matrix.removed_entries = Vec::with_capacity(self.removed_entries.capacity());
        for old_removed_ptr in self.removed_entries.iter() {
            let old_ptr = old_removed_ptr.ptr;
            let new_ptr = if old_ptr.is_null() {
                std::ptr::null_mut()
            } else {
                *entry_map.get(&old_ptr).unwrap()
            };
            new_matrix.removed_entries.push(EntryBasePtr { ptr: new_ptr });
        }

        new_matrix
    }
}

impl<T: Clone + Default> SparseMatrixBase<T> {
    /// Constructs a new `SparseMatrixBase` with default values.
    pub fn new() -> Self {
        SparseMatrixBase {
            m: 0,
            n: 0,
            node_count: 0,
            entry_block_size: 0,
            allocated_entry_count: 0,
            released_entry_count: 0,
            block_position: 0,
            block_idx: 0,
            entries: Vec::new(),
            removed_entries: Vec::new(),
            row_heads: Vec::new(),
            column_heads: Vec::new(),
            memory_allocated: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Constructs a sparse matrix with the given dimensions.
    pub fn initialise_sparse_matrix(&mut self, check_count: usize, bit_count: usize, entry_count: usize) {
        self.reset_matrix();
        self.m = check_count;
        self.n = bit_count;
        self.block_idx = -1;
        self.released_entry_count = 0;
        self.allocated_entry_count = 0;
        self.entry_block_size = self.m + self.n + entry_count;
        self.allocate_memory();
        self.entry_block_size = self.m + self.n;
    }

    /// Resets the matrix to its initial state.
    pub fn reset_matrix(&mut self) {
        if self.memory_allocated {
            self.column_heads.clear();
            self.row_heads.clear();
            self.removed_entries.clear();
            for entry_block in &mut self.entries {
                entry_block.clear();
            }
            self.entries.clear();
        }
        self.m = 0;
        self.n = 0;
        self.block_idx = -1;
        self.released_entry_count = 0;
        self.allocated_entry_count = 0;
        self.entry_block_size = 0;
        self.memory_allocated = false;
    }

    /// Allocates memory for the row and column header nodes.
    pub fn allocate_memory(&mut self) {
        self.memory_allocated = true;

        self.row_heads.resize(self.m, EntryBasePtr::null_mut());
        self.column_heads.resize(self.n, EntryBasePtr::null_mut());

        for i in 0..self.m {
            let row_entry_ptr = self.allocate_new_entry();
            unsafe {
                (*row_entry_ptr).row_index = -100;
                (*row_entry_ptr).col_index = -100;
                (*row_entry_ptr).up = row_entry_ptr;
                (*row_entry_ptr).down = row_entry_ptr;
                (*row_entry_ptr).left = row_entry_ptr;
                (*row_entry_ptr).right = row_entry_ptr;
            }
            self.row_heads[i] = EntryBasePtr::from_raw(row_entry_ptr);
        }

        for i in 0..self.n {
            let col_entry_ptr = self.allocate_new_entry();
            unsafe {
                (*col_entry_ptr).row_index = -100;
                (*col_entry_ptr).col_index = -100;
                (*col_entry_ptr).up = col_entry_ptr;
                (*col_entry_ptr).down = col_entry_ptr;
                (*col_entry_ptr).left = col_entry_ptr;
                (*col_entry_ptr).right = col_entry_ptr;
            }
            self.column_heads[i] = EntryBasePtr::from_raw(col_entry_ptr);
        }
    }

    /// Allocates a new entry object and returns a pointer to it.
    pub fn allocate_new_entry(&mut self) -> *mut EntryBase<T> {
        if !self.removed_entries.is_empty() {
            return self.removed_entries.pop().unwrap().ptr;
        }

        if self.released_entry_count == self.allocated_entry_count {
            let new_entry = EntryBase::<T>::new();
            self.entries.push(vec![new_entry; self.entry_block_size]);
            self.allocated_entry_count += self.entry_block_size;
            self.block_idx += 1;
            self.block_position = 0;
        }

        let e_ptr = &mut self.entries[self.block_idx as usize][self.block_position] as *mut EntryBase<T>;
        self.block_position += 1;
        self.released_entry_count += 1;
        e_ptr
    }

    /// Returns the number of non-zero entries in the matrix.
    pub fn entry_count(&self) -> usize {
        self.released_entry_count - self.n - self.m - self.removed_entries.len()
    }

    /// Computes the sparsity of the matrix.
    pub fn sparsity(&self) -> f64 {
        self.entry_count() as f64 / (self.m * self.n) as f64
    }

    /// Swaps two rows of the matrix.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        self.row_heads.swap(i, j);
        for e in self.iterate_row_mut(i) {
            unsafe {
                (*e).row_index = i as isize;
            }
        }
        for e in self.iterate_row_mut(j) {
            unsafe {
                (*e).row_index = j as isize;
            }
        }
    }

    /// Reorders the rows of the matrix based on the provided order.
    pub fn reorder_rows(&mut self, rows: &Vec<usize>) {
        let temp_row_heads = self.row_heads.clone();
        for i in 0..self.m {
            self.row_heads[i] = temp_row_heads[rows[i]];
            for e in self.iterate_row_mut(i) {
                unsafe {
                    (*e).row_index = i as isize;
                }
            }
        }
    }

    /// Swaps two columns in the sparse matrix.
    pub fn swap_columns(&mut self, i: usize, j: usize) {
        self.column_heads.swap(i, j);
        for e in self.iterate_column_mut(i) {
            unsafe {
                (*e).col_index = i as isize;
            }
        }
        for e in self.iterate_column_mut(j) {
            unsafe {
                (*e).col_index = j as isize;
            }
        }
    }

    /// Gets the number of non-zero entries in a row of the matrix.
    pub fn get_row_degree(&self, row: usize) -> isize {
        self.row_heads[row].as_ref().unwrap().col_index.abs() - 100
    }

    /// Gets the number of non-zero entries in a column of the matrix.
    pub fn get_col_degree(&self, col: usize) -> isize {
        self.column_heads[col].as_ref().unwrap().col_index.abs() - 100
    }

    /// Removes an entry from the matrix.
    pub fn remove_entry(&mut self, i: usize, j: usize) {
        let e = self.get_entry_mut(i, j);
        unsafe { self.remove(e) };
    }

    /// Removes an entry from the matrix and updates the row/column weights.
    pub unsafe fn remove(&mut self, e_ptr: *mut EntryBase<T>) {
        unsafe {
            if !(*e_ptr).at_end() {
                let e_left_ptr = (*e_ptr).left;
                let e_right_ptr = (*e_ptr).right;
                let e_up_ptr = (*e_ptr).up;
                let e_down_ptr = (*e_ptr).down;

                (*e_left_ptr).right = e_right_ptr;
                (*e_right_ptr).left = e_left_ptr;
                (*e_up_ptr).down = e_down_ptr;
                (*e_down_ptr).up = e_up_ptr;

                self.row_heads[(*e_ptr).row_index as usize].as_mut().unwrap().col_index += 1;
                self.column_heads[(*e_ptr).col_index as usize].as_mut().unwrap().col_index += 1;

                (*e_ptr).reset();
            }
            self.removed_entries.push(EntryBasePtr::from_raw(e_ptr));
        }
    }

    /// Inserts a new entry in the matrix at position (i, j).
    pub fn insert_entry(&mut self, j: usize, i: usize) -> *mut EntryBase<T> {
        // println!("new invokation, {i}, {j}");
        if j >= self.m || i >= self.n {
            panic!("Index i or j is out of bounds");
        }

        let mut left_entry_ptr = self.row_heads[j];
        let mut right_entry_ptr = self.row_heads[j];
        for e in self.reverse_iterate_row_mut(j) {
            let index = unsafe { (*e).col_index as usize };
            if index == i {
                return e;
            }
            if index > i {
                right_entry_ptr = EntryBasePtr::from_raw(e);
            }
            if index < i {
                left_entry_ptr = EntryBasePtr::from_raw(e);
                break;
            }
        }

        let mut up_entry_ptr = self.column_heads[i];
        let mut down_entry_ptr = self.column_heads[i];
        for e in self.reverse_iterate_column_mut(i) {
            let row_index = unsafe { (*e).row_index as usize };
            if row_index > j {
                down_entry_ptr = EntryBasePtr::from_raw(e);
            }
            if row_index < j {
                up_entry_ptr = EntryBasePtr::from_raw(e);
                break;
            }
        }

        let e_ptr = self.allocate_new_entry();
        self.node_count += 1;
        unsafe {
            (*e_ptr).row_index = j as isize;
            (*e_ptr).col_index = i as isize;
            (*e_ptr).right = right_entry_ptr.ptr;
            (*e_ptr).left = left_entry_ptr.ptr;
            (*e_ptr).up = up_entry_ptr.ptr;
            (*e_ptr).down = down_entry_ptr.ptr;
            (*left_entry_ptr).right = e_ptr;
            (*right_entry_ptr).left = e_ptr;
            (*up_entry_ptr).down = e_ptr;
            (*down_entry_ptr).up = e_ptr;

            (*self.row_heads[(*e_ptr).row_index as usize]).col_index -= 1;
            (*self.column_heads[(*e_ptr).col_index as usize]).col_index -= 1;
        }
        e_ptr
    }

    /// Gets an entry at row j and column i.
    pub fn get_entry_mut(&mut self, j: usize, i: usize) -> *mut EntryBase<T> {
        if j >= self.m || i >= self.n {
            panic!("Index i or j is out of bounds");
        }

        for e in self.reverse_iterate_column(i) {
            unsafe {
                if (*e).row_index as usize == j {
                    return e;
                }
            }
        }
        self.column_heads[i].ptr
    }

    pub fn get_entry(&self, j: usize, i: usize) -> *const EntryBase<T> {
        if j >= self.m || i >= self.n {
            panic!("Index i or j is out of bounds");
        }

        for e in self.reverse_iterate_column(i) {
            unsafe {
                if (*e).row_index as usize == j {
                    return e;
                }
            }
        }
        self.column_heads[i].ptr
    }

    /// Inserts a new row at row_index with entries at column indices col_indices.
    pub fn insert_row(&mut self, row_index: usize, col_indices: &Vec<usize>) -> *mut EntryBase<T> {
        for &j in col_indices {
            self.insert_entry(row_index, j);
        }
        self.row_heads[row_index].ptr
    }

    /// Returns the coordinates of all non-zero entries in the matrix.
    pub fn nonzero_coordinates(&mut self) -> Vec<(usize, usize)> {
        let mut nonzero = Vec::new();
        let mut node_count = 0;

        for i in 0..self.m {
            for e in self.iterate_row_mut(i) {
                node_count += 1;
                nonzero.push(unsafe { ((*e).row_index as usize, (*e).col_index as usize) });
            }
        }
        self.node_count = node_count;
        nonzero
    }

    /// Returns row adjacency list as vector of vectors.
    pub fn row_adjacency_list(&mut self) -> Vec<Vec<usize>> {
        let mut adj_list = Vec::new();
        for i in 0..self.m {
            let mut row = Vec::new();
            for e in self.iterate_row_mut(i) {
                unsafe {
                    row.push((*e).col_index as usize);
                }
            }
            adj_list.push(row);
        }
        adj_list
    }

    /// Returns column adjacency list as vector of vectors.
    pub fn col_adjacency_list(&mut self) -> Vec<Vec<usize>> {
        let mut adj_list = Vec::new();
        for i in 0..self.n {
            let mut col = Vec::new();
            for e in self.iterate_column_mut(i) {
                unsafe {
                    col.push((*e).row_index as usize);
                }
            }
            adj_list.push(col);
        }
        adj_list
    }

    /// Return a single column as 1D csc_matrix.
    pub fn get_column_csc(&mut self, col_index: usize) -> Vec<usize> {
        let mut col = Vec::new();
        for e in self.iterate_column_mut(col_index) {
            unsafe {
                col.push((*e).row_index as usize);
            }
        }
        col
    }

    /// Converts the sparse matrix to CSR format.
    pub fn to_csr_matrix(&mut self) -> CsrMatrix {
        CsrMatrix {
            m: self.m,
            n: self.n,
            entry_count: self.entry_count(),
            row_adjacency_list: self.row_adjacency_list(),
        }
    }

    /// Returns a vector of vectors, where each vector contains the column indices of the non-zero entries in a row.
    pub fn nonzero_rows(&mut self) -> Vec<Vec<usize>> {
        let mut nonzero = Vec::new();
        let mut node_count = 0;

        for i in 0..self.m {
            let mut row = Vec::new();
            for e in self.iterate_row_mut(i) {
                node_count += 1;
                row.push(unsafe { (*e).col_index as usize });
            }
            nonzero.push(row);
        }
        self.node_count = node_count;
        nonzero
    }

    /// Returns an iterator that iterates over the given row of the sparse matrix in a forward direction.
    pub fn iterate_row_mut(&mut self, i: usize) -> RowIterator<'_, T> {
        if i >= self.m {
            panic!("Iterator index out of bounds");
        }
        RowIterator::new(self, i)
    }

    pub fn iterate_row(&self, i: usize) -> RowIterator<'_, T> {
        if i >= self.m {
            panic!("Iterator index out of bounds");
        }
        RowIterator::new(self, i)
    }

    /// Returns an iterator that iterates over the given row of the sparse matrix in a reverse direction.
    pub fn reverse_iterate_row_mut(&mut self, i: usize) -> ReverseRowIterator<'_, T> {
        if i >= self.m {
            panic!("Iterator index out of bounds");
        }
        ReverseRowIterator::new(self, i)
    }

    pub fn reverse_iterate_row(&self, i: usize) -> ReverseRowIterator<'_, T> {
        if i >= self.m {
            panic!("Iterator index out of bounds");
        }
        ReverseRowIterator::new(self, i)
    }

    /// Returns an iterator that iterates over the given column of the sparse matrix in a forward direction.
    pub fn iterate_column_mut(&mut self, i: usize) -> ColumnIterator<'_, T> {
        if i >= self.n {
            panic!("Iterator index out of bounds");
        }
        ColumnIterator::new(self, i)
    }

    pub fn iterate_column(&self, i: usize) -> ColumnIterator<'_, T> {
        if i >= self.n {
            panic!("Iterator index out of bounds");
        }
        ColumnIterator::new(self, i)
    }

    /// Returns an iterator that iterates over the given column of the sparse matrix in a reverse direction.
    pub fn reverse_iterate_column_mut(&mut self, i: usize) -> ReverseColumnIterator<'_, T> {
        if i >= self.n {
            panic!("Iterator index out of bounds");
        }
        ReverseColumnIterator::new(self, i)
    }

    pub fn reverse_iterate_column(&self, i: usize) -> ReverseColumnIterator<'_, T> {
        if i >= self.n {
            panic!("Iterator index out of bounds");
        }
        ReverseColumnIterator::new(self, i)
    }
}

/// Iterator for iterating over rows in a sparse matrix.
pub struct RowIterator<'a, T: Clone + Default> {
    _matrix: &'a SparseMatrixBase<T>,
    it_count: isize,
    entry_count: isize,
    e: *mut EntryBase<T>,
}

impl<'a, T: Clone + Default> RowIterator<'a, T> {
    fn new(matrix: &'a SparseMatrixBase<T>, i: usize) -> Self {
        RowIterator {
            _matrix: matrix,
            it_count: 0,
            entry_count: matrix.get_row_degree(i),
            e: matrix.row_heads[i].ptr,
        }
    }
}

impl<'a, T: Clone + Default> Iterator for RowIterator<'a, T> {
    type Item = *mut EntryBase<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.it_count >= self.entry_count {
            return None;
        }
        unsafe {
            self.e = (*self.e).right;
        }
        self.it_count += 1;
        Some(self.e)
    }
}

/// Reverse iterator for iterating over rows in a sparse matrix.
pub struct ReverseRowIterator<'a, T: Clone + Default> {
    _matrix: &'a SparseMatrixBase<T>,
    it_count: isize,
    entry_count: isize,
    e: *mut EntryBase<T>,
}

impl<'a, T: Clone + Default> ReverseRowIterator<'a, T> {
    fn new(matrix: &'a SparseMatrixBase<T>, i: usize) -> Self {
        ReverseRowIterator {
            _matrix: matrix,
            it_count: 0,
            entry_count: matrix.get_row_degree(i),
            e: matrix.row_heads[i].ptr,
        }
    }
}

impl<'a, T: Clone + Default> Iterator for ReverseRowIterator<'a, T> {
    type Item = *mut EntryBase<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.it_count >= self.entry_count {
            return None;
        }
        unsafe {
            self.e = (*self.e).left;
        }
        self.it_count += 1;
        Some(self.e)
    }
}

/// Iterator for iterating over columns in a sparse matrix.
pub struct ColumnIterator<'a, T: Clone + Default> {
    _matrix: &'a SparseMatrixBase<T>,
    it_count: isize,
    entry_count: isize,
    e: *mut EntryBase<T>,
}

impl<'a, T: Clone + Default> ColumnIterator<'a, T> {
    fn new(matrix: &'a SparseMatrixBase<T>, i: usize) -> Self {
        ColumnIterator {
            _matrix: matrix,
            it_count: 0,
            entry_count: matrix.get_col_degree(i),
            e: matrix.column_heads[i].ptr,
        }
    }
}

impl<'a, T: Clone + Default> Iterator for ColumnIterator<'a, T> {
    type Item = *mut EntryBase<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.it_count >= self.entry_count {
            return None;
        }
        unsafe {
            self.e = (*self.e).down;
        }
        self.it_count += 1;
        Some(self.e)
    }
}

/// Reverse iterator for iterating over columns in a sparse matrix.
pub struct ReverseColumnIterator<'a, T: Clone + Default> {
    _matrix: &'a SparseMatrixBase<T>,
    it_count: isize,
    entry_count: isize,
    e: *mut EntryBase<T>,
}

impl<'a, T: Clone + Default> ReverseColumnIterator<'a, T> {
    fn new(matrix: &'a SparseMatrixBase<T>, i: usize) -> Self {
        ReverseColumnIterator {
            _matrix: matrix,
            it_count: 0,
            entry_count: matrix.get_col_degree(i),
            e: matrix.column_heads[i].ptr,
        }
    }
}

impl<'a, T: Clone + Default> Iterator for ReverseColumnIterator<'a, T> {
    type Item = *mut EntryBase<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.it_count >= self.entry_count {
            return None;
        }
        unsafe {
            self.e = (*self.e).up;
        }
        self.it_count += 1;
        Some(self.e)
    }
}
