//! Parity Matrix Row
//!
//! A single row in the parity matrix, providing operations in Z_2 linear system
//!

use super::interface::*;
use derivative::Derivative;
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;

pub type BitArrayUnit = usize;
pub const BIT_UNIT_LENGTH: usize = std::mem::size_of::<BitArrayUnit>() * 8;
pub type DualVariableTag = usize;

/// optimize for small clusters where there are no more than 63 edges
#[derive(Clone, Debug, Derivative, PartialEq, Eq)]
#[derivative(Default(new = "true"))]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
pub struct ParityRow {
    /// the first BIT_UNIT_LENGTH-1 edges are stored here, and the last bit is used the right hand bit value
    first: BitArrayUnit,
    /// the other edges
    others: Vec<BitArrayUnit>,
}

impl ParityRow {
    pub fn new_length(variable_count: usize) -> Self {
        let mut row = ParityRow::new();
        let others_len = variable_count / BIT_UNIT_LENGTH;
        if others_len > 0 {
            row.others = vec![0; others_len];
        }
        row
    }

    pub fn set_left(&mut self, var_index: usize, value: bool) {
        if var_index < BIT_UNIT_LENGTH - 1 {
            // common case
            if value {
                self.first |= 0x01 << var_index;
            } else {
                self.first &= !(0x01 << var_index);
            }
        } else {
            let bias_index = var_index - (BIT_UNIT_LENGTH - 1);
            let others_idx = bias_index / BIT_UNIT_LENGTH;
            let bit_idx = bias_index % BIT_UNIT_LENGTH;
            if value {
                self.others[others_idx] |= 0x01 << bit_idx;
            } else {
                self.others[others_idx] &= !(0x01 << bit_idx);
            }
        }
    }

    pub fn get_left(&self, var_index: usize) -> bool {
        if var_index < BIT_UNIT_LENGTH - 1 {
            // common case
            self.first & (0x01 << var_index) != 0
        } else {
            let bias_index = var_index - (BIT_UNIT_LENGTH - 1);
            let others_idx = bias_index / BIT_UNIT_LENGTH;
            let bit_idx = bias_index % BIT_UNIT_LENGTH;
            self.others[others_idx] & (0x01 << bit_idx) != 0
        }
    }

    pub fn set_right(&mut self, value: bool) {
        if value {
            self.first |= 0x01 << (BIT_UNIT_LENGTH - 1);
        } else {
            self.first &= !(0x01 << (BIT_UNIT_LENGTH - 1));
        }
    }

    pub fn get_right(&self) -> bool {
        self.first & (0x01 << (BIT_UNIT_LENGTH - 1)) != 0
    }

    pub fn add(&mut self, other: &Self) {
        debug_assert_eq!(self.others.len(), other.others.len(), "size must be the same");
        self.first ^= other.first;
        for i in 0..self.others.len() {
            self.others[i] ^= other.others[i];
        }
    }

    pub fn is_left_all_zero(&self) -> bool {
        if self.first & !(0x01usize << (BIT_UNIT_LENGTH - 1)) != 0 {
            return false;
        }
        for &other in self.others.iter() {
            if other != 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(feature = "python_binding")]
#[pymethods]
impl ParityRow {
    #[new]
    fn py_new_length(variable_count: usize) -> Self {
        Self::new_length(variable_count)
    }
    #[pyo3(name = "set_left")]
    fn py_set_left(&mut self, var_index: usize, value: bool) {
        self.set_left(var_index, value);
    }
    #[pyo3(name = "get_left")]
    fn py_get_left(&self, var_index: usize) -> bool {
        self.get_left(var_index)
    }
    #[pyo3(name = "set_right")]
    fn py_set_right(&mut self, value: bool) {
        self.set_right(value);
    }
    #[pyo3(name = "get_right")]
    fn py_get_right(&self) -> bool {
        self.get_right()
    }
    #[pyo3(name = "add")]
    fn py_add(&mut self, other: &Self) {
        self.add(other);
    }
    #[pyo3(name = "is_left_all_zero")]
    fn py_is_left_all_zero(&self) -> bool {
        self.is_left_all_zero()
    }
}

impl ParityRow {
    /// only trigger updates when the new `variable_count` is enough;
    #[inline]
    fn add_one_variable_should_append(variable_count: usize) -> bool {
        variable_count % BIT_UNIT_LENGTH == 0
    }

    /// make sure this function is called exactly once when adding a new variable
    pub(super) fn add_one_variable(rows: &mut [Self], variable_count: usize) {
        if Self::add_one_variable_should_append(variable_count) {
            let others_len = variable_count / BIT_UNIT_LENGTH;
            for row in rows {
                debug_assert_eq!(row.others.len() + 1, others_len);
                row.others.push(0);
            }
        }
    }

    pub(super) fn xor_two_rows(rows: &mut [Self], target: RowIndex, source: RowIndex) {
        if target < source {
            let (slice_1, slice_2) = rows.split_at_mut(source);
            let source = &slice_2[0];
            let target = &mut slice_1[target];
            target.add(source);
        } else {
            let (slice_1, slice_2) = rows.split_at_mut(target);
            let source = &slice_1[source];
            let target = &mut slice_2[0];
            target.add(source);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use rand::prelude::*;

    struct RowTester {
        verbose: bool,
        variable_count: usize,
        lhs: Vec<bool>,
        rhs: bool,
        // the object to be tested
        row: ParityRow,
    }

    impl RowTester {
        fn set_left(&mut self, var_index: usize, value: bool) {
            self.row.set_left(var_index, value);
            self.lhs[var_index] = value;
            self.assert_equal();
        }
        fn set_right(&mut self, value: bool) {
            self.row.set_right(value);
            self.rhs = value;
            self.assert_equal();
        }
        fn add(&mut self, other: &ParityRow) {
            self.row.add(other);
            for i in 0..self.variable_count {
                self.lhs[i] ^= other.get_left(i);
            }
            self.rhs ^= other.get_right();
            self.assert_equal();
        }
        fn is_left_all_zero(&self) -> bool {
            !self.lhs.iter().any(|x| *x)
        }
        fn add_one_variable(&mut self) {
            self.variable_count += 1;
            self.lhs.push(false);
            // update row
            let mut new_rows = [self.row.clone()];
            ParityRow::add_one_variable(&mut new_rows, self.variable_count);
            self.row = new_rows[0].clone();
        }
    }

    impl RowTester {
        fn new_length(variable_count: usize) -> Self {
            let row = ParityRow::new_length(variable_count);
            assert!(row.is_left_all_zero());
            Self {
                verbose: false,
                variable_count,
                lhs: vec![false; variable_count],
                rhs: false,
                row,
            }
        }
        fn assert_equal(&self) {
            if self.verbose {
                self.print();
            }
            for i in 0..self.variable_count {
                assert_eq!(self.lhs[i], self.row.get_left(i));
            }
            assert_eq!(self.rhs, self.row.get_right());
            assert_eq!(self.is_left_all_zero(), self.row.is_left_all_zero());
        }
        fn print(&self) {
            for i in 0..self.variable_count {
                print!("{}", if self.lhs[i] { 1 } else { 0 });
            }
            println!("={}", if self.rhs { 1 } else { 0 });
        }
        fn c2b(c: char) -> bool {
            c == '1'
        }
        fn load_from_str(lhs: &str, rhs: char) -> Self {
            let mut tester = Self::new_length(lhs.len());
            for (i, c) in lhs.chars().enumerate() {
                tester.set_left(i, Self::c2b(c));
            }
            tester.set_right(Self::c2b(rhs));
            tester
        }
        fn randomize(&mut self) -> &mut Self {
            let mut rng = rand::thread_rng();
            for i in 0..self.variable_count {
                let value = rng.gen();
                self.lhs[i] = value;
                self.row.set_left(i, value);
            }
            let value = rng.gen();
            self.rhs = value;
            self.row.set_right(value);
            self
        }
    }

    #[test]
    fn parity_matrix_row_simple_case() {
        // cargo test parity_matrix_row_simple_case -- --nocapture
        let mut tester = RowTester::new_length(8);
        println!("{:?}", tester.row.clone());
        tester.verbose = true;
        tester.set_left(0, true);
        assert!(!tester.row.is_left_all_zero());
        tester.set_left(4, true);
        assert!(!tester.row.is_left_all_zero());
        tester.set_left(0, false);
        assert!(!tester.row.is_left_all_zero());
        tester.set_right(true);
        tester.set_right(false);
        tester.set_left(4, false);
        assert!(tester.row.is_left_all_zero());
    }

    #[test]
    fn parity_matrix_row_random_operations() {
        // cargo test parity_matrix_row_random_operations -- --nocapture
        let mut rng = rand::thread_rng();
        for variable_count in 0..200 {
            let mut tester = RowTester::new_length(variable_count);
            for _ in 0..1000 {
                let value = rng.gen();
                let var_index = rng.gen::<usize>() % (variable_count + 1);
                if var_index < variable_count {
                    tester.set_left(var_index, value);
                } else {
                    tester.set_right(value);
                }
            }
        }
    }

    /// bug found in parity_matrix_row_2
    /// 01110011010001101000001110000011111110111010010010111111010011111=0
    /// 01110011010001101000001110000011111110111010010010111111010011110=0
    /// bug cause: write logic:
    ///     self.others[others_idx] &= (!0x01) << bit_idx;  # wrong!!!
    ///     self.others[others_idx] &= !(0x01 << bit_idx);
    #[test]
    fn parity_matrix_row_random_failed_1() {
        // cargo test parity_matrix_row_random_failed_1 -- --nocapture
        let mut tester = RowTester::load_from_str("01110011010001101000001110000011111110111010010010111111010011111", '0');
        tester.verbose = true;
        tester.set_left(tester.variable_count - 1, false);
    }

    #[test]
    fn parity_matrix_row_simple_add() {
        // cargo test parity_matrix_row_simple_add -- --nocapture
        let mut tester = RowTester::new_length(8);
        tester.randomize();
        tester.verbose = true;
        tester.add(&RowTester::new_length(8).randomize().row);
    }

    #[test]
    fn parity_matrix_row_random_adds() {
        // cargo test parity_matrix_row_random_adds -- --nocapture
        for variable_count in 0..200 {
            let mut tester = RowTester::new_length(variable_count);
            for _ in 0..500 {
                tester.add(&RowTester::new_length(variable_count).randomize().row);
            }
        }
    }

    #[test]
    fn parity_matrix_row_add_variables() {
        // cargo test parity_matrix_row_add_variables -- --nocapture
        let mut rng = rand::thread_rng();
        let mut tester = RowTester::new_length(0);
        for variable_count in 0..2000 {
            // a few random operations
            for _ in 0..20 {
                let value = rng.gen();
                let var_index = rng.gen::<usize>() % (variable_count + 1);
                if var_index < variable_count {
                    tester.set_left(var_index, value);
                } else {
                    tester.set_right(value);
                }
            }
            // add a new variable
            tester.add_one_variable();
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "size must be the same"))]
    fn parity_matrix_row_add_different_length() {
        // cargo test parity_matrix_row_add_different_length -- --nocapture
        let mut row1 = ParityRow::new_length(10);
        let row2 = ParityRow::new_length(BIT_UNIT_LENGTH + 10);
        row1.add(&row2);
    }
}
