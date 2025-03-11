//! bp.rs
//!
//! Belief Propagation (BP) decoder implementation.

use std::error::Error;
use std::f64;

use crate::custom_rng::RandomListShuffle;
use crate::gf2sparse::GF2Sparse;
use crate::sparse_matrix_base::EntryBase;

pub type BpEntry = EntryBase<_BpEntry>;
pub type BpSparse = GF2Sparse<_BpEntry>;

// Placeholder types for external modules (to add the fields to the struct, in place of C++ derivation)
#[derive(Clone, Default, Debug)]
pub struct _BpEntry {
    pub bit_to_check_msg: f64,
    pub check_to_bit_msg: f64,
}

#[derive(PartialEq, Clone, Debug)]
pub enum BpMethod {
    ProductSum = 0,
    MinimumSum = 1,
}

#[derive(PartialEq, Clone, Debug)]
pub enum BpSchedule {
    Serial = 0,
    Parallel = 1,
    SerialRelative = 2,
}

#[derive(PartialEq, Clone, Debug)]
pub enum BpInputType {
    Syndrome = 0,
    ReceivedVector = 1,
    Auto = 2,
}

#[derive(Clone, Debug)]
pub struct BpDecoder {
    pub pcm: BpSparse,
    pub channel_probabilities: Vec<f64>,
    pub check_count: usize,
    pub bit_count: usize,
    pub maximum_iterations: usize,
    pub method: BpMethod,
    pub schedule: BpSchedule,
    pub bp_input_type: BpInputType,
    pub ms_scaling_factor: f64,
    pub decoding: Vec<u8>,
    pub candidate_syndrome: Vec<u8>,
    pub log_prob_ratios: Vec<f64>,
    pub initial_log_prob_ratios: Vec<f64>,
    pub soft_syndrome: Vec<f64>,
    pub serial_schedule_order: Vec<usize>,
    pub iterations: usize,
    pub omp_thread_count: usize,
    pub converge: bool,
    pub random_schedule_seed: i32,
    pub random_schedule_at_every_iteration: bool,
    pub rng_list_shuffle: RandomListShuffle<usize>,
}

impl<'a> BpDecoder {
    /// Create a new BP decoder instance.
    pub fn new(
        parity_check_matrix: BpSparse,
        channel_probabilities: Vec<f64>,
        maximum_iterations: usize,
        method: BpMethod,
        schedule: BpSchedule,
        min_sum_scaling_factor: f64,
        omp_threads: usize,
        serial_schedule: Option<&Vec<usize>>,
        mut random_schedule_seed: i32,
        random_schedule_at_every_iteration: bool,
        bp_input_type: BpInputType,
    ) -> Result<BpDecoder, Box<dyn Error>> {
        let pcm = parity_check_matrix;
        let check_count = pcm.base.m;
        let bit_count = pcm.base.n;

        let initial_log_prob_ratios = vec![0.0; bit_count];
        let log_prob_ratios = vec![0.0; bit_count];
        let candidate_syndrome = vec![0u8; check_count];
        let decoding = vec![0u8; bit_count];
        let soft_syndrome = vec![0.0; check_count];
        let converge = false;
        let omp_thread_count = omp_threads;
        let iterations = 0;

        if channel_probabilities.len() != bit_count {
            return Err("Channel probabilities vector must have length equal to the number of bits".into());
        }

        let serial_schedule_order: Vec<usize>;
        let mut rng_list_shuffle = RandomListShuffle::new();

        if let Some(schedule) = serial_schedule {
            serial_schedule_order = schedule.clone();
            random_schedule_seed = -1;
        } else {
            serial_schedule_order = (0..bit_count).collect();
            rng_list_shuffle.seed(random_schedule_seed as u64);
        }

        Ok(BpDecoder {
            pcm,
            channel_probabilities,
            check_count,
            bit_count,
            maximum_iterations,
            method,
            schedule,
            bp_input_type,
            ms_scaling_factor: min_sum_scaling_factor,
            decoding,
            candidate_syndrome,
            log_prob_ratios,
            initial_log_prob_ratios,
            soft_syndrome,
            serial_schedule_order,
            iterations,
            omp_thread_count,
            converge,
            random_schedule_seed,
            random_schedule_at_every_iteration,
            rng_list_shuffle,
        })
    }

    /// Create a new BP decoder instance with some default parameters.
    pub fn new_3(
        parity_check_matrix: BpSparse,
        channel_probabilities: Vec<f64>,
        maximum_iterations: usize,
    ) -> Result<BpDecoder, Box<dyn Error>> {
        let method = BpMethod::ProductSum;
        let schedule = BpSchedule::Parallel;
        let min_sum_scaling_factor = 0.625;
        let omp_threads = 16;
        let serial_schedule = None;
        let random_schedule_seed = -1;
        let random_schedule_at_every_iteration = true;
        let bp_input_type = BpInputType::Auto;

        BpDecoder::new(
            parity_check_matrix,
            channel_probabilities,
            maximum_iterations,
            method,
            schedule,
            min_sum_scaling_factor,
            omp_threads,
            serial_schedule,
            random_schedule_seed,
            random_schedule_at_every_iteration,
            bp_input_type,
        )
    }

    /// Set the number of threads to use for OpenMP parallelization.
    pub fn set_omp_thread_count(&mut self, count: usize) {
        self.omp_thread_count = count;
    }

    /// Set initial log domain BP messages and log probabilities.
    pub fn initialise_log_domain_bp(&mut self) {
        for i in 0..self.bit_count {
            self.initial_log_prob_ratios[i] = ((1.0 - self.channel_probabilities[i]) / self.channel_probabilities[i]).ln();
            for e in self.pcm.base.iterate_column_mut(i) {
                unsafe { (*e).inner.bit_to_check_msg = self.initial_log_prob_ratios[i] };
            }
        }
    }

    /// update the log domain BP messages and log probabilities.
    pub fn set_log_domain_bp(&mut self, input_vector: &Vec<f64>) {
        for i in 0..self.bit_count {
            self.initial_log_prob_ratios[i] = input_vector[i];
            for e in self.pcm.base.iterate_column_mut(i) {
                unsafe { (*e).inner.bit_to_check_msg = self.initial_log_prob_ratios[i] };
            }
        }
    }

    /// Decode
    pub fn decode(&mut self, input_vector: &Vec<u8>) -> Vec<u8> {
        if (self.bp_input_type == BpInputType::Auto && input_vector.len() == self.bit_count)
            || (self.bp_input_type == BpInputType::ReceivedVector)
        {
            // println!("this is invoked");
            let syndrome = self.pcm.mulvec(input_vector);
            let rv_decoding = if self.schedule == BpSchedule::Parallel {
                self.bp_decode_parallel(&syndrome)
            } else if self.schedule == BpSchedule::Serial || self.schedule == BpSchedule::SerialRelative {
                self.bp_decode_serial(&syndrome)
            } else {
                panic!("Invalid BP schedule");
            };

            for i in 0..self.bit_count {
                self.decoding[i] = rv_decoding[i] ^ input_vector[i];
            }

            return self.decoding.clone();
        }

        if self.schedule == BpSchedule::Parallel {
            return self.bp_decode_parallel(input_vector);
        } else if self.schedule == BpSchedule::Serial || self.schedule == BpSchedule::SerialRelative {
            return self.bp_decode_serial(input_vector);
        } else {
            panic!("Invalid BP schedule");
        }
    }

    /// Decode parallelly
    pub fn bp_decode_parallel(&mut self, syndrome: &Vec<u8>) -> Vec<u8> {
        self.converge = false;

        for it in 1..=self.maximum_iterations {
            if self.method == BpMethod::ProductSum {
                for i in 0..self.check_count {
                    self.candidate_syndrome[i] = 0;

                    let mut temp = 1.0;
                    for e in self.pcm.base.iterate_row_mut(i) {
                        unsafe {
                            (*e).inner.check_to_bit_msg = temp;
                            temp *= ((*e).inner.bit_to_check_msg / 2.0).tanh();
                        }
                    }

                    temp = 1.0;
                    for e in self.pcm.base.reverse_iterate_row_mut(i) {
                        unsafe {
                            (*e).inner.check_to_bit_msg *= temp;
                            let message_sign = if syndrome[i] != 0 { -1.0 } else { 1.0 };
                            // (*e).inner.check_to_bit_msg = message_sign
                            //     * ((1.0 + (*e).inner.check_to_bit_msg) / (1.0 - (*e).inner.check_to_bit_msg)).ln();
                            // Note: However, shouldn't we just return? (-inf should be the actual result)
                            // TODO: Test with this
                            let eps = 1e-10;
                            let safe_msg = (*e).inner.check_to_bit_msg.clamp(-1.0 + eps, 1.0 - eps);
                            (*e).inner.check_to_bit_msg = message_sign * ((1.0 + safe_msg) / (1.0 - safe_msg)).ln();

                            temp *= ((*e).inner.bit_to_check_msg / 2.0).tanh();
                        }
                    }
                }
            } else if self.method == BpMethod::MinimumSum {
                for i in 0..self.check_count {
                    self.candidate_syndrome[i] = 0;
                    let mut total_sgn = syndrome[i] as i32;
                    let mut temp = f64::MAX;

                    for e in self.pcm.base.iterate_row_mut(i) {
                        unsafe {
                            if (*e).inner.bit_to_check_msg <= 0.0 {
                                total_sgn += 1;
                            }
                            (*e).inner.check_to_bit_msg = temp;
                            let abs_bit_to_check_msg = (*e).inner.bit_to_check_msg.abs();
                            if abs_bit_to_check_msg < temp {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }

                    temp = f64::MAX;
                    for e in self.pcm.base.reverse_iterate_row_mut(i) {
                        let mut sgn = total_sgn;
                        unsafe {
                            if (*e).inner.bit_to_check_msg <= 0.0 {
                                sgn += 1;
                            }
                            if temp < (*e).inner.check_to_bit_msg {
                                (*e).inner.check_to_bit_msg = temp;
                            }

                            let message_sign = if sgn % 2 == 0 { 1.0 } else { -1.0 };
                            (*e).inner.check_to_bit_msg *= message_sign * self.ms_scaling_factor;

                            let abs_bit_to_check_msg = (*e).inner.bit_to_check_msg.abs();
                            if abs_bit_to_check_msg < temp {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }
                }
            }

            for i in 0..self.bit_count {
                let mut temp = self.initial_log_prob_ratios[i];
                for e in self.pcm.base.iterate_column_mut(i) {
                    unsafe {
                        (*e).inner.bit_to_check_msg = temp;
                        temp += (*e).inner.check_to_bit_msg;
                    }
                }

                self.log_prob_ratios[i] = temp;
                if temp <= 0.0 {
                    self.decoding[i] = 1;
                    for e in self.pcm.base.iterate_column(i) {
                        unsafe {
                            self.candidate_syndrome[(*e).row_index as usize] ^= 1;
                        }
                    }
                } else {
                    self.decoding[i] = 0;
                }
            }

            if self.candidate_syndrome == *syndrome {
                self.converge = true;
            }

            self.iterations = it;

            if self.converge {
                return self.decoding.clone();
            }

            for i in 0..self.bit_count {
                let mut temp = 0.0;
                for e in self.pcm.base.reverse_iterate_column_mut(i) {
                    unsafe {
                        (*e).inner.bit_to_check_msg += temp;
                        temp += (*e).inner.check_to_bit_msg;
                    }
                }
            }
        }

        self.decoding.clone()
    }

    /// Decode serially
    pub fn bp_decode_serial(&mut self, syndrome: &Vec<u8>) -> Vec<u8> {
        self.converge = false;
        for it in 1..=self.maximum_iterations {
            if self.random_schedule_seed > -1 {
                self.rng_list_shuffle.shuffle(&mut self.serial_schedule_order);
            } else if self.schedule == BpSchedule::SerialRelative {
                self.serial_schedule_order.sort_by(|&bit1, &bit2| {
                    if it != 1 {
                        // This replicates the descending order comparison by reversing the comparison.
                        // use this result self.log_prob_ratios[bit1] > self.log_prob_ratios[bit2]
                        self.log_prob_ratios[bit2]
                            .partial_cmp(&self.log_prob_ratios[bit1])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        let prob1 = (1.0 - self.channel_probabilities[bit1]) / self.channel_probabilities[bit1];
                        let prob2 = (1.0 - self.channel_probabilities[bit2]) / self.channel_probabilities[bit2];

                        // Calculate logs and compare, reversing for descending order.
                        prob2.ln().partial_cmp(&prob1.ln()).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }

            for &bit_index in &self.serial_schedule_order {
                // Initialize log probabilities for each bit
                self.log_prob_ratios[bit_index] =
                    ((1.0 - self.channel_probabilities[bit_index]) / self.channel_probabilities[bit_index]).ln();

                // First, gather all check information for the current bit
                let mut checks = Vec::new();
                for e in self.pcm.base.iterate_column(bit_index) {
                    unsafe {
                        let check_index = (*e).row_index as usize;
                        let check_messages: Vec<f64> = self
                            .pcm
                            .base
                            .iterate_row(check_index)
                            .filter(|&g| g != e) // Skip the current bit's entry to avoid borrow conflict
                            .map(|g| (*g).inner.bit_to_check_msg)
                            .collect();
                        checks.push((e, check_index, check_messages));
                    }
                }

                // Then update check-to-bit messages based on gathered data
                for (e, check_index, check_messages) in checks {
                    unsafe {
                        if self.method == BpMethod::ProductSum {
                            let product_sum = check_messages.iter().map(|&msg| (msg / 2.0).tanh()).product::<f64>();
                            let check_msg = ((-1.0f64).powi(syndrome[check_index] as i32))
                                * ((1.0 + product_sum) / (1.0 - product_sum)).ln();
                            (*e).inner.check_to_bit_msg = check_msg;
                        } else if self.method == BpMethod::MinimumSum {
                            let mut sgn = syndrome[check_index] as i32;
                            let mut min_abs_msg = f64::MAX;
                            for &msg in &check_messages {
                                if msg <= 0.0 {
                                    sgn += 1;
                                }
                                min_abs_msg = min_abs_msg.min(msg.abs());
                            }
                            let message_sign = if sgn % 2 == 0 { 1.0 } else { -1.0 };
                            (*e).inner.check_to_bit_msg = self.ms_scaling_factor * message_sign * min_abs_msg;
                        }
                        (*e).inner.bit_to_check_msg = self.log_prob_ratios[bit_index];
                        self.log_prob_ratios[bit_index] += (*e).inner.check_to_bit_msg;
                    }
                }

                // Finalize the bit value based on the updated log probability ratios
                if self.log_prob_ratios[bit_index] <= 0.0 {
                    self.decoding[bit_index] = 1;
                } else {
                    self.decoding[bit_index] = 0;
                }

                // Update the bit-to-check messages for the next iteration
                let mut temp = 0.0;
                for e in self.pcm.base.reverse_iterate_column_mut(bit_index) {
                    unsafe {
                        (*e).inner.bit_to_check_msg += temp;
                        temp += (*e).inner.check_to_bit_msg;
                    }
                }
            }

            // Check for convergence
            self.candidate_syndrome = self.pcm.mulvec(&self.decoding);
            if self.candidate_syndrome == *syndrome {
                self.converge = true;
                break;
            }
            self.iterations = it;
        }

        self.decoding.clone()
    }

    /// Decode single scan
    pub fn bp_decode_single_scan(&mut self, syndrome: &Vec<u8>) -> Vec<u8> {
        self.converge = false;
        let mut converged = false;

        let mut log_prob_ratios_old = vec![0.0; self.bit_count];

        for i in 0..self.bit_count {
            self.initial_log_prob_ratios[i] = ((1.0 - self.channel_probabilities[i]) / self.channel_probabilities[i]).ln();
            self.log_prob_ratios[i] = self.initial_log_prob_ratios[i];
        }

        for it in 1..=self.maximum_iterations {
            if converged {
                continue;
            }

            log_prob_ratios_old.clone_from_slice(&self.log_prob_ratios);

            if it != 1 {
                self.log_prob_ratios.clone_from_slice(&self.initial_log_prob_ratios);
            }

            for i in 0..self.check_count {
                self.candidate_syndrome[i] = 0;

                let mut total_sgn = syndrome[i] as i32;
                let mut temp = f64::MAX;
                let mut bit_to_check_msg;

                for e in self.pcm.base.iterate_row_mut(i) {
                    unsafe {
                        if it == 1 {
                            (*e).inner.check_to_bit_msg = 0.0;
                        }
                        bit_to_check_msg = log_prob_ratios_old[(*e).col_index as usize] - (*e).inner.check_to_bit_msg;
                        if bit_to_check_msg <= 0.0 {
                            total_sgn += 1;
                        }
                        (*e).inner.bit_to_check_msg = temp;
                        let abs_bit_to_check_msg = bit_to_check_msg.abs();
                        if abs_bit_to_check_msg < temp {
                            temp = abs_bit_to_check_msg;
                        }
                    }
                }

                temp = f64::MAX;

                for e in self.pcm.base.reverse_iterate_row_mut(i) {
                    unsafe {
                        let mut sgn = total_sgn;
                        if it == 1 {
                            (*e).inner.check_to_bit_msg = 0.0;
                        }
                        bit_to_check_msg = log_prob_ratios_old[(*e).col_index as usize] - (*e).inner.check_to_bit_msg;
                        if bit_to_check_msg <= 0.0 {
                            sgn += 1;
                        }
                        if temp < (*e).inner.bit_to_check_msg {
                            (*e).inner.bit_to_check_msg = temp;
                        }
                        let message_sign = if sgn % 2 == 0 { 1.0 } else { -1.0 };
                        (*e).inner.check_to_bit_msg = message_sign * self.ms_scaling_factor * (*e).inner.bit_to_check_msg;
                        self.log_prob_ratios[(*e).col_index as usize] += (*e).inner.check_to_bit_msg;

                        let abs_bit_to_check_msg = bit_to_check_msg.abs();
                        if abs_bit_to_check_msg < temp {
                            temp = abs_bit_to_check_msg;
                        }
                    }
                }
            }

            for i in 0..self.bit_count {
                unsafe {
                    if self.log_prob_ratios[i] <= 0.0 {
                        self.decoding[i] = 1;
                        for e in self.pcm.base.iterate_column(i) {
                            self.candidate_syndrome[(*e).row_index as usize] ^= 1;
                        }
                    } else {
                        self.decoding[i] = 0;
                    }
                }
            }

            converged = self.candidate_syndrome == *syndrome;
            self.iterations = it;

            if converged {
                self.converge = true;
                return self.decoding.clone();
            }
        }

        self.converge = converged;
        self.decoding.clone()
    }

    /// Decode with soft information
    pub fn soft_info_decode_serial(&mut self, soft_info_syndrome: &Vec<f64>, cutoff: f64, sigma: f64) -> Vec<u8> {
        let mut syndrome = Vec::with_capacity(self.check_count);
        self.soft_syndrome = soft_info_syndrome.clone();

        for i in 0..self.check_count {
            self.soft_syndrome[i] = 2.0 * self.soft_syndrome[i] / (sigma * sigma);
            syndrome.push(if self.soft_syndrome[i] <= 0.0 { 1 } else { 0 });
        }

        self.converge = false;
        let mut converged = false;

        self.initialise_log_domain_bp();

        for it in 1..=self.maximum_iterations {
            if converged {
                continue;
            }

            if self.random_schedule_at_every_iteration && self.omp_thread_count == 1 {
                // Reorder schedule elements randomly
                self.rng_list_shuffle.shuffle(&mut self.serial_schedule_order);
            }

            for &bit_index in &self.serial_schedule_order {
                self.log_prob_ratios[bit_index] =
                    ((1.0 - self.channel_probabilities[bit_index]) / self.channel_probabilities[bit_index]).ln();

                let mut checks = Vec::new();
                for e in self.pcm.base.iterate_column(bit_index) {
                    unsafe {
                        let check_index = (*e).row_index as usize;
                        let check_messages: Vec<f64> = self
                            .pcm
                            .base
                            .iterate_row(check_index)
                            .filter(|&g| g != e)
                            .map(|g| (*g).inner.bit_to_check_msg)
                            .collect();
                        checks.push((e, check_index, check_messages));
                    }
                }

                for (e, check_index, check_messages) in checks {
                    unsafe {
                        let mut sgn = syndrome[check_index] as i32;
                        let temp = check_messages.iter().map(|&msg| msg.abs()).fold(f64::MAX, f64::min);

                        for &msg in &check_messages {
                            if msg <= 0.0 {
                                sgn ^= 1;
                            }
                        }

                        let soft_syndrome_magnitude = self.soft_syndrome[check_index].abs();
                        let propagated_msg = if soft_syndrome_magnitude < cutoff && soft_syndrome_magnitude < temp {
                            soft_syndrome_magnitude
                        } else {
                            temp
                        };

                        let message_sign = if sgn % 2 == 0 { 1.0 } else { -1.0 };
                        (*e).inner.check_to_bit_msg = self.ms_scaling_factor * message_sign * propagated_msg;
                        (*e).inner.bit_to_check_msg = self.log_prob_ratios[bit_index];
                        self.log_prob_ratios[bit_index] += (*e).inner.check_to_bit_msg;
                    }
                }

                if self.log_prob_ratios[bit_index] <= 0.0 {
                    self.decoding[bit_index] = 1;
                } else {
                    self.decoding[bit_index] = 0;
                }

                let mut temp = 0.0;
                for e in self.pcm.base.reverse_iterate_column_mut(bit_index) {
                    unsafe {
                        (*e).inner.bit_to_check_msg += temp;
                        temp += (*e).inner.check_to_bit_msg;
                    }
                }
            }

            self.candidate_syndrome = self.pcm.mulvec(&self.decoding);
            converged = self.candidate_syndrome == syndrome;
            self.iterations = it;
            if converged {
                self.converge = true;
                break;
            }
        }

        self.decoding.clone()
    }
}
