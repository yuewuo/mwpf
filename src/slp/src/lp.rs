use crate::slp::{Number, Solution};
use rayon::prelude::*;

/// Represents an LP instance.
pub struct LP<N> {
    pub n_constraints: usize,
    pub n_vars: usize,
    pub basic_indices: Vec<usize>,
    pub tableau: Vec<Vec<N>>, // Row major format
}

impl<N: Number> LP<N> {
    pub fn solve(&mut self, parallel: bool) -> Solution<N> {
        if self.is_dual_feasible() {
            return self.dual_simplex(parallel);
        }

        let mut is_b_negative = vec![false; self.n_constraints + 1];
        let no_b_negative = if parallel {
            is_b_negative[1..]
                .par_iter_mut()
                .zip(&self.tableau[1..])
                .map(|(is_b_i_neg, constr)| {
                    if constr[0] < N::zero() {
                        *is_b_i_neg = true;
                        1
                    } else {
                        0
                    }
                })
                .sum()
        } else {
            is_b_negative[1..]
                .iter_mut()
                .zip(&self.tableau[1..])
                .map(|(is_b_i_neg, constr)| {
                    if constr[0] < N::zero() {
                        *is_b_i_neg = true;
                        1
                    } else {
                        0
                    }
                })
                .sum()
        };

        let tot_col = self.tableau[0].len();
        if no_b_negative != 0 {
            let mut auxi_lp = self.create_auxi_lp(is_b_negative, no_b_negative);
            match auxi_lp.simplex(parallel) {
                Solution::Infeasible => return Solution::Infeasible,
                Solution::Unbounded => return Solution::Unbounded,
                Solution::Optimal(obj, _) => {
                    if obj != N::zero() {
                        return Solution::Infeasible;
                    }
                    if parallel {
                        self.tableau[1..=self.n_constraints]
                            .par_iter_mut()
                            .zip(&auxi_lp.tableau[1..=self.n_constraints])
                            .for_each(|(t, a)| {
                                t[..tot_col].clone_from_slice(&a[..tot_col]);
                            });
                        self.basic_indices
                            .par_iter_mut()
                            .zip(&auxi_lp.basic_indices)
                            .for_each(|(b, &a)| {
                                *b = a;
                            });
                    } else {
                        self.tableau[1..=self.n_constraints]
                            .iter_mut()
                            .zip(&auxi_lp.tableau[1..=self.n_constraints])
                            .for_each(|(t, a)| {
                                t[..tot_col].clone_from_slice(&a[..tot_col]);
                            });
                        self.basic_indices
                            .iter_mut()
                            .zip(&auxi_lp.basic_indices)
                            .for_each(|(b, &a)| {
                                *b = a;
                            });
                    }
                    for i in 1..=self.n_constraints {
                        let multipler = self.tableau[0][self.basic_indices[i]].clone();
                        for j in 0..tot_col {
                            let num_to_sub = multipler.clone() * self.tableau[i][j].clone();
                            self.tableau[0][j] -= num_to_sub;
                        }
                    }
                }
            }
        }
        self.simplex(parallel)
    }

    pub fn create_auxi_lp(&self, is_b_negative: Vec<bool>, no_b_negative: usize) -> LP<N> {
        let mut tableau = vec![];
        let tot_col = self.tableau[0].len();

        tableau.push(vec![]);

        let mut curr_neg_index = 1;
        for (i, &is_b_i_neg) in is_b_negative.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let mut row = vec![];
            for j in 0..tot_col {
                row.push(if is_b_i_neg {
                    -self.tableau[i][j].clone()
                } else {
                    self.tableau[i][j].clone()
                });
            }
            for j in 1..=no_b_negative {
                if is_b_i_neg && curr_neg_index == j {
                    row.push(N::one());
                } else {
                    row.push(N::zero());
                }
            }
            if is_b_i_neg {
                curr_neg_index += 1;
            }
            tableau.push(row);
        }

        let mut auxi_obj = vec![N::zero(); tot_col + no_b_negative];
        for j in 1..=self.n_constraints {
            if is_b_negative[j] {
                for (k, v) in auxi_obj.iter_mut().enumerate() {
                    *v -= tableau[j][k].clone();
                }
            }
        }
        for j in 0..no_b_negative {
            auxi_obj[tot_col + j] = N::one();
        }
        tableau[0] = auxi_obj;

        let mut auxi_basic_indices = self.basic_indices.clone();
        let mut curr_neg_index = 0;
        for (j, &v) in is_b_negative.iter().enumerate() {
            if v {
                auxi_basic_indices[j] = tot_col + curr_neg_index;
                curr_neg_index += 1;
            }
        }

        LP {
            n_constraints: self.n_constraints,
            n_vars: self.n_vars + no_b_negative,
            basic_indices: auxi_basic_indices,
            tableau,
        }
    }

    pub fn simplex(&mut self, parallel: bool) -> Solution<N> {
        loop {
            let mut entering_var = 1;
            for (i, v) in self.tableau[0].iter().enumerate() {
                if *v < N::zero() && i != 0 && *v < self.tableau[0][entering_var] {
                    entering_var = i;
                }
            }

            if self.tableau[0][entering_var] >= N::zero() {
                let mut model = vec![];
                for i in 1..=self.n_vars {
                    let mut found = 0;
                    for (j, &v) in self.basic_indices.iter().enumerate() {
                        if i != 0 && i == v {
                            found = j;
                            break;
                        }
                    }
                    if found == 0 {
                        model.push(N::zero());
                    } else {
                        model.push(self.tableau[found][0].clone());
                    }
                }
                break Solution::Optimal(self.tableau[0][0].clone(), model);
            }

            let mut leaving_var = 1;
            for i in 1..=self.n_constraints {
                if self.tableau[i][entering_var] > N::zero()
                    && (self.tableau[leaving_var][entering_var] <= N::zero()
                        || self.tableau[i][0].clone() / self.tableau[i][entering_var].clone()
                            < self.tableau[leaving_var][0].clone()
                                / self.tableau[leaving_var][entering_var].clone())
                {
                    leaving_var = i;
                }
            }

            if self.tableau[leaving_var][entering_var] <= N::zero() {
                break Solution::Unbounded;
            }

            LP::pivot(&mut self.tableau, entering_var, leaving_var, parallel);
            self.basic_indices[leaving_var] = entering_var;
        }
    }

    pub fn dual_simplex(&mut self, parallel: bool) -> Solution<N> {
        loop {
            let mut leaving_var = 1;
            for i in 2..=self.n_constraints {
                if self.tableau[i][0] < self.tableau[leaving_var][0] {
                    leaving_var = i;
                }
            }

            if self.tableau[leaving_var][0] >= N::zero() {
                let mut model = vec![];
                for i in 1..=self.n_vars {
                    let mut found = 0;
                    for (j, &v) in self.basic_indices.iter().enumerate() {
                        if i != 0 && i == v {
                            found = j;
                            break;
                        }
                    }
                    if found == 0 {
                        model.push(N::zero());
                    } else {
                        model.push(self.tableau[found][0].clone());
                    }
                }
                break Solution::Optimal(self.tableau[0][0].clone(), model);
            }

            let mut entering_var = 1;
            for i in 1..self.tableau[0].len() {
                if self.tableau[leaving_var][entering_var] == N::zero() {
                    entering_var = i;
                    continue;
                }
                if self.tableau[leaving_var][i] < N::zero()
                    && (-self.tableau[0][i].clone() / self.tableau[leaving_var][i].clone()
                        < -self.tableau[0][entering_var].clone()
                            / self.tableau[leaving_var][entering_var].clone())
                {
                    entering_var = i;
                }
            }

            if self.tableau[leaving_var][entering_var] >= N::zero() {
                break Solution::Infeasible;
            }

            LP::pivot(&mut self.tableau, entering_var, leaving_var, parallel);
            self.basic_indices[leaving_var] = entering_var;
        }
    }

    pub fn pivot(
        tableau: &mut Vec<Vec<N>>,
        entering_var: usize,
        leaving_var: usize,
        parallel: bool,
    ) {
        let pivot_coeff = tableau[leaving_var][entering_var].clone();
        if parallel {
            tableau[leaving_var].par_iter_mut().for_each(|v| {
                *v /= pivot_coeff.clone();
            });
        } else {
            tableau[leaving_var].iter_mut().for_each(|v| {
                *v /= pivot_coeff.clone();
            });
        }
        for k in 0..tableau.len() {
            if k != leaving_var {
                let multiplier = tableau[k][entering_var].clone();
                for i in 0..tableau[k].len() {
                    let num_to_sub = multiplier.clone() * tableau[leaving_var][i].clone();
                    tableau[k][i] -= num_to_sub;
                }
            }
        }
    }

    pub fn is_dual_feasible(&self) -> bool {
        for v in &self.tableau[0] {
            if *v < N::zero() {
                return false;
            }
        }
        true
    }
}