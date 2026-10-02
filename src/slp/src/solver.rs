use crate::slp::lp::*;
use crate::slp::parser::{LpProblem, ObjectiveType};
use crate::slp::{Number, Solution, SolverOptions, SolverSettings};

/// Linear Programming Solver.
pub struct Solver<N> {
    lp: LP<N>,
    options: SolverOptions,
    is_int_constraints: Vec<bool>,
    negate_objective: bool,
}

impl<N: Number> Solver<N> {
    /// Creates a new Solver instance from mprog input format.
    pub fn new(input: &str) -> Solver<N>
    where
        N::Err: std::fmt::Debug,
    {
        crate::slp::parser::parse_lp_problem(input).unwrap().into()
    }

    /// Creates a new Solver instance with integer constraints.
    pub(crate) fn new_with_int_constraints(
        lp: LP<N>,
        is_int_constraints: Vec<bool>,
        negate_objective: bool,
    ) -> Self {
        Solver {
            lp,
            options: SolverOptions { parallel: false },
            is_int_constraints,
            negate_objective,
        }
    }

    /// Enable a setting.
    pub fn setting(&mut self, setting: SolverSettings) {
        match setting {
            SolverSettings::EnableDataParallelism => self.options.parallel = true,
        }
    }

    /// Solves the LP.
    ///
    /// Uses naive version of simplex method.
    ///
    /// Returns [a solution](enum.Solution.html).
    pub fn solve(&mut self) -> Solution<N> {
        match self.lp.solve(self.options.parallel) {
            Solution::Infeasible => Solution::Infeasible,
            Solution::Unbounded => Solution::Unbounded,
            Solution::Optimal(opt, model) => {
                let solution = Self::branch_and_bound(
                    &self.lp,
                    self.options.parallel,
                    opt,
                    model,
                    &self.is_int_constraints,
                    None,
                );
                if let Solution::Optimal(opt, model) = solution {
                    if self.negate_objective {
                        Solution::Optimal(-opt, model)
                    } else {
                        Solution::Optimal(opt, model)
                    }
                } else {
                    solution
                }
            }
        }
    }

    fn branch_and_bound(
        lp: &LP<N>,
        parallel: bool,
        lp_opt: N,
        model: Vec<N>,
        is_int_constraints: &[bool],
        mut known_opt: Option<N>,
    ) -> Solution<N> {
        let mut non_int_index = 0;
        for (i, v) in model.iter().enumerate() {
            if is_int_constraints[i] && !v.is_integer() {
                non_int_index = i + 1;
                break;
            }
        }
        if non_int_index == 0 {
            return Solution::Optimal(lp_opt, model);
        }

        let mut basic_index = 0;
        for i in 1..=lp.n_constraints {
            if lp.basic_indices[i] == non_int_index {
                basic_index = i;
                break;
            }
        }

        let mut tableau = lp.tableau.clone();
        for row in &mut tableau {
            row.push(N::zero());
        }
        let mut new_constr = vec![N::zero(); tableau[0].len()];
        new_constr[non_int_index] = N::one();
        new_constr[0] = model[non_int_index - 1].floor();
        new_constr[tableau[0].len() - 1] = N::one();
        if basic_index != 0 {
            for (i, v) in new_constr.iter_mut().enumerate() {
                *v -= tableau[basic_index][i].clone();
            }
        }
        tableau.push(new_constr);
        let mut basic_indices = lp.basic_indices.clone();
        basic_indices.push(tableau[0].len() - 1);

        let mut new_lp = LP {
            n_constraints: lp.n_constraints + 1,
            n_vars: lp.n_vars,
            tableau,
            basic_indices,
        };

        let sol1 = new_lp.dual_simplex(parallel);
        let sol1_int = match sol1 {
            Solution::Infeasible => Solution::Infeasible,
            Solution::Unbounded => Solution::Unbounded,
            Solution::Optimal(opt, model) => Self::branch_and_bound(
                &new_lp,
                parallel,
                opt,
                model,
                is_int_constraints,
                known_opt.clone(),
            ),
        };

        if let Solution::Optimal(opt, _) = &sol1_int {
            known_opt = match known_opt {
                None => Some(opt.clone()),
                Some(k_opt) => Some(if k_opt > *opt { k_opt } else { opt.clone() }),
            };
        }

        tableau = lp.tableau.clone();
        for row in &mut tableau {
            row.push(N::zero());
        }
        let mut new_constr = vec![N::zero(); tableau[0].len()];
        new_constr[non_int_index] = -N::one();
        new_constr[0] = -model[non_int_index - 1].ceil();
        new_constr[tableau[0].len() - 1] = N::one();
        if basic_index != 0 {
            for (i, v) in new_constr.iter_mut().enumerate() {
                *v += tableau[basic_index][i].clone();
            }
        }
        tableau.push(new_constr);
        basic_indices = lp.basic_indices.clone();
        basic_indices.push(tableau[0].len() - 1);

        let mut new_lp = LP {
            n_constraints: lp.n_constraints + 1,
            n_vars: lp.n_vars,
            tableau,
            basic_indices,
        };
        let sol2 = new_lp.dual_simplex(parallel);
        let sol2_int = match sol2 {
            Solution::Infeasible => Solution::Infeasible,
            Solution::Unbounded => Solution::Unbounded,
            Solution::Optimal(opt, model) => {
                Self::branch_and_bound(&new_lp, parallel, opt, model, is_int_constraints, known_opt)
            }
        };

        match (sol1_int, sol2_int) {
            (Solution::Infeasible, Solution::Infeasible) => Solution::Infeasible,
            (Solution::Unbounded, _) | (_, Solution::Unbounded) => Solution::Unbounded,
            (Solution::Optimal(opt1, model1), Solution::Optimal(opt2, model2)) => {
                if opt1 > opt2 {
                    Solution::Optimal(opt1, model1)
                } else {
                    Solution::Optimal(opt2, model2)
                }
            }
            (Solution::Optimal(opt, model), _) => Solution::Optimal(opt, model),
            (_, Solution::Optimal(opt, model)) => Solution::Optimal(opt, model),
        }
    }
}

impl<N: Number> From<LpProblem<N>> for Solver<N> {
    fn from(mut lp_problem: LpProblem<N>) -> Self {
        let mut tableau = vec![];
        let mut basic_indices = vec![0];
        let n_constraints = lp_problem.constraints.len();
        let n_vars = lp_problem.vars_list.len();
        let mut obj = lp_problem.objective;
        for i in obj.iter_mut() {
            *i = -i.clone();
        }
        obj.insert(0, N::zero());
        for _ in 0..n_constraints {
            obj.push(N::zero());
        }
        tableau.push(obj);
        for (i, constr) in lp_problem.constraints.iter_mut().enumerate() {
            constr.0.insert(0, constr.1.clone());
            for j in 0..n_constraints {
                constr.0.push(if i == j { N::one() } else { N::zero() });
            }
            // TODO Remove clone
            tableau.push(constr.0.clone());
            basic_indices.push(n_vars + i + 1);
        }

        let lp = LP {
            n_constraints,
            n_vars,
            basic_indices,
            tableau,
        };
        Solver::new_with_int_constraints(
            lp,
            lp_problem.is_int_constraints,
            lp_problem.objective_type == ObjectiveType::Min,
        )
    }
}