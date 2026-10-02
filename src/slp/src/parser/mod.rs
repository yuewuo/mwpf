use crate::slp::Number;
use pest::error::Error;
use pest::Parser;

mod lp_parser {
    #[derive(pest_derive::Parser)]
    #[grammar = "src/slp/src/parser/grammar.pest"]
    pub struct LpParser;
}

use lp_parser::*;

/// LP Problem instance.
pub struct LpProblem<N> {
    /// Variables list.
    pub(crate) vars_list: Vec<String>,
    /// ith value is true if ith variable has insteger constraint.
    pub(crate) is_int_constraints: Vec<bool>,
    /// Constraints.
    pub(crate) constraints: Vec<(Vec<N>, N)>,
    /// Objective to be maximized.
    pub(crate) objective: Vec<N>,
    /// Objective type.
    pub(crate) objective_type: ObjectiveType,
}

#[derive(PartialEq)]
enum OperatorType {
    LtEq,
    GtEq,
}

/// Objective type.
#[derive(PartialEq)]
pub enum ObjectiveType {
    /// Maximize.
    Max,
    /// Minimize.
    Min,
}

enum AstNode<N> {
    Lp {
        objective: Vec<N>,
        constraints: Vec<(Vec<N>, N)>,
    },
    Variable(String),
    VariableInt(String),
    Objective(Vec<N>),
    Constraints(Vec<(Vec<N>, N)>),
    Constraint(Vec<N>, N),
    Expression(Vec<N>),
    Term(N, usize),
    Number(N),
    None,
}

struct AstInternal {
    n_vars: usize,
    variables: Vec<String>,
    is_int_constraints: Vec<bool>,
    objective_type: ObjectiveType,
}

/// Parses LP Problem from given input.
pub fn parse_lp_problem<N>(input: &str) -> Result<LpProblem<N>, Error<Rule>>
where
    N: Number,
    N::Err: std::fmt::Debug,
{
    let lp_problem = LpParser::parse(Rule::lp_problem, input)?.next().unwrap();
    use pest::iterators::Pair;

    let mut internal = AstInternal {
        n_vars: 0,
        variables: vec![],
        is_int_constraints: vec![],
        objective_type: ObjectiveType::Max,
    };

    fn parse_pair<N>(pair: Pair<Rule>, internal: &mut AstInternal) -> AstNode<N>
    where
        N: Number,
        N::Err: std::fmt::Debug,
    {
        match pair.as_rule() {
            Rule::lp_problem => {
                let mut inner_pairs = pair.into_inner();
                let vars_rule = inner_pairs.next().unwrap();
                parse_pair::<N>(vars_rule, internal);
                let obj_rule = inner_pairs.next().unwrap();
                let obj = parse_pair(obj_rule, internal);
                let cons_rule = inner_pairs.next().unwrap();
                let cons = parse_pair(cons_rule, internal);
                AstNode::Lp {
                    objective: if let AstNode::Objective(vs) = obj {
                        vs
                    } else {
                        unreachable!()
                    },
                    constraints: if let AstNode::Constraints(cs) = cons {
                        cs
                    } else {
                        unreachable!()
                    },
                }
            }
            Rule::variables => {
                let mut vars = vec![];
                let mut is_int_constraints = vec![];
                for rule in pair.into_inner() {
                    match parse_pair::<N>(rule, internal) {
                        AstNode::Variable(var) => {
                            vars.push(var);
                            is_int_constraints.push(false);
                        }
                        AstNode::VariableInt(var) => {
                            vars.push(var);
                            is_int_constraints.push(true);
                        }
                        _ => unreachable!(),
                    }
                }
                internal.n_vars = vars.len();
                internal.variables = vars;
                internal.is_int_constraints = is_int_constraints;
                AstNode::None
            }
            Rule::variable_real => {
                let var = pair.into_inner().next().unwrap().as_str();
                AstNode::Variable(var.to_string())
            }
            Rule::variable_int => {
                let var = pair.into_inner().next().unwrap().as_str();
                AstNode::VariableInt(var.to_string())
            }
            Rule::variable => parse_pair(pair.into_inner().next().unwrap(), internal),
            Rule::max_objective => {
                internal.objective_type = ObjectiveType::Max;
                let exp = pair.into_inner().next().unwrap();
                if let AstNode::Expression(exp) = parse_pair(exp, internal) {
                    AstNode::Objective(exp)
                } else {
                    unreachable!()
                }
            }
            Rule::min_objective => {
                let exp = pair.into_inner().next().unwrap();
                internal.objective_type = ObjectiveType::Min;
                if let AstNode::Expression(mut exp) = parse_pair::<N>(exp, internal) {
                    for v in exp.iter_mut() {
                        *v = -v.clone();
                    }
                    AstNode::Objective(exp)
                } else {
                    unreachable!()
                }
            }
            Rule::objective => parse_pair(pair.into_inner().next().unwrap(), internal),
            Rule::constraints => {
                let mut cons = vec![];
                for rule in pair.into_inner() {
                    if let AstNode::Constraint(exp, rhs) = parse_pair(rule, internal) {
                        cons.push((exp, rhs));
                    } else {
                        unreachable!()
                    }
                }
                AstNode::Constraints(cons)
            }
            Rule::constraint => {
                let mut inner_pairs = pair.into_inner();
                let exp_rule = inner_pairs.next().unwrap();
                let exp = parse_pair::<N>(exp_rule, internal);
                let opr_rule = inner_pairs.next().unwrap();
                let oper = match opr_rule.as_str() {
                    "<=" => OperatorType::LtEq,
                    ">=" => OperatorType::GtEq,
                    _ => unreachable!(),
                };
                let rhs_rule = inner_pairs.next().unwrap();
                let rhs = parse_pair::<N>(rhs_rule, internal);
                match (exp, rhs) {
                    (AstNode::Expression(mut exp), AstNode::Number(mut rhs)) => {
                        if oper == OperatorType::GtEq {
                            for t in exp.iter_mut() {
                                *t = -t.clone();
                            }
                            rhs = -rhs;
                        }
                        AstNode::Constraint(exp, rhs)
                    }
                    _ => unreachable!(),
                }
            }
            Rule::expression => {
                let mut terms = vec![N::zero(); internal.n_vars];
                for rule in pair.into_inner() {
                    if let AstNode::Term(r, i) = parse_pair(rule, internal) {
                        terms[i] = r;
                    } else {
                        unreachable!();
                    }
                }
                AstNode::Expression(terms)
            }
            Rule::signed_term => parse_pair(pair.into_inner().next().unwrap(), internal),
            Rule::pos_signed_term => parse_pair(pair.into_inner().next().unwrap(), internal),
            Rule::neg_signed_term => {
                let term = parse_pair::<N>(pair.into_inner().next().unwrap(), internal);
                if let AstNode::Term(r, i) = term {
                    AstNode::Term(-r, i)
                } else {
                    unreachable!()
                }
            }
            Rule::unsigned_term => {
                let mut inner_pairs = pair.into_inner();
                let coeff_rule = inner_pairs.next().unwrap();
                if let AstNode::Number(r) = parse_pair(coeff_rule, internal) {
                    let var = inner_pairs.next().unwrap().as_str();
                    let mut index = internal.variables.len();
                    for (i, v) in internal.variables.iter().enumerate() {
                        if v == var {
                            index = i;
                            break;
                        }
                    }
                    if index == internal.variables.len() {
                        panic!("Unknown identifier {}", var);
                    }
                    AstNode::Term(r, index)
                } else {
                    unreachable!()
                }
            }
            Rule::coefficient => {
                let rule = pair.into_inner().next();
                match rule {
                    Some(rule) => parse_pair(rule, internal),
                    None => AstNode::Number(N::one()),
                }
            }
            Rule::number | Rule::pos_number | Rule::neg_number => {
                AstNode::Number(pair.as_str().parse().unwrap())
            }
            _ => AstNode::None,
        }
    }

    let parsed = parse_pair(lp_problem, &mut internal);

    match parsed {
        AstNode::Lp {
            constraints,
            objective,
        } => Ok(LpProblem {
            vars_list: internal.variables,
            is_int_constraints: internal.is_int_constraints,
            constraints,
            objective,
            objective_type: internal.objective_type,
        }),
        _ => unreachable!(),
    }
}