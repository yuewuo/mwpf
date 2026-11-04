use super::{Model, RowProblem};
use highs_sys::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type SparseRow = (f64, f64, Vec<usize>, Vec<f64>);
type SolveResult = (i32, i32, bool, bool, f64, f64, Vec<f64>);

fn check_status(status: HighsInt, operation: &str) -> PyResult<()> {
    match status {
        STATUS_OK => Ok(()),
        STATUS_WARNING => {
            log::warn!("HiGHS emitted a warning: {operation}");
            Ok(())
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "HiGHS {operation} failed with status {status}"
        ))),
    }
}

fn check_bounds(lower: f64, upper: f64) -> PyResult<()> {
    if lower.is_nan() || upper.is_nan() || lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
        return Err(PyValueError::new_err("invalid HiGHS bounds"));
    }
    Ok(())
}

#[pyfunction]
fn _highs_solve(
    py: Python<'_>,
    costs: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    rows: Vec<SparseRow>,
    time_limit: Option<f64>,
    output_flag: bool,
    mip_rel_gap: f64,
    mip_abs_gap: f64,
) -> PyResult<SolveResult> {
    if costs.len() != lower.len() || costs.len() != upper.len() {
        return Err(PyValueError::new_err("column array lengths must match"));
    }
    let nonzeros = rows.iter().try_fold(0usize, |total, row| total.checked_add(row.2.len()));
    if HighsInt::try_from(costs.len()).is_err()
        || HighsInt::try_from(rows.len()).is_err()
        || nonzeros.and_then(|count| HighsInt::try_from(count).ok()).is_none()
    {
        return Err(PyValueError::new_err("model exceeds HiGHS index limits"));
    }
    for ((&cost, &low), &high) in costs.iter().zip(&lower).zip(&upper) {
        if !cost.is_finite() {
            return Err(PyValueError::new_err("objective coefficients must be finite"));
        }
        check_bounds(low, high)?;
    }
    for (low, high, indices, values) in &rows {
        check_bounds(*low, *high)?;
        if indices.len() != values.len()
            || indices.iter().any(|&index| index >= costs.len())
            || values.iter().any(|value| !value.is_finite())
            || indices.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != indices.len()
        {
            return Err(PyValueError::new_err("invalid sparse row indices or coefficients"));
        }
    }
    if time_limit.is_some_and(|value| !value.is_finite() || value < 0.)
        || !mip_rel_gap.is_finite()
        || mip_rel_gap < 0.
        || !mip_abs_gap.is_finite()
        || mip_abs_gap < 0.
    {
        return Err(PyValueError::new_err("solver limits and gaps must be finite and nonnegative"));
    }

    py.detach(move || {
        let mut problem = RowProblem::new();
        let columns: Vec<_> = costs
            .into_iter()
            .zip(lower)
            .zip(upper)
            .map(|((cost, low), high)| problem.add_integer_column(cost, low..=high))
            .collect();
        for (low, high, indices, values) in rows {
            problem.add_row(
                low..=high,
                indices.into_iter().zip(values).map(|(index, value)| (columns[index], value)),
            );
        }
        let mut model = Model::try_new(problem)
            .map_err(|status| PyRuntimeError::new_err(format!("HiGHS rejected model: {status:?}")))?;
        let ptr = model.highs.mut_ptr();
        // Match MWPF's own solver settings; do not reset the shared global scheduler.
        unsafe {
            check_status(Highs_setIntOptionValue(ptr, c"threads".as_ptr(), 1), "setting threads")?;
            check_status(
                Highs_setStringOptionValue(ptr, c"parallel".as_ptr(), c"off".as_ptr()),
                "disabling parallel search",
            )?;
            check_status(
                Highs_setBoolOptionValue(ptr, c"output_flag".as_ptr(), i32::from(output_flag)),
                "setting output_flag",
            )?;
            check_status(
                Highs_setBoolOptionValue(ptr, c"log_to_console".as_ptr(), i32::from(output_flag)),
                "setting log_to_console",
            )?;
            check_status(
                Highs_setDoubleOptionValue(ptr, c"mip_rel_gap".as_ptr(), mip_rel_gap),
                "setting mip_rel_gap",
            )?;
            check_status(
                Highs_setDoubleOptionValue(ptr, c"mip_abs_gap".as_ptr(), mip_abs_gap),
                "setting mip_abs_gap",
            )?;
            if let Some(limit) = time_limit {
                check_status(
                    Highs_setDoubleOptionValue(ptr, c"time_limit".as_ptr(), limit),
                    "setting time_limit",
                )?;
            }
            let run_status = Highs_run(ptr);
            check_status(run_status, "run")?;
            let model_status = Highs_getModelStatus(ptr);
            let mut primal_status = 0;
            let info_status = Highs_getIntInfoValue(ptr, c"primal_solution_status".as_ptr(), &mut primal_status);
            check_status(info_status, "reading primal_solution_status")?;
            let info_valid = info_status == STATUS_OK;
            let value_valid = info_valid && primal_status == SOLUTION_STATUS_FEASIBLE;
            let mut dual_bound = f64::NAN;
            let mut objective = f64::NAN;
            if info_valid {
                check_status(
                    Highs_getDoubleInfoValue(ptr, c"mip_dual_bound".as_ptr(), &mut dual_bound),
                    "reading mip_dual_bound",
                )?;
                objective = Highs_getObjectiveValue(ptr);
            }
            let mut values = Vec::new();
            if value_valid {
                values.resize(columns.len(), 0.);
                check_status(
                    Highs_getSolution(
                        ptr,
                        values.as_mut_ptr(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    ),
                    "reading solution",
                )?;
            }
            Ok((
                run_status,
                model_status,
                info_valid,
                value_valid,
                dual_bound,
                objective,
                values,
            ))
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_highs_solve, module)?)?;
    Ok(())
}
