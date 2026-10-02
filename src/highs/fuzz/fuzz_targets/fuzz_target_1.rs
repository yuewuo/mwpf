#![no_main]
use mwpf::highs::{RowProblem, Sense};
use libfuzzer_sys::arbitrary;
use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::ops::Range;

#[derive(Arbitrary)]
struct ColData {
    val: f64,
    range: Range<f64>,
    integrality: bool,
}

fn test(u: &mut Unstructured) -> arbitrary::Result<()> {
    let mut pb = RowProblem::default();
    let vars = u
        .arbitrary_iter::<ColData>()?
        .map(|cd| {
            let cd = cd?;
            Ok(pb.add_column_with_integrality(cd.val, cd.range, cd.integrality))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let num_rows = u.arbitrary::<u8>()? as usize;

    for _ in 0..num_rows {
        let range = u.arbitrary::<Range<f64>>()?;
        pb.add_row(
            range,
            &[
                (*u.choose(&vars)?, u.arbitrary()?),
                (*u.choose(&vars)?, u.arbitrary()?),
                (*u.choose(&vars)?, u.arbitrary()?),
            ],
        );
    }
    if let Ok(solved) = pb
        .try_optimise(*u.choose(&[Sense::Maximise, Sense::Minimise])?)
        .and_then(|p| p.try_solve())
    {
        let solution = solved.get_solution();
        assert_eq!(solution.columns().len(), vars.len());
        assert_eq!(solution.rows().len(), num_rows);
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let _ = test(&mut u);
});
