use criterion::{black_box, criterion_group, criterion_main, BatchSize, Bencher, Criterion};
use mwpf::heapz::{DecreaseKey, Heap, RankPairingHeap};

fn is_empty_benchmark(b: &mut Bencher) {
    let mut heap = RankPairingHeap::multi_pass_min();
    heap.push(black_box(1), black_box(1));
    b.iter(|| heap.is_empty());
}

fn size_benchmark(b: &mut Bencher) {
    let mut heap = RankPairingHeap::multi_pass_min();
    heap.push(1, 1);
    b.iter(|| heap.size());
}

fn push_benchmark(b: &mut Bencher) {
    let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
    b.iter_batched(
        || RankPairingHeap::<i32, i32>::multi_pass_min(),
        |mut heap| arr.iter().for_each(|num| heap.push(black_box(*num), black_box(*num))),
        BatchSize::SmallInput,
    );
}

fn top_benchmark(b: &mut Bencher) {
    let mut heap = RankPairingHeap::multi_pass_min();
    heap.push(1, 1);
    b.iter(|| {
        let _ = heap.top();
    });
}

fn top_mut_benchmark(b: &mut Bencher) {
    let mut heap = RankPairingHeap::multi_pass_min();
    heap.push(1, 1);
    b.iter(|| {
        let _ = heap.top_mut();
    });
}

fn pop_benchmark(b: &mut Bencher) {
    b.iter_batched(
        || {
            let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
            let mut heap = RankPairingHeap::multi_pass_min();
            arr.iter().for_each(|num| heap.push(black_box(*num), black_box(*num)));
            (heap, arr.len())
        },
        |(mut heap, len)| {
            for _ in 0..len {
                let _ = heap.pop();
            }
        },
        BatchSize::SmallInput,
    );
}

fn update_benchmark(b: &mut Bencher) {
    let mut i = 0;
    b.iter_batched(
        || {
            let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
            let mut heap = RankPairingHeap::multi_pass_min();
            let key = arr[(i % arr.len()) as usize];
            let value = if i % 2 == 0 { -1 } else { 2 };
            arr.iter().for_each(|num| heap.push(black_box(*num), black_box(*num)));
            i += 1;
            (heap, (key, value))
        },
        |(mut heap, (key, value))| heap.update(&key, value),
        BatchSize::SmallInput,
    );
}

fn delete_benchmark(b: &mut Bencher) {
    let mut i = 0;
    b.iter_batched(
        || {
            let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
            let mut heap = RankPairingHeap::multi_pass_min();
            let key = arr[(i % arr.len()) as usize];
            arr.iter().for_each(|num| heap.push(black_box(*num), black_box(*num)));
            i += 1;
            (heap, key)
        },
        |(mut heap, key)| {
            let _ = heap.delete(&key);
        },
        BatchSize::SmallInput,
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("RankPairingHeap.is_empty", is_empty_benchmark);
    c.bench_function("RankPairingHeap.size", size_benchmark);
    c.bench_function("RankPairingHeap.push", push_benchmark);
    c.bench_function("RankPairingHeap.top", top_benchmark);
    c.bench_function("RankPairingHeap.top_mut", top_mut_benchmark);
    c.bench_function("RankPairingHeap.pop", pop_benchmark);
    c.bench_function("RankPairingHeap.update", update_benchmark);
    c.bench_function("RankPairingHeap.delete", delete_benchmark);
}

criterion_group!(rank_pairing_heap, criterion_benchmark);
criterion_main!(rank_pairing_heap);
