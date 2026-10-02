use criterion::{black_box, criterion_group, criterion_main, BatchSize, Bencher, Criterion};
use mwpf::heapz::{Heap, PairingHeap};

fn is_empty_benchmark(b: &mut Bencher) {
    let mut heap = PairingHeap::min();
    heap.push(black_box(1), black_box(1));
    b.iter(|| heap.is_empty());
}

fn size_benchmark(b: &mut Bencher) {
    let mut heap = PairingHeap::min();
    heap.push(1, 1);
    b.iter(|| heap.size());
}

fn push_benchmark(b: &mut Bencher) {
    let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
    b.iter_batched(
        || PairingHeap::<i32, i32>::min(),
        |mut heap| arr.iter().for_each(|num| heap.push(black_box(*num), black_box(*num))),
        BatchSize::SmallInput,
    );
}

fn top_benchmark(b: &mut Bencher) {
    let mut heap = PairingHeap::min();
    heap.push(1, 1);
    b.iter(|| {
        let _ = heap.top();
    });
}

pub fn top_mut_benchmark(b: &mut Bencher) {
    let mut heap = PairingHeap::min();
    heap.push(1, 1);
    b.iter(|| {
        let _ = heap.top_mut();
    });
}

pub fn pop_benchmark(b: &mut Bencher) {
    b.iter_batched(
        || {
            let arr = vec![1, 3, 5, -2, 6, -7, 9, 10, 13, 4, 12, 115, 500, 132, 67, 334];
            let mut heap = PairingHeap::min();
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

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("PairingHeap.is_empty", is_empty_benchmark);
    c.bench_function("PairingHeap.size", size_benchmark);
    c.bench_function("PairingHeap.push", push_benchmark);
    c.bench_function("PairingHeap.top", top_benchmark);
    c.bench_function("PairingHeap.top_mut", top_mut_benchmark);
    c.bench_function("PairingHeap.pop", pop_benchmark);
}

criterion_group!(pairing_heap, criterion_benchmark);
criterion_main!(pairing_heap);
