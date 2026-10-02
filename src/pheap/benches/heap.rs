use criterion::{criterion_group, criterion_main, Criterion};
use keyed_priority_queue::KeyedPriorityQueue;
use mwpf::pheap::PairingHeap;
use priority_queue::PriorityQueue;

enum Queue {
    PairingHeap(PairingHeap<i32, i32>),
    AdPairingHeap(addressable_pairing_heap::PairingHeap<i32, i32>),
    A422PairingHeap(pairing_heap::PairingHeap<i32>),
    PriorityQueue(PriorityQueue<i32, i32>),
    KeyedPriorityQueue(KeyedPriorityQueue<i32, i32>),
}

impl Queue {
    fn insert(&mut self, key: i32) {
        match self {
            Queue::PairingHeap(ph) => ph.insert(key, key),
            Queue::AdPairingHeap(ph) => {
                ph.push(key, key);
            }
            Queue::PriorityQueue(pq) => {
                pq.push(key, key);
            }
            Queue::A422PairingHeap(ph) => {
                ph.push(key);
            }
            Queue::KeyedPriorityQueue(pq) => {
                pq.push(key, key);
            }
        }
    }

    fn update(&mut self, key: i32, delta: i32) {
        match self {
            Queue::PairingHeap(ph) => ph.decrease_prio(&key, delta),
            Queue::AdPairingHeap(_) => unimplemented!(),
            Queue::PriorityQueue(pq) => {
                if let Some(q) = pq.get_priority(&key).cloned() {
                    pq.change_priority(&key, q - 10);
                }
            }
            Queue::A422PairingHeap(_) => unimplemented!(),
            Queue::KeyedPriorityQueue(pq) => {
                if let Some(q) = pq.get_priority(&key).cloned() {
                    pq.set_priority(&key, q - 10).unwrap();
                }
            }
        }
    }

    fn delete(&mut self) {
        match self {
            Queue::PairingHeap(ph) => {
                ph.delete_min();
            }
            Queue::AdPairingHeap(ph) => {
                ph.pop();
            }
            Queue::PriorityQueue(pq) => {
                pq.pop();
            }
            Queue::A422PairingHeap(ph) => {
                ph.pop();
            }
            Queue::KeyedPriorityQueue(pq) => {
                pq.pop();
            }
        };
    }
}

enum QueueKind {
    PairingHeap,
    AdPairingHeap,
    A422PairingHeap,
    PriorityQueue,
    KeyedPriorityQueue,
}

impl QueueKind {
    fn create(&self) -> Queue {
        match self {
            QueueKind::PairingHeap => Queue::PairingHeap(PairingHeap::<i32, i32>::new()),
            QueueKind::AdPairingHeap => Queue::AdPairingHeap(addressable_pairing_heap::PairingHeap::<i32, i32>::new()),
            QueueKind::PriorityQueue => Queue::PriorityQueue(PriorityQueue::<i32, i32>::new()),
            QueueKind::A422PairingHeap => Queue::A422PairingHeap(pairing_heap::PairingHeap::<i32>::new()),
            QueueKind::KeyedPriorityQueue => Queue::KeyedPriorityQueue(KeyedPriorityQueue::<i32, i32>::new()),
        }
    }

    fn bench(&self, op: i32, update: bool) {
        let mut insert_op = op + 1;
        let mut update_op = insert_op;
        let mut delete_op = 0;
        let mut count = 0;

        let mut q = self.create();

        while insert_op > -1 {
            for _ in 0..insert_op {
                q.insert(count);
                count += 1;
            }

            if update {
                for ii in 0..update_op {
                    q.update(count - ii, 10);
                }
            }

            for _ in 0..delete_op {
                q.delete();
            }

            insert_op -= 1;
            update_op -= 1;
            delete_op += 1;
        }
    }
}

fn no_change_prio(c: &mut Criterion) {
    c.bench_function("Pairing Heap (1000)", |b| {
        b.iter(|| QueueKind::PairingHeap.bench(1_000, false))
    });
    c.bench_function("Addressable Pairing Heap (1000)", |b| {
        b.iter(|| QueueKind::AdPairingHeap.bench(1_000, false))
    });
    c.bench_function("APasel422's Pairing Heap (1000)", |b| {
        b.iter(|| QueueKind::A422PairingHeap.bench(1_000, false))
    });
    c.bench_function("Priority Queue (1000)", |b| {
        b.iter(|| QueueKind::PriorityQueue.bench(1_000, false))
    });
    c.bench_function("Keyed Priority Queue (1000)", |b| {
        b.iter(|| QueueKind::KeyedPriorityQueue.bench(1_000, false))
    });
}

fn with_change_prio(c: &mut Criterion) {
    c.bench_function("Pairing Heap (1000) | Priority Update", |b| {
        b.iter(|| QueueKind::PairingHeap.bench(1_000, true))
    });
    c.bench_function("Priority Queue (1000) | Priority Update", |b| {
        b.iter(|| QueueKind::PriorityQueue.bench(1_000, true))
    });
    c.bench_function("Keyed Priority Queue (1000) | Priority Update", |b| {
        b.iter(|| QueueKind::KeyedPriorityQueue.bench(1_000, true))
    });
}

criterion_group!(benches, no_change_prio, with_change_prio);
criterion_main!(benches);
