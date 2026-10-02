# Pairing Heap

This modified copy is built as `mwpf::pheap`, not as a separate crate. Original
package metadata, including the upstream author and license declaration, is
retained in `Cargo.toml.orig`. Run its tests from the repository root with
`cargo test --lib pheap::` and its benchmarks with `cargo bench --bench pheap`.
The graph examples are now `pheap_dijkstra`, `pheap_mst`, and `pheap_stress`;
the historical benchmark results and standalone helper scripts below describe
the upstream package layout.

[![Crates.io](https://img.shields.io/crates/v/pheap)](https://crates.io/crates/pheap) [![Documentation](https://docs.rs/pheap/badge.svg)](https://docs.rs/pheap) 

From [Wikipedia](https://en.wikipedia.org/wiki/Pairing_heap):
> A pairing heap is a type of heap data structure with relatively simple implementation and excellent practical amortized performance.
> Pairing heaps are heap-ordered multiway tree structures, and can be considered simplified Fibonacci heaps. They are considered a "robust choice" for implementing such algorithms as Prim's MST algorithm.

A min-pairing heap supports the following operations:
- ```find_min```: finds the minimum element of the heap, which is the root.
- ```merge```: combines two heaps together.
- ```insert```: adds a new element into the heap.
- ```delete_min```: remove the root and reorder its children nodes.
- ```decrease_key```: decrease the priority of an element. Standard implementation of a heap data structure does not support searching for a key efficiently (which is the case in this crate). Thus, this operation can take very long time, with an upper bound of ```O(2^(sqrt(log log n)))```.

The crate also comes with an efficient implementation of Dijkstra's algorithm to solve the single source shortest path problem and Prim's algorithm for finding minimum spanning tree.

## Benchmarks
To measure the performance of this implementation, I choose the following libraries that are available on [crates.io](https://crates.io/) to experiment:
- [Addressable pairing heap](https://crates.io/crates/addressable-pairing-heap)
- [Apasel422's implementation of pairing heap](https://crates.io/crates/pairing-heap)
- [Priority queue](https://crates.io/crates/priority-queue)
- [Keyed priority queue](https://crates.io/crates/keyed_priority_queue)

*If I miss any libraries, please let me know.*

The experiments are conducted on my PC with the following spec:
> OS: Fedora 34 64-bit  
> CPU: AMD® Ryzen 7 3800x 8-core processor   
> RAM: 32 GB  

### Experiment 1
> Each implementation is tasked to execute 1000 insertions / 0 deletes, then 999 insertions / 1 deletes (remove the top element), until the number of deletes is 1000. This means each implementation has to execute 500_500 insertions and 500_500 deletions.

For this experiment, I use the crate [```criterion```](https://crates.io/crates/criterion) to measure the performance of each implementation. 

|  | Pairing heap<br>(this crate)  | Addressable pairing heap | Pairing heap <br>(Apasel422) | Priority queue | Keyed priority queue
--- | --- | --- | --- | --- | ---
|Average time<br>(milliseconds)|20.37|56.6|24.18|116.84|111.30|

### Experiment 2
> Each implementation is tasked to execute 1000 insertions / 1000 priority update / 0 deletes, then 999 insertions / 999 priority updates | 1 deletes (remove the top element), until the number of deletes is 1000.

|  | Pairing heap<br>(this crate)  | Addressable pairing heap | Pairing heap <br>(Apasel422) | Priority queue | Keyed priority queue
--- | --- | --- | --- | --- | ---
|Average time<br>(seconds)|1.399|No implementation|No implementation|0.171|0.142|

For this experiment, the pairing heap fairs worse than other two libraries. This is due to the fact that pairing heap data structures must search for keys, which in worse cases takes ```O(n)``` time, while other implementations leverage the fast lookup power from hash map.

### Experiment 3
> Each implementation is tasked to insert 1 million elements and the memory consumption will be measured.

For this experiment, I write a simple ```main``` (in ```examples/stress.rs```) and use ```valgrind``` with ```massif``` for the evaluation purpose.

To compile:
```batch
cargo build --examples --release
```

To run valgrind:
```batch
valgrind --tool=massif ./target/release/examples/pheap_stress <implementation> <number of nodes to be inserted>
```

The commandline argument ```<implementation>``` accepts the following options:
- ```pairing_heap```
- ```priority_queue```
- ```keyed_priority_queue```
- ```addressable_pairing_heap```
- ```ap422_pairing_heap```

|  | Pairing heap<br>(this crate)  | Addressable pairing heap | Pairing heap <br>(Apasel422) | Priority queue | Keyed priority queue
--- | --- | --- | --- | --- | ---
|Peak heap<br>memory consumption<br> (MB)|30.5|72.0|segfault|62|76|

The image outputs of ```massif-visualiser``` are stored in the folder ```img```.

## Dijkstra's algorithm
To test the performance of Dijkstra's algorithm with pairing heap, I use the [DIMACS dataset](http://www.diag.uniroma1.it/challenge9/download.shtml). You can download all datasets by using the python script with the following command:

```python
python3 scripts/download.py -d dimacs-all --dest data/
```

On [crates.io](https://crates.io/) there are several libraries that have Dijkstra's algorithm but I only find the crate [`pathfinding`](https://crates.io/crates/pathfinding) performant (please let me know if I miss any crate).

For this experiment, all implementations are tasked to solve the shortest path problem on all DIMACS dataset and I take the average runtime after ten runs.

**Note:** the function `dijkstra_all` of `pathfinding` returns only the direct parent node for a queried node, instead of an entire path, the function `sssp_dijkstra_lazy` is used for my implementation of Dijkstra's algorithm. This function returns a result which is (kind of) equivalent to what `pathfinding` delivers. By doing so, we can compare the solving time of both implementations, while ignoring the path building time.

Time is measured in millisecond:

|  | Number of nodes | Number of edges | pheap  | pathfinding
--- | --- | --- | --- | --- |
|DIMACS-NY| 264_346 | 733_846 | 88 |110|
|DIMACS-BAY| 321_270 | 800_172 | 94 |127|
|DIMACS-COL| 435_666 | 1_057_066 | 126 |172|
|DIMACS-FLA| 1_070_376 | 2_712_798 | 377 |626|
|DIMACS-NW| 1_207_945 | 2_840_208 | 456 |665|
|DIMACS-NE| 1_524_453 | 3_897_636 | 619 |852|
|DIMACS-CAL| 1_890_815 | 4_657_742 | 740 |1_246|
|DIMACS-LKS| 2_758_119 | 6_885_658 | 1_141 |1_695|
|DIMACS-E| 3_598_623 | 8_778_114 | 1_548 |2_151|
|DIMACS-W| 6_262_104 | 15_248_146 | 3_098 |4_460|
|DIMACS-CTR| 14_081_816 | 34_292_496 | 10_183 |11_256|
|DIMACS-USA| 23_947_347 | 58_333_344 | 16_678 |20_896|

## Minimum spanning tree
In this experiment, I measure the performance of both libraries in finding the MST. However, there are several differences between two crates that are worth mentioning: firstly, while ```pathfinding``` uses Kruskal's algorithm, I implement only the Prim's algorithm using the pairing heap. Secondly, ```pathfinding```'s implementation returns only the iterators of edges and it is the task of users to collect these iterators and (re)construct the MST. On the other hand, my implementation returns the complete graph and total weight of an MST. Thus, I run two experiments for ```pheap```, one solving without building MST, and the other for both solving and building MST.

Average time after ten runs, measured in milliesecond:

|  | Number of nodes | Number of edges | pheap <br> (Solve) | pheap <br> (Solve + Build) | pathfinding
--- | --- | --- | --- | --- | --- |
|DIMACS-NY| 264_346 | 733_846 | 78 |140 | 132|
|DIMACS-BAY| 321_270 | 800_172 | 93 | 170 | 140|
|DIMACS-COL| 435_666 | 1_057_066 | 132 | 243 | 191|
|DIMACS-FLA| 1_070_376 | 2_712_798 | 358 | 727 |598|
|DIMACS-NW| 1_207_945 | 2_840_208 | 409 | 863 | 622|
|DIMACS-NE| 1_524_453 | 3_897_636 | 565 | 1_144 | 845|
|DIMACS-CAL| 1_890_815 | 4_657_742 | 715 | 1_553 | 1_148|
|DIMACS-LKS| 2_758_119 | 6_885_658 | 1_093 | 2_307 | 1_641|
|DIMACS-E| 3_598_623 | 8_778_114 | 1_452 | 3_100 | 2_125|
|DIMACS-W| 6_262_104 | 15_248_146 | 2_618 | 5_732 | 4_042|
|DIMACS-CTR| 14_081_816 | 34_292_496 | 7_371 | 16_470 |9_712|
|DIMACS-USA| 23_947_347 | 58_333_344 | 11_785 | 25_450 |17_943|

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.