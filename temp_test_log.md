# unit test: primal_module_parallel_circuit_level_noise_qec_playground_2
On MacBook Air, Apple M2, 8-core CPU, 8GB-memory (M-series Macs do not support Hyper-Threading)

**Note**: I did not really speacify `timeout` here. So perhaps that's what caused the fluctuation in the resolve time
Question/To-be-investigated: use data parallelism techniques inside each serial unit, will this increase or decrease performance?

|     Config         |  ARC_pointers |    unsafe_pointer       |                                              unsafe_pointer(for_loop)|
|--------------------|---------------|-------------------------|----------------------------------------------------------------------|
|serial              |116.051417ms    | 105.300417ms    | |
|2-core(split-4)     |               |108.460792ms | |
|2-core(split-2)     |               | 68.950167ms | |
|3-core(split-4)     |               | 56.544458ms | |
|4-core(split-4)     | 71.006959ms   |  49.95225ms        | |  
|5-core(split-4)     |               | 57.16975ms | |
|6-core(split-4)     |               |58.851334ms | |
|8-core(split-8)     |               | 51.1085ms | |
|3-core(split-3)     |               | 60.036583ms | |
|3-core(split-3, w/o print)|          | 62.062208ms     | 44.858125ms |
|4-core(split-4, w/o print)|          |38.930125ms /40.798666ms /40.984459ms /52.91975ms/51.7205ms    |        59.234167ms / 38.461375ms /40.251542ms /94.73175ms /36.394709ms /35.105958ms /46.176292ms |
|5-core(split-5, w/o print)  |         |50.592333ms /101.778208ms /46.0205ms    | |

# Cli testing (preliminary)
On MacBook Air, Apple M2, 8-core CPU, 8GB-memory (M-series Macs do not support Hyper-Threading)
## Serial: 
`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type union-find --use-deterministic-seed`
total: 1.938e-2, round: 1.938e-2, syndrome: 3.696e-5
total resolve time 19.381953502999934
total tuning time 0.0

## Parallel:
`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 4 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 4, "pin_threads_to_cores": true, "timeout": 1, "cluster_node_limit": 40}}' --use-deterministic-seed`

parameters:
* `code_type`: `RotatedPlanarCode`
* `nm`: `500`
* `solver-type`: `parallel-union-find`
* `split-num`: `4` 
* `thread_pool_size`: `4`
* `timeout`: `1`
* `cluster_node_limit`: `40`

| Config | Total Resolve Time (1000 seeds) |
|--------|---------------------------------|
|2-core(split-2) |    13.736988631999987   |
|3-core(split-3) |   10.913757047000011s   |
|4-core(split-4) |    10.821974638999999s  |
|5-core(split-5) |   panic occurs (see below)|


### 5-core
`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 5 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 5, "pin_threads_to_cores": true, "timeout": 1, "cluster_node_limit": 40}}' --use-deterministic-seed`

`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 5 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 5, "pin_threads_to_cores": true, "timeout": 1, "cluster_node_limit": 40}}' --single-seed 3`
panic occurs, 
sometimes, thread '<unnamed>' panicked at src/primal_module_serial.rs:752:101:
called `Option::unwrap()` on a `None` value
sometimes, thread '<unnamed>' panicked at src/primal_module_serial.rs:421:86:
satisfiable

`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 5 --solver-config '{"dual": {"enable_parallel_execution": false},"primal": {"thread_pool_size": 5, "pin_threads_to_cores": false, "timeout": 1, "cluster_node_limit": 40}}' --single-seed 3`
serial module is fine, does not panic


### Other Parallel Solvers
`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-single-hair --split-num 4 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 4}}' --use-deterministic-seed`
prints to stderr, did not wait for this to finish. 

`cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-joint-single-hair --split-num 4 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 4}}' --use-deterministic-seed`
also prints to stderr, did not wait for this to finish.


# Adapt Partition Profile visualizer 
We adapted the partition profile visualizer from Fusion Blossom to visualize the execution windows of base partitions and fusion operations on multiple threads. We first need to generate the partition profile as follows:

```sh
cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 4 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 4, "pin_threads_to_cores": true, "timeout": 1, "cluster_node_limit": 40}}' --use-deterministic-seed --benchmark-profiler-output visualize/data/7-500-0.001-rotated-planar/core-4-split-4.profile
```
Note that we need to save the `.profile` files in the `/visualize/data` directory. 

Next, we need to start a local server by running 

```sh
cd visualize
python3 server.py
``` 

Now, we can open the visualizer at the following address: 
```sh
http://localhost:8088/partition-profile.html?filename=7-500-0.001-rotated-planar/core-4-split-4.profile
```
Note that the actual server port number needs to match that in `server.py`. The `filename` needs to be under the `visualize/data` directory.

# Evaluate both implementation (Odd/Even parity and For Loop) for their accuracy
On MacBook Air, Apple M2, 8-core CPU, 8GB-memory (M-series Macs do not support Hyper-Threading)

```sh
cargo run --bin mwpf -r  benchmark 7 0.001 --code-type qec-playground-code --code-config '{"code_type": "RotatedPlanarCode", "nm": 500}' --solver-type parallel-union-find --split-num 4 --solver-config '{"dual": {"enable_parallel_execution": true},"primal": {"thread_pool_size": 4, "pin_threads_to_cores": true, "timeout": 1, "cluster_node_limit": 40}}' --use-deterministic-seed --benchmark-profiler-output visualize/data/7-500-0.001-rotated-planar/core-4-split-4.profile
```



# Fix the bug in 5-core (likely threads trying to access the same address?)

