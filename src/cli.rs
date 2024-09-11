use crate::example_codes::*;
use crate::matrix::*;
use crate::mwpf_solver::*;
use crate::util::*;
use crate::visualize::*;
use clap::builder::{StringValueParser, TypedValueParser, ValueParser};
use clap::error::{ContextKind, ContextValue, ErrorKind};
use clap::{Parser, Subcommand, ValueEnum};
use itertools::partition;
use more_asserts::assert_le;
use num_traits::FromPrimitive;
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::RngCore;
use rand::{thread_rng, Rng, SeedableRng};
use serde::Serialize;
use serde_variant::to_variant_name;
use std::env;
use std::collections::BTreeSet;
use std::usize::MAX;

const TEST_EACH_ROUNDS: usize = 100;

#[derive(Parser, Clone)]
#[clap(author = clap::crate_authors!(", "))]
#[clap(version = env!("CARGO_PKG_VERSION"))]
#[clap(about = "Minimum-Weight Parity Factor Algorithm for Quantum Error Correction Decoding")]
#[clap(color = clap::ColorChoice::Auto)]
#[clap(propagate_version = true)]
#[clap(subcommand_required = true)]
#[clap(arg_required_else_help = true)]
pub struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// benchmark the speed (and also correctness if enabled)
    Benchmark(BenchmarkParameters),
    /// benchmark the matrix speed
    MatrixSpeed(MatrixSpeedParameters),
    /// built-in tests
    Test {
        #[clap(subcommand)]
        command: TestCommands,
    },
}

#[derive(Parser, Clone)]
pub struct BenchmarkParameters {
    /// code distance
    #[clap(value_parser)]
    d: VertexNum,
    /// physical error rate: the probability of each edge to
    #[clap(value_parser)]
    p: f64,
    /// rounds of noisy measurement, valid only when multiple rounds
    #[clap(short = 'e', long, default_value_t = 0.)]
    pe: f64,
    /// rounds of noisy measurement, valid only when multiple rounds
    #[clap(short = 'n', long, default_value_t = 0)]
    noisy_measurements: VertexNum,
    /// maximum weight of edges
    #[clap(long, default_value_t = 1000)]
    max_weight: Weight,
    /// example code type
    #[clap(short = 'c', long, value_enum, default_value_t = ExampleCodeType::CodeCapacityTailoredCode)]
    code_type: ExampleCodeType,
    /// the configuration of the code builder
    #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
    code_config: serde_json::Value,
    /// logging to the default visualizer file at visualize/data/visualizer.json
    #[clap(long, action)]
    enable_visualizer: bool,
    /// print syndrome patterns
    #[clap(long, action)]
    print_syndrome_pattern: bool,
    /// print error patterns
    #[clap(long, action)]
    print_error_pattern: bool,
    /// the method to verify the correctness of the decoding result
    #[clap(long, value_enum, default_value_t = Verifier::ActualError)]
    verifier: Verifier,
    /// the number of iterations to run
    #[clap(short = 'r', long, default_value_t = 1000)]
    total_rounds: usize,
    /// select the combination of primal and dual module
    #[clap(short = 'p', long, value_enum, default_value_t = PrimalDualType::UnionFind)]
    primal_dual_type: PrimalDualType,
    /// the configuration of primal and dual module
    #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
    primal_dual_config: serde_json::Value,
    /// message on the progress bar
    #[clap(long, default_value_t = format!(""))]
    pb_message: String,
    /// use deterministic seed for debugging purpose
    #[clap(long, action)]
    use_deterministic_seed: bool,
    /// the benchmark profile output file path
    #[clap(long)]
    benchmark_profiler_output: Option<String>,
    /// skip some iterations, useful when debugging
    #[clap(long, default_value_t = 0)]
    starting_iteration: usize,
    /// apply deterministic seed for debugging purpose
    #[clap(long, action)]
    apply_deterministic_seed: Option<u64>,
    /// single seed for debugging purposes
    #[clap(long, action)]
    single_seed: Option<u64>,
    #[clap(long, value_enum, default_value_t = PartitionStrategy::None)]
    pub partition_strategy: PartitionStrategy,
    #[clap(long, default_value_t = 0)]
    split_num: usize,
}

#[derive(Subcommand, Clone, Debug)]
pub enum TestCommands {
    /// test common cases
    Common,
    /// test various codes using code capacity noise model
    CodeCapacity {
        /// print out the command to test
        #[clap(short = 'c', long, action)]
        print_command: bool,
        /// enable visualizer
        #[clap(short = 'v', long, action)]
        enable_visualizer: bool,
        /// use strict verifier to check whether the result is always optimal
        #[clap(short = 'u', long, action)]
        use_strict: bool,
        /// enable print syndrome pattern
        #[clap(short = 's', long, action)]
        print_syndrome_pattern: bool,
        /// select the combination of primal and dual module
        #[clap(short = 'p', long, value_enum, default_value_t = PrimalDualType::UnionFind)]
        primal_dual_type: PrimalDualType,
        /// the configuration of primal and dual module
        #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
        primal_dual_config: serde_json::Value,
    },
}

/// note that these code type is only for example, to test and demonstrate the correctness of the algorithm, but not for real QEC simulation;
/// for real simulation, please refer to <https://github.com/yuewuo/QEC-Playground>
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
pub enum ExampleCodeType {
    /// quantum repetition code with perfect stabilizer measurement
    CodeCapacityRepetitionCode,
    /// planar code with perfect stabilizer measurement, only one type of stabilizer's decoding graph (thus normal graph)
    CodeCapacityPlanarCode,
    /// color code with perfect stabilizer measurement
    CodeCapacityColorCode,
    /// tailored surface code, which is essentially rotated planar code with depolarizing noise model
    CodeCapacityTailoredCode,
    /// read from error pattern file, generated using option `--primal-dual-type error-pattern-logger`
    ErrorPatternReader,
    /// code constructed by QEC-Playground, pass configurations using `--code-config`
    #[cfg(feature = "qecp_integrate")]
    #[serde(rename = "qec-playground-code")]
    QECPlaygroundCode,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
pub enum PartitionStrategy {
    /// no partition
    None,
    /// partition a planar code into top half and bottom half
    CodeCapacityPlanarCodeVerticalPartitionHalf,
    /// partition a planar code into 4 pieces: top left and right, bottom left and right
    CodeCapacityPlanarCodeVerticalPartitionFour,
    /// partition a repetition code into left and right half
    CodeCapacityRepetitionCodePartitionHalf,
    /// partition a phenomenological (or circuit-level) planar code with time axis
    PhenomenologicalPlanarCodeTimePartition,
    /// partition a phenomenological (or circuit-level) rotated code with time axis
    PhenomenologicalRotatedCodeTimePartition,
}



#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum PrimalDualType {
    /// the solver from Union-Find decoder
    UnionFind,
    /// the single-hair solver
    SingleHair,
    /// joint single-hair solver
    JointSingleHair,
    /// log error into a file for later fetch
    ErrorPatternLogger,
    /// parallel primal and parallel dual, Union-Find decoder
    ParallelUnionFind,
    /// parallel primal and parallel dual, single-hair decoder
    ParallelSingleHair,
    /// parallel primal and parallel dual, joint single-hair solver
    ParallelJointSingleHair,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
pub enum Verifier {
    /// disable verifier
    None,
    /// use the serial version of fusion blossom library to verify
    FusionSerial,
    /// if the actual error has smaller weight than the lower bound of the weight range then fail
    ActualError,
    /// if the actual error has smaller weight than the solved subgraph then fail
    StrictActualError,
}

#[derive(Parser, Clone)]
pub struct MatrixSpeedParameters {
    #[clap(short = 'c', long, value_enum, default_value_t = MatrixSpeedClass::EchelonTailTight)]
    matrix_type: MatrixSpeedClass,
    #[clap(long, default_value_t = 50)]
    width: usize,
    #[clap(long, default_value_t = 50)]
    height: usize,
    #[clap(long, default_value_t = 0.1)]
    one_density: f64,
    #[clap(short = 'r', long, default_value_t = 100000)]
    total_rounds: usize,
    #[clap(long)]
    deterministic_seed: Option<u64>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
pub enum MatrixSpeedClass {
    EchelonTailTight,
    EchelonTight,
    Echelon,
}

#[derive(Clone)]
struct SerdeJsonParser;
impl TypedValueParser for SerdeJsonParser {
    type Value = serde_json::Value;
    fn parse_ref(
        &self,
        cmd: &clap::Command,
        arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let inner = StringValueParser::new();
        let val = inner.parse_ref(cmd, arg, value)?;
        match serde_json::from_str::<serde_json::Value>(&val) {
            Ok(vector) => Ok(vector),
            Err(error) => {
                let mut err = clap::Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
                if let Some(arg) = arg {
                    err.insert(ContextKind::InvalidArg, ContextValue::String(arg.to_string()));
                }
                err.insert(
                    ContextKind::InvalidValue,
                    ContextValue::String(format!("should be like {{\"a\":1}}, parse error: {error}")),
                );
                Err(err)
            }
        }
    }
}

// impl MatrixSpeedClass {
//     pub fn run(&self, parameters: MatrixSpeedParameters, samples: Vec<Vec<(Vec<usize>, bool)>>) {
//         match *self {
//             MatrixSpeedClass::EchelonTailTight => {
//                 let mut matrix = Echelon::<Tail<Tight<BasicMatrix>>>::new();
//                 for edge_index in 0..parameters.width {
//                     matrix.add_tight_variable(edge_index);
//                 }
//                 Self::run_on_matrix_interface(&matrix, samples)
//             }
//             MatrixSpeedClass::EchelonTight => {
//                 let mut matrix = Echelon::<Tight<BasicMatrix>>::new();
//                 for edge_index in 0..parameters.width {
//                     matrix.add_tight_variable(edge_index);
//                 }
//                 Self::run_on_matrix_interface(&matrix, samples)
//             }
//             MatrixSpeedClass::Echelon => {
//                 let mut matrix = Echelon::<BasicMatrix>::new();
//                 for edge_index in 0..parameters.width {
//                     matrix.add_variable(edge_index);
//                 }
//                 Self::run_on_matrix_interface(&matrix, samples)
//             }
//         }
//     }

//     pub fn run_on_matrix_interface<M: MatrixView + Clone>(matrix: &M, samples: Vec<Vec<(Vec<usize>, bool)>>) {
//         for parity_checks in samples.iter() {
//             let mut matrix = matrix.clone();
//             for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
//                 matrix.add_constraint(vertex_index, incident_edges, *parity);
//             }
//             // for a MatrixView, visiting the columns and rows is sufficient to update its internal state
//             matrix.columns();
//             matrix.rows();
//         }
//     }
// }

/// test for time partition
#[allow(clippy::unnecessary_cast)]
pub fn graph_time_partition(initializer: &SolverInitializer, positions: &Vec<VisualizePosition>, defect_vertices: &Vec<VertexIndex>, split_num: usize) -> PartitionConfig  {
    assert!(positions.len() > 0, "positive number of positions");
    let mut partition_config = PartitionConfig::new(initializer.vertex_num);
    let mut last_t = positions[0].t;
    let mut t_list: Vec<f64> = vec![];
    t_list.push(last_t);
    for position in positions {
        assert!(position.t >= last_t, "t not monotonically increasing, vertex reordering must be performed before calling this");
        if position.t != last_t {
            t_list.push(position.t);
        }
        last_t = position.t;
    }

    // pick the t value in the middle to split it
    let mut t_split_vec: Vec<f64> = vec![0.0; split_num - 1];
    for i in 0..(split_num - 1) {
        let index: usize = t_list.len()/split_num * (i + 1);
        t_split_vec[i] = t_list[index];
    }
    // find the vertices indices
    let mut split_start_index_vec = vec![MAX; split_num - 1];
    let mut split_end_index_vec = vec![MAX; split_num - 1];
    let mut start_index = 0;
    let mut end_index = 0;
    for (vertex_index, position) in positions.iter().enumerate() {
        if start_index < split_num - 1 {
            if split_start_index_vec[start_index] == MAX && position.t == t_split_vec[start_index] {
                split_start_index_vec[start_index] = vertex_index;
                if start_index != 0 {
                    end_index += 1;
                }
                start_index += 1;
            }
        }
        
        if end_index < split_num - 1 {
            if position.t == t_split_vec[end_index] {
                split_end_index_vec[end_index] = vertex_index + 1;
                // end_index += 1;
            }
        }
    }

    assert!(split_start_index_vec.iter().all(|&x| x != MAX), "Some elements in split_start_index_vec are equal to MAX");
    
    // partitions are found
    let mut graph_nodes = vec![];
    let mut partitions_vec = vec![];
    for i in 0..split_num  {
        if i == 0 {
            partitions_vec.push(VertexRange::new(0, split_start_index_vec[0]));
        } else if i == split_num - 1 {
            partitions_vec.push(VertexRange::new(split_end_index_vec[i - 1], positions.len()));
        } else {
            partitions_vec.push(VertexRange::new(split_end_index_vec[i - 1], split_start_index_vec[i]));
        }

        if i < split_num - 1 {
            partition_config.fusions.push((i, i+1));
        }
        
        let a = partition_config.dag_partition_units.add_node(());
        graph_nodes.push(a.clone());
    }
    partition_config.partitions = partitions_vec;

    for i in 0..split_num {
        if i < split_num - 1 {
            partition_config.dag_partition_units.add_edge(graph_nodes[i], graph_nodes[i+1], false);
        }
    }
    partition_config.defect_vertices = BTreeSet::from_iter(defect_vertices.clone());

    partition_config
}

impl Cli {
    pub fn run(self) {
        match self.command {
            Commands::Benchmark(BenchmarkParameters {
                d,
                p,
                pe,
                noisy_measurements,
                max_weight,
                code_type,
                enable_visualizer,
                verifier,
                total_rounds,
                primal_dual_type,
                pb_message,
                primal_dual_config,
                code_config,
                use_deterministic_seed,
                benchmark_profiler_output,
                print_syndrome_pattern,
                starting_iteration,
                print_error_pattern,
                apply_deterministic_seed,
                single_seed,
                partition_strategy,
                split_num,
            }) => {
                // whether to disable progress bar, useful when running jobs in background
                let disable_progress_bar = env::var("DISABLE_PROGRESS_BAR").is_ok();
                let mut code: Box<dyn ExampleCode> = code_type.build(d, p, noisy_measurements, max_weight, code_config);
                if pe != 0. {
                    code.set_erasure_probability(pe);
                }
                if enable_visualizer {
                    // print visualizer file path only once
                    print_visualize_link(static_visualize_data_filename());
                }
                // create initializer and solver
                let initializer = code.get_initializer();

                // let partition_config = graph_time_partition(&initializer, &code.get_positions(), &defect_vertices, split_num);
                // let partition_info = partition_config.info();

                // let mut primal_dual_solver = primal_dual_type.build(&initializer, &*code, primal_dual_config);
                // let mut result_verifier = verifier.build(&initializer);
                // prepare progress bar display
                let mut pb = if !disable_progress_bar {
                    let mut pb = ProgressBar::on(std::io::stderr(), total_rounds as u64);
                    pb.message(format!("{pb_message} ").as_str());
                    Some(pb)
                } else {
                    if !pb_message.is_empty() {
                        print!("{pb_message} ");
                    }
                    None
                };

                if let Some(seed) = single_seed {
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);
                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }

                    let defect_vertices = code.generate_random_errors(seed).0.defect_vertices;
                    let mut partition_config = PartitionConfig::new(initializer.vertex_num);
                    let mut partition_info = partition_config.info();

                    if split_num > 0 {
                        partition_config = graph_time_partition(&initializer, &code.get_positions(), &defect_vertices, split_num);
                        partition_info = partition_config.info();
                    } 
                    let mut primal_dual_solver = primal_dual_type.build(&initializer, &partition_info, &*code, primal_dual_config);
                    let mut result_verifier = verifier.build(&initializer);
                    // create a new visualizer each round
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer = Visualizer::new(
                            Some(visualize_data_folder() + static_visualize_data_filename().as_str()),
                            code.get_positions(),
                            true,
                        )
                        .unwrap();
                        visualizer = Some(new_visualizer);
                    }

                    // let begin_time = std::time::Instant::now();
                    primal_dual_solver.solve_visualizer(&syndrome_pattern, None, seed); // FIXME: for release, remove the seed that is passed in for debugging purposes

                    // solver load the defect vertices from their indices
                    result_verifier.verify(
                        &mut primal_dual_solver,
                        &syndrome_pattern,
                        &error_pattern,
                        None,
                        seed,
                    );
                    primal_dual_solver.clear(); // also count the clear operation
                    // let end_time = std::time::Instant::now();
                    // let resolve_time = (end_time - begin_time).as_secs_f64();
                    // println!("resolve time {:?}", resolve_time);

                    return;
                }

                let mut benchmark_profiler = BenchmarkProfiler::new(noisy_measurements, benchmark_profiler_output);
                // let mut rng = thread_rng();
                thread_rng().gen::<u64>();
                let mut seed = match apply_deterministic_seed {
                    Some(seed) => seed,
                    None => thread_rng().gen::<u64>(),
                };
                let mut rng = SmallRng::seed_from_u64(seed);
                // println!("OG_s: {:?}", seed);
                for round in (starting_iteration as u64)..(total_rounds as u64) {
                    pb.as_mut().map(|pb| pb.set(round));
                    seed = if use_deterministic_seed { round } else { rng.next_u64() };
                    // println!("NEW rng seed: {:?}", seed);
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);
                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }

                    let defect_vertices = code.generate_random_errors(seed).0.defect_vertices;
                    let mut partition_config = PartitionConfig::new(initializer.vertex_num);
                    let mut partition_info = partition_config.info();

                    if split_num > 0 {
                        partition_config = graph_time_partition(&initializer, &code.get_positions(), &defect_vertices, split_num);
                        partition_info = partition_config.info();
                    } 
                    let mut primal_dual_solver = primal_dual_type.build(&initializer, &partition_info, &*code, primal_dual_config.clone());
                    let mut result_verifier = verifier.build(&initializer);

                    // create a new visualizer each round
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer = Visualizer::new(
                            Some(visualize_data_folder() + static_visualize_data_filename().as_str()),
                            code.get_positions(),
                            true,
                        )
                        .unwrap();
                        visualizer = Some(new_visualizer);
                    }
                    benchmark_profiler.begin(&syndrome_pattern, &error_pattern);
                    primal_dual_solver.solve_visualizer(&syndrome_pattern, visualizer.as_mut(), seed); // FIXME: for release, remove the seed that is passed in for debugging purposes
                    benchmark_profiler.event("decoded".to_string());
                    result_verifier.verify(
                        &mut primal_dual_solver,
                        &syndrome_pattern,
                        &error_pattern,
                        visualizer.as_mut(),
                        seed,
                    );
                    benchmark_profiler.event("verified".to_string());
                    primal_dual_solver.clear(); // also count the clear operation

                    benchmark_profiler.end(Some(&*primal_dual_solver));
                    if let Some(pb) = pb.as_mut() {
                        if pb_message.is_empty() {
                            pb.message(format!("{} ", benchmark_profiler.brief()).as_str());
                        }
                    }
                }
                if disable_progress_bar {
                    // always print out brief
                    println!("{}", benchmark_profiler.brief());
                } else {
                    if let Some(pb) = pb.as_mut() {
                        pb.finish()
                    }
                    println!();
                }

                eprintln!("total resolve time {:?}", benchmark_profiler.sum_round_time);
            }
            Commands::MatrixSpeed(parameters) => {
                let MatrixSpeedParameters {
                    matrix_type,
                    width,
                    height,
                    one_density,
                    total_rounds,
                    deterministic_seed,
                } = parameters.clone();
                // fist generate the parity samples
                let mut samples = Vec::with_capacity(total_rounds);
                let deterministic_seed = deterministic_seed.unwrap_or_else(|| rand::thread_rng().gen());
                let mut rng = DeterministicRng::seed_from_u64(deterministic_seed);
                for _ in 0..total_rounds {
                    let mut parity_checks: Vec<(Vec<usize>, bool)> = Vec::with_capacity(height);
                    for _ in 0..height {
                        parity_checks.push((
                            (0..width).filter(|_| rng.next_f64() < one_density).collect(),
                            rng.next_f64() < one_density,
                        ))
                    }
                    samples.push(parity_checks);
                }
                // call the matrix operation
                // matrix_type.run(parameters, samples);
            }
            Commands::Test { command } => match command {
                TestCommands::Common => {
                    println!("[Common Test] Union-Find on Code Capacity Noise");
                    execute_in_cli(["".to_owned(), "test".to_owned(), "code-capacity".to_owned()].iter(), true);
                }
                TestCommands::CodeCapacity {
                    print_command,
                    enable_visualizer,
                    use_strict,
                    print_syndrome_pattern,
                    primal_dual_type,
                    primal_dual_config,
                } => {
                    let mut parameters = vec![];
                    for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                        for d in [3, 7, 11, 15, 19] {
                            parameters.push(vec![
                                format!("{d}"),
                                format!("{p}"),
                                format!("--code-type"),
                                format!("code-capacity-repetition-code"),
                                format!("--pb-message"),
                                format!("repetition {d} {p}"),
                            ]);
                        }
                    }
                    for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                        for d in [3, 7, 11, 15, 19] {
                            parameters.push(vec![
                                format!("{d}"),
                                format!("{p}"),
                                format!("--code-type"),
                                format!("code-capacity-planar-code"),
                                format!("--pb-message"),
                                format!("planar {d} {p}"),
                            ]);
                        }
                    }
                    for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                        for d in [3, 7, 11, 15, 19] {
                            parameters.push(vec![
                                format!("{d}"),
                                format!("{p}"),
                                format!("--code-type"),
                                format!("code-capacity-tailored-code"),
                                format!("--pb-message"),
                                format!("tailored {d} {p}"),
                            ]);
                        }
                    }
                    for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                        for d in [3, 7, 11, 15, 19] {
                            parameters.push(vec![
                                format!("{d}"),
                                format!("{p}"),
                                format!("--code-type"),
                                format!("code-capacity-color-code"),
                                format!("--pb-message"),
                                format!("color {d} {p}"),
                            ]);
                        }
                    }
                    let command_head = vec![format!(""), format!("benchmark")];
                    let mut command_tail = vec!["--total-rounds".to_string(), format!("{TEST_EACH_ROUNDS}")];
                    if use_strict {
                        command_tail.append(&mut vec![format!("--verifier"), format!("strict-actual-error")]);
                    } else {
                        command_tail.append(&mut vec![format!("--verifier"), format!("actual-error")]);
                    }
                    if enable_visualizer {
                        command_tail.append(&mut vec![format!("--enable-visualizer")]);
                    }
                    if print_syndrome_pattern {
                        command_tail.append(&mut vec![format!("--print-syndrome-pattern")]);
                    }
                    command_tail.append(&mut vec![
                        format!("--primal-dual-type"),
                        format!("{}", to_variant_name(&primal_dual_type).unwrap()),
                        format!("--primal-dual-config"),
                        serde_json::to_string(&primal_dual_config).unwrap(),
                    ]);
                    for parameter in parameters.iter() {
                        execute_in_cli(
                            command_head.iter().chain(parameter.iter()).chain(command_tail.iter()),
                            print_command,
                        );
                    }
                }
            },
        }
    }
}

pub fn execute_in_cli<'a>(iter: impl Iterator<Item = &'a String> + Clone, print_command: bool) {
    if print_command {
        print!("[command]");
        for word in iter.clone() {
            if word.contains(char::is_whitespace) {
                print!("'{word}' ")
            } else {
                print!("{word} ")
            }
        }
        println!();
    }
    Cli::parse_from(iter).run();
}

impl ExampleCodeType {
    fn build(
        &self,
        d: VertexNum,
        p: f64,
        _noisy_measurements: VertexNum,
        max_weight: Weight,
        mut code_config: serde_json::Value,
    ) -> Box<dyn ExampleCode> {
        match self {
            Self::CodeCapacityRepetitionCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityRepetitionCode::new(d, p, max_weight))
            }
            Self::CodeCapacityPlanarCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityPlanarCode::new(d, p, max_weight))
            }
            Self::CodeCapacityTailoredCode => {
                let mut pxy = 0.; // default to infinite bias
                let config = code_config.as_object_mut().expect("config must be JSON object");
                if let Some(value) = config.remove("pxy") {
                    pxy = value.as_f64().expect("code_count number");
                }
                Box::new(CodeCapacityTailoredCode::new(d, pxy, p, max_weight))
            }
            Self::CodeCapacityColorCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityColorCode::new(d, p, max_weight))
            }
            Self::ErrorPatternReader => Box::new(ErrorPatternReader::new(code_config)),
            #[cfg(feature = "qecp_integrate")]
            Self::QECPlaygroundCode => Box::new(QECPlaygroundCode::new(d, p, code_config)),
        }
    }
}

// implement this after printing/finalizing the poster
// impl PartitionStrategy {
//     pub fn build(
//         &self,
//         code: &mut dyn ExampleCode,
//         d: VertexNum,
//         noisy_measurements: VertexNum,
//         mut partition_config: serde_json::Value,
//     ) -> (SolverInitializer, PartitionConfig) {
//         use example_partition::*;
//         let partition_config = match self {
//             Self::None => {
//                 assert_eq!(partition_config, json!({}), "config not supported");
//                 NoPartition::new().build_apply(code)
//             }
//             Self::CodeCapacityPlanarCodeVerticalPartitionHalf => {
//                 assert_eq!(partition_config, json!({}), "config not supported");
//                 CodeCapacityPlanarCodeVerticalPartitionHalf::new(d, d / 2).build_apply(code)
//             }
//             Self::CodeCapacityPlanarCodeVerticalPartitionFour => {
//                 assert_eq!(partition_config, json!({}), "config not supported");
//                 CodeCapacityPlanarCodeVerticalPartitionFour::new(d, d / 2, d / 2).build_apply(code)
//             }
//             Self::CodeCapacityRepetitionCodePartitionHalf => {
//                 assert_eq!(partition_config, json!({}), "config not supported");
//                 CodeCapacityRepetitionCodePartitionHalf::new(d, d / 2).build_apply(code)
//             }
//             Self::PhenomenologicalPlanarCodeTimePartition => {
//                 let config = partition_config.as_object_mut().expect("config must be JSON object");
//                 let mut partition_num = 10;
//                 let mut enable_tree_fusion = false;
//                 let mut maximum_tree_leaf_size = usize::MAX;
//                 if let Some(value) = config.remove("partition_num") {
//                     partition_num = value.as_u64().expect("partition_num: usize") as usize;
//                 }
//                 if let Some(value) = config.remove("enable_tree_fusion") {
//                     enable_tree_fusion = value.as_bool().expect("enable_tree_fusion: bool");
//                 }
//                 if let Some(value) = config.remove("maximum_tree_leaf_size") {
//                     maximum_tree_leaf_size = value.as_u64().expect("maximum_tree_leaf_size: usize") as usize;
//                 }
//                 if !config.is_empty() {
//                     panic!("unknown config keys: {:?}", config.keys().collect::<Vec<&String>>());
//                 }
//                 PhenomenologicalPlanarCodeTimePartition::new_tree(
//                     d,
//                     noisy_measurements,
//                     partition_num,
//                     enable_tree_fusion,
//                     maximum_tree_leaf_size,
//                 )
//                 .build_apply(code)
//             }
//             Self::PhenomenologicalRotatedCodeTimePartition => {
//                 let config = partition_config.as_object_mut().expect("config must be JSON object");
//                 let mut partition_num = 10;
//                 let mut enable_tree_fusion = false;
//                 let mut maximum_tree_leaf_size = usize::MAX;
//                 if let Some(value) = config.remove("partition_num") {
//                     partition_num = value.as_u64().expect("partition_num: usize") as usize;
//                 }
//                 if let Some(value) = config.remove("enable_tree_fusion") {
//                     enable_tree_fusion = value.as_bool().expect("enable_tree_fusion: bool");
//                 }
//                 if let Some(value) = config.remove("maximum_tree_leaf_size") {
//                     maximum_tree_leaf_size = value.as_u64().expect("maximum_tree_leaf_size: usize") as usize;
//                 }
//                 if !config.is_empty() {
//                     panic!("unknown config keys: {:?}", config.keys().collect::<Vec<&String>>());
//                 }
//                 PhenomenologicalRotatedCodeTimePartition::new_tree(
//                     d,
//                     noisy_measurements,
//                     partition_num,
//                     enable_tree_fusion,
//                     maximum_tree_leaf_size,
//                 )
//                 .build_apply(code)
//             }
//         };
//         (code.get_initializer(), partition_config)
//     }
// }

impl PrimalDualType {
    fn build(
        &self,
        initializer: &SolverInitializer,
        partition_info: &PartitionInfo,
        code: &dyn ExampleCode,
        primal_dual_config: serde_json::Value,
    ) -> Box<dyn PrimalDualSolver> {
        match self {
            Self::UnionFind => Box::new(SolverSerialUnionFind::new(initializer, primal_dual_config)),
            Self::SingleHair => Box::new(SolverSerialSingleHair::new(initializer, primal_dual_config)),
            Self::JointSingleHair => Box::new(SolverSerialJointSingleHair::new(initializer, primal_dual_config)),
            Self::ErrorPatternLogger => Box::new(SolverErrorPatternLogger::new(initializer, code, primal_dual_config)),
            Self::ParallelUnionFind => Box::new(SolverParallelUnionFind::new(initializer, partition_info, primal_dual_config)),
            Self::ParallelSingleHair => Box::new(SolverParallelSingleHair::new(initializer, partition_info, primal_dual_config)),
            Self::ParallelJointSingleHair => Box::new(SolverParallelJointSingleHair::new(initializer, partition_info, primal_dual_config)),
        }
    }
}

impl Verifier {
    fn build(&self, initializer: &SolverInitializer) -> Box<dyn ResultVerifier> {
        match self {
            Self::None => Box::new(VerifierNone {}),
            Self::FusionSerial => Box::new(VerifierFusionSerial {
                initializer: initializer.clone(),
            }),
            Self::ActualError => Box::new(VerifierActualError {
                initializer: initializer.clone(),
                is_strict: false,
            }),
            Self::StrictActualError => Box::new(VerifierActualError {
                initializer: initializer.clone(),
                is_strict: true,
            }),
        }
    }
}

trait ResultVerifier {
    fn verify(
        &mut self,
        primal_dual_solver: &mut Box<dyn PrimalDualSolver>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Vec<usize>,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    );
}

struct VerifierNone {}

impl ResultVerifier for VerifierNone {
    fn verify(
        &mut self,
        _primal_dual_solver: &mut Box<dyn PrimalDualSolver>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Vec<usize>,
        _visualizer: Option<&mut Visualizer>,
        _seed: u64,
    ) {
    }
}

struct VerifierFusionSerial {
    pub initializer: SolverInitializer,
}

impl ResultVerifier for VerifierFusionSerial {
    fn verify(
        &mut self,
        _primal_dual_solver: &mut Box<dyn PrimalDualSolver>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Vec<usize>,
        _visualizer: Option<&mut Visualizer>,
        _seed: u64,
    ) {
        println!("{}", self.initializer.vertex_num);
        unimplemented!()
    }
}

struct VerifierActualError {
    initializer: SolverInitializer,
    pub is_strict: bool,
}

impl ResultVerifier for VerifierActualError {
    fn verify(
        &mut self,
        primal_dual_solver: &mut Box<dyn PrimalDualSolver>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Vec<usize>,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    ) {
        if !syndrome_pattern.erasures.is_empty() {
            unimplemented!()
        }
        let actual_weight = if error_pattern.is_empty() && !syndrome_pattern.defect_vertices.is_empty() {
            // error pattern is not generated by the simulator
            Rational::from_usize(usize::MAX).unwrap()
        } else {
            self.initializer.get_subgraph_index_total_weight(error_pattern)
        };
        let (subgraph, weight_range) = primal_dual_solver.subgraph_range_visualizer(visualizer, seed);
        assert!(
            self.initializer
                .matches_subgraph_syndrome(&subgraph, &syndrome_pattern.defect_vertices),
            "bug: the result subgraph does not match the syndrome || the seed is {seed:?}"
        );
        assert_le!(
            weight_range.lower,
            actual_weight,
            "bug: the lower bound of weight range is larger than the actual weight || the seed is {seed:?}"
        );
        if self.is_strict {
            let subgraph_weight = self.initializer.get_subgraph_total_weight(&subgraph);
            assert_le!(subgraph_weight, actual_weight, "it's not a minimum-weight parity subgraph: the actual error pattern has smaller weight, range: {weight_range:?}");
            assert_eq!(
                weight_range.lower, weight_range.upper,
                "the weight range must be optimal: lower = upper || the seed is {seed:?}"
            );
        }
    }
}
