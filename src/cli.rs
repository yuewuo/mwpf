use crate::example_codes::*;
use crate::matrix::*;
use crate::mwpf_solver::*;
use crate::util::*;
use crate::visualize::*;
use bp::bp::BpSparse;
use clap::builder::{StringValueParser, TypedValueParser, ValueParser};
use clap::error::{ContextKind, ContextValue, ErrorKind};
use clap::{Parser, Subcommand, ValueEnum};
use more_asserts::assert_le;
use num_traits::FromPrimitive;
#[cfg(feature = "progress_bar")]
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::RngCore;
use rand::{thread_rng, Rng, SeedableRng};
use serde::Serialize;
use serde_variant::to_variant_name;
use std::env;
use std::sync::Arc;

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
    /// decoder speed
    DecoderSpeed(DecoderSpeedParameters),
    /// custom test command, for porting failed python test
    CustomTest,
    /// custom test command, for comparing with ported failed python test
    CustomTest2,
    /// built-in tests
    Test {
        #[clap(subcommand)]
        command: TestCommands,
    },
}

#[derive(Parser, Clone)]
pub struct DecoderSpeedParameters {
    #[clap(short = 'f', long)]
    file_path: String,
    /// select the combination of primal and dual module
    #[clap(short = 'p', long, value_enum, default_value_t = SolverType::JointSingleHair)]
    solver_type: SolverType,
    /// the configuration of primal and dual module
    #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
    solver_config: serde_json::Value,
    /// to use bp or not
    #[clap(long, action)]
    use_bp: bool,
    #[clap(long, action)]
    bp_application_ratio: Option<f64>,
    #[clap(long, action)]
    bp_max_iter: Option<usize>,
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
    /// example code type
    #[clap(short = 'c', long, value_enum, default_value_t = ExampleCodeType::CodeCapacityTailoredCode)]
    code_type: ExampleCodeType,
    /// the configuration of the code builder
    #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
    code_config: serde_json::Value,
    /// logging to the default visualizer file at visualize/data/visualizer.json
    #[clap(long, action)]
    enable_visualizer: bool,
    /// generate json from the visualizer; if not provided, we will use 256MB max memory to store the JSON in memory, and move it to a temporary file (which is deleted when the program exit) if the size of the JSON exceeds 256MB.
    #[clap(long)]
    visualizer_json_filepath: Option<String>,
    /// generate html from the visualizer
    #[clap(long)]
    visualizer_html_filepath: Option<String>,
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
    #[clap(short = 'p', long, value_enum, default_value_t = SolverType::UnionFind)]
    solver_type: SolverType,
    /// the configuration of primal and dual module
    #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
    solver_config: serde_json::Value,
    /// message on the progress bar
    #[clap(long, default_value_t = format!(""))]
    pb_message: String,
    /// use deterministic seed for debugging purpose (round number is the seed)
    #[clap(long, action)]
    use_deterministic_seed: bool,
    /// the benchmark profile output file path
    #[clap(long)]
    benchmark_profiler_output: Option<String>,
    /// skip some iterations, useful when debugging
    #[clap(long, default_value_t = 0)]
    starting_iteration: usize,
    /// apply a deterministic seed for debugging purposes
    #[clap(long, action)]
    apply_deterministic_seed: Option<u64>,
    /// only execute a single seed for debugging purposes
    #[clap(long, action)]
    single_seed: Option<u64>,
    /// to use bp or not
    #[clap(long, action)]
    use_bp: bool,
    #[clap(long, action)]
    bp_application_ratio: Option<f64>,
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
        #[clap(short = 'p', long, value_enum, default_value_t = SolverType::UnionFind)]
        solver_type: SolverType,
        /// the configuration of primal and dual module
        #[clap(long, default_value_t = json!({}), value_parser = ValueParser::new(SerdeJsonParser))]
        solver_config: serde_json::Value,
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
    /// read from error pattern file, generated using option `--solver-type error-pattern-logger`
    ErrorPatternReader,
    /// code constructed by QEC-Playground, pass configurations using `--code-config`
    #[cfg(feature = "qecp_integrate")]
    #[serde(rename = "qec-playground-code")]
    QECPlaygroundCode,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SolverType {
    /// the solver from Union-Find decoder
    UnionFind,
    /// the single-hair solver
    SingleHair,
    /// joint single-hair solver
    JointSingleHair,
    /// log error into a file for later fetch
    ErrorPatternLogger,
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

impl MatrixSpeedClass {
    pub fn run(&self, parameters: MatrixSpeedParameters, samples: Vec<Vec<(Vec<usize>, bool)>>) {
        match *self {
            MatrixSpeedClass::EchelonTailTight => {
                let mut matrix = Echelon::<Tail<Tight<BasicMatrix>>>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_tight_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
            MatrixSpeedClass::EchelonTight => {
                let mut matrix = Echelon::<Tight<BasicMatrix>>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_tight_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
            MatrixSpeedClass::Echelon => {
                let mut matrix = Echelon::<BasicMatrix>::new();
                for edge_index in 0..parameters.width {
                    matrix.add_variable(edge_index);
                }
                Self::run_on_matrix_interface(&matrix, samples)
            }
        }
    }

    pub fn run_on_matrix_interface<M: MatrixView + Clone>(matrix: &M, samples: Vec<Vec<(Vec<usize>, bool)>>) {
        for parity_checks in samples.iter() {
            let mut matrix = matrix.clone();
            for (vertex_index, (incident_edges, parity)) in parity_checks.iter().enumerate() {
                matrix.add_constraint(vertex_index, incident_edges, *parity);
            }
            // for a MatrixView, visiting the columns and rows is sufficient to update its internal state
            matrix.columns();
            matrix.rows();
        }
    }
}

impl Cli {
    pub fn run(self) {
        match self.command {
            Commands::Benchmark(BenchmarkParameters {
                d,
                p,
                pe,
                noisy_measurements,
                code_type,
                enable_visualizer,
                visualizer_json_filepath,
                visualizer_html_filepath,
                verifier,
                total_rounds,
                solver_type,
                #[cfg(feature = "progress_bar")]
                pb_message,
                #[cfg(not(feature = "progress_bar"))]
                    pb_message: _,
                solver_config,
                code_config,
                use_deterministic_seed,
                benchmark_profiler_output,
                print_syndrome_pattern,
                starting_iteration,
                print_error_pattern,
                apply_deterministic_seed,
                single_seed,
                use_bp,
                bp_application_ratio,
            }) => {
                // whether to disable progress bar, useful when running jobs in background
                #[cfg(feature = "progress_bar")]
                let disable_progress_bar = env::var("DISABLE_PROGRESS_BAR").is_ok();
                let mut code: Box<dyn ExampleCode> = code_type.build(d, p, noisy_measurements, code_config);

                // setting up the BP decoder
                let mut pcm = BpSparse::new(code.vertex_num(), code.edge_num(), 0);
                for (col_index, edge) in code.edges().iter().enumerate() {
                    for &row_index in edge.vertices.iter() {
                        pcm.insert_entry(row_index, col_index);
                    }
                }

                if pe != 0. {
                    code.set_erasure_probability(pe);
                }
                // create initializer and solver
                let initializer = Arc::new(code.get_initializer());
                let mut solver = solver_type.build(&initializer, &*code, solver_config.clone());
                if use_bp {
                    solver = match SolverBPWrapper::new(solver.solver_base(), 1, bp_application_ratio.unwrap_or(0.1))
                        .solver
                        .inner
                    {
                        SolverEnum::SolverSerialUnionFind(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialJointSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverErrorPatternLogger(_) => panic!("not supported"),
                    };
                }
                let mut result_verifier = verifier.build(&initializer);
                // prepare progress bar display
                #[cfg(feature = "progress_bar")]
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

                // single seed mode, intended only execute a single failing round
                if let Some(seed) = single_seed {
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);

                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer =
                            Visualizer::new(visualizer_json_filepath.clone(), code.get_positions(), true).unwrap();
                        visualizer = Some(new_visualizer);
                    }

                    solver.solve_visualizer(syndrome_pattern.clone(), visualizer.as_mut());
                    result_verifier.verify(&mut solver, &syndrome_pattern, &error_pattern, visualizer.as_mut(), seed);
                    if let Some(html_path) = &visualizer_html_filepath {
                        if let Some(visualizer) = visualizer.as_mut() {
                            visualizer.save_html(html_path);
                        }
                    }
                    solver.clear();

                    return;
                }

                let mut benchmark_profiler = BenchmarkProfiler::new(noisy_measurements, benchmark_profiler_output);
                thread_rng().gen::<u64>();
                let mut seed = match apply_deterministic_seed {
                    Some(seed) => seed,
                    None => thread_rng().gen::<u64>(),
                };
                let mut rng = SmallRng::seed_from_u64(seed);
                for round in (starting_iteration as u64)..(total_rounds as u64) {
                    #[cfg(feature = "progress_bar")]
                    pb.as_mut().map(|pb| pb.set(round));
                    seed = if use_deterministic_seed { round } else { rng.next_u64() };
                    let (syndrome_pattern, error_pattern) = code.generate_random_errors(seed);

                    if print_syndrome_pattern {
                        println!("syndrome_pattern: {:?}", syndrome_pattern);
                    }
                    if print_error_pattern {
                        println!("error_pattern: {:?}", error_pattern);
                    }
                    // create a new visualizer each round
                    let mut visualizer = None;
                    if enable_visualizer {
                        let new_visualizer =
                            Visualizer::new(visualizer_json_filepath.clone(), code.get_positions(), true).unwrap();
                        visualizer = Some(new_visualizer);
                    }
                    benchmark_profiler.begin(&syndrome_pattern, &error_pattern);
                    solver.solve_visualizer(syndrome_pattern.clone(), visualizer.as_mut());
                    benchmark_profiler.event("decoded".to_string());
                    result_verifier.verify(&mut solver, &syndrome_pattern, &error_pattern, visualizer.as_mut(), seed);
                    benchmark_profiler.event("verified".to_string());
                    solver.clear(); // also count the clear operation

                    benchmark_profiler.end(Some(&*solver));

                    if solver.get_tuning_time().is_some() {
                        solver.clear_tuning_time();
                    }
                    if let Some(html_path) = &visualizer_html_filepath {
                        if let Some(visualizer) = visualizer.as_mut() {
                            visualizer.save_html(html_path);
                        }
                    }
                    #[cfg(feature = "progress_bar")]
                    if let Some(pb) = pb.as_mut() {
                        if pb_message.is_empty() {
                            pb.message(format!("{} ", benchmark_profiler.brief()).as_str());
                        }
                    }
                }
                #[cfg(feature = "progress_bar")]
                if disable_progress_bar {
                    // always print out brief
                    println!("{}", benchmark_profiler.brief());
                } else {
                    if let Some(pb) = pb.as_mut() {
                        pb.finish()
                    }
                    println!();
                }

                // printing the total round time and total tuning time for benchmark purpose
                eprintln!("total resolve time {:?}", benchmark_profiler.sum_round_time);
                eprintln!("total tuning time {:?}", benchmark_profiler.sum_tuning_time);
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
                matrix_type.run(parameters, samples);
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
                    solver_type,
                    solver_config,
                } => {
                    let mut parameters = vec![];
                    let code_types = ["repetition", "planar", "tailored", "color"];

                    for code_type in code_types.iter() {
                        for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.499] {
                            for d in [3, 7, 11, 15, 19] {
                                parameters.push(vec![
                                    format!("{d}"),
                                    format!("{p}"),
                                    format!("--code-type"),
                                    format!("code-capacity-{code_type}-code"),
                                    format!("--pb-message"),
                                    format!("{code_type} {d} {p}"),
                                ]);
                            }
                        }
                    }

                    let command_head = vec![format!(""), format!("benchmark")];
                    let mut command_tail = vec!["--total-rounds".to_string(), format!("{TEST_EACH_ROUNDS}")];
                    command_tail.append(&mut vec![
                        format!("--verifier"),
                        if use_strict {
                            "strict-actual-error".to_string()
                        } else {
                            "actual-error".to_string()
                        },
                    ]);
                    if enable_visualizer {
                        command_tail.append(&mut vec![format!("--enable-visualizer")]);
                    }
                    if print_syndrome_pattern {
                        command_tail.append(&mut vec![format!("--print-syndrome-pattern")]);
                    }
                    command_tail.append(&mut vec![
                        format!("--solver-type"),
                        format!("{}", to_variant_name(&solver_type).unwrap()),
                        format!("--solver-config"),
                        serde_json::to_string(&solver_config).unwrap(),
                    ]);
                    for parameter in parameters.iter() {
                        execute_in_cli(
                            command_head.iter().chain(parameter.iter()).chain(command_tail.iter()),
                            print_command,
                        );
                    }
                }
            },
            Commands::DecoderSpeed(DecoderSpeedParameters {
                file_path,
                solver_type,
                solver_config,
                use_bp,
                bp_application_ratio,
                bp_max_iter,
            }) => {
                if !file_path.ends_with("cbor") {
                    eprintln!(
                        "only support cbor file, the file path \"{file_path}\" does not end with cbor, operations may fail"
                    );
                }
                let BenchmarkSuite {
                    initializer,
                    syndrome_patterns,
                } = BenchmarkSuite::from_cbor(&file_path);

                // time construction time
                let start = std::time::Instant::now();
                let initializer = Arc::new(initializer);
                let mut decoder = solver_type.bench_build(&initializer, solver_config);
                if use_bp {
                    decoder = match SolverBPWrapper::new(
                        decoder.solver_base(),
                        bp_max_iter.unwrap_or(1),
                        bp_application_ratio.unwrap_or(0.1),
                    )
                    .solver
                    .inner
                    {
                        SolverEnum::SolverSerialUnionFind(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverSerialJointSingleHair(x) => Box::new(x) as Box<dyn SolverTrait>,
                        SolverEnum::SolverErrorPatternLogger(_) => panic!("not supported"),
                    };
                }
                let construction_time = start.elapsed();
                eprintln!("construction time {:?}", construction_time);

                // time solving time
                let start = std::time::Instant::now();
                for syndrome_pattern in syndrome_patterns.into_iter() {
                    decoder.solve(syndrome_pattern);
                    decoder.clear();
                }
                let solving_time = start.elapsed();
                eprintln!("solving time {:?}", solving_time);
            }
            Commands::CustomTest => {
                // custom test: ported from test_override_weights_decoding
                let vertex_num = 3;
                let edge1 = HyperEdge {
                    vertices: vec![0, 1],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge2 = HyperEdge {
                    vertices: vec![1, 2],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge3 = HyperEdge {
                    vertices: vec![2, 0],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let initializer = SolverInitializer::new(vertex_num, vec![edge1, edge2, edge3]);
                let initializer = Arc::new(initializer);
                let solver: SolverSerialJointSingleHair = SolverSerialJointSingleHair::new(&initializer, json!({}));
                let mut solver = SolverBPWrapper::new(solver.solver_base(), 1, 0.625);

                // // force set wieghts to [0, 0, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));

                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 0.0);

                // // mix weights of ratio = 0.8
                // // 0.8 * [0, 0, 100] + 0.2 * [100, 100, 100] = [20, 20, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     0.8.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 40.0);

                // set to negative weight
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![(-20.0).into(), 10.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.upper, (-10.0));

                solver.solve(SyndromePattern::new_vertices(vec![0, 2]));
                let (subgraph, bound) = solver.subgraph_range();
                println!("{:?}", subgraph);
                println!("({}, {})", bound.lower, bound.upper);
                assert_eq!(bound.lower, bound.upper);
                assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![2]);
                assert_eq!(bound.upper, Weight::from_float(100.0).unwrap());
            }
            Commands::CustomTest2 => {
                // custom test: ported from test_override_weights_decoding
                let vertex_num = 3;
                let edge1 = HyperEdge {
                    vertices: vec![0, 1],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge2 = HyperEdge {
                    vertices: vec![1, 2],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let edge3 = HyperEdge {
                    vertices: vec![2, 0],
                    weight: Weight::from_float(100.0).unwrap(),
                };
                let initializer = SolverInitializer::new(vertex_num, vec![edge1, edge2, edge3]);
                let initializer = Arc::new(initializer);
                let mut solver: SolverSerialJointSingleHair = SolverSerialJointSingleHair::new(&initializer, json!({}));
                // let mut solver = SolverBPWrapper::new(solver.solver_base(), 1, 0.625);

                // // force set wieghts to [0, 0, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));

                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 0.0);

                // // mix weights of ratio = 0.8
                // // 0.8 * [0, 0, 100] + 0.2 * [100, 100, 100] = [20, 20, 100]
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![0.0.into(), 0.0.into(), 100.0.into()],
                //     0.8.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(bound.upper, 40.0);

                // set to negative weight
                // solver.solve(SyndromePattern::new_with_override_weights(
                //     vec![0, 2],
                //     vec![(-20.0).into(), 10.0.into(), 100.0.into()],
                //     1.0.into(),
                //     None,
                // ));
                // let (subgraph, bound) = solver.subgraph_range();
                // println!("{:?}", subgraph);
                // println!("({}, {})", bound.lower, bound.upper);
                // assert_eq!(bound.lower, bound.upper);
                // assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![0, 1]);
                // assert_eq!(bound.upper, (-10.0));

                solver.solve(SyndromePattern::new_vertices(vec![0, 2]));
                let (subgraph, bound) = solver.subgraph_range();
                println!("{:?}", subgraph);
                println!("({}, {})", bound.lower, bound.upper);
                assert_eq!(bound.lower, bound.upper);
                assert_eq!(subgraph.into_iter().collect::<Vec<_>>(), vec![2]);
                assert_eq!(bound.upper, Weight::from_float(100.0).unwrap());
            }
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
        mut code_config: serde_json::Value,
    ) -> Box<dyn ExampleCode> {
        match self {
            Self::CodeCapacityRepetitionCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityRepetitionCode::new(d, p))
            }
            Self::CodeCapacityPlanarCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityPlanarCode::new(d, p))
            }
            Self::CodeCapacityTailoredCode => {
                let mut pxy = 0.; // default to infinite bias
                let config = code_config.as_object_mut().expect("config must be JSON object");
                if let Some(value) = config.remove("pxy") {
                    pxy = value.as_f64().expect("code_count number");
                }
                Box::new(CodeCapacityTailoredCode::new(d, pxy, p))
            }
            Self::CodeCapacityColorCode => {
                assert_eq!(code_config, json!({}), "config not supported");
                Box::new(CodeCapacityColorCode::new(d, p))
            }
            Self::ErrorPatternReader => Box::new(ErrorPatternReader::new(code_config)),
            #[cfg(feature = "qecp_integrate")]
            Self::QECPlaygroundCode => Box::new(QECPlaygroundCode::new(d, p, code_config)),
        }
    }
}

impl SolverType {
    pub fn build(
        &self,
        initializer: &Arc<SolverInitializer>,
        code: &dyn ExampleCode,
        solver_config: serde_json::Value,
    ) -> Box<dyn SolverTrait> {
        match self {
            Self::UnionFind => Box::new(SolverSerialUnionFind::new(initializer, solver_config)),
            Self::SingleHair => Box::new(SolverSerialSingleHair::new(initializer, solver_config)),
            Self::JointSingleHair => Box::new(SolverSerialJointSingleHair::new(initializer, solver_config)),
            Self::ErrorPatternLogger => Box::new(SolverErrorPatternLogger::new(initializer, code, solver_config)),
        }
    }

    pub fn bench_build(
        &self,
        initializer: &Arc<SolverInitializer>,
        solver_config: serde_json::Value,
    ) -> Box<dyn SolverTrait> {
        match self {
            Self::UnionFind => Box::new(SolverSerialUnionFind::new(initializer, solver_config)),
            Self::SingleHair => Box::new(SolverSerialSingleHair::new(initializer, solver_config)),
            Self::JointSingleHair => Box::new(SolverSerialJointSingleHair::new(initializer, solver_config)),
            Self::ErrorPatternLogger => panic!("error pattern logger does not support decoder (stable) speed benchmark"),
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
        solver: &mut Box<dyn SolverTrait>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Subgraph,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    );
}

struct VerifierNone {}

impl ResultVerifier for VerifierNone {
    fn verify(
        &mut self,
        _solver: &mut Box<dyn SolverTrait>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Subgraph,
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
        _solver: &mut Box<dyn SolverTrait>,
        _syndrome_pattern: &SyndromePattern,
        _error_pattern: &Subgraph,
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
        solver: &mut Box<dyn SolverTrait>,
        syndrome_pattern: &SyndromePattern,
        error_pattern: &Subgraph,
        visualizer: Option<&mut Visualizer>,
        seed: u64,
    ) {
        if !syndrome_pattern.erasures.is_empty()
            || !syndrome_pattern.heralds.is_empty()
            || syndrome_pattern.override_weights.is_some()
        {
            unimplemented!()
        }
        let actual_weight = if error_pattern.is_empty() && !syndrome_pattern.defect_vertices.is_empty() {
            // error pattern is not generated by the simulator
            Rational::from_usize(usize::MAX).unwrap()
        } else {
            Rational::from(
                self.initializer
                    .get_subgraph_total_weight(&OutputSubgraph::new(error_pattern.clone(), Default::default())),
            )
        };
        let (subgraph, weight_range) = solver.subgraph_range_visualizer(visualizer);

        // solver.print_clusters();
        // assert!(
        //     self.initializer
        //         .matches_subgraph_syndrome(&subgraph, &syndrome_pattern.defect_vertices),
        //     "bug: the result subgraph does not match the syndrome || the seed is {seed:?}"
        // );
        assert_le!(
            weight_range.lower,
            actual_weight,
            "bug: the lower bound of weight range is larger than the actual weight || the seed is {seed:?}"
        );
        if self.is_strict {
            let subgraph_weight = Rational::from(self.initializer.get_subgraph_total_weight(&subgraph));
            assert_le!(subgraph_weight, actual_weight, "it's not a minimum-weight parity subgraph: the actual error pattern has smaller weight, range: {weight_range:?}");
            assert_eq!(
                weight_range.lower, weight_range.upper,
                "the weight range must be optimal: lower = upper || the seed is {seed:?}"
            );
        }
    }
}
