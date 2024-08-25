//! Parallel Primal Module
//! 
//! A parallel implementation of the primal module, by calling functions provided by the serial primal module
//! 
//! 


use super::dual_module::*;
use super::dual_module_parallel::*;
use crate::dual_module_pq::EdgeWeak;
use crate::dual_module_pq::{FutureQueueMethods, Obstacle};
use super::pointers::*;
use super::primal_module::*;
use super::primal_module_serial::*;
use super::util::*;
use std::cmp::Ordering;
use super::visualize::*;
use crate::model_hypergraph::ModelHyperGraph;
use crate::rayon::prelude::*;
use rand::rngs::adapter;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::DerefMut;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use crate::num_traits::FromPrimitive;
use crate::plugin::*;
use crate::num_traits::One;


pub struct PrimalModuleParallel {
    /// the basic wrapped serial modules at the beginning, afterwards the fused units are appended after them
    pub units: Vec<PrimalModuleParallelUnitPtr>,
    /// local configuration
    pub config: PrimalModuleParallelConfig,
    /// partition information generated by the config
    pub partition_info: Arc<PartitionInfo>,
    /// thread pool used to execute async functions in parallel
    pub thread_pool: Arc<rayon::ThreadPool>,
}

pub struct PrimalModuleParallelUnit {
    /// the index
    pub unit_index: usize,
    /// the dual module interface, for constant-time clear
    pub interface_ptr: DualModuleInterfacePtr,
    /// partition information generated by the config
    pub partition_info: Arc<PartitionInfo>,
    /// the owned serial primal module
    pub serial_module: PrimalModuleSerial,
    /// adjacent parallel units of this unit, and whether they each are fused with this unit
    pub adjacent_parallel_units: BTreeMap<PrimalModuleParallelUnitPtr, bool>,
    /// whether this unit is solved 
    pub is_solved: bool,
}


pub type PrimalModuleParallelUnitPtr = ArcRwLock<PrimalModuleParallelUnit>;
pub type PrimalModuleParallelUnitWeak = WeakRwLock<PrimalModuleParallelUnit>;

impl std::fmt::Debug for PrimalModuleParallelUnitPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let unit = self.read_recursive();
        write!(f, "{}", unit.unit_index)
    }
}

impl std::fmt::Debug for PrimalModuleParallelUnitWeak {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.upgrade_force().fmt(f)
    }
}

impl Ord for PrimalModuleParallelUnitPtr {
    fn cmp(&self, other: &Self) -> Ordering {
        // compare the pointer address 
        let ptr1 = Arc::as_ptr(self.ptr());
        let ptr2 = Arc::as_ptr(other.ptr());
        // https://doc.rust-lang.org/reference/types/pointer.html
        // "When comparing raw pointers they are compared by their address, rather than by what they point to."
        ptr1.cmp(&ptr2)
    }
}

impl PartialOrd for PrimalModuleParallelUnitPtr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimalModuleParallelConfig {
    /// enable async execution of dual operations; only used when calling top-level operations, not used in individual units
    #[serde(default = "primal_module_parallel_default_configs::thread_pool_size")]
    pub thread_pool_size: usize,
    /// pin threads to cores sequentially
    #[serde(default = "primal_module_parallel_default_configs::pin_threads_to_cores")]
    pub pin_threads_to_cores: bool,
}

impl Default for PrimalModuleParallelConfig {
    fn default() -> Self {
        serde_json::from_value(json!({})).unwrap()
    }
}

pub mod primal_module_parallel_default_configs {
    pub fn thread_pool_size() -> usize {
        0
    } // by default to the number of CPU cores
      // pub fn thread_pool_size() -> usize { 1 }  // debug: use a single core
    pub fn pin_threads_to_cores() -> bool {
        false
    } // pin threads to cores to achieve most stable results
}

impl PrimalModuleParallel {
    pub fn new_config(
        initializer: &SolverInitializer,
        partition_info: &PartitionInfo,
        config: PrimalModuleParallelConfig,
    ) -> Self {
        let partition_info = Arc::new(partition_info.clone());
        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        if config.thread_pool_size != 0 {
            thread_pool_builder = thread_pool_builder.num_threads(config.thread_pool_size);
        }
        if config.pin_threads_to_cores {
            let core_ids = core_affinity::get_core_ids().unwrap();
            // println!("core_ids: {core_ids:?}");
            thread_pool_builder = thread_pool_builder.start_handler(move |thread_index| {
                // https://stackoverflow.com/questions/7274585/linux-find-out-hyper-threaded-core-id
                if thread_index < core_ids.len() {
                    crate::core_affinity::set_for_current(core_ids[thread_index]);
                } // otherwise let OS decide which core to execute
            });
        }

        let thread_pool = thread_pool_builder.build().expect("creating thread pool failed");
        let mut units = vec![];
        let unit_count = partition_info.units.len();
        thread_pool.scope(|_| {
            (0..unit_count)
                .into_par_iter()
                .map(|unit_index| {
                    // println!("unit_index: {unit_index}");
                    let primal_module = PrimalModuleSerial::new_empty(initializer);
                    let interface_ptr = DualModuleInterfacePtr::new();

                    PrimalModuleParallelUnitPtr::new_value(PrimalModuleParallelUnit {
                        unit_index,
                        interface_ptr, 
                        partition_info: partition_info.clone(),
                        serial_module: primal_module,
                        adjacent_parallel_units: BTreeMap::new(),
                        is_solved: false,
                    })
                })
                .collect_into_vec(&mut units);
        });

        // we need to fill in the BTreeMap of adjacent_parallel_units
        // we need to fill in the adjacent_parallel_units here 
        for unit_index in 0..partition_info.units.len() {
            // println!("for unit {:?}", unit_index);
            let mut unit = units[unit_index].write();
            for adjacent_unit_index in &partition_info.units[unit_index].adjacent_parallel_units {
                // println!("adjacent_parallel_unit: {:?}", adjacent_unit_index);
                let adjacnet_unit_pointer = &units[*adjacent_unit_index];
                unit.adjacent_parallel_units.insert(adjacnet_unit_pointer.clone(), false); 
                // println!("adjacent_parallel_unit ptr: {:?}", Arc::as_ptr(pointer.clone().ptr()));
            }
            drop(unit);
        }

        Self {
            units,
            config,
            partition_info,
            thread_pool: Arc::new(thread_pool),
        }
    }
}

impl PrimalModuleParallelUnitPtr {

    // syndrome pattern is created in this function. This function could not be used for dynamic fusion
    fn individual_solve<DualSerialModule: DualModuleImpl + Send + Sync, Queue, F: Send + Sync>(
        &self,
        primal_module_parallel: &PrimalModuleParallel,
        partitioned_syndrome_pattern: PartitionedSyndromePattern,
        parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
        callback: &mut Option<&mut F>,
    ) where
        F: FnMut(
            &DualModuleInterfacePtr,
            &DualModuleParallelUnit<DualSerialModule, Queue>,
            &PrimalModuleSerial,
            Option<&GroupMaxUpdateLength>,
        ),
        Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        let mut primal_unit = self.write();
        let unit_index = primal_unit.unit_index;
        println!("unit index: {}", primal_unit.unit_index);
        let dual_module_ptr = &parallel_dual_module.units[unit_index];
        let mut dual_unit = dual_module_ptr.write();
        let partition_unit_info = &primal_unit.partition_info.units[unit_index];
        let (owned_defect_range, _) = partitioned_syndrome_pattern.partition(partition_unit_info);
        let interface_ptr = primal_unit.interface_ptr.clone();

        // solve the individual unit first 
        if !primal_unit.is_solved {
            // we solve the individual unit first
            let syndrome_pattern = Arc::new(owned_defect_range.expand());
            primal_unit.serial_module.solve_step_callback(
                &interface_ptr,
                syndrome_pattern,
                dual_unit.deref_mut(),
                |interface, dual_module, primal_module, group_max_update_length| {
                    if let Some(callback) = callback.as_mut() {
                        callback(interface, dual_module, primal_module, Some(group_max_update_length));
                    }
                },
            );
            primal_unit.is_solved = true;
            if let Some(callback) = callback.as_mut() {
                callback(&primal_unit.interface_ptr, &dual_unit, &primal_unit.serial_module, None);
            }
        }
        drop(primal_unit);
        drop(dual_unit);
    }

    /// call this only if children is guaranteed to be ready and solved
    #[allow(clippy::unnecessary_cast)]
    fn fuse_and_solve<DualSerialModule: DualModuleImpl + Send + Sync, Queue, F: Send + Sync>(
        &self,
        primal_module_parallel: &PrimalModuleParallel,
        partitioned_syndrome_pattern: PartitionedSyndromePattern,
        parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
        callback: &mut Option<&mut F>,
    ) where
        F: FnMut(
            &DualModuleInterfacePtr,
            &DualModuleParallelUnit<DualSerialModule, Queue>,
            &PrimalModuleSerial,
            Option<&GroupMaxUpdateLength>,
        ),
        Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        assert!(self.read_recursive().is_solved, "this unit must have been solved before we fuse it with its neighbors");
        
        // this unit has been solved, we can fuse it with its adjacent units
        // we iterate through the dag_partition_unit to fuse units together 
        let self_dual_ptr = &parallel_dual_module.units[self.read_recursive().unit_index];
        self.fuse_operation_on_adjacent_units(self_dual_ptr, parallel_dual_module);

        let mut primal_unit = self.write();
        primal_unit.fuse_operation_on_self(self_dual_ptr, parallel_dual_module);

        // now we have finished fusing self with all adjacent units, we run solve again

        let mut dual_unit = self_dual_ptr.write();
        // let partition_unit_info = &primal_unit.partition_info.units[unit_index];
        // let (owned_defect_range, _) = partitioned_syndrome_pattern.partition(partition_unit_info);
        let interface_ptr = primal_unit.interface_ptr.clone();

        primal_unit.serial_module.solve_step_callback_interface_loaded(
            &interface_ptr,
            dual_unit.deref_mut(),
            |interface, dual_module, primal_module, group_max_update_length| {
                if let Some(callback) = callback.as_mut() {
                    callback(interface, dual_module, primal_module, Some(group_max_update_length));
                }
            },
        );
        if let Some(callback) = callback.as_mut() {
            callback(&primal_unit.interface_ptr, &dual_unit, &primal_unit.serial_module, None);
        }
    }

    fn fuse_operation_on_adjacent_units<DualSerialModule: DualModuleImpl + Send + Sync, Queue>
    (&self, 
    self_dual_ptr: &DualModuleParallelUnitPtr<DualSerialModule, Queue>,
    parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
    ) 
    where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        // we need to fuse this unit with all of its adjacent units
        // this is for the adjacent unit
        for (adjacent_unit_ptr, is_fused) in self.read_recursive().adjacent_parallel_units.iter() {
            if *is_fused {
                // if already fused, then skip
                continue;
            } else {
                let mut adjacent_unit = adjacent_unit_ptr.write();
                if let Some(is_fused_with_self) = adjacent_unit.adjacent_parallel_units.get_mut(self) {
                    *is_fused_with_self = true;
                } else {
                    panic!("this adjacent unit does not have self as its adjacent unit, check new_config");
                }

                // after setting the bool in BTreeMap of PrimalModuleParallelUnit, we need to add the corresponding DualModuleParallelUnit 
                let adjacent_dual_unit_ptr = &parallel_dual_module.units[adjacent_unit.unit_index];
                let mut adjacent_dual_unit = adjacent_dual_unit_ptr.write();
                adjacent_dual_unit.adjacent_parallel_units.push(self_dual_ptr.clone());
                
                // we also need to change the `is_fusion` of all vertices to true. 
                for vertex_ptr in adjacent_dual_unit.serial_module.vertices.iter() {
                    let mut vertex = vertex_ptr.write();
                    vertex.fusion_done = true;
                }
                drop(adjacent_unit);
            }

        }
    }
}

impl PrimalModuleParallelUnit {
    fn fuse_operation_on_self<DualSerialModule: DualModuleImpl + Send + Sync, Queue>
    (&mut self,
    self_dual_ptr: &DualModuleParallelUnitPtr<DualSerialModule, Queue>,
    parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
    ) 
    where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        let mut self_dual_unit = self_dual_ptr.write();
        for (adjacent_unit_ptr, is_fused) in self.adjacent_parallel_units.iter_mut() {
            if *is_fused {
                // if already fused, then skip
                continue;
            } else {
                *is_fused = true;

                // we need to add the DualModuleParallelUnitPtr to the adjacent_parallel_units of self
                let adjacent_dual_unit_ptr = &parallel_dual_module.units[adjacent_unit_ptr.read_recursive().unit_index];
                self_dual_unit.adjacent_parallel_units.push(adjacent_dual_unit_ptr.clone());
            }
        }
        drop(self_dual_unit);
    }
}

impl PrimalModuleParallel {
    pub fn parallel_solve<DualSerialModule: DualModuleImpl + Send + Sync, Queue>(
        &mut self,
        syndrome_pattern: Arc<SyndromePattern>,
        parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
    ) where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        self.parallel_solve_step_callback(syndrome_pattern, parallel_dual_module, |_, _, _, _| {});
    }

    pub fn parallel_solve_visualizer<DualSerialModule: DualModuleImpl + Send + Sync + MWPSVisualizer, Queue>(
        &mut self,
        syndrome_pattern: Arc<SyndromePattern>,
        parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
        visualizer: Option<&mut Visualizer>,
    ) where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        if let Some(visualizer) = visualizer {
            self.parallel_solve_step_callback(
                syndrome_pattern,
                parallel_dual_module,
                |interface, dual_module, primal_module, group_max_update_length| {
                    if let Some(group_max_update_length) = group_max_update_length {
                        if cfg!(debug_assertions) {
                            println!("group_max_update_length: {:?}", group_max_update_length);
                        }
                        if group_max_update_length.is_unbounded() {
                            visualizer
                                .snapshot_combined("unbounded grow".to_string(), vec![interface, dual_module, primal_module])
                                .unwrap();
                        } else if let Some(length) = group_max_update_length.get_valid_growth() {
                            visualizer
                                .snapshot_combined(format!("grow {length}"), vec![interface, dual_module, primal_module])
                                .unwrap();
                        } else {
                            let first_conflict = format!("{:?}", group_max_update_length.peek().unwrap());
                            visualizer
                                .snapshot_combined(
                                    format!("resolve {first_conflict}"),
                                    vec![interface, dual_module, primal_module],
                                )
                                .unwrap();
                        };
                    } else {
                        visualizer
                            .snapshot_combined("unit solved".to_string(), vec![interface, dual_module, primal_module])
                            .unwrap();
                    }
                    
                },
            );
            let last_unit = self.units.last().unwrap().read_recursive();
            visualizer
                .snapshot_combined(
                    "solved".to_string(),
                    vec![&last_unit.interface_ptr, parallel_dual_module, self],
                )
                .unwrap();
        } else {
            self.parallel_solve(syndrome_pattern, parallel_dual_module);
        }
    }

    pub fn parallel_solve_step_callback<DualSerialModule: DualModuleImpl + Send + Sync, Queue, F: Send + Sync>(
        &mut self,
        syndrome_pattern: Arc<SyndromePattern>,
        parallel_dual_module: &DualModuleParallel<DualSerialModule, Queue>,
        mut callback: F,
    ) where
        F: FnMut(
            &DualModuleInterfacePtr,
            &DualModuleParallelUnit<DualSerialModule, Queue>,
            &PrimalModuleSerial,
            Option<&GroupMaxUpdateLength>,
        ),
        Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        // let thread_pool = Arc::clone(&self.thread_pool);
        for unit_index in 0..self.partition_info.units.len() {
            let unit_ptr = self.units[unit_index].clone();
            unit_ptr.individual_solve::<DualSerialModule, Queue, F>(
                self, 
                PartitionedSyndromePattern::new(&syndrome_pattern), 
                parallel_dual_module, 
                &mut Some(&mut callback),
            );
        }

        for unit_index in 0..self.partition_info.units.len() {
            let unit_ptr = self.units[unit_index].clone();
            unit_ptr.fuse_and_solve::<DualSerialModule, Queue, F>(
                self, 
                PartitionedSyndromePattern::new(&syndrome_pattern), 
                parallel_dual_module, 
                &mut Some(&mut callback),
            );
        }
    }


}

impl PrimalModuleImpl for PrimalModuleParallel {
    /// create a primal module given the dual module
    fn new_empty(solver_initializer: &SolverInitializer) -> Self {
        Self::new_config(
            solver_initializer,
            &PartitionConfig::new(solver_initializer.vertex_num).info(),
            PrimalModuleParallelConfig::default(),
        )
    }

    /// clear all states; however this method is not necessarily called when load a new decoding problem, so you need to call it yourself
    fn clear(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().enumerate().for_each(|(unit_idx, unit_ptr)| {
                let mut unit = unit_ptr.write();
                unit.clear();
            });
        });
    }

    /// load a new decoding problem given dual interface: note that all nodes MUST be defect node
    /// this function needs to be written to allow dynamic fusion
    fn load<D: DualModuleImpl>(&mut self, interface_ptr: &DualModuleInterfacePtr, dual_module: &mut D) {
        panic!("load interface directly into the parallel primal module is forbidden, use `individual_solve` instead");
    }

    /// analyze the reason why dual module cannot further grow, update primal data structure (alternating tree, temporary matches, etc)
    /// and then tell dual module what to do to resolve these conflicts;
    /// note that this function doesn't necessarily resolve all the conflicts, but can return early if some major change is made.
    /// when implementing this function, it's recommended that you resolve as many conflicts as possible.
    ///
    /// note: this is only ran in the "search" mode
    fn resolve(
        &mut self,
        group_max_update_length: GroupMaxUpdateLength,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        panic!("parallel primal module cannot handle global resolve requests, use `individual_solve` instead");
    }

    /// resolve the conflicts in the "tune" mode
    fn resolve_tune(
        &mut self,
        _group_max_update_length: BTreeSet<MaxUpdateLength>,
        _interface: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> (BTreeSet<MaxUpdateLength>, bool) {
        panic!("`resolve_tune` not implemented, this primal module does not work with tuning mode");
    }

    fn solve(
        &mut self,
        interface: &DualModuleInterfacePtr,
        syndrome_pattern: Arc<SyndromePattern>,
        dual_module: &mut impl DualModuleImpl,
    ) {
        self.solve_step_callback(interface, syndrome_pattern, dual_module, |_, _, _, _| {})
    }

    fn subgraph(&mut self, interface: &DualModuleInterfacePtr, seed: u64)
        -> Subgraph 
    {
        // implementation using rayon, however, this didnt work for since I need to update the trait of dual_module input in primal_module
        self.thread_pool.scope(|_| {
            let results: Vec<_> = 
                self.units.par_iter().filter_map(| unit_ptr| {
                    let mut unit = unit_ptr.write();
                    Some(unit.subgraph(interface, seed))
                }).collect();
            let mut final_subgraph: Vec<EdgeWeak> = vec![];
            for local_subgraph in results.into_iter() {
                final_subgraph.extend(local_subgraph);
            }
            final_subgraph
        })
    }
}

impl PrimalModuleImpl for PrimalModuleParallelUnit {
    /// create a primal module given the dual module
    /// this function needs to be implemented for dynamic fusion
    fn new_empty(solver_initializer: &SolverInitializer) -> Self {
        panic!("creating parallel unit directly from initializer is forbidden, use `PrimalModuleParallel::new` instead");
    }

    /// clear all states; however this method is not necessarily called when load a new decoding problem, so you need to call it yourself
    fn clear(&mut self) {
        self.serial_module.clear();
        self.interface_ptr.clear();
    }

    /// load a new decoding problem given dual interface: note that all nodes MUST be defect node
    fn load<D: DualModuleImpl>(&mut self, interface_ptr: &DualModuleInterfacePtr, dual_module: &mut D) {
        self.serial_module.load(interface_ptr, dual_module);
    }

    /// analyze the reason why dual module cannot further grow, update primal data structure (alternating tree, temporary matches, etc)
    /// and then tell dual module what to do to resolve these conflicts;
    /// note that this function doesn't necessarily resolve all the conflicts, but can return early if some major change is made.
    /// when implementing this function, it's recommended that you resolve as many conflicts as possible.
    ///
    /// note: this is only ran in the "search" mode
    fn resolve(
        &mut self,
        group_max_update_length: GroupMaxUpdateLength,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        self.serial_module.resolve(group_max_update_length, interface, dual_module)
    }

    /// resolve the conflicts in the "tune" mode
    fn resolve_tune(
        &mut self,
        group_max_update_length: BTreeSet<MaxUpdateLength>,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> (BTreeSet<MaxUpdateLength>, bool) {
        self.serial_module.resolve_tune(group_max_update_length, interface, dual_module)
    }

    fn subgraph(&mut self, interface: &DualModuleInterfacePtr, seed: u64)
        -> Subgraph 
    {
        self.serial_module.subgraph(interface, seed)
    }

    fn subgraph_range(
        &mut self,
        interface: &DualModuleInterfacePtr,
        seed: u64,
    ) -> (Subgraph, WeightRange) {
        self.serial_module.subgraph_range(interface, seed)
    }
}



impl MWPSVisualizer for PrimalModuleParallel {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        // do the sanity check first before taking snapshot
        // self.sanity_check().unwrap();
        let mut value = json!({});
        for unit_ptr in self.units.iter() {
            let unit = unit_ptr.read_recursive();
            // if !unit.is_active {
            //     continue;
            // } // do not visualize inactive units
            let value_2 = unit.snapshot(abbrev);
            snapshot_combine_values(&mut value, value_2, abbrev);
        }
        value
    }
}

impl MWPSVisualizer for PrimalModuleParallelUnit {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        self.serial_module.snapshot(abbrev)
    }
}



#[cfg(test)]
pub mod tests {
    use super::super::example_codes::*;
    use super::super::primal_module::*;

    use super::super::primal_module_serial::*;
    use crate::decoding_hypergraph::*;
    use super::*;
    use crate::num_traits::FromPrimitive;

    use crate::plugin_single_hair::PluginSingleHair;
    use crate::plugin_union_find::PluginUnionFind;
    use crate::plugin::PluginVec;
    use crate::dual_module_serial::*;
    use crate::dual_module_pq::*;
    
    pub fn primal_module_parallel_basic_standard_syndrome(
        code: impl ExampleCode,
        visualize_filename: String,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        growing_strategy: GrowingStrategy,
    ) -> (
        PrimalModuleParallel,
        impl DualModuleImpl + MWPSVisualizer,
    ) {
        println!("{defect_vertices:?}");
        let visualizer = {
            let visualizer = Visualizer::new(
                Some(visualize_data_folder() + visualize_filename.as_str()),
                code.get_positions(),
                true,
            )
            .unwrap();
            print_visualize_link(visualize_filename.clone());
            visualizer
        };

        // create dual module
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 18),   // unit 0
            VertexRange::new(24, 42), // unit 1
        ];
        partition_config.fusions = vec![
                    (0, 1), // unit 2, by fusing 0 and 1
                ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);
        partition_config.defect_vertices = BTreeSet::from_iter(defect_vertices.clone());

        let partition_info = partition_config.info();


        let mut dual_module_parallel_config = DualModuleParallelConfig::default();
        // dual_module_parallel_config.enable_parallel_execution = true;
        let mut dual_module: DualModuleParallel<DualModulePQ<FutureObstacleQueue<Rational>>, FutureObstacleQueue<Rational>> =
            DualModuleParallel::new_config(&initializer, &partition_info, dual_module_parallel_config);

        // create primal module
        let primal_config = PrimalModuleParallelConfig {..Default::default()};
        let primal_module = PrimalModuleParallel::new_config(&model_graph.initializer, &partition_info, primal_config.clone());
        // primal_module.growing_strategy = growing_strategy;
        // primal_module.plugins = Arc::new(plugins);
        // primal_module.config = serde_json::from_value(json!({"timeout":1})).unwrap();

        primal_module_parallel_basic_standard_syndrome_optional_viz(
            code,
            defect_vertices,
            final_dual,
            plugins,
            growing_strategy,
            dual_module,
            primal_module,
            model_graph,
            Some(visualizer),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn primal_module_parallel_basic_standard_syndrome_optional_viz<Queue>
    (
        _code: impl ExampleCode,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        growing_strategy: GrowingStrategy,
        mut dual_module: DualModuleParallel<DualModulePQ<Queue>, Queue>,
        mut primal_module: PrimalModuleParallel,
        model_graph: Arc<crate::model_hypergraph::ModelHyperGraph>,
        mut visualizer: Option<Visualizer>,
    ) -> (
        PrimalModuleParallel,
        impl DualModuleImpl + MWPSVisualizer,
    ) 
    where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
    {
        // try to work on a simple syndrome
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph, defect_vertices.clone());
        primal_module.parallel_solve_visualizer(
            decoding_graph.syndrome_pattern.clone(),
            &mut dual_module,
            visualizer.as_mut(),
        );


        // let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, 0);
        // if let Some(visualizer) = visualizer.as_mut() {
        //     let last_interface_ptr = &primal_module.units.last().unwrap().read_recursive().interface_ptr;
        //     visualizer
        //         .snapshot_combined(
        //             "subgraph".to_string(),
        //             vec![last_interface_ptr, &dual_module, &subgraph, &weight_range],
        //         )
        //         .unwrap();
        // }
        // assert!(
        //     decoding_graph
        //         .model_graph
        //         .matches_subgraph_syndrome(&subgraph, &defect_vertices),
        //     "the result subgraph is invalid"
        // );
        // assert_eq!(
        //     Rational::from_usize(final_dual).unwrap(),
        //     weight_range.upper,
        //     "unmatched sum dual variables"
        // );
        // assert_eq!(
        //     Rational::from_usize(final_dual).unwrap(),
        //     weight_range.lower,
        //     "unexpected final dual variable sum"
        // );
        (primal_module, dual_module)
    }

    /// test a simple case
    #[test]
    fn primal_module_parallel_tentative_test_1() {
        // RUST_BACKTRACE=1 cargo test primal_module_parallel_tentative_test_1 -- --nocapture
        let weight = 1; // do not change, the data is hard-coded
        let code = CodeCapacityPlanarCode::new(7, 0.1, weight);
        let defect_vertices = vec![19, 35];

        let visualize_filename = "primal_module_parallel_tentative_test_1.json".to_string();
        primal_module_parallel_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            4,
            vec![],
            GrowingStrategy::SingleCluster,
        );
    }
}