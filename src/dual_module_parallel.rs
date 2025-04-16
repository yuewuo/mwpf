/// Parallel Implementation of Dual Module PQ

#[cfg_attr(feature = "unsafe_pointer", allow(dropping_references))]
use super::dual_module_pq::*;
use super::pointers::*;
use super::util::*;
use super::visualize::*;
use crate::dual_module::DualModuleImpl;
use crate::num_traits::{ToPrimitive, Zero};
use crate::ordered_float::OrderedFloat;
use crate::primal_module::Affinity;
use crate::primal_module_serial::PrimalClusterPtr;
use crate::rayon::prelude::*;
use crate::serde_json;
use crate::{add_shared_methods, dual_module::*};
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use rayon::vec;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone)]
pub struct DualModuleParallel<SerialModule: DualModuleImpl + Send + Sync, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    /// the set of all DualModuleParallelUnits, one for each partition
    /// we set the read-write lock
    pub units: Vec<DualModuleParallelUnitPtr<SerialModule, Queue>>,
    /// configuration such as thread_pool_size
    pub config: DualModuleParallelConfig,
    /// partition information
    pub partition_info: Arc<PartitionInfo>,
    /// thread pool used to execute async functions in parallel
    pub thread_pool: Arc<rayon::ThreadPool>,
    // /// an empty sync requests queue just to implement the trait
    // pub empty_sync_request: Vec<SyncRequest>,
    /// a dynamic (to-be-update) undirected graph (DAG) to keep track of the relationship between different partition units, assumed to be acylic if we partition
    /// along the time axis, but could be cyclic depending on the partition and fusion strategy
    pub dag_partition_units: FastIterSet<(usize, usize, bool)>, // (unit_index0, unit_index1, is_fused)
    /// partitioned initializers, used in both primal and dual parallel modules
    pub partitioned_initializers: Vec<PartitionedSolverInitializer>,

    /// should think more about whether having this makes sense
    /// the current mode of the dual module
    ///     note: currently does not have too much functionality
    mode: DualModuleMode,

    /// PhantomData to account for the SerialModule parameter
    _phantom: PhantomData<SerialModule>,
}

// impl std::fmt::Debug for DualNodePtr {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         let dual_node = self.read_recursive(); // reading index is consistent
//         f.debug_struct("DualNode")
//             .field("index", &dual_node.index)
//             .field("dual_variable", &dual_node.get_dual_variable())
//             .field("grow_rate", &dual_node.grow_rate)
//             .field("hair", &dual_node.invalid_subgraph.hair.iter().map(|e| e.read_recursive().edge_index).collect::<Vec<_>>())
//             .finish()
//         // let new = ArcRwLock::new_value(Rational::zero());
//         // let global_time = dual_node.global_time.as_ref().unwrap_or(&new).read_recursive();
//         // write!(
//         //     f,
//         //     "\n\t\tindex: {}, global_time: {:?}, grow_rate: {:?}, dual_variable: {}\n\t\tdual_variable_at_last_updated_time: {}, last_updated_time: {}\n\timpacted_edges: {:?}\n",
//         //     dual_node.index,
//         //     global_time,
//         //     dual_node.grow_rate,
//         //     dual_node.get_dual_variable(),
//         //     dual_node.dual_variable_at_last_updated_time,
//         //     dual_node.last_updated_time,
//         //     dual_node.invalid_subgraph.hair
//         // )
//     }
// }

pub struct DualModuleParallelUnit<SerialModule: DualModuleImpl + Send + Sync, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    pub unit_index: usize,
    /// The corresponding serial_module, in this case, the serial module with priority queue implementation
    pub serial_module: DualModulePQGeneric<Queue>,
    /// * The serial units being fused with this serial unit.
    /// * For non-boundary unit, the initial state of this vector contains the DualModuleParallelUnit of the boundary unit (aka
    /// the unit formed by the boundary vertices of this unit). When more than one such boundary vertices units are present at initialization,
    /// we should insert them based on their respective orientation in the time-space chunk block.
    /// * For boundary unit, the initial state of this vector is the non-boundary unit it connects to.
    /// * When we fuse 2 DualModuleParallelUnit, we could only fuse a non-boundary unit with a boundary unit
    pub adjacent_parallel_units: Vec<DualModuleParallelUnitWeak<SerialModule, Queue>>,
    /// Whether this unit is a boundary unit
    pub is_boundary_unit: bool,
    /// partition info
    pub partition_info: Arc<PartitionInfo>,
    /// owning_range
    pub owning_range: VertexRange,
    pub enable_parallel_execution: bool,
    /// should think a bit more about whether having this makes sense
    /// the current mode of the dual module
    ///     note: currently does not have too much functionality
    mode: DualModuleMode,
    /// PhantomData to account for the SerialModule parameter
    _phantom: PhantomData<SerialModule>,
}

pub type DualModuleParallelUnitPtr<SerialModule, Queue> = ArcManualSafeLock<DualModuleParallelUnit<SerialModule, Queue>>;
pub type DualModuleParallelUnitWeak<SerialModule, Queue> = WeakManualSafeLock<DualModuleParallelUnit<SerialModule, Queue>>;

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> std::fmt::Debug for DualModuleParallelUnitPtr<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let unit = self.read_recursive();
        write!(f, "{}", unit.unit_index)
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> std::fmt::Debug for DualModuleParallelUnitWeak<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.upgrade_force().fmt(f)
    }
}

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> Clone for DualModuleParallel<SerialModule: DualModuleImpl + Send + Sync, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn clone(&self) -> Self {
//         Self {
//             units: self.units.clone(),

//         }
//     }
// }

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> Ord for DualModuleParallelUnitPtr<SerialModule, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn cmp(&self, other: &Self) -> Ordering {
//         // compare the pointer address
//         let ptr1 = Arc::as_ptr(self.ptr());
//         let ptr2 = Arc::as_ptr(other.ptr());
//         // https://doc.rust-lang.org/reference/types/pointer.html
//         // "When comparing raw pointers they are compared by their address, rather than by what they point to."
//         ptr1.cmp(&ptr2)
//     }
// }

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> PartialOrd for DualModuleParallelUnitPtr<SerialModule, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> Ord for DualModuleParallelUnitWeak<SerialModule, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn cmp(&self, other: &Self) -> Ordering {
//         // compare the pointer address
//         let ptr1 = Weak::as_ptr(self.ptr());
//         let ptr2 = Weak::as_ptr(other.ptr());
//         // https://doc.rust-lang.org/reference/types/pointer.html
//         // "When comparing raw pointers they are compared by their address, rather than by what they point to."
//         // println!("ptr1: {:?}", ptr1);
//         // println!("ptr2: {:?}", ptr2);
//         ptr1.cmp(&ptr2)
//     }
// }

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> PartialOrd for DualModuleParallelUnitWeak<SerialModule, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl<SerialModule: DualModuleImpl + Send + Sync, Queue> Clone for DualModuleParallelUnit<SerialModule, Queue>
// where Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
// {
//     fn clone(&self) -> Self {
//         Self {
//             unit_index: self.unit_index.clone(),
//             serial_module: self.serial_module.clone(),
//             adjacent_parallel_units: self.adjacent_parallel_units.clone(),
//             is_boundary_unit: self.is_boundary_unit.clone(),
//             partition_info: self.partition_info.clone(),
//             owning_range: self.owning_range.clone(),
//             enable_parallel_execution: self.enable_parallel_execution.clone(),
//             mode: self.mode.clone(),
//             _phantom: PhantomData,
//         }
//     }
// }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DualModuleParallelConfig {
    /// enable async execution of dual operations; only used when calling top-level operations, not used in individual units
    #[serde(default = "dual_module_parallel_default_configs::thread_pool_size")]
    pub thread_pool_size: usize,
    /// enable parallel execution of a fused dual module
    #[serde(default = "dual_module_parallel_default_configs::enable_parallel_execution")]
    pub enable_parallel_execution: bool,
}

impl Default for DualModuleParallelConfig {
    fn default() -> Self {
        serde_json::from_value(json!({})).unwrap()
    }
}

pub mod dual_module_parallel_default_configs {
    pub fn thread_pool_size() -> usize {
        0
    } // by default to the number of CPU cores
    pub fn enable_parallel_execution() -> bool {
        false
    } // by default disabled: parallel execution may cause too much context switch, yet not much speed benefit
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleParallel<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    #[allow(clippy::unnecessary_cast)]
    pub fn new_config(
        initializer: &SolverInitializer,
        partition_info: &PartitionInfo,
        config: DualModuleParallelConfig,
    ) -> Self {
        // automatic reference counter for partition info
        let partition_info = Arc::new(partition_info.clone());

        // build thread pool
        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        if config.thread_pool_size != 0 {
            thread_pool_builder = thread_pool_builder.num_threads(config.thread_pool_size);
        }
        let thread_pool = thread_pool_builder.build().expect("creating thread pool failed");

        // build partition initializer
        let mut units = vec![];
        let unit_count = partition_info.units.len();
        let mut partitioned_initializers: Vec<PartitionedSolverInitializer> = (0..unit_count)
            .map(|unit_index| {
                let unit_partition_info = &partition_info.units[unit_index];
                let owning_range = &unit_partition_info.owning_range;
                let boundary_vertices = &unit_partition_info.boundary_vertices;

                PartitionedSolverInitializer {
                    unit_index,
                    vertex_num: initializer.vertex_num,
                    edge_num: initializer.weighted_edges.len(),
                    owning_range: *owning_range,
                    weighted_edges: vec![],
                    boundary_vertices: boundary_vertices.clone(),
                    is_boundary_unit: unit_partition_info.is_boundary_unit,
                    defect_vertices: partition_info.config.defect_vertices.clone(),
                    // boundary_vertices: unit_partition_info.boundary_vertices.clone(),
                    // adjacent_partition_units: unit_partition_info.adjacent_partition_units.clone(),
                    // owning_interface: Some(partition_units[unit_index].downgrade()),
                    heralds: vec![],
                }
            })
            .collect::<Vec<_>>();

        assert!(
            initializer.heralds.is_empty(),
            "non-empty heralds are yet to be supported in DualModuleParallel",
        );

        // now we assign each edge to its unique partition
        // println!("edge num: {}", initializer.weighted_edges.len());
        for (edge_index, hyper_edge) in initializer.weighted_edges.iter().enumerate() {
            let mut vertices_unit_indices: HashMap<usize, Vec<usize>> = HashMap::new();
            let mut boundary_vertices_adjacent_units_index: HashMap<usize, Vec<usize>> = HashMap::new(); // key: unit_index; value: all vertex indices belong to this unit
            let mut exist_boundary_vertex = false;
            let mut exist_boundary_unit_index = 0;
            for vertex_index in hyper_edge.vertices.iter() {
                let unit_index = partition_info.vertex_to_owning_unit.get(vertex_index).unwrap();
                let unit = &partition_info.units[*unit_index];
                if unit.is_boundary_unit {
                    exist_boundary_vertex = true;
                    exist_boundary_unit_index = unit.unit_index;
                    if let Some(x) = boundary_vertices_adjacent_units_index.get_mut(unit_index) {
                        x.push(*vertex_index);
                    } else {
                        let mut vertices = vec![];
                        vertices.push(*vertex_index);
                        boundary_vertices_adjacent_units_index.insert(*unit_index, vertices.clone());
                    }
                } else {
                    if let Some(x) = vertices_unit_indices.get_mut(unit_index) {
                        x.push(*vertex_index);
                    } else {
                        let mut vertices = vec![];
                        vertices.push(*vertex_index);
                        vertices_unit_indices.insert(*unit_index, vertices.clone());
                    }
                }
            }

            // println!("hyper_edge index: {edge_index}");
            // println!("vertices_unit_indices: {vertices_unit_indices:?}");
            // println!("boundary vertices adjacent unit indices: {boundary_vertices_adjacent_units_index:?}");

            // if all vertices are the boundary vertices
            if vertices_unit_indices.len() == 0 {
                // we add the hyperedge to the boundary unit
                let unit_index = boundary_vertices_adjacent_units_index.keys().next().unwrap();
                partitioned_initializers[*unit_index]
                    .weighted_edges
                    .push((hyper_edge.clone(), edge_index));
            } else {
                // some vertices connected by this hyperedge are not boundary vertices
                let first_vertex_unit_index = *vertices_unit_indices.keys().next().unwrap();
                let all_vertex_from_same_unit = vertices_unit_indices.len() == 1; // whether the rest (exluding boundary vertices) are from the same unit
                if !exist_boundary_vertex {
                    // all within owning range of one unit (since for the vertices to span multiple units, one of them has to be the boundary vertex)
                    // we assume that for vertices of a hyperedge, if there aren't any boundary vertices among them, they must belong to the same partition unit
                    assert!(all_vertex_from_same_unit, "For the vertices of hyperedge {}, there does not exist boundary vertex but all the vertices do not belong to the same unit", edge_index);
                    // since all vertices this hyperedge connects to belong to the same unit, we can assign this hyperedge to that partition unit
                    partitioned_initializers[first_vertex_unit_index]
                        .weighted_edges
                        .push((hyper_edge.clone(), edge_index));
                } else {
                    // there exist boundary vertex (among the vertices this hyper_edge connects to), the rest vertices span multiple units
                    // println!("vertices span multiple units");
                    if all_vertex_from_same_unit {
                        let mut hyper_edge_clone = hyper_edge.clone();
                        hyper_edge_clone.connected_to_boundary_vertex = true;
                        partitioned_initializers[first_vertex_unit_index]
                            .weighted_edges
                            .push((hyper_edge_clone, edge_index));

                        // if vertices_unit_indices.get(&first_vertex_unit_index).unwrap().len() == 1 {
                        //     // insert this edge to the non-boundary unit
                        //     // println!("edge_index: {:?}, unit_index: {:?}", edge_index, first_vertex_unit_index);
                        //     let mut hyper_edge_clone = hyper_edge.clone();
                        //     hyper_edge_clone.connected_to_boundary_vertex = true;
                        //     partitioned_initializers[first_vertex_unit_index].weighted_edges.push((hyper_edge_clone, edge_index));
                        // } else if vertices_unit_indices.get(&first_vertex_unit_index).unwrap().len() > 1 {
                        //     // insert this edge to the boundary unit
                        //     partitioned_initializers[exist_boundary_unit_index].weighted_edges.push((hyper_edge.clone(), edge_index));
                        // } else {
                        //     panic!("cannot find the corresponding vertices in unit");
                        // }
                    } else {
                        // println!("exist boundary vertices, vertices unit indices {vertices_unit_indices:?}");
                        // if the vertices of this hyperedge (excluding the boundary vertices) belong to 2 different partition unit
                        // sanity check: there really are only 2 unique partition units
                        // let mut sanity_check = HashSet::new();
                        // for (_vertex_index, vertex_unit_index) in &vertices_unit_indices {
                        //     sanity_check.insert(vertex_unit_index);
                        // }
                        // assert!(sanity_check.len() == 2, "there are fewer than 2 or more than 2 partition units");

                        // we create new hyperedge with the boundary vertex + verticies exlusive for one partition unit
                        for (unit_index, vertices) in vertices_unit_indices.iter_mut() {
                            if let Some(boundary_vertices) = boundary_vertices_adjacent_units_index.get(unit_index) {
                                vertices.extend(boundary_vertices);
                            }
                        }

                        // now we add the boundary vertices in
                        for (unit_index, vertices) in vertices_unit_indices.iter() {
                            let mut hyper_edge_new = HyperEdge::new(vertices.clone(), hyper_edge.weight.clone());
                            hyper_edge_new.connected_to_boundary_vertex = true;
                            partitioned_initializers[*unit_index]
                                .weighted_edges
                                .push((hyper_edge_new, edge_index));
                        }
                    }
                }
            }
        }

        // now that we are done with assigning hyperedge to its unique partitions, we proceed to initialize DualModuleParallelUnit for every partition
        // print function for check during dev
        // println!("partitioned_initializers: {:?}", partitioned_initializers);
        thread_pool.scope(|_| {
            (0..unit_count)
                .into_par_iter()
                .map(|unit_index| {
                    // println!("unit_index: {unit_index}");
                    let mut dual_module: DualModulePQGeneric<Queue> =
                        DualModulePQGeneric::new_partitioned(&partitioned_initializers[unit_index]);

                    DualModuleParallelUnitPtr::new_value(DualModuleParallelUnit {
                        unit_index,
                        partition_info: Arc::clone(&partition_info),
                        owning_range: partition_info.units[unit_index].owning_range,
                        serial_module: dual_module,
                        enable_parallel_execution: config.enable_parallel_execution,
                        adjacent_parallel_units: vec![],
                        is_boundary_unit: partition_info.units[unit_index].is_boundary_unit,
                        mode: DualModuleMode::default(),
                        _phantom: PhantomData,
                    })
                })
                .collect_into_vec(&mut units);
        });

        // we iterate through the boundary units
        for boundary_unit_index in partition_info.config.partitions.len()..unit_count {
            let unit = units[boundary_unit_index].read_recursive();
            // we iterate through the vertices of the boundary unit
            for (index, vertex_ptr) in unit.serial_module.vertices.iter().enumerate() {
                let vertex_index = vertex_ptr.read_recursive().vertex_index;
                let mut vertex = vertex_ptr.write();
                // fill in the `mirrored_vertices` of vertcies for boundary-unit
                for adjacent_unit_index in partition_info.units[boundary_unit_index].adjacent_parallel_units.iter() {
                    let adjacent_unit = units[*adjacent_unit_index].read_recursive();
                    let mut offset_corresponding_mirrored_vertex = adjacent_unit.owning_range.len();
                    for adjacent_boundary_index_range in
                        partitioned_initializers[*adjacent_unit_index].boundary_vertices.iter()
                    {
                        if adjacent_boundary_index_range.contains(vertex_index) {
                            break;
                        } else {
                            offset_corresponding_mirrored_vertex += adjacent_boundary_index_range.len();
                        }
                    }

                    let corresponding_mirrored_vertex =
                        &adjacent_unit.serial_module.vertices[offset_corresponding_mirrored_vertex + index];
                    vertex.mirrored_vertices.push(corresponding_mirrored_vertex.downgrade());
                }

                // fill in the `mirrored_vertices` of vertices for non-boundary-unit
                for adjacent_unit_index in partition_info.units[boundary_unit_index].adjacent_parallel_units.iter() {
                    let adjacent_unit = units[*adjacent_unit_index].read_recursive();
                    let mut offset_corresponding_mirrored_vertex = adjacent_unit.owning_range.len();
                    for adjacent_boundary_index_range in
                        partitioned_initializers[*adjacent_unit_index].boundary_vertices.iter()
                    {
                        if adjacent_boundary_index_range.contains(vertex_index) {
                            break;
                        } else {
                            offset_corresponding_mirrored_vertex += adjacent_boundary_index_range.len();
                        }
                    }

                    // println!("offset_corresponding_mirrored_vertex: {:?}", offset_corresponding_mirrored_vertex);
                    let corresponding_mirrored_vertex_ptr =
                        &adjacent_unit.serial_module.vertices[offset_corresponding_mirrored_vertex + index];
                    let mut corresponding_mirrored_vertex = corresponding_mirrored_vertex_ptr.write();
                    // add the mirrored vertices of the boundary vertex (excluding the vertex itself) to the mirrored vertices of the corresponding mirrored vertex of the non-boundary-unit
                    for vertex_ptr0 in vertex.mirrored_vertices.iter() {
                        if !vertex_ptr0.eq(&corresponding_mirrored_vertex_ptr.downgrade()) {
                            corresponding_mirrored_vertex.mirrored_vertices.push(vertex_ptr0.clone());
                        }
                    }
                    corresponding_mirrored_vertex.mirrored_vertices.push(vertex_ptr.downgrade());
                }
            }
            drop(unit);
        }

        // // debug print
        // for vertex_ptr in units[2].read_recursive().serial_module.vertices.iter() {
        //     let vertex = vertex_ptr.read_recursive();
        //     println!("vertex {:?} in unit 2, mirrored vertices: {:?}, incident edges: {:?}", vertex.vertex_index, vertex.mirrored_vertices, vertex.edges);
        // }

        // for (edge, edge_index) in partitioned_initializers[2].weighted_edges.iter() {
        //     println!("edge index: {:?}", edge_index);
        // }

        // now we are initializing dag_partition_units
        let mut dag_partition_units = FastIterSet::new();
        let graph = &partition_info.config.dag_partition_units;
        for edge_index in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge_index).unwrap();
            dag_partition_units.insert((source.index(), target.index(), false));
        }

        Self {
            units,
            config,
            partition_info,
            thread_pool: Arc::new(thread_pool),
            dag_partition_units,
            partitioned_initializers,
            mode: DualModuleMode::default(),
            _phantom: PhantomData,
        }
    }

    /// find the parallel unit that handles this dual node, should be unique
    pub fn find_handling_parallel_unit(
        &self,
        dual_node_ptr: &DualNodePtr,
    ) -> DualModuleParallelUnitPtr<SerialModule, Queue> {
        let defect_ptr = dual_node_ptr.get_representative_vertex();
        let owning_unit_index = self
            .partition_info
            .vertex_to_owning_unit
            .get(&defect_ptr.read_recursive().vertex_index);
        match owning_unit_index {
            Some(x) => {
                let owning_unit_ptr = self.units[*x].clone();
                return owning_unit_ptr;
            }
            None => {
                panic!("This dual node {} is not contained in any partition, we cannot find a parallel unit that handles this dual node.", defect_ptr.read_recursive().vertex_index)
            }
        }
    }

    // statically fuse all units
    pub fn static_fuse_all(&mut self) {
        // we need to set the unit_active to true for all units
        for unit_index in 0..self.units.len() {
            let unit = self.units[unit_index].write();
            let mut unit_active = unit.serial_module.unit_active.write();
            *unit_active = true;
        }
        // we need to fill in the adjacent_parallel_units here
        for unit_index in 0..self.units.len() {
            let mut unit = self.units[unit_index].write();
            for adjacent_unit_index in &self.partition_info.units[unit_index].adjacent_parallel_units {
                let pointer = &self.units[*adjacent_unit_index];
                unit.adjacent_parallel_units.push(pointer.downgrade());
                // println!("adjacent_parallel_unit ptr: {:?}", Arc::as_ptr(pointer.clone().ptr()));
            }
        }
    }

    /// the current implementation iterates through all vertices. This is not efficient.
    /// What if we get rid of `boundary-unit` completely and only have `non-boundary-unit`? For example, unit 0 and unit 1
    /// could still both share the same line of `boundary_vertices`, when fusing the two units, we combine the `incident_edges` and `incident_vertices` of
    /// the boundary vertices of unit 0 and unit 1.
    pub fn static_combine_all_mirrored_vertices(&mut self) {
        for unit_index in 0..self.units.len() {
            let mut unit = self.units[unit_index].read_recursive();
            if unit.is_boundary_unit {
                for i in 0..unit.serial_module.vertices.len() {
                    let vertex_ptr = &unit.serial_module.vertices[i];
                    let mut edges_to_add = Vec::new();
                    for corresponding_mirrored_vertex in vertex_ptr.read_recursive().mirrored_vertices.iter() {
                        for edge_weak in corresponding_mirrored_vertex.upgrade_force().read_recursive().edges.iter() {
                            let edge_ptr = edge_weak.upgrade_force();
                            let mut edge = edge_ptr.write();
                            for local_vertex in edge.vertices.iter_mut() {
                                if local_vertex.eq(&corresponding_mirrored_vertex) {
                                    *local_vertex = vertex_ptr.downgrade();
                                }
                            }
                        }
                        edges_to_add.extend(
                            corresponding_mirrored_vertex
                                .upgrade_force()
                                .read_recursive()
                                .edges
                                .iter()
                                .cloned(),
                        );
                    }
                    vertex_ptr.write().edges.extend(edges_to_add);
                }
            } else {
                for i in 0..unit.serial_module.all_mirrored_vertices.len() {
                    let vertex_ptr = &unit.serial_module.all_mirrored_vertices[i];
                    let mut edges_to_add = Vec::new();
                    for corresponding_mirrored_vertex in vertex_ptr.read_recursive().mirrored_vertices.iter() {
                        for edge_weak in corresponding_mirrored_vertex.upgrade_force().read_recursive().edges.iter() {
                            let edge_ptr = edge_weak.upgrade_force();
                            let mut edge = edge_ptr.write();
                            for local_vertex in edge.vertices.iter_mut() {
                                if local_vertex.eq(&corresponding_mirrored_vertex) {
                                    *local_vertex = vertex_ptr.downgrade();
                                }
                            }
                        }
                        edges_to_add.extend(
                            corresponding_mirrored_vertex
                                .upgrade_force()
                                .read_recursive()
                                .edges
                                .iter()
                                .cloned(),
                        );
                    }
                    vertex_ptr.write().edges.extend(edges_to_add);
                }
            }
        }

        // // previous implementation
        // for unit_index in 0..self.units.len() {
        //     let mut unit = self.units[unit_index].read_recursive();
        //     for i in 0..unit.serial_module.vertices.len() {
        //         let vertex_ptr = &unit.serial_module.vertices[i];
        //         let mut edges_to_add = Vec::new();
        //         for corresponding_mirrored_vertex in vertex_ptr.read_recursive().mirrored_vertices.iter() {
        //             for edge_weak in corresponding_mirrored_vertex.upgrade_force().read_recursive().edges.iter() {
        //                 let edge_ptr = edge_weak.upgrade_force();
        //                 let mut edge = edge_ptr.write();
        //                 for local_vertex in edge.vertices.iter_mut() {
        //                     if local_vertex.eq(&corresponding_mirrored_vertex) {
        //                         *local_vertex = vertex_ptr.downgrade();
        //                     }
        //                 }
        //             }
        //             edges_to_add.extend(corresponding_mirrored_vertex.upgrade_force().read_recursive().edges.iter().cloned());

        //         }
        //         vertex_ptr.write().edges.extend(edges_to_add);
        //     }
        // }
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleImpl for DualModuleParallel<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    /// create a new dual module with empty syndrome
    fn new_empty(initializer: &Arc<SolverInitializer>) -> Self {
        Self::new_config(
            initializer,
            &PartitionConfig::new(initializer.vertex_num).info(),
            DualModuleParallelConfig::default(),
        )
    }

    /// clear all growth and existing dual nodes, prepared for the next decoding
    #[inline(never)]
    fn clear(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().enumerate().for_each(|(unit_index, unit_ptr)| {
                let mut unit = unit_ptr.write();
                unit.clear(); // to be implemented in DualModuleParallelUnit
            })
        })
    }

    /// CHECK!! add defect node
    fn add_defect_node(&mut self, dual_node_ptr: &DualNodePtr) {
        let mut unit_ptr = self.find_handling_parallel_unit(dual_node_ptr);
        self.thread_pool.scope(|_| {
            let mut unit = unit_ptr.write();
            unit.add_defect_node(dual_node_ptr);
        })
    }

    /// CHECK!! add corresponding dual node, note that the `internal_vertices` and `hair_edges` are not set
    fn add_dual_node(&mut self, dual_node_ptr: &DualNodePtr) {
        let mut unit_ptr = self.find_handling_parallel_unit(dual_node_ptr);
        self.thread_pool.scope(|_| {
            let mut unit = unit_ptr.write();
            unit.add_dual_node(dual_node_ptr);
        })
    }

    /// update grow rate
    fn set_grow_rate(&mut self, dual_node_ptr: &DualNodePtr, grow_rate: Rational) {
        let mut unit_ptr = self.find_handling_parallel_unit(dual_node_ptr);
        self.thread_pool.scope(|_| {
            let mut unit = unit_ptr.write();
            unit.set_grow_rate(dual_node_ptr, grow_rate); // to be implemented in DualModuleParallelUnit
        })
    }

    /// check the maximum length to grow (shrink) for all nodes, return a list of conflicting reason and a single number indicating the maximum rate to grow:
    /// this number will be 0 if any conflicting reason presents
    fn report(&mut self) -> DualReport {
        let mut dual_report = DualReport::new();
        let unit_ptr = &self.units[0];
        unit_ptr.bfs_report(&mut dual_report);
        dual_report
    }

    /// An optional function that can manipulate individual dual node, not necessarily supported by all implementations
    fn grow_dual_node(&mut self, dual_node_ptr: &DualNodePtr, length: Rational) {
        let mut unit_ptr = self.find_handling_parallel_unit(dual_node_ptr);
        self.thread_pool.scope(|_| {
            let mut unit = unit_ptr.write();
            unit.grow_dual_node(dual_node_ptr, length) // to be implemented in DualModuleParallelUnit
        })
    }

    // the pq implementation (serial) simply updates the global time
    fn grow(&mut self, length: Rational) {
        let unit = &self.units[0];
        unit.bfs_grow(length.clone());
    }

    /// come back later to fix the owning_edge_range contains
    fn get_edge_nodes(&self, edge_ptr: EdgePtr) -> Vec<DualNodePtr> {
        edge_ptr
            .read_recursive()
            .dual_nodes
            .iter()
            .map(|x| x.upgrade_force().ptr)
            .collect::<Vec<_>>()
    }

    fn get_edge_slack(&self, edge_ptr: EdgePtr) -> Rational {
        let edge = edge_ptr.read_recursive();
        let unit_ptr = &self.units[edge.unit_index.unwrap()];
        let mut unit = unit_ptr.write();
        unit.get_edge_slack(edge_ptr.clone())
    }

    fn is_edge_tight(&self, edge_ptr: EdgePtr) -> bool {
        self.get_edge_slack(edge_ptr).is_zero()
    }

    /* New tuning-related methods */
    // tuning mode shared methods
    add_shared_methods!();

    /// the individual units change their mode, queue is cleared, but for DualModuleParallel, it does not
    /// have a `mode` per se. Thus, this function is not implemented.
    fn advance_mode(&mut self) {
        unimplemented!()
    }

    /// reset the mode to the default. For DualModuleParallel, it does not have a `mode` per se.
    /// Thus, this function is not implemented.
    fn reset_mode(&mut self) {
        unimplemented!()
    }

    /// document the end of tuning for getting the total tuning time. For DualModuleParallel, it does not have a `mode` per se.
    /// Thus, this function is not implemented.
    fn end_tuning(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().for_each(|unit_ptr| {
                let mut unit = unit_ptr.write();
                unit.end_tuning();
            });
        })
    }

    /// get the total tuning time. For DualModuleParallel, it does not have a `mode` per se.
    /// Hence, we just simply sum up the `total_tuning_time` of all units
    fn get_total_tuning_time(&self) -> Option<f64> {
        let total: f64 = self
            .units
            .par_iter()
            .map(|unit_ptr| {
                // Use a thread-safe way to access the data if necessary
                unit_ptr.read_recursive().get_total_tuning_time().unwrap_or(0.0)
            })
            .sum();

        if total.is_zero() {
            None
        } else {
            Some(total)
        }
        // serial implementation
        // let mut total = f64::zero();
        // for unit_ptr in self.units.iter() {
        //     match unit_ptr.read_recursive().get_total_tuning_time() {
        //         Some(tuning_time) => {total += tuning_time;},
        //         None => continue,
        //     }
        // }
        // if total.is_zero() {
        //     None
        // } else {
        //     Some(total)
        // }
    }

    /// Reset: clear the tuning time. We clear the tuning time of all individual units
    fn clear_tuning_time(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().for_each(|unit_ptr| {
                let mut unit = unit_ptr.write();
                unit.clear_tuning_time();
            });
        })
    }

    /// CHECK!! "add_dual_node", but in tuning phase, don't modify the pq or the grow rates
    fn add_dual_node_tune(&mut self, dual_node_ptr: &DualNodePtr) {
        unimplemented!()
    }

    /// CHECK!! syncing all possible states (dual_variable and edge_weights) with global time, so global_time can be discarded later
    fn sync(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().for_each(|unit_ptr| {
                let mut unit = unit_ptr.write();
                unit.sync(); // to be implemented in DualModuleParallelUnit
            });
        })
    }

    /// grow a specific edge on the spot
    fn grow_edge(&self, edge_ptr: EdgePtr, amount: &Rational) {
        let mut edge = edge_ptr.write();
        edge.growth_at_last_updated_time += amount;
    }

    /// `is_edge_tight` but in tuning phase
    fn is_edge_tight_tune(&self, edge_ptr: EdgePtr) -> bool {
        let edge = edge_ptr.read_recursive();
        edge.weight == edge.growth_at_last_updated_time
    }

    /// `get_edge_slack` but in tuning phase
    fn get_edge_slack_tune(&self, edge_ptr: EdgePtr) -> Rational {
        let edge = edge_ptr.read_recursive();
        edge.weight.clone() - edge.growth_at_last_updated_time.clone()
    }

    /* miscs */

    /// print all the states for the current dual module
    fn debug_print(&self) {
        println!("this dual_module doesn't support debug print");
    }

    /* affinity */

    /// calculate affinity based on the following metric
    ///     Clusters with larger primal-dual gaps will receive high affinity because working on those clusters
    ///     will often reduce the gap faster. However, clusters with a large number of dual variables, vertices,
    ///     and hyperedges will receive a lower affinity
    fn calculate_cluster_affinity(&mut self, cluster: PrimalClusterPtr) -> Option<Affinity> {
        let mut start = 0.0;
        let cluster = cluster.read_recursive();
        start -= cluster.edges.len() as f64 + cluster.nodes.len() as f64;

        let mut weight = Rational::zero();
        for edge_ptr in cluster.edges.iter() {
            // let edge_ptr = self.edges[edge_index].read_recursive();
            let edge = edge_ptr.read_recursive();
            weight += &edge.weight - &edge.growth_at_last_updated_time;
        }
        for node in cluster.nodes.iter() {
            let dual_node = node.read_recursive().dual_node_ptr.clone();
            weight -= &dual_node.read_recursive().dual_variable_at_last_updated_time;
        }
        if weight.is_zero() {
            return None;
        }
        start += weight.to_f64().unwrap();
        Some(OrderedFloat::from(start))
    }

    // fn get_obstacles_tune(
    //     &self,
    //     optimizer_result: OptimizerResult,
    //     dual_node_deltas: FastIterMap<OrderedDualNodePtr, (Rational, NodeIndex)>,
    // ) -> FastIterSet<Obstacle> {
    //     unimplemented!()
    // }

    /// get the edge free weight, for each edge what is the weight that are free to use by the given participating dual variables
    fn get_edge_free_weight(&self, edge_ptr: EdgePtr, participating_dual_variables: &hashbrown::HashSet<usize>) -> Rational {
        let edge = edge_ptr.read_recursive();
        let mut free_weight = edge.weight.clone();
        for dual_node in edge.dual_nodes.iter() {
            if participating_dual_variables.contains(&dual_node.index) {
                continue;
            }
            let dual_node = dual_node.upgrade_force();
            free_weight -= &dual_node.ptr.read_recursive().dual_variable_at_last_updated_time;
        }

        free_weight
    }

    fn get_edge_weight(&self, edge_ptr: EdgePtr) -> Rational {
        edge_ptr.read_recursive().weight.clone()
    }

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights(&self, edge_ptr: EdgePtr, cluster_index: NodeIndex, grow_rate: Rational);

    #[cfg(feature = "incr_lp")]
    fn get_edge_free_weight_cluster(&self, edge_ptr: EdgePtr, cluster_index: NodeIndex) -> Rational;

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights_union(
        &self,
        dual_node_ptr: &DualNodePtr,
        drained_cluster_index: NodeIndex,
        absorbing_cluster_index: NodeIndex,
    );

    // CHECK!!
    fn adjust_weights_for_negative_edges(&mut self) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().for_each(|unit_ptr| {
                let mut unit = unit_ptr.write();
                unit.adjust_weights_for_negative_edges();
            });
        })
    }

    /// CHECK!! update weights of dual_module;
    /// the weight of the dual module is set to be `old_weight + mix_ratio * (new_weight - old_weight)`
    fn update_weights(&mut self, new_weights: Vec<Rational>, mix_ratio: Weight) {
        self.thread_pool.scope(|_| {
            self.units.par_iter().for_each(|unit_ptr| {
                let mut unit = unit_ptr.write();
                unit.update_weights(new_weights.clone(), mix_ratio);
            });
        })
    }

    fn get_negative_weight_sum(&self) -> Rational {
        let mut negative_weight_sum = Rational::zero();
        for unit in self.units.iter() {
            negative_weight_sum += unit.read_recursive().get_negative_weight_sum();
        }
        negative_weight_sum
    }

    fn get_negative_edges(&self) -> HashSet<EdgeIndex> {
        let mut negative_edges = HashSet::new();
        for unit in self.units.iter() {
            negative_edges.extend(unit.read_recursive().get_negative_edges());
        }
        negative_edges
    }

    fn get_flip_vertices(&self) -> HashSet<VertexIndex> {
        let mut flip_vertices = HashSet::new();
        for unit in self.units.iter() {
            flip_vertices.extend(unit.read_recursive().get_flip_vertices());
        }
        flip_vertices
    }

    /// exist for testing purposes
    fn get_vertex_ptr(&self, vertex_index: VertexIndex) -> VertexPtr {
        for unit in self.units.iter() {
            if unit.read_recursive().owning_range.contains(vertex_index) {
                return unit.read_recursive().get_vertex_ptr(vertex_index);
            }
        }
        panic!("none of the units in DualModuleParallel contain vertex_index, cannot find the corresponding vertex pointer");
    }

    /// exist for testing purposes
    fn get_edge_ptr(&self, edge_index: EdgeIndex) -> EdgePtr {
        for unit in self.units.iter() {
            if unit.read_recursive().owning_range.contains(edge_index) {
                return unit.read_recursive().get_edge_ptr(edge_index);
            }
        }
        panic!("none of the units in DualModuleParallel contain vertex_index, cannot find the corresponding vertex pointer");
    }

    fn get_vertex_ptr_vec(&self, vertex_indices: &[VertexIndex]) -> Vec<VertexPtr> {
        unimplemented!()
    }

    fn get_edge_ptr_vec(&self, edge_indices: &[EdgeIndex]) -> Vec<EdgePtr> {
        unimplemented!()
    }

    fn get_vertex_num(&self) -> usize {
        unimplemented!()
    }

    fn get_edge_num(&self) -> usize {
        unimplemented!()
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleParallelUnitPtr<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    /// check the maximum length to grow (shrink) for all nodes, return a list of conflicting reason and a single number indicating the maximum rate to grow:
    /// this number will be 0 if any conflicting reason presents
    pub fn report(&mut self) -> DualReport {
        let mut group_max_update_length = DualReport::new();
        self.bfs_report(&mut group_max_update_length);
        group_max_update_length
    }

    /// grow a specific length globally, length must be positive.
    /// note that a negative growth should be implemented by reversing the speed of each dual node
    pub fn grow(&mut self, length: Rational) {
        // println!("grow by length: {:?}", length);
        self.bfs_grow(length.clone());
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleImpl for DualModuleParallelUnit<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    /// create a new dual module with empty syndrome
    fn new_empty(initializer: &Arc<SolverInitializer>) -> Self {
        // tentative, but in the future, I need to modify this so that I can create a new PartitionUnit and fuse it with an existing bigger block
        panic!("creating parallel unit directly from initializer is forbidden, use `DualModuleParallel::new` instead");
    }

    /// clear all growth and existing dual nodes, prepared for the next decoding
    fn clear(&mut self) {
        self.serial_module.clear();
    }

    /// add defect node
    fn add_defect_node(&mut self, dual_node_ptr: &DualNodePtr) {
        self.serial_module.add_defect_node(dual_node_ptr);
    }

    /// add corresponding dual node, note that the `internal_vertices` and `hair_edges` are not set
    fn add_dual_node(&mut self, dual_node_ptr: &DualNodePtr) {
        self.serial_module.add_dual_node(dual_node_ptr);
    }

    /// update grow rate
    fn set_grow_rate(&mut self, dual_node_ptr: &DualNodePtr, grow_rate: Rational) {
        self.serial_module.set_grow_rate(dual_node_ptr, grow_rate);
    }

    /// check the maximum length to grow (shrink) for all nodes, return a list of conflicting reason and a single number indicating the maximum rate to grow:
    /// this number will be 0 if any conflicting reason presents
    fn report(&mut self) -> DualReport {
        // we need this to run on the newly (additionally) added unit
        self.serial_module.report()
        // we should not need this, refer to the `compute_maximum_update_length()` implementation in DualModuleParallelUnitPtr
        // unimplemented!()
    }

    /// An optional function that can manipulate individual dual node, not necessarily supported by all implementations
    fn grow_dual_node(&mut self, _dual_node_ptr: &DualNodePtr, _length: Rational) {
        unimplemented!()
    }

    // /// An optional function that can manipulate individual dual node, not necessarily supported by all implementations
    // fn grow_dual_node(&mut self, dual_node_ptr: &DualNodePtr, length: Rational) {
    //     let defect_vertex = dual_node_ptr.get_representative_vertex();
    //     println!("grow_dual_node: defect vertex found from dual node ptr is {}", defect_vertex.read_recursive().vertex_index);
    //     let mut visited: HashSet<usize> = HashSet::new();
    //     self.dfs_grow_dual_node(dual_node_ptr, length, defect_vertex, &mut visited);
    // }

    /// grow a specific length globally, length must be positive.
    /// note that a negative growth should be implemented by reversing the speed of each dual node
    fn grow(&mut self, length: Rational) {
        // we need this to run on the newly (additionally) added unit
        self.serial_module.grow(length);
        // we should not need this, refer to the `grow()` implementation in DualModuleParallelUnitPtr
        // unimplemented!()
    }

    fn get_edge_nodes(&self, edge_ptr: EdgePtr) -> Vec<DualNodePtr> {
        self.serial_module.get_edge_nodes(edge_ptr)
    }
    fn get_edge_slack(&self, edge_ptr: EdgePtr) -> Rational {
        self.serial_module.get_edge_slack(edge_ptr)
    }
    fn is_edge_tight(&self, edge_ptr: EdgePtr) -> bool {
        self.serial_module.is_edge_tight(edge_ptr)
    }

    /* New tuning-related methods */
    /// mode mangements
    // tuning mode shared methods
    // self.write().serial_module.add_shared_methods!();
    /// Returns a reference to the mode field.
    fn mode(&self) -> &DualModuleMode {
        &self.mode
    }

    /// Returns a mutable reference to the mode field.
    fn mode_mut(&mut self) -> &mut DualModuleMode {
        &mut self.mode
    }

    fn advance_mode(&mut self) {
        self.serial_module.advance_mode();
    }

    /// reset the mode to the default
    fn reset_mode(&mut self) {
        *self.mode_mut() = DualModuleMode::default();
    }

    /// document the end of tuning for getting the total tuning time
    fn end_tuning(&mut self) {
        self.serial_module.end_tuning()
    }

    /// get the total tuning time
    fn get_total_tuning_time(&self) -> Option<f64> {
        self.serial_module.get_total_tuning_time()
    }

    /// Reset: clear the tuning time
    fn clear_tuning_time(&mut self) {
        self.serial_module.clear_tuning_time();
    }

    /// "add_dual_node", but in tuning phase, don't modify the pq or the grow rates
    fn add_dual_node_tune(&mut self, dual_node_ptr: &DualNodePtr) {
        eprintln!("this dual_module does not implement tuning");
        self.add_dual_node(dual_node_ptr);
    }

    /// syncing all possible states (dual_variable and edge_weights) with global time, so global_time can be discarded later
    fn sync(&mut self) {
        self.serial_module.sync();
    }

    /// grow a specific edge on the spot
    fn grow_edge(&self, edge_ptr: EdgePtr, amount: &Rational) {
        self.serial_module.grow_edge(edge_ptr, amount);
    }

    /// `is_edge_tight` but in tuning phase
    fn is_edge_tight_tune(&self, edge_ptr: EdgePtr) -> bool {
        self.serial_module.is_edge_tight_tune(edge_ptr)
    }

    /// `get_edge_slack` but in tuning phase
    fn get_edge_slack_tune(&self, edge_ptr: EdgePtr) -> Rational {
        self.serial_module.get_edge_slack_tune(edge_ptr)
    }

    /* miscs */

    /// print all the states for the current dual module
    fn debug_print(&self) {
        self.serial_module.debug_print();
    }

    /* affinity */

    /// calculate affinity based on the following metric
    ///     Clusters with larger primal-dual gaps will receive high affinity because working on those clusters
    ///     will often reduce the gap faster. However, clusters with a large number of dual variables, vertices,
    ///     and hyperedges will receive a lower affinity
    fn calculate_cluster_affinity(&mut self, cluster: PrimalClusterPtr) -> Option<Affinity> {
        self.serial_module.calculate_cluster_affinity(cluster)
    }

    // fn get_obstacles_tune(
    //     &self,
    //     optimizer_result: OptimizerResult,
    //     dual_node_deltas: FastIterMap<OrderedDualNodePtr, (Rational, NodeIndex)>,
    // ) -> FastIterSet<Obstacle> {
    //     unimplemented!()
    // }

    /// get the edge free weight, for each edge what is the weight that are free to use by the given participating dual variables
    fn get_edge_free_weight(&self, edge_ptr: EdgePtr, participating_dual_variables: &hashbrown::HashSet<usize>) -> Rational {
        self.serial_module
            .get_edge_free_weight(edge_ptr, participating_dual_variables)
    }

    fn get_edge_weight(&self, edge_ptr: EdgePtr) -> Rational {
        unimplemented!()
    }

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights(&self, edge_ptr: EdgePtr, cluster_index: NodeIndex, grow_rate: Rational);

    #[cfg(feature = "incr_lp")]
    fn get_edge_free_weight_cluster(&self, edge_ptr: EdgePtr, cluster_index: NodeIndex) -> Rational;

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights_union(
        &self,
        dual_node_ptr: &DualNodePtr,
        drained_cluster_index: NodeIndex,
        absorbing_cluster_index: NodeIndex,
    );

    fn adjust_weights_for_negative_edges(&mut self) {
        self.serial_module.adjust_weights_for_negative_edges();
    }

    /// update weights of dual_module;
    /// the weight of the dual module is set to be `old_weight + mix_ratio * (new_weight - old_weight)`
    fn update_weights(&mut self, _new_weights: Vec<Rational>, _mix_ratio: Weight) {
        unimplemented!()
    }

    fn get_negative_weight_sum(&self) -> Rational {
        self.serial_module.get_negative_weight_sum()
    }

    fn get_negative_edges(&self) -> HashSet<EdgeIndex> {
        self.serial_module.get_negative_edges()
    }

    fn get_flip_vertices(&self) -> HashSet<VertexIndex> {
        self.serial_module.get_flip_vertices()
    }

    /// exist for testing purposes
    fn get_vertex_ptr(&self, vertex_index: VertexIndex) -> VertexPtr {
        let local_vertex_index = vertex_index - self.owning_range.start();
        self.serial_module.get_vertex_ptr(local_vertex_index)
    }

    /// exist for testing purposes
    fn get_edge_ptr(&self, edge_index: EdgeIndex) -> EdgePtr {
        let local_edge_index = edge_index - self.owning_range.start();
        self.serial_module.get_edge_ptr(local_edge_index)
    }

    fn get_vertex_ptr_vec(&self, vertex_indices: &[VertexIndex]) -> Vec<VertexPtr> {
        unimplemented!()
    }

    fn get_edge_ptr_vec(&self, edge_indices: &[EdgeIndex]) -> Vec<EdgePtr> {
        unimplemented!()
    }

    fn get_vertex_num(&self) -> usize {
        unimplemented!()
    }

    fn get_edge_num(&self) -> usize {
        unimplemented!()
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleParallelUnit<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    pub fn new_seperate_config(
        initializer: &SolverInitializer,
        seperate_partition_info: &PartitionInfo,
        config: DualModuleParallelConfig,
    ) -> Self {
        let seperate_partition_info = Arc::new(seperate_partition_info.clone());

        let unit_partition_info = &seperate_partition_info.units[0];
        let owning_range = &unit_partition_info.owning_range;
        let boundary_vertices = &unit_partition_info.boundary_vertices;

        let partitioned_solver_initializer = PartitionedSolverInitializer {
            unit_index: unit_partition_info.unit_index,
            vertex_num: initializer.vertex_num,
            edge_num: initializer.weighted_edges.len(),
            owning_range: *owning_range,
            weighted_edges: initializer
                .weighted_edges
                .iter()
                .enumerate()
                .map(|e| (e.1.clone(), e.0))
                .collect::<Vec<_>>(),
            boundary_vertices: boundary_vertices.clone(),
            is_boundary_unit: unit_partition_info.is_boundary_unit,
            defect_vertices: seperate_partition_info.config.defect_vertices.clone(),
            // boundary_vertices: unit_partition_info.boundary_vertices.clone(),
            // adjacent_partition_units: unit_partition_info.adjacent_partition_units.clone(),
            // owning_interface: Some(partition_units[unit_index].downgrade()),
            heralds: vec![],
        };

        assert!(
            initializer.heralds.is_empty(),
            "non-empty heralds are yet to be supported in DualModuleParallel (new_separate_config",
        );

        let mut dual_module: DualModulePQGeneric<Queue> =
            DualModulePQGeneric::new_seperate_unit(&partitioned_solver_initializer);

        DualModuleParallelUnit {
            unit_index: partitioned_solver_initializer.unit_index,
            partition_info: Arc::clone(&seperate_partition_info),
            owning_range: *owning_range,
            serial_module: dual_module,
            enable_parallel_execution: config.enable_parallel_execution,
            adjacent_parallel_units: vec![],
            is_boundary_unit: unit_partition_info.is_boundary_unit,
            mode: DualModuleMode::default(),
            _phantom: PhantomData,
        }
    }
}

impl<SerialModule: DualModuleImpl + Send + Sync, Queue> DualModuleParallelUnitPtr<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    // I do need to iteratively grow all the neighbors, instead I only grow this unit
    // this helps me to reduce the time complexity of copying all the nodes from one interface to the other during fusion
    pub fn bfs_grow(&self, length: Rational) {
        let mut dual_module_unit = self.write();
        if dual_module_unit.enable_parallel_execution {
            // println!("enable parallel execution");
            // implementation using rayon without locks
            // early terminate if no active dual nodes in this partition unit
            // if !self.has_active_node {
            //     return;
            // }
            // println!("bfs grow");

            // dual_module_unit.serial_module.grow(length.clone());
            // drop(dual_module_unit);
            // let dual_module_unit = self.read_recursive();

            // // could potentially use rayon to optimize it
            // // implement a breadth first search to grow all connected (fused) neighbors
            // let mut queue = VecDeque::new();
            // let mut visited = FastIterSet::new();
            // visited.insert(self.clone());
            // queue.push_back(self.clone());
            // drop(dual_module_unit);

            // while let Some(node) = {
            //     queue.pop_front()
            // } {
            //     let neighbors = &node.read_recursive().adjacent_parallel_units;

            //     neighbors.par_iter().for_each(|neighbor| {
            //         if !visited.contains(&neighbor) {
            //             neighbor.write().serial_module.grow(length.clone());
            //             visited.insert(neighbor.clone());
            //             queue.push_back(neighbor.clone());
            //         }
            //     });
            // }

            // implementation using rayon with locks
            // early terminate if no active dual nodes in this partition unit
            // if !self.has_active_node {
            //     return;
            // }
            // println!("bfs grow");

            dual_module_unit.serial_module.grow(length.clone());
            drop(dual_module_unit);
            let dual_module_unit = self.read_recursive();

            // could potentially use rayon to optimize it
            // implement a breadth first search to grow all connected (fused) neighbors
            let queue = Arc::new(Mutex::new(VecDeque::new()));
            let visited = Arc::new(Mutex::new(FastIterSet::new()));

            let mut visited_lock = visited.lock().unwrap();
            visited_lock.insert(self.downgrade());
            drop(visited_lock);

            let mut queue_lock = queue.lock().unwrap();
            queue_lock.push_back(self.downgrade());
            drop(queue_lock);
            drop(dual_module_unit);

            while let Some(node) = {
                let mut queue_lock = queue.lock().unwrap();
                queue_lock.pop_front()
            } {
                let neighbors_ptr = &node.upgrade_force();
                let neighbors = &neighbors_ptr.read_recursive().adjacent_parallel_units;

                neighbors.par_iter().for_each(|neighbor| {
                    let mut visited_lock = visited.lock().unwrap();
                    let mut queue_lock = queue.lock().unwrap();

                    if !visited_lock.contains(&neighbor) {
                        if *neighbor
                            .upgrade_force()
                            .read_recursive()
                            .serial_module
                            .unit_active
                            .read_recursive()
                        {
                            neighbor.upgrade_force().write().serial_module.grow(length.clone());
                            queue_lock.push_back(neighbor.clone());
                        }
                        visited_lock.insert(neighbor.clone());
                    }
                });
            }
        } else {
            //  implementation using sequential for loop, we need to compare the resolve time of this and the version using rayon
            dual_module_unit.serial_module.grow(length.clone());
            drop(dual_module_unit);
            let dual_module_unit = self.read_recursive();
            // could potentially use rayon to optimize it
            // implement a breadth first search to grow all connected (fused) neighbors
            let mut frontier: VecDeque<_> = VecDeque::new();
            let mut visited = FastIterSet::new();
            // println!("index: {:?}", self.unit_index);
            // visited.insert(Arc::as_ptr(self.ptr()));
            visited.insert(self.downgrade());
            // println!("self pointer: {:?}", Arc::as_ptr(self.ptr()));

            for neighbor in dual_module_unit.adjacent_parallel_units.iter() {
                // println!("first neighbor pointer: {:?}", Arc::as_ptr(neighbor.ptr()));
                frontier.push_front(neighbor.clone());
            }

            drop(dual_module_unit);
            while !frontier.is_empty() {
                // println!("frontier len: {:?}", frontier.len());
                let temp = frontier.pop_front().unwrap();
                // println!("frontier len: {:?}", frontier.len());

                if *temp
                    .upgrade_force()
                    .read_recursive()
                    .serial_module
                    .unit_active
                    .read_recursive()
                {
                    temp.upgrade_force().write().serial_module.grow(length.clone());
                }

                // visited.insert(Arc::as_ptr(temp.ptr()));
                visited.insert(temp.clone());
                // println!("temp pointer: {:?}",  Arc::as_ptr(temp.ptr()));
                // println!("temp index: {:?}", temp.unit_index);
                // println!("len: {:?}", temp.adjacent_parallel_units.len());

                for neighbor in temp.upgrade_force().read_recursive().adjacent_parallel_units.iter() {
                    // println!("hihi");
                    // println!("neighbor pointer: {:?}", Arc::as_ptr(neighbor.ptr()));
                    // if !visited.contains(&Arc::as_ptr(neighbor.ptr())) {
                    //     frontier.push_back(neighbor.clone());
                    // }
                    if !visited.contains(neighbor) {
                        frontier.push_back(neighbor.clone());
                    }
                    // println!("frontier len: {:?}", frontier.len());
                }
                drop(temp);
                // println!("after for loop");
            }
        }
    }

    fn bfs_report(&self, dual_report: &mut DualReport) {
        let mut dual_module_unit = self.write();
        if dual_module_unit.enable_parallel_execution {
            let serial_module_dual_report = dual_module_unit.serial_module.report();
            // println!("serial_module group max_update length: {:?}", serial_module_group_max_update_length);
            drop(dual_module_unit);
            let dual_module_unit = self.read_recursive();
            dual_report.extend(serial_module_dual_report);

            // implement a breadth first search to grow all connected (fused) neighbors
            let queue = Arc::new(Mutex::new(VecDeque::new()));
            let visited = Arc::new(Mutex::new(FastIterSet::new()));

            let mut visited_lock = visited.lock().unwrap();
            visited_lock.insert(self.downgrade());
            drop(visited_lock);

            let mut queue_lock = queue.lock().unwrap();
            queue_lock.push_back(self.downgrade());
            drop(queue_lock);
            drop(dual_module_unit);

            let local_dual_report = Arc::new(Mutex::new(DualReport::new()));
            while let Some(node) = {
                let mut queue_lock = queue.lock().unwrap();
                queue_lock.pop_front()
            } {
                let neighbors_ptr = node.upgrade_force();
                let neighbors = &neighbors_ptr.read_recursive().adjacent_parallel_units;

                neighbors.par_iter().for_each(|neighbor| {
                    let mut visited_lock = visited.lock().unwrap();
                    let mut queue_lock = queue.lock().unwrap();

                    if !visited_lock.contains(&neighbor) {
                        if *neighbor
                            .upgrade_force()
                            .read_recursive()
                            .serial_module
                            .unit_active
                            .read_recursive()
                        {
                            let serial_module_dual_report = neighbor.upgrade_force().write().serial_module.report();
                            local_dual_report.lock().unwrap().extend(serial_module_dual_report);
                            queue_lock.push_back(neighbor.clone());
                        }

                        visited_lock.insert(neighbor.clone());
                    }
                });
            }

            let final_local_dual_report = local_dual_report.lock().unwrap();
            dual_report.extend(final_local_dual_report.clone());
        } else {
            // implementation with sequential iteration of neighbors
            // early terminate if no active dual nodes anywhere in the descendant

            // println!("bfs_compute_max_update_length");

            let serial_module_dual_report = dual_module_unit.serial_module.report();
            // println!("serial_module group max_update length: {:?}", serial_module_group_max_update_length);
            drop(dual_module_unit);
            let dual_module_unit = self.read_recursive();

            dual_report.extend(serial_module_dual_report);

            // we need to find the maximum update length of all connected (fused) units
            // so we run a bfs, we could potentially use rayon to optimize it
            let mut frontier: VecDeque<_> = VecDeque::new();
            let mut visited = FastIterSet::new();
            visited.insert(self.downgrade());
            // println!("self pointer: {:?}", Arc::as_ptr(self.ptr()));

            for neighbor in dual_module_unit.adjacent_parallel_units.iter() {
                // println!("first neighbor pointer: {:?}", Weak::as_ptr(neighbor.ptr()));
                frontier.push_front(neighbor.clone());
            }

            while !frontier.is_empty() {
                // println!("frontier len: {:?}", frontier.len());
                let temp = frontier.pop_front().unwrap();
                // println!("frontier len: {:?}", frontier.len());
                if *temp
                    .upgrade_force()
                    .read_recursive()
                    .serial_module
                    .unit_active
                    .read_recursive()
                {
                    let serial_module_dual_report = temp.upgrade_force().write().serial_module.report();
                    // println!("temp serial_module_group_max_update_length: {:?}", serial_module_group_max_update_length);
                    dual_report.extend(serial_module_dual_report);
                    visited.insert(temp.clone());
                    for neighbor in temp.upgrade_force().read_recursive().adjacent_parallel_units.iter() {
                        // println!("hihi");
                        // println!("neighbor pointer: {:?}", Arc::as_ptr(neighbor.ptr()));
                        if !visited.contains(neighbor) {
                            frontier.push_back(neighbor.clone());
                        }
                        // println!("frontier len: {:?}", frontier.len());
                    }
                } else {
                    visited.insert(temp.clone());
                }

                // println!("temp pointer: {:?}",  Arc::as_ptr(temp.ptr()));

                drop(temp);
                // println!("after for loop");
            }
        }
    }
}

// now we implement the visualization functions
impl<SerialModule: DualModuleImpl + MWPSVisualizer + Send + Sync, Queue> MWPSVisualizer
    for DualModuleParallel<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        // do the sanity check first before taking snapshot
        // self.sanity_check().unwrap();
        let mut value = json!({});
        for unit_ptr in self.units.iter() {
            let unit = unit_ptr.read_recursive();
            let value_2 = unit.snapshot(abbrev);
            // println!("value in unit {}: {}", unit.unit_index, value_2);
            // let value = value.as_object_mut().expect("snapshot must be an object");
            // let value_2 = value_2.as_object_mut().expect("snapshot must be an object");
            snapshot_combine_values(&mut value, value_2, abbrev);
            // println!("\n\n");
            // println!("after combine: {}", value);
        }
        value
    }
}

// now we proceed to implement the visualization tool
impl<SerialModule: DualModuleImpl + MWPSVisualizer + Send + Sync, Queue> MWPSVisualizer
    for DualModuleParallelUnit<SerialModule, Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Send + Sync + Clone,
{
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        // println!("snapshot unit index {}", self.unit_index);
        self.serial_module.snapshot(abbrev)
    }
}

#[cfg(test)]
pub mod tests {
    use std::usize::MAX;

    // use slp::Solver;

    use super::super::example_codes::*;
    use super::super::primal_module::*;
    use super::super::primal_module_serial::*;
    use super::*;
    use crate::decoding_hypergraph::*;
    use crate::num_traits::FromPrimitive;

    use crate::model_hypergraph::ModelHyperGraph;
    use crate::plugin::PluginVec;
    use crate::plugin_single_hair::PluginSingleHair;
    use crate::plugin_union_find::PluginUnionFind;

    #[test]
    fn dual_module_parallel_test_1() {
        // cargo test dual_module_parallel_test_1 -- --nocapture
        let visualize_filename = "dual_module_parallel_test_1.json".to_string();
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let weight: f64 = 2.1972245773362196;
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        // visualizer.snapshot("code".to_string(), &code).unwrap();

        // create dual module
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        println!("initializer: {:?}", initializer);
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 18),  // unit 0
            VertexRange::new(24, 42), // unit 1
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 2, by fusing 0 and 1
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);

        let partition_info = partition_config.info();
        println!("partition_info: {:?}", partition_info);

        // create dual module
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph.clone(), vec![3, 29, 30]);
        let mut dual_module: DualModuleParallel<
            DualModulePQGeneric<FutureObstacleQueue<Rational>>,
            FutureObstacleQueue<Rational>,
        > = DualModuleParallel::new_config(&initializer, &partition_info, DualModuleParallelConfig::default());
        dual_module.static_fuse_all();
        dual_module.static_combine_all_mirrored_vertices();

        // try to work on a simple syndrome
        let interface_ptr = DualModuleInterfacePtr::new_load(decoding_graph, &mut dual_module);

        // println!("interface_ptr json: {}", interface_ptr.snapshot(false));
        // println!("dual_module json: {}", dual_module.snapshot(false));

        visualizer
            .snapshot_combined("syndrome".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        println!("done first visualization");

        // grow them each by half
        let begin_time = std::time::Instant::now();
        let dual_node_3_ptr = interface_ptr.read_recursive().nodes[0].clone();
        let dual_node_12_ptr = interface_ptr.read_recursive().nodes[1].clone();
        let dual_node_30_ptr = interface_ptr.read_recursive().nodes[2].clone();
        dual_module.set_grow_rate(&dual_node_3_ptr, Rational::from_usize(1).unwrap());
        dual_module.set_grow_rate(&dual_node_12_ptr, Rational::from_usize(1).unwrap());
        dual_module.set_grow_rate(&dual_node_30_ptr, Rational::from_usize(1).unwrap());

        dual_module.grow(Rational::from_f64(weight / 2.).unwrap());
        // dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);

        println!("start second visualization");

        visualizer
            .snapshot_combined("grow".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // cluster becomes solved
        println!("start third visualization");
        dual_module.grow(Rational::from_f64(weight / 2.).unwrap());
        visualizer
            .snapshot_combined("solved".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();
        let end_time = std::time::Instant::now();
        let resolve_time = end_time - begin_time;

        // the result subgraph
        let subgraph_weak = vec![
            dual_module.get_edge_ptr(15).downgrade(),
            dual_module.get_edge_ptr(20).downgrade(),
        ];
        let subgraph = subgraph_weak
            .iter()
            .map(|e| e.upgrade_force().read_recursive().edge_index)
            .collect::<Vec<_>>();
        visualizer
            .snapshot_combined("subgraph".to_string(), vec![&interface_ptr, &dual_module, &subgraph])
            .unwrap();
        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());

        println!("resolve time {:?}", resolve_time);
    }

    /// this test panics because the vertices of the decoding graph are not numbered continuously. Hence, we cannot
    /// rely on vertex range to partition the graph. Revisit the partition design to perhaps add in IndexSet to acoomodate this issue.
    #[test]
    fn dual_module_parallel_test_2() {
        // cargo test dual_module_parallel_test_2 -- --nocapture
        let visualize_filename = "dual_module_parallel_test_2.json".to_string();
        let weight = 2.1972245773362196;
        let code = CodeCapacityTailoredCode::new(7, 0., 0.1);
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();

        // create dual module
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        println!("initializer: {:?}", initializer);
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 20),  // unit 0
            VertexRange::new(28, 48), // unit 1
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 2, by fusing 0 and 1
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);

        let partition_info = partition_config.info();
        println!("partition_info: {:?}", partition_info);

        // create dual module
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph.clone(), vec![23, 24, 29, 30]);
        visualizer
            .snapshot_combined("decoding_graph".to_string(), vec![Arc::new(decoding_graph.clone()).as_ref()])
            .unwrap();
        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());

        let mut dual_module: DualModuleParallel<
            DualModulePQGeneric<FutureObstacleQueue<Rational>>,
            FutureObstacleQueue<Rational>,
        > = DualModuleParallel::new_config(&initializer, &partition_info, DualModuleParallelConfig::default());
        dual_module.static_fuse_all();

        // try to work on a simple syndrome
        let interface_ptr = DualModuleInterfacePtr::new_load(decoding_graph, &mut dual_module);

        // println!("interface_ptr json: {}", interface_ptr.snapshot(false));
        // println!("dual_module json: {}", dual_module.snapshot(false));

        visualizer
            .snapshot_combined("syndrome".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        println!("done first visualization");

        {
            let interface_ptr_read = interface_ptr.read_recursive();
            let dual_node_ptrs = interface_ptr_read.nodes.iter().take(4).cloned();
            dual_node_ptrs.for_each(|node_ptr| dual_module.set_grow_rate(&node_ptr, Rational::from_usize(1).unwrap()));
        }

        // grow them each by a quarter
        let begin_time = std::time::Instant::now();
        dual_module.grow(Rational::from_f64(weight / 4.).unwrap());
        visualizer
            .snapshot_combined("solved".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();
        let end_time = std::time::Instant::now();
        let resolve_time = end_time - begin_time;

        // the result subgraph
        let subgraph_weak = vec![dual_module.get_edge_ptr(24).downgrade()];
        let subgraph = subgraph_weak
            .iter()
            .map(|e| e.upgrade_force().read_recursive().edge_index)
            .collect::<Vec<_>>();
        visualizer
            .snapshot_combined("subgraph".to_string(), vec![&interface_ptr, &dual_module, &subgraph])
            .unwrap();
        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());

        println!("resolve time {:?}", resolve_time);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_module_parallel_basic_standard_syndrome_optional_viz(
        _code: impl ExampleCode,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        mut dual_module: impl DualModuleImpl + MWPSVisualizer + Send + Sync,
        model_graph: Arc<crate::model_hypergraph::ModelHyperGraph>,
        mut visualizer: Option<Visualizer>,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleSerial,
        impl DualModuleImpl + MWPSVisualizer,
    ) {
        // create primal module
        let mut primal_module = PrimalModuleSerial::new_empty(&model_graph.initializer);
        primal_module.plugins = Arc::new(plugins);
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph.clone(), defect_vertices.clone());
        let interface_ptr = DualModuleInterfacePtr::new(model_graph.clone());

        let begin_time = std::time::Instant::now();
        primal_module.solve_visualizer(
            &interface_ptr,
            decoding_graph.syndrome_pattern.clone(),
            &mut dual_module,
            visualizer.as_mut(),
        );

        let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
        let end_time = std::time::Instant::now();
        let resolve_time = begin_time - end_time;
        println!("resolve time: {:?}", resolve_time);
        println!("subgraph: {:?}", subgraph.subgraph);
        if let Some(visualizer) = visualizer.as_mut() {
            visualizer
                .snapshot_combined(
                    "subgraph".to_string(),
                    vec![&interface_ptr, &dual_module, &subgraph, &weight_range],
                )
                .unwrap();
            visualizer.save_html_along_json();
            println!("open visualizer at {}", visualizer.html_along_json_path());
        }

        assert!(
            decoding_graph
                .model_graph
                .matches_subgraph_syndrome(&subgraph, &defect_vertices),
            "the result subgraph is invalid"
        );
        assert_eq!(
            weight_range.lower, weight_range.upper,
            "lower and upper of weight_range do not match"
        );
        assert_eq!(final_dual, weight_range.upper, "unmatched sum dual variables");
        assert_eq!(final_dual, weight_range.lower, "unexpected final dual variable sum");
        (interface_ptr, primal_module, dual_module)
    }

    pub fn dual_module_parallel_basic_standard_syndrome(
        code: impl ExampleCode,
        visualize_filename: String,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        initializer: &Arc<SolverInitializer>,
        partition_info: PartitionInfo,
        model_graph: &Arc<ModelHyperGraph>,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleSerial,
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
            visualizer
        };

        // create dual module
        let mut dual_module_parallel_config = DualModuleParallelConfig::default();
        // dual_module_parallel_config.enable_parallel_execution = true;
        let mut dual_module: DualModuleParallel<
            DualModulePQGeneric<FutureObstacleQueue<Rational>>,
            FutureObstacleQueue<Rational>,
        > = DualModuleParallel::new_config(&initializer, &partition_info, dual_module_parallel_config);
        dual_module.static_fuse_all();
        dual_module.static_combine_all_mirrored_vertices();
        // let mut dual_module: DualModulePQ<FutureObstacleQueue<Rational>> = DualModulePQ::new_empty(&model_graph.initializer);

        dual_module_parallel_basic_standard_syndrome_optional_viz(
            code,
            defect_vertices,
            final_dual,
            plugins,
            dual_module,
            model_graph.clone(),
            Some(visualizer),
        )
    }

    /// test a simple case, split into 2, no defect vertex in boundary-unit, clusters do not grow into other units
    #[test]
    fn dual_module_parallel_basic_test_2() {
        // cargo test dual_module_parallel_basic_test_2 -- --nocapture
        let visualize_filename = "dual_module_parallel_basic_test_2.json".to_string();
        // let weight = 1; // do not change, the data is hard-coded
        // let pxy = 0.0602828812732227;
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let defect_vertices = vec![2, 35];

        // create model graph
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 18),  // unit 0
            VertexRange::new(24, 42), // unit 1
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 2, by fusing 0 and 1
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);
        partition_config.defect_vertices = FastIterSet::from_iter(defect_vertices.clone());

        let partition_info = partition_config.info();

        dual_module_parallel_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(8.788898309344878),
            vec![],
            initializer,
            partition_info,
            &model_graph,
        );
    }

    /// test a simple case, split into 2, a defect vertex in boundary-unit, clusters do grow into other units
    #[test]
    fn dual_module_parallel_basic_test_3() {
        // cargo test dual_module_parallel_basic_test_3 -- --nocapture
        let visualize_filename = "dual_module_parallel_basic_test_3.json".to_string();
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let defect_vertices = vec![19, 35];

        // create model graph
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 18),  // unit 0
            VertexRange::new(24, 42), // unit 1
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 2, by fusing 0 and 1
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);
        partition_config.defect_vertices = FastIterSet::from_iter(defect_vertices.clone());

        let partition_info = partition_config.info();

        dual_module_parallel_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(6.591673732008658),
            vec![],
            initializer,
            partition_info,
            &model_graph,
        );
    }

    /// test a simple case, split into 2, a defect vertex in boundary-unit, clusters grow into other units
    #[test]
    fn dual_module_parallel_basic_test_4() {
        // cargo test dual_module_parallel_basic_test_4 -- --nocapture
        let visualize_filename = "dual_module_parallel_basic_test_4.json".to_string();
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let defect_vertices = vec![12, 18];

        // create model graph
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 18),  // unit 0
            VertexRange::new(24, 42), // unit 1
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 2, by fusing 0 and 1
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);
        partition_config.defect_vertices = FastIterSet::from_iter(defect_vertices.clone());

        let partition_info = partition_config.info();

        dual_module_parallel_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(2.1972245773362196),
            vec![],
            initializer,
            partition_info,
            &model_graph,
        );
    }

    /// test a simple case, split into 4, a defect vertex in boundary-unit, clusters grow into other units
    #[test]
    fn dual_module_parallel_basic_test_5() {
        // cargo test dual_module_parallel_basic_test_5 -- --nocapture
        let visualize_filename = "dual_module_parallel_basic_test_5.json".to_string();
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let defect_vertices = vec![16, 19, 28];

        // create model graph
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        partition_config.partitions = vec![
            VertexRange::new(0, 6),   // unit 0
            VertexRange::new(12, 18), // unit 1
            VertexRange::new(24, 30), // unit 2
            VertexRange::new(36, 42), // unit 3
        ];
        partition_config.fusions = vec![
            (0, 1), // unit 4, by fusing 0 and 1
            (1, 2), // unit 5,
            (2, 3), // unit 6
        ];
        let a = partition_config.dag_partition_units.add_node(());
        let b = partition_config.dag_partition_units.add_node(());
        let c = partition_config.dag_partition_units.add_node(());
        let d = partition_config.dag_partition_units.add_node(());
        partition_config.dag_partition_units.add_edge(a, b, false);
        partition_config.dag_partition_units.add_edge(b, c, false);
        partition_config.dag_partition_units.add_edge(c, d, false);

        partition_config.defect_vertices = FastIterSet::from_iter(defect_vertices.clone());

        let partition_info = partition_config.info();

        dual_module_parallel_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(8.788898309344878),
            vec![],
            initializer,
            partition_info,
            &model_graph,
        );
    }

    /// now, we test the serial version of the above test case to see whether the lower and upper bound of the weight range match
    #[test]
    fn dual_module_parallel_basic_test_serial_5() {
        // cargo test dual_module_parallel_basic_test_serial_5 -- --nocapture
        let visualize_filename = "dual_module_parallel_basic_test_serial_5.json".to_string();
        let code = CodeCapacityPlanarCode::new(7, 0.1);
        let defect_vertices = vec![16, 19, 28];

        use crate::primal_module_serial::tests::primal_module_serial_basic_standard_syndrome;
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(8.788898309344878),
            vec![],
        );
    }

    /// test for time partition
    #[allow(clippy::unnecessary_cast)]
    pub fn graph_time_partition(
        initializer: &SolverInitializer,
        positions: &Vec<VisualizePosition>,
        defect_vertices: &Vec<VertexIndex>,
        split_num: usize,
    ) -> PartitionConfig {
        assert!(positions.len() > 0, "positive number of positions");
        let mut partition_config = PartitionConfig::new(initializer.vertex_num);
        let mut last_t = positions[0].t;
        let mut t_list: Vec<f64> = vec![];
        t_list.push(last_t);
        for position in positions {
            assert!(
                position.t >= last_t,
                "t not monotonically increasing, vertex reordering must be performed before calling this"
            );
            if position.t != last_t {
                t_list.push(position.t);
            }
            last_t = position.t;
        }

        // pick the t value in the middle to split it
        println!("t_list first: {:?}, t_list last: {:?}", t_list[0], t_list.last().unwrap());
        let mut t_split_vec: Vec<f64> = vec![0.0; split_num - 1];
        for i in 0..(split_num - 1) {
            let index: usize = t_list.len() / split_num * (i + 1);
            t_split_vec[i] = t_list[index];
        }
        println!("t_split_vec: {:?}", t_split_vec);

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

        println!("split_start_index_vec: {:?}", split_start_index_vec);
        println!("split_end_index_vec: {:?}", split_end_index_vec);
        assert!(
            split_start_index_vec.iter().all(|&x| x != MAX),
            "Some elements in split_start_index_vec are equal to MAX"
        );

        // partitions are found
        let mut graph_nodes = vec![];
        let mut partitions_vec = vec![];
        for i in 0..split_num {
            if i == 0 {
                partitions_vec.push(VertexRange::new(0, split_start_index_vec[0]));
            } else if i == split_num - 1 {
                partitions_vec.push(VertexRange::new(split_end_index_vec[i - 1], positions.len()));
            } else {
                partitions_vec.push(VertexRange::new(split_end_index_vec[i - 1], split_start_index_vec[i]));
            }

            if i < split_num - 1 {
                partition_config.fusions.push((i, i + 1));
            }

            let a = partition_config.dag_partition_units.add_node(());
            graph_nodes.push(a.clone());
        }
        partition_config.partitions = partitions_vec;

        for i in 0..split_num {
            if i < split_num - 1 {
                partition_config
                    .dag_partition_units
                    .add_edge(graph_nodes[i], graph_nodes[i + 1], false);
            }
        }
        partition_config.defect_vertices = FastIterSet::from_iter(defect_vertices.clone());

        partition_config
    }

    pub fn dual_module_parallel_evaluation_qec_playground_helper(
        code: impl ExampleCode,
        visualize_filename: String,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        split_num: usize,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleSerial,
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
            visualizer
        };

        // create dual module
        let model_graph = code.get_model_graph();
        let initializer = &model_graph.initializer;
        let partition_config = graph_time_partition(&initializer, &code.get_positions(), &defect_vertices, split_num);
        let partition_info = partition_config.info();

        // create dual module
        // let decoding_graph = DecodingHyperGraph::new_defects(model_graph.clone(), vec![3, 29, 30]);
        let mut dual_module_parallel_config = DualModuleParallelConfig::default();
        dual_module_parallel_config.enable_parallel_execution = true;
        let mut dual_module: DualModuleParallel<
            DualModulePQGeneric<FutureObstacleQueue<Rational>>,
            FutureObstacleQueue<Rational>,
        > = DualModuleParallel::new_config(&initializer, &partition_info, dual_module_parallel_config);
        dual_module.static_fuse_all();
        dual_module.static_combine_all_mirrored_vertices();

        dual_module_parallel_basic_standard_syndrome_optional_viz(
            code,
            defect_vertices,
            final_dual,
            plugins,
            dual_module,
            model_graph,
            Some(visualizer),
        )
    }

    #[test]
    fn dual_module_parallel_circuit_level_noise_qec_playground_1() {
        // cargo test dual_module_parallel_circuit_level_noise_qec_playground_1 -- --nocapture
        let config = json!({
            "code_type": qecp::code_builder::CodeType::RotatedPlanarCode,
            "nm": 8,
        });

        let code = QECPlaygroundCode::new(3, 0.1, config);
        let defect_vertices = vec![3, 10, 18, 19, 31, 35, 43];

        let visualize_filename = "dual_module_parallel_circuit_level_noise_qec_playground_1.json".to_string();
        dual_module_parallel_evaluation_qec_playground_helper(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(7.067635986959294),
            vec![],
            2,
        );
    }

    #[test]
    fn dual_module_parallel_circuit_level_noise_qec_playground_serial_1() {
        // cargo test dual_module_parallel_circuit_level_noise_qec_playground_serial_1 -- --nocapture
        let config = json!({
            "code_type": qecp::code_builder::CodeType::RotatedPlanarCode,
            "nm": 8,
        });

        let code = QECPlaygroundCode::new(3, 0.001, config);
        let defect_vertices = vec![3, 10, 18, 19, 31, 35, 43];

        let visualize_filename = "dual_module_parallel_circuit_level_noise_qec_playground_serial_1.json".to_string();
        use crate::primal_module_serial::tests::primal_module_serial_basic_standard_syndrome;
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(7.067635986959294),
            vec![],
        );
    }

    /// test solver on circuit level noise with random errors, split into 2
    #[test]
    fn dual_module_parallel_circuit_level_noise_qec_playground_2() {
        // cargo test dual_module_parallel_circuit_level_noise_qec_playground_2 -- --nocapture
        let config = json!({
            "code_type": qecp::code_builder::CodeType::RotatedPlanarCode
        });

        let mut code = QECPlaygroundCode::new(7, 0.001, config);
        let defect_vertices = code.generate_random_errors(132).0.defect_vertices;

        let visualize_filename = "dual_module_parallel_circuit_level_noise_qec_playground_2.json".to_string();
        dual_module_parallel_evaluation_qec_playground_helper(
            code,
            visualize_filename,
            defect_vertices.clone(),
            Rational::from(19.412934685147327),
            vec![],
            4,
        );
    }

    /// test solver on circuit level noise with random errors, split into 4
    /// for now, this test case does not pass, the lower and upper bound of the weight range do not match
    /// even when split to 2, the lower and upper bound of the weight range still do not match (nm = 18)
    /// but somehow, when nm = 8, the lower and upper bound of the weight range match
    /// it turns out that, if p=0.001, the lower and upper bound of the weight range will match
    #[test]
    fn dual_module_parallel_circuit_level_noise_qec_playground_3() {
        // cargo test dual_module_parallel_circuit_level_noise_qec_playground_3 -- --nocapture
        let config = json!({
            "code_type": qecp::code_builder::CodeType::RotatedPlanarCode,
            "nm": 18,
        });

        let mut code = QECPlaygroundCode::new(7, 0.001, config);
        // [83, 95, 108, 111, 120, 152, 155, 168, 179, 180, 203, 209, 211, 217, 218, 238, 286, 287, 310, 311, 314, 318, 321, 322, 366]
        let defect_vertices = code.generate_random_errors(132).0.defect_vertices;

        let visualize_filename = "dual_module_parallel_circuit_level_noise_qec_playground_3.json".to_string();
        dual_module_parallel_evaluation_qec_playground_helper(
            code,
            visualize_filename,
            defect_vertices.clone(),
            Rational::from(49.84751933558799),
            vec![],
            4,
        );
    }

    /// now, we test the serial version of the above test case to see whether the lower and upper bound of the weight range match
    #[test]
    fn dual_module_parallel_circuit_level_noise_qec_playground_serial_3() {
        // cargo test dual_module_parallel_circuit_level_noise_qec_playground_serial_3 -- --nocapture
        let config = json!({
            "code_type": qecp::code_builder::CodeType::RotatedPlanarCode,
            "nm": 18,
        });

        let mut code = QECPlaygroundCode::new(7, 0.001, config);
        let defect_vertices = code.generate_random_errors(132).0.defect_vertices;

        let visualize_filename = "dual_module_parallel_circuit_level_noise_qec_playground_serial_3.json".to_string();
        use crate::primal_module_serial::tests::primal_module_serial_basic_standard_syndrome;
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from(49.84751933558799),
            vec![],
        );
    }
}
