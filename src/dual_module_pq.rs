//! Dual Module with Priority Queue
//!
//! A serial implementation of the dual module with priority queue optimization
//!
//! Only debug tests are failing, which aligns with the dual_module_serial behavior
//!
#![cfg_attr(feature = "unsafe_pointer", allow(dropping_references))]

use crate::num_traits::{ToPrimitive, Zero};
use crate::pointers::*;
use crate::primal_module::Affinity;
use crate::primal_module_serial::PrimalClusterPtr;
use crate::util::*;
use crate::visualize::*;
use crate::{add_shared_methods, dual_module::*};

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    sync::Arc,
    time::Instant,
};

#[cfg(feature = "unsafe_pointer")]
use crate::pointers::UnsafePtr;
use derivative::Derivative;
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use heapz::RankPairingHeap;
use heapz::{DecreaseKey, Heap};
use num_traits::Signed;
use parking_lot::{lock_api::RwLockWriteGuard, RawRwLock};
use pheap::PairingHeap;
use priority_queue::PriorityQueue;

/* Helper structs for events/obstacles during growing */
#[derive(Debug, Clone)]
pub struct FutureEvent<T: Ord + PartialEq + Eq, E> {
    /// when the event will happen
    pub time: T,
    /// the event
    pub event: E,
}

// `impl`s to allow the object to be compared according to time
impl<T: Ord + PartialEq + Eq, E> PartialEq for FutureEvent<T, E> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}
impl<T: Ord + PartialEq + Eq, E> Eq for FutureEvent<T, E> {}
impl<T: Ord + PartialEq + Eq, E> Ord for FutureEvent<T, E> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time.cmp(&other.time)
    }
}
impl<T: Ord + PartialEq + Eq, E> PartialOrd for FutureEvent<T, E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type FutureObstacle<T> = FutureEvent<T, Obstacle>;
pub type MinBinaryHeap<F> = BinaryHeap<Reverse<F>>;
pub type _FutureObstacleQueue<T> = MinBinaryHeap<FutureObstacle<T>>;

pub type MinPriorityQueue<O, T> = PriorityQueue<O, Reverse<T>>;
pub type FutureObstacleQueue<T> = MinPriorityQueue<Obstacle, T>;

impl<T: Ord + PartialEq + Eq + std::fmt::Debug> FutureQueueMethods<T, Obstacle> for FutureObstacleQueue<T> {
    fn will_happen(&mut self, time: T, event: Obstacle) {
        self.push(event, Reverse(time));
    }
    fn peek_event(&self) -> Option<(&T, &Obstacle)> {
        self.peek().map(|future| (&future.1 .0, future.0))
    }
    fn pop_event(&mut self) -> Option<(T, Obstacle)> {
        self.pop().map(|future| (future.1 .0, future.0))
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn len(&self) -> usize {
        self.len()
    }
}

pub trait FutureQueueMethods<T: Ord + PartialEq + Eq + std::fmt::Debug, E: std::fmt::Debug> {
    /// Append an event at time T
    ///     Note: this may have multiple distinct yet valid behaviors, e,g, weather there are duplicates allowed in the data strcture, default to allow
    fn will_happen(&mut self, time: T, event: E);

    /// peek for a queue
    fn peek_event(&self) -> Option<(&T, &E)>;

    /// pop for a queue
    fn pop_event(&mut self) -> Option<(T, E)>;

    /// clear for a queue
    fn clear(&mut self);

    /// length of the queue
    fn len(&self) -> usize;

    /// is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Ord + PartialEq + Eq + std::fmt::Debug, E: std::fmt::Debug> FutureQueueMethods<T, E>
    for MinBinaryHeap<FutureEvent<T, E>>
{
    fn will_happen(&mut self, time: T, event: E) {
        self.push(Reverse(FutureEvent { time, event }))
    }
    fn peek_event(&self) -> Option<(&T, &E)> {
        self.peek().map(|future| (&future.0.time, &future.0.event))
    }
    fn pop_event(&mut self) -> Option<(T, E)> {
        self.pop().map(|future| (future.0.time, future.0.event))
    }
    fn clear(&mut self) {
        self.clear();
    }
    fn len(&self) -> usize {
        self.len()
    }
}

/* Vertices and Edges */
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Vertex {
    /// the index of this vertex in the decoding graph, not necessary the index in [`DualModulePQ::vertices`] if it's partitioned
    pub vertex_index: VertexIndex,
    /// if a vertex is defect, then [`Vertex::propagated_dual_node`] always corresponds to that root
    pub is_defect: bool,
    /// all neighbor edges, in surface code this should be constant number of edges
    pub edges: Vec<EdgeWeak>,
    /// if this vertex is in boundary unit, find its corresponding mirror vertices in the other units. If this vertex is in non-boundary unit but a mirrored vertex,
    /// find its other mirrored vertices in other units (both boundary and non-boundary units)
    pub mirrored_vertices: Vec<VertexWeak>,
}

impl Vertex {
    fn clear(&mut self) {
        self.is_defect = false;
    }
}

pub type VertexPtr = ArcManualSafeLock<Vertex>;
pub type VertexWeak = WeakManualSafeLock<Vertex>;

impl std::fmt::Debug for VertexPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let vertex = self.read_recursive();
        write!(f, "{}", vertex.vertex_index)
    }
}

impl std::fmt::Debug for VertexWeak {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let vertex_ptr = self.upgrade_force();
        let vertex = vertex_ptr.read_recursive();
        write!(f, "{}", vertex.vertex_index)
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Edge {
    /// global edge index
    pub edge_index: EdgeIndex,
    /// total weight of this edge
    pub weight: Rational,

    pub vertices: Vec<VertexWeak>, // note: consider using/constructing ordered vertex, this will speed up `adjust_weights_for_negative_edges`
    /// the dual nodes that contributes to this edge
    pub dual_nodes: Vec<OrderedDualNodeWeak>,

    /// the speed of growth, at the current time
    ///     Note: changing this should cause the `growth_at_last_updated_time` and `last_updated_time` to update
    pub grow_rate: Rational,
    /// the last time this Edge is synced/updated with the global time
    pub last_updated_time: Rational,
    /// growth value at the last updated time, also, growth_at_last_updated_time <= weight
    pub growth_at_last_updated_time: Rational,
    /// the partition unit this edge belongs to. For non-parallel implementation, this value is set to None.
    pub unit_index: Option<usize>,
    /// whether this edge is connected to a boundary vertex, (this edges must belong to non-boundary unit)
    pub connected_to_boundary_vertex: bool,

    #[cfg(feature = "incr_lp")]
    /// storing the weights of the clusters that are currently contributing to this edge
    pub cluster_weights: hashbrown::HashMap<usize, Rational>,
}

impl Edge {
    fn clear(&mut self) {
        self.growth_at_last_updated_time = Rational::zero();
        self.last_updated_time = Rational::zero();
        self.dual_nodes.clear();
        self.grow_rate = Rational::zero();
        #[cfg(feature = "incr_lp")]
        self.cluster_weights.clear();
    }
}

pub type EdgePtr = ArcManualSafeLock<Edge>;
pub type EdgeWeak = WeakManualSafeLock<Edge>;

impl std::fmt::Debug for EdgePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let edge = self.read_recursive();
        // write!(
        //     f,
        //     "[edge: {}]: weight: {}, grow_rate: {}, growth_at_last_updated_time: {}, last_updated_time: {}\n\tdual_nodes: {:?}\n",
        //     edge.edge_index,
        //     edge.weight,
        //     edge.grow_rate,
        //     edge.growth_at_last_updated_time,
        //     edge.last_updated_time,
        //     edge.dual_nodes.iter().filter(|node| !node.weak_ptr.upgrade_force().read_recursive().grow_rate.is_zero()).collect::<Vec<_>>()
        // )
        write!(f, "[edge: {}]", edge.edge_index,)
    }
}

impl std::fmt::Debug for EdgeWeak {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let edge_ptr = self.upgrade_force();
        let edge = edge_ptr.read_recursive();
        write!(
            f,
            "[edge: {}]: weight: {}, grow_rate: {}, growth_at_last_updated_time: {}, last_updated_time: {}\n\tdual_nodes: {:?}\n",
            edge.edge_index,
            edge.weight,
            edge.grow_rate,
            edge.growth_at_last_updated_time,
            edge.last_updated_time,
            edge.dual_nodes.iter().filter(|node| !node.weak_ptr.upgrade_force().read_recursive().grow_rate.is_zero()).collect::<Vec<_>>()
        )
    }
}

pub type DualModulePQ = DualModulePQGeneric<FutureObstacleQueue<Rational>>;

/* the actual dual module */
#[derive(Clone)]
pub struct DualModulePQGeneric<Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug,
{
    /// all vertices including virtual ones
    pub vertices: Vec<VertexPtr>,
    /// keep edges, which can also be accessed in [`Self::vertices`]
    pub edges: Vec<EdgePtr>,
    /// storing all the anticipated events in a priority queue structure
    ///     ordering: the event in the most recent future is at the head of the queue
    obstacle_queue: Queue,
    /// the global time of this dual module
    ///     Note: Wrap-around edge case is not currently considered
    global_time: ArcManualSafeLock<Rational>,

    /// the current mode of the dual module
    mode: DualModuleMode,

    // tuning mode statistics
    tuning_start_time: Option<Instant>,
    total_tuning_time: Option<f64>,

    // negative weight handling
    negative_weight_sum: Rational,
    negative_edges: HashSet<EdgeIndex>,
    flip_vertices: HashSet<VertexIndex>,

    // remember the initializer for original weights and heralded weighted edges
    pub initializer: Arc<SolverInitializer>,

    /// the number of all vertices (including those partitioned into other serial module)
    pub vertex_num: VertexNum,
    /// the number of all edges (including those partitioned into other seiral module)
    pub edge_num: usize,
    /// all mirrored vertices of this unit, mainly for parallel implementation
    pub all_mirrored_vertices: Vec<VertexPtr>,

    /// unit is active if it has an edge connected to a boundary vertex with non-zero growth
    pub unit_active: ArcManualSafeLock<bool>,
}

impl<Queue> DualModulePQGeneric<Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Clone,
{
    /// helper function to bring an edge update to speed with current time if needed
    fn update_edge_if_necessary(&self, edge_ptr: &EdgePtr) {
        let global_time = self.global_time.read_recursive();
        let mut edge = edge_ptr.write();
        if global_time.eq(&edge.last_updated_time) {
            // the edge is not behind
            return;
        }

        // debug_assert!(
        //     global_time.ge(&edge.last_updated_time),
        //     "global time is behind, maybe a wrap-around has happened"
        // );

        let time_diff = global_time.clone() - &edge.last_updated_time;
        let newly_grown_amount = &time_diff * &edge.grow_rate;
        edge.growth_at_last_updated_time += newly_grown_amount;
        if edge.connected_to_boundary_vertex && edge.growth_at_last_updated_time.is_positive() {
            let mut unit_active = self.unit_active.write();
            *unit_active = true;
        }
        edge.last_updated_time = global_time.clone();

        debug_assert!(
            edge.growth_at_last_updated_time <= edge.weight,
            "growth larger than weight: check if events are 1) inserted and 2) handled correctly",
        );
        drop(edge);
    }

    /// helper function to bring a dual node update to speed with current time if needed
    fn update_dual_node_if_necessary(&mut self, node_ptr: &DualNodePtr) {
        let global_time = self.global_time.read_recursive();
        let mut node = node_ptr.write();
        if global_time.eq(&node.last_updated_time) {
            // the edge is not behind
            return;
        }

        // debug_assert!(
        //     global_time.ge(&node.last_updated_time),
        //     "global time is behind, maybe a wrap-around has happened"
        // );

        let dual_variable = node.get_dual_variable();
        node.set_dual_variable(dual_variable);
        node.last_updated_time = global_time.clone();
        debug_assert!(
            !node.get_dual_variable().is_negative(),
            "negative dual variable: check if events are 1) inserted and 2) handled correctly"
        );
        drop(node);
    }

    /// debugging function
    #[allow(dead_code)]
    fn debug_update_all(&mut self, dual_node_ptrs: &[DualNodePtr]) {
        // updating all edges
        for edge_ptr in self.edges.iter() {
            self.update_edge_if_necessary(edge_ptr);
        }
        // updating all dual nodes
        for dual_node_ptr in dual_node_ptrs.iter() {
            self.update_dual_node_if_necessary(dual_node_ptr);
        }
    }

    pub fn compute_max_valid_grow(&mut self) -> Option<Rational> {
        let global_time = self.global_time.read_recursive().clone();
        // getting rid of all the invalid events
        while let Some((time, event)) = self.obstacle_queue.peek_event() {
            // found a valid event
            if self.is_valid_obstacle(event, time) {
                // valid grow
                if time != &global_time {
                    return Some(time - global_time.clone());
                }
                // goto else
                break;
            }
            self.obstacle_queue.pop_event();
        }
        None
    }

    /// return if the current obstacle is valid
    ///     note: even when the pq cannot hold duplicate events, `is_invalid` approach is more efficient than needing to remove items from the q
    fn is_valid_obstacle(
        &self,
        obstacle: &Obstacle,
        event_time: &Rational, // time associated with the obstacle
    ) -> bool {
        #[allow(clippy::unnecessary_cast)]
        return match obstacle {
            Obstacle::Conflict { edge_ptr } => {
                let edge = edge_ptr.read_recursive();
                // not changing, cannot have conflict
                if !edge.grow_rate.is_positive() {
                    return false;
                }
                let growth_at_event_time =
                    &edge.growth_at_last_updated_time + (event_time - &edge.last_updated_time) * &edge.grow_rate;

                // we have a postivie grow rate, should become tight
                growth_at_event_time == edge.weight
            }
            Obstacle::ShrinkToZero { dual_node_ptr } => {
                let node = dual_node_ptr.ptr.read_recursive();
                // only negative grow rates can shrink to zero
                if !node.grow_rate.is_negative() {
                    return false;
                }
                let growth_at_event_time =
                    &node.dual_variable_at_last_updated_time + (event_time - &node.last_updated_time) * &node.grow_rate;

                // we have a negative grow rate, should become zero
                growth_at_event_time.is_zero()
            }
        };
    }
}

pub type DualModulePQlPtr<Queue> = ArcManualSafeLock<DualModulePQGeneric<Queue>>;
pub type DualModulePQWeak<Queue> = WeakManualSafeLock<DualModulePQGeneric<Queue>>;

impl<Queue> DualModuleImpl for DualModulePQGeneric<Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Clone,
{
    /// initialize the dual module, which is supposed to be reused for multiple decoding tasks with the same structure
    #[allow(clippy::unnecessary_cast)]
    fn new_empty(initializer: &Arc<SolverInitializer>) -> Self {
        #[cfg(not(feature = "loose_sanity_check"))]
        initializer.sanity_check().unwrap();

        // create vertices
        let vertices: Vec<VertexPtr> = (0..initializer.vertex_num)
            .map(|vertex_index| {
                VertexPtr::new_value(Vertex {
                    vertex_index,
                    is_defect: false,
                    edges: vec![],
                    mirrored_vertices: vec![],
                })
            })
            .collect();
        // set edges
        let mut edges = Vec::<EdgePtr>::new();
        for hyperedge in initializer.weighted_edges.iter() {
            let edge = Edge {
                edge_index: edges.len() as EdgeIndex,
                weight: hyperedge.weight.clone(),
                dual_nodes: vec![],
                vertices: hyperedge
                    .vertices
                    .iter()
                    .map(|i| vertices[*i as usize].downgrade())
                    .collect::<Vec<_>>(),
                last_updated_time: Rational::zero(),
                growth_at_last_updated_time: Rational::zero(),
                grow_rate: Rational::zero(),
                unit_index: None,
                connected_to_boundary_vertex: false,
                #[cfg(feature = "incr_lp")]
                cluster_weights: hashbrown::HashMap::new(),
            };

            let edge_ptr = EdgePtr::new_value(edge);

            for &vertex_index in hyperedge.vertices.iter() {
                vertices[vertex_index as usize].write().edges.push(edge_ptr.downgrade());
            }

            edges.push(edge_ptr);
        }
        Self {
            vertices,
            edges,
            obstacle_queue: Queue::default(),
            global_time: ArcManualSafeLock::new_value(Rational::zero()),
            mode: DualModuleMode::default(),
            tuning_start_time: None,
            total_tuning_time: None,
            negative_weight_sum: Default::default(),
            negative_edges: Default::default(),
            flip_vertices: Default::default(),
            initializer: initializer.clone(),
            vertex_num: initializer.vertex_num,
            edge_num: initializer.weighted_edges.len(),
            all_mirrored_vertices: vec![],
            unit_active: ArcManualSafeLock::new_value(false),
        }
    }

    /// clear all growth and existing dual nodes
    fn clear(&mut self) {
        // todo: try parallel clearing, if a core supports hyper-threading then this may benefit
        self.vertices.iter().for_each(|p| p.write().clear());

        self.edges
            .iter()
            .zip(self.initializer.weighted_edges.iter().map(|x| &x.weight))
            .for_each(|(p, og_weight)| {
                let mut p_write = p.write();
                p_write.clear();
                p_write.weight = og_weight.clone();
            });

        self.obstacle_queue.clear();
        self.global_time.write().set_zero();
        self.mode_mut().reset();

        self.tuning_start_time = None;

        self.negative_edges.clear();
        self.negative_weight_sum = Rational::zero();
        self.flip_vertices.clear();
    }

    #[allow(clippy::unnecessary_cast)]
    /// Adding a defect node to the DualModule
    fn add_defect_node(&mut self, dual_node_ptr: &DualNodePtr) {
        let dual_node = dual_node_ptr.read_recursive();
        debug_assert!(dual_node.invalid_subgraph.edges.is_empty());
        debug_assert!(
            dual_node.invalid_subgraph.vertices.len() == 1,
            "defect node (without edges) should only work on a single vertex, for simplicity"
        );
        drop(dual_node);
        // let vertex_ptr = dual_node.invalid_subgraph.vertices.iter().next().unwrap();
        // let mut vertex = vertex_ptr.write();
        // assert!(!vertex.is_defect, "defect should not be added twice");
        // vertex.is_defect = true;
        // drop(dual_node);
        // drop(vertex);
        self.add_dual_node(dual_node_ptr);
    }

    #[allow(clippy::unnecessary_cast)]
    /// Mostly invoked by `add_defect_node`, triggering a pq update, and edges updates
    fn add_dual_node(&mut self, dual_node_ptr: &DualNodePtr) {
        dual_node_ptr.write().init_time(self.global_time.clone());
        let global_time = self.global_time.read_recursive();
        let dual_node_weak = dual_node_ptr.downgrade();
        let dual_node = dual_node_ptr.read_recursive();

        if dual_node.grow_rate.is_negative() {
            self.obstacle_queue.will_happen(
                // it is okay to use global_time now, as this must be up-to-speed
                dual_node.get_dual_variable().clone() / (-dual_node.grow_rate.clone()) + global_time.clone(),
                Obstacle::ShrinkToZero {
                    dual_node_ptr: OrderedDualNodePtr::new(dual_node.index, dual_node_ptr.clone()),
                },
            );
        }

        for edge_ptr in dual_node.invalid_subgraph.hair.iter() {
            // should make sure the edge is up-to-speed before making its variables change
            self.update_edge_if_necessary(edge_ptr);
            let mut edge = edge_ptr.write();

            edge.grow_rate += &dual_node.grow_rate;
            edge.dual_nodes
                .push(OrderedDualNodeWeak::new(dual_node.index, dual_node_weak.clone()));

            if edge.grow_rate.is_positive() {
                self.obstacle_queue.will_happen(
                    // it is okay to use global_time now, as this must be up-to-speed
                    (edge.weight.clone() - edge.growth_at_last_updated_time.clone()) / edge.grow_rate.clone()
                        + global_time.clone(),
                    Obstacle::Conflict {
                        edge_ptr: edge_ptr.clone(),
                    },
                );
            }
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn add_dual_node_tune(&mut self, dual_node_ptr: &DualNodePtr) {
        let dual_node_weak = dual_node_ptr.downgrade();
        let dual_node = dual_node_ptr.read_recursive();

        for edge_ptr in dual_node.invalid_subgraph.hair.iter() {
            let mut edge = edge_ptr.write();

            edge.dual_nodes
                .push(OrderedDualNodeWeak::new(dual_node.index, dual_node_weak.clone()));
        }
    }

    #[allow(clippy::unnecessary_cast)]
    fn set_grow_rate(&mut self, dual_node_ptr: &DualNodePtr, grow_rate: Rational) {
        self.update_dual_node_if_necessary(dual_node_ptr);
        let mut dual_node = dual_node_ptr.write();

        // it is okay to use global_time now, as this must be up-to-speed
        let global_time = self.global_time.read_recursive().clone();
        let grow_rate_diff = &grow_rate - &dual_node.grow_rate;

        dual_node.grow_rate = grow_rate.clone();
        if dual_node.grow_rate.is_negative() {
            let time = dual_node.get_dual_variable().clone() / (-grow_rate) + global_time.clone();
            let event = Obstacle::ShrinkToZero {
                dual_node_ptr: OrderedDualNodePtr::new(dual_node.index, dual_node_ptr.clone()),
            };
            self.obstacle_queue.will_happen(time, event);
        }

        // don't reacquire the read guard
        for edge_ptr in dual_node.invalid_subgraph.hair.iter() {
            self.update_edge_if_necessary(edge_ptr);
            let mut edge = edge_ptr.write();

            edge.grow_rate += &grow_rate_diff;
            if edge.grow_rate.is_positive() {
                self.obstacle_queue.will_happen(
                    // it is okay to use global_time now, as this must be up-to-speed
                    (edge.weight.clone() - edge.growth_at_last_updated_time.clone()) / edge.grow_rate.clone()
                        + global_time.clone(),
                    Obstacle::Conflict {
                        edge_ptr: edge_ptr.clone(),
                    },
                );
            }
        }
    }

    fn report(&mut self) -> DualReport {
        // self.debug_print();

        if let Some(max_valid_grow) = self.compute_max_valid_grow() {
            return DualReport::ValidGrow(max_valid_grow);
        }

        let global_time = self.global_time.read_recursive().clone();

        // else , it is a valid conflict to resolve
        if let Some((_, event)) = self.obstacle_queue.pop_event() {
            // this is used, since queues are not sets, and can contain duplicate events
            // Note: check that this is the assumption, though not much more overhead anyway
            // let mut group_max_update_length_set = FastIterSet::default();

            // Note: With de-dup queue implementation, we could use vectors here
            let mut dual_report = DualReport::new();
            dual_report.add_obstacle(event);

            // append all conflicts that happen at the same time as now
            // Note: This is Top-of-K operation, BTreeSets could be used.
            while let Some((time, _)) = self.obstacle_queue.peek_event() {
                if global_time == *time {
                    let (time, event) = self.obstacle_queue.pop_event().unwrap();
                    if !self.is_valid_obstacle(&event, &time) {
                        continue;
                    }
                    // add
                    dual_report.add_obstacle(event);
                } else {
                    break;
                }
            }

            for obstacle in dual_report.iter().unwrap() {
                self.obstacle_queue.will_happen(global_time.clone(), obstacle.clone());
            }
            return dual_report;
        }

        // nothing useful could be done, return unbounded
        DualReport::new()
    }

    /// for pq implementation, simply updating the global time is enough, could be part of the `report` function
    fn grow(&mut self, length: Rational) {
        if length.is_negative() {
            println!("{:?}", self.obstacle_queue);
        }
        assert!(
            length.is_positive(),
            "growth should be positive; if desired, please set grow rate to negative for shrinking"
        );
        let mut global_time_write = self.global_time.write();
        *global_time_write = global_time_write.clone() + length;
    }

    /* identical with the dual_module_serial */
    #[allow(clippy::unnecessary_cast)]
    fn get_edge_nodes(&self, edge_ptr: EdgePtr) -> Vec<DualNodePtr> {
        edge_ptr
            .read_recursive()
            .dual_nodes
            .iter()
            .map(|x| x.upgrade_force().ptr)
            .collect()
    }

    #[allow(clippy::unnecessary_cast)]
    /// how much away from saturated is the edge
    fn get_edge_slack(&self, edge_ptr: EdgePtr) -> Rational {
        let edge = edge_ptr.read_recursive();
        edge.weight.clone()
            - (self.global_time.read_recursive().clone() - edge.last_updated_time.clone()) * edge.grow_rate.clone()
            - edge.growth_at_last_updated_time.clone()
    }

    /// is the edge saturated
    fn is_edge_tight(&self, edge_ptr: EdgePtr) -> bool {
        self.get_edge_slack(edge_ptr).is_zero()
    }

    /* tuning mode related new methods */

    // tuning mode shared methods
    add_shared_methods!();

    /// is the edge tight, but for tuning mode
    fn is_edge_tight_tune(&self, edge_ptr: EdgePtr) -> bool {
        let edge = edge_ptr.read_recursive();
        edge.weight == edge.growth_at_last_updated_time
    }

    fn get_edge_slack_tune(&self, edge_ptr: EdgePtr) -> Rational {
        let edge = edge_ptr.read_recursive();
        edge.weight.clone() - edge.growth_at_last_updated_time.clone()
    }

    /// change mode, clear queue as queue is no longer needed. also sync to get rid off the need for global time
    fn advance_mode(&mut self) {
        self.tuning_start_time = Some(Instant::now());
        self.mode_mut().advance();
        self.obstacle_queue.clear();
        self.sync();
    }

    /// at the end of tuning mode, record the total time spent on tuning
    fn end_tuning(&mut self) {
        self.total_tuning_time = Some(self.tuning_start_time.unwrap().elapsed().as_secs_f64());
    }

    /// get the total time spent on tuning
    fn get_total_tuning_time(&self) -> Option<f64> {
        self.total_tuning_time
    }

    /// clear the tuning time
    fn clear_tuning_time(&mut self) {
        self.total_tuning_time = None;
    }

    /// grow specific amount for a specific edge
    fn grow_edge(&self, edge_ptr: EdgePtr, amount: &Rational) {
        let mut edge = edge_ptr.write();
        edge.growth_at_last_updated_time += amount;
    }

    /// sync all states and global time so the concept of time and pq can retire
    fn sync(&mut self) {
        // note: we can either set the global time to be zero, or just not change it anymore

        let mut nodes_touched = FastIterSet::new();

        for edges in self.edges.iter_mut() {
            let mut edge = edges.write();

            // update if necessary
            let global_time = self.global_time.read_recursive();
            if edge.last_updated_time != global_time.clone() {
                // the edge is behind
                debug_assert!(
                    global_time.clone() >= edge.last_updated_time,
                    "global time is behind, maybe a wrap-around has happened"
                );

                let time_diff = global_time.clone() - &edge.last_updated_time;
                let newly_grown_amount = &time_diff * &edge.grow_rate;
                edge.growth_at_last_updated_time += newly_grown_amount;
                edge.last_updated_time = global_time.clone();
                debug_assert!(
                    edge.growth_at_last_updated_time <= edge.weight,
                    "growth larger than weight: check if events are 1) inserted and 2) handled correctly"
                );
            }

            for dual_node_ptr in edge.dual_nodes.iter() {
                if nodes_touched.contains(&dual_node_ptr.index) {
                    continue;
                }
                let _dual_node_ptr = dual_node_ptr.upgrade_force();
                let node = _dual_node_ptr.ptr.read_recursive();
                nodes_touched.insert(node.index);

                // update if necessary
                let global_time = self.global_time.read_recursive();
                if node.last_updated_time != global_time.clone() {
                    // the node is behind
                    debug_assert!(
                        global_time.clone() >= node.last_updated_time,
                        "global time is behind, maybe a wrap-around has happened"
                    );

                    drop(node);
                    let mut node = _dual_node_ptr.ptr.write();

                    let dual_variable = node.get_dual_variable();
                    node.set_dual_variable(dual_variable);
                    node.last_updated_time = global_time.clone();
                    debug_assert!(
                        !node.get_dual_variable().is_negative(),
                        "negative dual variable: check if events are 1) inserted and 2) handled correctly"
                    );
                }
            }
        }
    }

    /// misc debug print statement
    fn debug_print(&self) {
        println!("\n[current states]");
        println!("global time: {:?}", self.global_time.read_recursive());
        println!(
            "edges: {:?}",
            self.edges // .iter()
                       // .filter(|e| !e.read_recursive().grow_rate.is_zero())
                       // .collect::<Vec<&EdgePtr>>()
        );
        if self.obstacle_queue.len() > 0 {
            println!("pq: {:?}", self.obstacle_queue.len());
        }

        let mut all_nodes = FastIterSet::default();
        for edge in self.edges.iter() {
            let edge = edge.read_recursive();
            for node in edge.dual_nodes.iter() {
                let node = node.upgrade_force();
                if node.ptr.read_recursive().grow_rate.is_zero() {
                    continue;
                }
                all_nodes.insert(node);
            }
        }
        println!("nodes: {:?}", all_nodes);
    }

    /* affinity */
    fn calculate_cluster_affinity(&mut self, cluster: PrimalClusterPtr) -> Option<Affinity> {
        let mut start = 0.0;
        let cluster = cluster.read_recursive();
        start -= cluster.edges.len() as f64 + cluster.nodes.len() as f64;
        let global_time = self.global_time.read_recursive().clone();

        let mut weight = Rational::zero();
        for edge_ptr in cluster.edges.iter() {
            let edge = edge_ptr.read_recursive();
            weight += &edge.growth_at_last_updated_time + (&global_time - &edge.last_updated_time) * &edge.grow_rate;
        }
        for node in cluster.nodes.iter() {
            let dual_node = node.read_recursive().dual_node_ptr.clone();
            let dual_node_read_ptr = dual_node.read_recursive();
            weight -= &dual_node_read_ptr.dual_variable_at_last_updated_time
                + (&global_time - &dual_node_read_ptr.last_updated_time) * &dual_node_read_ptr.grow_rate;
        }
        if weight.is_zero() {
            return None;
        }
        start += weight.to_f64().unwrap();
        Some(Affinity::from(start))
    }

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
        let edge = edge_ptr.read_recursive();
        edge.weight.clone()
    }

    #[cfg(feature = "incr_lp")]
    fn get_edge_free_weight_cluster(&self, edge_ptr: EdgePtr, cluster_index: NodeIndex) -> Rational {
        let edge = edge_ptr.read_recursive();
        edge.weight.clone()
            - edge
                .cluster_weights
                .iter()
                .filter_map(|(c_idx, y)| if cluster_index.ne(c_idx) { Some(y) } else { None })
                .sum::<Rational>()
    }

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights_union(
        &self,
        dual_node_ptr: &DualNodePtr,
        drained_cluster_index: NodeIndex,
        absorbing_cluster_index: NodeIndex,
    ) {
        let dual_node = dual_node_ptr.read_recursive();
        for edge_ptr in dual_node.invalid_subgraph.hair.iter() {
            let mut edge = edge_ptr.write();
            if let Some(removed) = edge.cluster_weights.remove(&drained_cluster_index) {
                *edge
                    .cluster_weights
                    .entry(absorbing_cluster_index)
                    .or_insert(Rational::zero()) += removed;
            }
        }
    }

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights(&self, edge_ptr: EdgePtr, cluster_index: usize, weight: Rational) {
        match edge_ptr.write().cluster_weights.entry(cluster_index) {
            hashbrown::hash_map::Entry::Occupied(mut o) => {
                *o.get_mut() += weight;
            }
            hashbrown::hash_map::Entry::Vacant(v) => {
                v.insert(weight);
            }
        }
    }

    fn adjust_weights_for_negative_edges(&mut self) {
        for edge in self.edges.iter() {
            let mut edge = edge.write();
            if edge.weight.is_negative() {
                self.negative_edges.insert(edge.edge_index);
                self.negative_weight_sum += edge.weight.clone();

                for vertex in edge.vertices.iter() {
                    let vertex = vertex.upgrade_force();
                    if self.flip_vertices.contains(&vertex.read_recursive().vertex_index) {
                        self.flip_vertices.remove(&vertex.read_recursive().vertex_index);
                    } else {
                        self.flip_vertices.insert(vertex.read_recursive().vertex_index);
                    }
                }

                edge.weight = -edge.weight.clone();
            }
        }
    }

    fn update_weights(&mut self, new_weights: Vec<Weight>, mix_ratio: Weight) {
        for (edge, new_weight) in self.edges.iter().zip(new_weights.iter()) {
            let mut edge = edge.write();

            let current_edge_weight = edge.weight.clone();
            let new_weight =
                Weight::from(current_edge_weight.clone() + mix_ratio.clone() * (new_weight - current_edge_weight));

            edge.weight = new_weight;
        }
    }

    fn set_weights(&mut self, new_weights: FastIterMap<EdgeIndex, Weight>) {
        for (edge_index, new_weight) in new_weights.into_iter() {
            let mut edge = self.edges[edge_index].write();
            edge.weight = new_weight;
        }
    }

    fn get_negative_weight_sum(&self) -> Rational {
        self.negative_weight_sum.clone()
    }

    fn get_negative_edges(&self) -> HashSet<EdgeIndex> {
        self.negative_edges.clone()
    }

    fn get_flip_vertices(&self) -> HashSet<VertexIndex> {
        self.flip_vertices.clone()
    }

    fn get_vertex_ptr(&self, vertex_index: VertexIndex) -> VertexPtr {
        self.vertices[vertex_index as usize].clone()
    }

    fn get_edge_ptr(&self, edge_index: EdgeIndex) -> EdgePtr {
        self.edges[edge_index as usize].clone()
    }

    fn get_vertex_ptr_vec(&self, vertex_indices: &[VertexIndex]) -> Vec<VertexPtr> {
        vertex_indices
            .to_vec()
            .iter()
            .map(|&i| self.vertices[i as usize].clone())
            .collect()
    }

    fn get_edge_ptr_vec(&self, edge_indices: &[EdgeIndex]) -> Vec<EdgePtr> {
        edge_indices
            .to_vec()
            .iter()
            .map(|&i| self.edges[i as usize].clone())
            .collect()
    }

    fn get_vertex_num(&self) -> usize {
        self.vertices.len()
    }

    fn get_edge_num(&self) -> usize {
        self.edges.len()
    }
}

impl<Queue> DualModulePQGeneric<Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Clone,
{
    /// to be called in dual_module_parallel.rs
    pub fn new_partitioned(partitioned_initializer: &PartitionedSolverInitializer) -> Self {
        // println!("///////////////////////////////////////////////////////////////////////////////");
        // println!("for new_partitioned: {partitioned_initializer:?}");
        // println!("///////////////////////////////////////////////////////////////////////////////");
        /// debug printing
        let mut all_defect_vertices = vec![];
        // create vertices
        let mut vertices: Vec<VertexPtr> = partitioned_initializer
            .owning_range
            .iter()
            .map(|vertex_index| {
                VertexPtr::new_value(Vertex {
                    vertex_index,
                    is_defect: if partitioned_initializer.defect_vertices.contains(&vertex_index) {
                        all_defect_vertices.push(vertex_index);
                        true
                    } else {
                        false
                    },
                    edges: Vec::new(),
                    mirrored_vertices: vec![], // initialized to empty, to be filled in `new_config()` in parallel implementation
                })
            })
            .collect::<Vec<_>>();

        // now we want to 'add' the boundary vertices into the vertices for this partition (if this partition is non-boundary unit). We create new (mirrored) vertices of these
        // boundary vertices, and add them to the vertices list of the non-boundary-unit. These mirrored vertices are not connected to any edges yet, as the edges are not created yet.
        // They are specifically tracked by `all_mirrored_vertices` of DualModulePQGeneric.
        let mut total_boundary_vertices = HashMap::<VertexIndex, VertexIndex>::new(); // all boundary vertices mapping to the specific local partition index
        let mut all_mirrored_vertices = vec![];
        if !partitioned_initializer.is_boundary_unit {
            // only the index_range matters here, the units of the adjacent partitions do not matter here
            for adjacent_index_range in partitioned_initializer.boundary_vertices.iter() {
                for vertex_index in adjacent_index_range.range[0]..adjacent_index_range.range[1] {
                    if !partitioned_initializer.owning_range.contains(vertex_index) {
                        total_boundary_vertices.insert(vertex_index, vertices.len() as VertexIndex);
                        let vertex_ptr0 = VertexPtr::new_value(Vertex {
                            vertex_index,
                            is_defect: if partitioned_initializer.defect_vertices.contains(&vertex_index) {
                                all_defect_vertices.push(vertex_index);
                                true
                            } else {
                                false
                            },
                            edges: Vec::new(),
                            mirrored_vertices: vec![], // set to empty, to be filled in `new_config()` in parallel implementation
                        });
                        vertices.push(vertex_ptr0.clone());
                        all_mirrored_vertices.push(vertex_ptr0);
                    }
                }
            }
        }

        // initialize global time
        let global_time = ArcManualSafeLock::new_value(Rational::zero());

        // set edges
        let mut edges = Vec::<EdgePtr>::new();
        for (hyper_edge, edge_index) in partitioned_initializer.weighted_edges.iter() {
            // above, we have created the vertices that follow its own numbering rule for the index
            // so we need to calculate the vertex indices of the hyper_edge to make it match the local index of each partition unit. then, we can create EdgePtr
            let mut local_hyper_edge_vertices = Vec::<VertexWeak>::new();
            for vertex_index in hyper_edge.vertices.iter() {
                // println!("vertex_index: {:?}", vertex_index);
                let local_index = if partitioned_initializer.owning_range.contains(*vertex_index) {
                    vertex_index - partitioned_initializer.owning_range.start()
                } else {
                    total_boundary_vertices[vertex_index]
                };
                local_hyper_edge_vertices.push(vertices[local_index].downgrade());
            }
            // now we create the edgeptr
            let edge_ptr = EdgePtr::new_value(Edge {
                edge_index: *edge_index,
                weight: hyper_edge.weight.clone(),
                dual_nodes: vec![],
                vertices: local_hyper_edge_vertices,
                last_updated_time: Rational::zero(),
                growth_at_last_updated_time: Rational::zero(),
                grow_rate: Rational::zero(),
                unit_index: Some(partitioned_initializer.unit_index),
                connected_to_boundary_vertex: hyper_edge.connected_to_boundary_vertex,
            });

            // we also need to update the vertices of this hyper_edge
            for vertex_index in hyper_edge.vertices.iter() {
                let local_index = if partitioned_initializer.owning_range.contains(*vertex_index) {
                    vertex_index - partitioned_initializer.owning_range.start()
                } else {
                    total_boundary_vertices[vertex_index]
                };
                vertices[local_index].write().edges.push(edge_ptr.downgrade());
            }

            edges.push(edge_ptr.clone());
            // println!("edge: {:?}, edge_weight: {:?}", edge_ptr.clone().read_recursive().edge_index, edge_ptr.read_recursive().weight);
        }

        Self {
            vertices,
            edges,
            obstacle_queue: Queue::default(),
            global_time: global_time.clone(),
            mode: DualModuleMode::default(),
            vertex_num: partitioned_initializer.vertex_num,
            edge_num: partitioned_initializer.edge_num,
            all_mirrored_vertices,
            unit_active: ArcManualSafeLock::new_value(false), // false by default, to be updated later when edge_growth are calcualted
            tuning_start_time: None,
            total_tuning_time: None,
            negative_weight_sum: Default::default(),
            negative_edges: Default::default(),
            flip_vertices: Default::default(),
            initializer: partitioned_initializer,
        }
    }

    pub fn new_seperate_unit(new_seperate_initializer: &PartitionedSolverInitializer) -> Self {
        // initializer.sanity_check().unwrap();
        // create vertices, might need to offset the vertex index by the total number of vertex in the already existing blocks
        let vertices: Vec<VertexPtr> = (0..new_seperate_initializer.vertex_num)
            .map(|vertex_index| {
                VertexPtr::new_value(Vertex {
                    vertex_index,
                    is_defect: false,
                    edges: vec![],
                    mirrored_vertices: vec![], // set to empty for non-parallel implementation
                })
            })
            .collect::<Vec<_>>();
        // set global time
        let global_time = ArcManualSafeLock::new_value(Rational::zero());
        // set edges
        let mut edges = Vec::<EdgePtr>::new();
        for (hyperedge, _) in new_seperate_initializer.weighted_edges.iter() {
            let edge_ptr = EdgePtr::new_value(Edge {
                edge_index: edges.len() as EdgeIndex,
                weight: hyperedge.weight.clone(),
                dual_nodes: vec![],
                vertices: hyperedge
                    .vertices
                    .iter()
                    .map(|i| vertices[*i as usize].downgrade())
                    .collect::<Vec<_>>(),
                last_updated_time: Rational::zero(),
                growth_at_last_updated_time: Rational::zero(),
                grow_rate: Rational::zero(),
                unit_index: None,
                connected_to_boundary_vertex: false,
                #[cfg(feature = "incr_lp")]
                cluster_weights: hashbrown::HashMap::new(),
            });
            for &vertex_index in hyperedge.vertices.iter() {
                vertices[vertex_index as usize].write().edges.push(edge_ptr.downgrade());
            }
            edges.push(edge_ptr.clone());
        }
        Self {
            vertices,
            edges,
            obstacle_queue: Queue::default(),
            global_time: global_time.clone(),
            mode: DualModuleMode::default(),
            vertex_num: new_seperate_initializer.vertex_num,
            edge_num: new_seperate_initializer.weighted_edges.len(),
            all_mirrored_vertices: vec![],
            // all_defect_vertices: vec![], // used only for parallel implementation
            unit_active: ArcManualSafeLock::new_value(false), // used only for parallel implementation
            tuning_start_time: None,
            total_tuning_time: None,
            negative_weight_sum: Default::default(),
            negative_edges: Default::default(),
            flip_vertices: Default::default(),
            initializer: new_seperate_initializer,
        }
    }
}

impl<Queue> MWPSVisualizer for DualModulePQGeneric<Queue>
where
    Queue: FutureQueueMethods<Rational, Obstacle> + Default + std::fmt::Debug + Clone,
{
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let mut vertices: Vec<serde_json::Value> = (0..self.vertex_num).map(|_| serde_json::Value::Null).collect();

        for vertex_ptr in self.vertices.iter() {
            let vertex = vertex_ptr.read_recursive();
            vertices[vertex.vertex_index] = json!({
                if abbrev { "s" } else { "is_defect" }: i32::from(vertex.is_defect),
            });
        }
        let mut edges: Vec<serde_json::Value> = (0..self.edge_num).map(|_| serde_json::Value::Null).collect();
        for edge_ptr in self.edges.iter() {
            let edge = edge_ptr.read_recursive();
            let current_growth = &edge.growth_at_last_updated_time
                + (&self.global_time.read_recursive().clone() - &edge.last_updated_time) * &edge.grow_rate;
            let unexplored = &edge.weight - &current_growth;
            assert!(!unexplored.is_negative());
            edges[edge.edge_index] = json!({
                if abbrev { "w" } else { "weight" }: edge.weight.to_f64(),
                "wn": numer_of(&edge.weight),
                "wd": denom_of(&edge.weight),
                if abbrev { "v" } else { "vertices" }: edge.vertices.iter().map(|x| x.upgrade_force().read_recursive().vertex_index).collect::<Vec<_>>(),
                if abbrev { "g" } else { "growth" }: current_growth.to_f64(),
                "gn": numer_of(&current_growth),
                "gd": denom_of(&current_growth),
                if abbrev { "u" } else { "unexplored" }: unexplored.to_f64(),
                "un": numer_of(&unexplored),
                "ud": denom_of(&unexplored),
            });
        }
        json!({
            "vertices": vertices,
            "edges": edges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoding_hypergraph::*;
    use crate::example_codes::*;
    use num_traits::FromPrimitive;

    #[test]
    fn dual_module_pq_learn_priority_queue_1() {
        // cargo test dual_module_pq_learn_priority_queue_1 -- --nocapture
        let mut future_obstacle_queue = _FutureObstacleQueue::<usize>::new();
        assert_eq!(0, future_obstacle_queue.len());
        macro_rules! ref_event {
            ($index:expr, $edges:expr) => {
                Some((
                    &$index,
                    &Obstacle::Conflict {
                        edge_ptr: $edges[$index].clone(),
                    },
                ))
            };
        }
        macro_rules! value_event {
            ($index:expr, $edges:expr) => {
                Some((
                    $index,
                    Obstacle::Conflict {
                        edge_ptr: $edges[$index].clone(),
                    },
                ))
            };
        }
        // initialize edges
        let edges: Vec<EdgePtr> = vec![0, 1, 2, 3]
            .into_iter()
            .map(|edge_index| {
                EdgePtr::new_value(Edge {
                    edge_index,
                    weight: Rational::zero(),
                    dual_nodes: vec![],
                    vertices: vec![],
                    last_updated_time: Rational::zero(),
                    growth_at_last_updated_time: Rational::zero(),
                    grow_rate: Rational::zero(),
                    unit_index: None,
                    connected_to_boundary_vertex: false,
                    #[cfg(feature = "incr_lp")]
                    cluster_weights: hashbrown::HashMap::new(),
                })
            })
            .collect();

        // test basic order
        future_obstacle_queue.will_happen(
            2,
            Obstacle::Conflict {
                edge_ptr: edges[2].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[1].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            3,
            Obstacle::Conflict {
                edge_ptr: edges[3].clone(),
            },
        );
        assert_eq!(future_obstacle_queue.peek_event(), ref_event!(1, edges));
        assert_eq!(future_obstacle_queue.peek_event(), ref_event!(1, edges));
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(1, edges));
        assert_eq!(future_obstacle_queue.peek_event(), ref_event!(2, edges));
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(2, edges));
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(3, edges));
        assert_eq!(future_obstacle_queue.peek_event(), None);
        // test duplicate elements, the queue must be able to hold all the duplicate events
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[1].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[1].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[1].clone(),
            },
        );
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(1, edges));
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(1, edges));
        assert_eq!(future_obstacle_queue.pop_event(), value_event!(1, edges));
        assert_eq!(future_obstacle_queue.peek_event(), None);
        // test order of events at the same time
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[2].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[1].clone(),
            },
        );
        future_obstacle_queue.will_happen(
            1,
            Obstacle::Conflict {
                edge_ptr: edges[3].clone(),
            },
        );
        let mut events = vec![];
        while let Some((time, event)) = future_obstacle_queue.pop_event() {
            assert_eq!(time, 1);
            events.push(event);
        }
        assert_eq!(events.len(), 3);
        println!("events: {events:?}");
    }

    #[test]
    fn dual_module_pq_basics_1() {
        // cargo test dual_module_pq_basics_1 -- --nocapture
        let visualize_filename = "dual_module_pq_basics_1.json".to_string();
        let weight: f64 = 2.1972245773362196;
        let code = CodeCapacityColorCode::new(7, 0.1);
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        // create dual module
        let model_graph = code.get_model_graph();
        let mut dual_module = DualModulePQ::new_empty(&model_graph.initializer);
        // try to work on a simple syndrome
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph, vec![3, 12]);
        let interface_ptr = DualModuleInterfacePtr::new_load(decoding_graph, &mut dual_module);

        visualizer
            .snapshot_combined("syndrome".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // grow them each by half
        let dual_node_3_ptr = interface_ptr.read_recursive().nodes[0].clone();
        let dual_node_12_ptr = interface_ptr.read_recursive().nodes[1].clone();
        dual_module.set_grow_rate(&dual_node_3_ptr, Rational::from_usize(1).unwrap());
        dual_module.set_grow_rate(&dual_node_12_ptr, Rational::from_usize(1).unwrap());

        dual_module.grow(Rational::from_f64(weight / 2.).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);
        visualizer
            .snapshot_combined("grow".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // cluster becomes solved
        dual_module.grow(Rational::from_f64(weight / 2.).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);
        visualizer
            .snapshot_combined("solved".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // the result subgraph
        let subgraph = vec![15, 20];
        visualizer
            .snapshot_combined("subgraph".to_string(), vec![&interface_ptr, &dual_module, &subgraph])
            .unwrap();

        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());
    }

    #[test]
    fn dual_module_pq_basics_2() {
        // cargo test dual_module_pq_basics_2 -- --nocapture
        let visualize_filename = "dual_module_pq_basics_2.json".to_string();
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
        let mut dual_module = DualModulePQ::new_empty(&model_graph.initializer);
        // try to work on a simple syndrome
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph, vec![23, 24, 29, 30]);
        let interface_ptr = DualModuleInterfacePtr::new_load(decoding_graph, &mut dual_module);
        visualizer
            .snapshot_combined("syndrome".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        {
            let interface_ptr_read = interface_ptr.read_recursive();
            let dual_node_ptrs = interface_ptr_read.nodes.iter().take(4).cloned();
            dual_node_ptrs.for_each(|node_ptr| dual_module.set_grow_rate(&node_ptr, Rational::from_usize(1).unwrap()));
        }

        // grow them each by a quarter
        dual_module.grow(Rational::from_f64(weight / 4.).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);
        visualizer
            .snapshot_combined("solved".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // the result subgraph
        let subgraph = vec![24];
        visualizer
            .snapshot_combined("subgraph".to_string(), vec![&interface_ptr, &dual_module, &subgraph])
            .unwrap();

        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());
    }

    #[test]
    fn dual_module_pq_basics_3() {
        // cargo test dual_module_pq_basics_3 -- --nocapture
        let visualize_filename = "dual_module_pq_basics_3.json".to_string();
        let pxy = 0.0602828812732227;
        let code = CodeCapacityTailoredCode::new(7, pxy, 0.1); // do not change probabilities: the data is hard-coded
        let mut visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        // create dual module
        let model_graph = code.get_model_graph();
        let mut dual_module = DualModulePQ::new_empty(&model_graph.initializer);
        // try to work on a simple syndrome
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph, vec![17, 23, 29, 30]);
        let interface_ptr = DualModuleInterfacePtr::new_load(decoding_graph, &mut dual_module);
        visualizer
            .snapshot_combined("syndrome".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();
        let dual_node_17_ptr = interface_ptr.read_recursive().nodes[0].clone();
        let dual_node_23_ptr = interface_ptr.read_recursive().nodes[1].clone();
        let dual_node_29_ptr = interface_ptr.read_recursive().nodes[2].clone();
        let dual_node_30_ptr = interface_ptr.read_recursive().nodes[3].clone();

        let unit_grow_rate = 2.1972245773362196 / 1000.;

        // first round of growth
        dual_module.set_grow_rate(&dual_node_17_ptr, Rational::from_f64(unit_grow_rate).unwrap());
        dual_module.set_grow_rate(&dual_node_23_ptr, Rational::from_f64(unit_grow_rate).unwrap());
        dual_module.set_grow_rate(&dual_node_29_ptr, Rational::from_f64(unit_grow_rate).unwrap());
        dual_module.set_grow_rate(&dual_node_30_ptr, Rational::from_f64(unit_grow_rate).unwrap());

        dual_module.grow(Rational::from_i64(160).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);

        visualizer
            .snapshot_combined("grow".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // reset everything
        dual_module.set_grow_rate(&dual_node_17_ptr, Rational::from_i64(0).unwrap());
        dual_module.set_grow_rate(&dual_node_23_ptr, Rational::from_i64(0).unwrap());
        dual_module.set_grow_rate(&dual_node_29_ptr, Rational::from_i64(0).unwrap());
        dual_module.set_grow_rate(&dual_node_30_ptr, Rational::from_i64(0).unwrap());

        // create cluster
        interface_ptr.create_node_vec_from_indices(&[24], &mut dual_module);
        let dual_node_cluster_ptr = interface_ptr.read_recursive().nodes[4].clone();
        dual_module.set_grow_rate(&dual_node_17_ptr, Rational::from_f64(unit_grow_rate).unwrap());
        dual_module.set_grow_rate(&dual_node_cluster_ptr, Rational::from_f64(unit_grow_rate).unwrap());
        dual_module.grow(Rational::from_i64(160).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);

        visualizer
            .snapshot_combined("grow".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // reset
        dual_module.set_grow_rate(&dual_node_17_ptr, Rational::from_i64(0).unwrap());
        dual_module.set_grow_rate(&dual_node_cluster_ptr, Rational::from_i64(0).unwrap());

        // create bigger cluster
        interface_ptr.create_node_vec_from_indices(&[18, 23, 24, 31], &mut dual_module);
        let dual_node_bigger_cluster_ptr = interface_ptr.read_recursive().nodes[5].clone();
        dual_module.set_grow_rate(&dual_node_bigger_cluster_ptr, Rational::from_f64(unit_grow_rate).unwrap());

        dual_module.grow(Rational::from_i64(120).unwrap());
        dual_module.debug_update_all(&interface_ptr.read_recursive().nodes);

        visualizer
            .snapshot_combined("solved".to_string(), vec![&interface_ptr, &dual_module])
            .unwrap();

        // the result subgraph
        let subgraph = vec![82, 24];
        visualizer
            .snapshot_combined("subgraph".to_string(), vec![&interface_ptr, &dual_module, &subgraph])
            .unwrap();

        visualizer.save_html_along_json();
        println!("open visualizer at {}", visualizer.html_along_json_path());
    }

    // TODO: write more tests here, perhaps unit tests
}

// Future Object Queues that are constructed with PQ libraries that are bugged

#[derive(Debug, Clone)]
pub struct PairingPQ<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone> {
    pub container: HashMap<Obstacle, T>,
    pub heap: PairingHeap<Obstacle, T>,
}

// implement default for PairingPQ
impl<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone> Default for PairingPQ<T> {
    fn default() -> Self {
        Self {
            container: HashMap::default(),
            heap: PairingHeap::new(),
        }
    }
}

impl<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone + std::ops::Sub<Output = T> + std::ops::SubAssign>
    FutureQueueMethods<T, Obstacle> for PairingPQ<T>
{
    fn will_happen(&mut self, time: T, event: Obstacle) {
        match self.container.entry(event.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(time.clone());
                self.heap.insert(event, time);
            }
            Entry::Occupied(mut entry) => {
                let old_time = entry.get().clone();
                *entry.get_mut() = time.clone();
                self.heap.decrease_prio(&event, time.clone() - old_time);
            }
        }
    }
    fn peek_event(&self) -> Option<(&T, &Obstacle)> {
        self.heap.find_min().map(|future| (future.1, future.0))
    }
    fn pop_event(&mut self) -> Option<(T, Obstacle)> {
        let res = self.heap.delete_min().map(|future| (future.1, future.0));
        match &res {
            Some((_, event)) => {
                self.container.remove(event);
            }
            None => {}
        }
        res
    }
    fn clear(&mut self) {
        self.container.clear();
        while !self.heap.is_empty() {
            self.heap.delete_min();
        }
    }
    fn len(&self) -> usize {
        self.heap.len()
    }
}

#[derive(Debug, Clone)]
pub struct RankPairingPQ<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone> {
    pub container: HashMap<Obstacle, T>,
    pub heap: RankPairingHeap<Obstacle, T>,
}

impl<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone> Default for RankPairingPQ<T> {
    fn default() -> Self {
        Self {
            container: HashMap::default(),
            heap: RankPairingHeap::multi_pass_min2(),
        }
    }
}

impl<T: Ord + PartialEq + Eq + std::fmt::Debug + Clone> FutureQueueMethods<T, Obstacle> for RankPairingPQ<T> {
    fn will_happen(&mut self, time: T, event: Obstacle) {
        if self.container.contains_key(&event) {
            self.heap.update(&event, time.clone());
            self.container.insert(event, time);
        } else {
            self.heap.push(event.clone(), time.clone());
            self.container.insert(event, time);
        }
    }
    fn peek_event(&self) -> Option<(&T, &Obstacle)> {
        self.heap.top().map(|key| (self.container.get(key).unwrap(), key))
    }
    fn pop_event(&mut self) -> Option<(T, Obstacle)> {
        match self.heap.pop() {
            None => None,
            Some(key) => Some((self.container.remove(&key).unwrap(), key)),
        }
    }
    fn clear(&mut self) {
        self.container.clear();
        while !self.heap.is_empty() {
            self.heap.pop();
        }
    }
    fn len(&self) -> usize {
        self.heap.size()
    }
}
