//! Dual Module
//!
//! Generics for dual modules
//!

use crate::decoding_hypergraph::*;
use crate::derivative::Derivative;
use crate::invalid_subgraph::*;
use crate::model_hypergraph::*;
use crate::num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};
use crate::pointers::*;
use crate::primal_module::Affinity;
use crate::primal_module_serial::PrimalClusterPtr;
use crate::relaxer_optimizer::OptimizerResult;
use crate::util::*;
use crate::visualize::*;
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;

use std::sync::Arc;

// this is not effectively doing much right now due to the My (Leo's) desire for ultra performance (inlining function > branches)
#[derive(Default, Debug, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", eq, eq_int))]
pub enum DualModuleMode {
    /// Mode 1
    #[default]
    Search, // Searching for a solution

    /// Mode 2
    Tune, // Tuning for the optimal solution
}

impl DualModuleMode {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&mut self) {
        match self {
            Self::Search => *self = Self::Tune,
            Self::Tune => panic!("dual module mode is already in tune mode"),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::Search;
    }
}

// Each dual_module impl should have mode and affinity_map, hence these methods should be shared
//      Note: Affinity Map is not implemented in this branch, but a different file/branch (there incurs performance overhead)
#[macro_export]
macro_rules! add_shared_methods {
    () => {
        /// Returns a reference to the mode field.
        fn mode(&self) -> &DualModuleMode {
            &self.mode
        }

        /// Returns a mutable reference to the mode field.
        fn mode_mut(&mut self) -> &mut DualModuleMode {
            &mut self.mode
        }
    };
}

pub struct DualNode {
    /// the index of this dual node, helps to locate internal details of this dual node
    pub index: NodeIndex,
    /// the corresponding invalid subgraph
    pub invalid_subgraph: Arc<InvalidSubgraph>,

    /// the strategy to grow the dual variables
    pub grow_rate: Rational,
    /// the pointer to the global time
    /// Note: may employ some unsafe features while being sound in performance-critical cases
    ///       and can remove option when removing dual_module_serial
    global_time: Option<ArcRwLock<Rational>>,
    /// the last time this dual_node is synced/updated with the global time
    pub last_updated_time: Rational,
    /// dual variable's value at the last updated time
    pub dual_variable_at_last_updated_time: Rational,
}

impl DualNode {
    /// get the current up-to-date dual_variable
    pub fn get_dual_variable(&self) -> Rational {
        // in the interest of performance/avoiding redundant work, this may be upgraded to taking in
        // `&mut self` and update the value if needed
        match self.global_time.clone() {
            Some(global_time) => {
                // Note: clone here to give up read lock?
                let global_time = global_time.read_recursive();
                if global_time.ge(&self.last_updated_time) {
                    (global_time.clone() - self.last_updated_time.clone()) * self.grow_rate.clone()
                        + self.dual_variable_at_last_updated_time.clone()
                } else {
                    self.dual_variable_at_last_updated_time.clone()
                }
            }
            None => self.dual_variable_at_last_updated_time.clone(),
        }
    }

    /// setter for current dual_variable
    pub fn set_dual_variable(&mut self, new_dual_variable: Rational) {
        self.dual_variable_at_last_updated_time = new_dual_variable;
    }

    /// initialize the global time pointer and the last_updated_time
    pub fn init_time(&mut self, global_time_ptr: ArcRwLock<Rational>) {
        self.last_updated_time = global_time_ptr.read_recursive().clone();
        self.global_time = Some(global_time_ptr);
    }
}

pub type DualNodePtr = ArcRwLock<DualNode>;
pub type DualNodeWeak = WeakRwLock<DualNode>;

impl std::fmt::Debug for DualNodePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let dual_node = self.read_recursive(); // reading index is consistent
        f.debug_struct("DualNode")
            .field("index", &dual_node.index)
            .field("dual_variable", &dual_node.get_dual_variable())
            .field("grow_rate", &dual_node.grow_rate)
            .field("hair", &dual_node.invalid_subgraph.hair)
            .finish()
        // let new = ArcRwLock::new_value(Rational::zero());
        // let global_time = dual_node.global_time.as_ref().unwrap_or(&new).read_recursive();
        // write!(
        //     f,
        //     "\n\t\tindex: {}, global_time: {:?}, grow_rate: {:?}, dual_variable: {}\n\t\tdual_variable_at_last_updated_time: {}, last_updated_time: {}\n\timpacted_edges: {:?}\n",
        //     dual_node.index,
        //     global_time,
        //     dual_node.grow_rate,
        //     dual_node.get_dual_variable(),
        //     dual_node.dual_variable_at_last_updated_time,
        //     dual_node.last_updated_time,
        //     dual_node.invalid_subgraph.hair
        // )
    }
}

impl std::fmt::Debug for DualNodeWeak {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.upgrade_force().fmt(f)
    }
}

/// an array of dual nodes
/// dual nodes, once created, will never be deconstructed until the next run
#[derive(Derivative)]
#[derivative(Debug)]
#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf"))]
pub struct DualModuleInterface {
    /// all the dual node that can be used to control a concrete dual module implementation
    // #[cfg_attr(feature = "python_binding", pyo3(get))]
    pub nodes: Vec<DualNodePtr>,
    /// given an invalid subgraph, find its corresponding dual node
    pub hashmap: HashMap<Arc<InvalidSubgraph>, NodeIndex>,
    /// the decoding graph
    pub decoding_graph: DecodingHyperGraph,
}

pub type DualModuleInterfacePtr = ArcRwLock<DualModuleInterface>;
pub type DualModuleInterfaceWeak = WeakRwLock<DualModuleInterface>;

impl std::fmt::Debug for DualModuleInterfacePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let interface = self.read_recursive();
        write!(f, "{}", interface.nodes.len())
    }
}

impl std::fmt::Debug for DualModuleInterfaceWeak {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.upgrade_force().fmt(f)
    }
}

/// a pair of node index and dual node pointer, used for comparison without acquiring the lock
/// useful for when inserting into sets
#[derive(Derivative, PartialEq, Eq, Clone, Debug)]
pub struct OrderedDualNodePtr {
    pub index: NodeIndex,
    pub ptr: DualNodePtr,
}

impl std::hash::Hash for OrderedDualNodePtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl OrderedDualNodePtr {
    pub fn new(index: NodeIndex, ptr: DualNodePtr) -> Self {
        Self { index, ptr }
    }
}
impl From<DualNodePtr> for OrderedDualNodePtr {
    fn from(ptr: DualNodePtr) -> Self {
        let index = ptr.read_recursive().index;
        Self { index, ptr }
    }
}
impl PartialOrd for OrderedDualNodePtr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.index.cmp(&other.index))
    }
}
impl Ord for OrderedDualNodePtr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

#[derive(Derivative, PartialEq, Eq, Clone, Debug)]
pub struct OrderedDualNodeWeak {
    pub index: NodeIndex,
    pub weak_ptr: DualNodeWeak,
}

impl OrderedDualNodeWeak {
    pub fn new(index: NodeIndex, weak_ptr: DualNodeWeak) -> Self {
        Self { index, weak_ptr }
    }

    pub fn upgrade_force(&self) -> OrderedDualNodePtr {
        OrderedDualNodePtr::new(self.index, self.weak_ptr.upgrade_force())
    }
}
impl PartialOrd for OrderedDualNodeWeak {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.index.cmp(&other.index))
    }
}
impl Ord for OrderedDualNodeWeak {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

#[derive(Derivative, Clone)]
#[derivative(Debug, Default(new = "true"))]
pub enum DualReport {
    /// unbounded
    #[derivative(Default)]
    Unbounded,
    /// non-zero maximum update length
    ValidGrow(Rational),
    /// conflicting reasons and pending VertexShrinkStop events (empty in a single serial dual module)
    Obstacles(Vec<Obstacle>),
}

/// common trait that must be implemented for each implementation of dual module
pub trait DualModuleImpl {
    /// create a new dual module with empty syndrome
    fn new_empty(initializer: &Arc<SolverInitializer>) -> Self;

    /// clear all growth and existing dual nodes, prepared for the next decoding
    fn clear(&mut self);

    /// add defect node
    fn add_defect_node(&mut self, dual_node_ptr: &DualNodePtr);

    /// add corresponding dual node, note that the `internal_vertices` and `hair_edges` are not set
    fn add_dual_node(&mut self, dual_node_ptr: &DualNodePtr);

    /// update grow rate
    fn set_grow_rate(&mut self, dual_node_ptr: &DualNodePtr, grow_rate: Rational);

    /// check the maximum length to grow (shrink) for all nodes, return a list of conflicting reason and a single number indicating the maximum rate to grow:
    /// this number will be 0 if any conflicting reason presents
    fn report(&mut self) -> DualReport;

    /// An optional function that can manipulate individual dual node, not necessarily supported by all implementations
    fn grow_dual_node(&mut self, _dual_node_ptr: &DualNodePtr, _length: Rational) {
        panic!("the dual module implementation doesn't support this function, please use another dual module")
    }

    /// grow a specific length globally, length must be positive.
    /// note that a negative growth should be implemented by reversing the speed of each dual node
    fn grow(&mut self, length: Rational);

    /// get all nodes contributing to the edge
    fn get_edge_nodes(&self, edge_index: EdgeIndex) -> Vec<DualNodePtr>;

    /// get the slack on a specific edge (weight - growth)
    fn get_edge_slack(&self, edge_index: EdgeIndex) -> Rational;

    /// check if the edge is tight
    fn is_edge_tight(&self, edge_index: EdgeIndex) -> bool;

    /* New tuning-related methods */
    // mode managements

    /// get the current mode of the dual module
    fn mode(&self) -> &DualModuleMode;

    /// get the mutable reference to the current mode of the dual module
    fn mode_mut(&mut self) -> &mut DualModuleMode;

    /// advance the mode from searching to tuning
    fn advance_mode(&mut self) {
        eprintln!("this dual_module does not implement different modes");
    }

    /// reset the mode to the default
    fn reset_mode(&mut self) {
        *self.mode_mut() = DualModuleMode::default();
    }

    /// document the end of tuning for getting the total tuning time
    fn end_tuning(&mut self) {
        panic!("this module doesn't work with tuning")
    }

    /// get the total tuning time
    fn get_total_tuning_time(&self) -> Option<f64> {
        panic!("this module doesn't work with tuning")
    }

    /// Reset: clear the tuning time
    fn clear_tuning_time(&mut self) {
        panic!("this module doesn't work with tuning")
    }

    /// "add_dual_node", but in tuning phase, don't modify the pq or the grow rates
    fn add_dual_node_tune(&mut self, dual_node_ptr: &DualNodePtr) {
        eprintln!("this dual_module does not implement tuning");
        self.add_dual_node(dual_node_ptr);
    }

    /// syncing all possible states (dual_variable and edge_weights) with global time, so global_time can be discarded later
    fn sync(&mut self) {
        panic!("this dual_module does not have global time and does not need to sync");
    }

    /// grow a specific edge on the spot
    fn grow_edge(&self, _edge_index: EdgeIndex, _amount: &Rational) {
        panic!("this dual_module doesn't support edge growth");
    }

    /// `is_edge_tight` but in tuning phase
    fn is_edge_tight_tune(&self, edge_index: EdgeIndex) -> bool {
        eprintln!("this dual_module does not implement tuning");
        self.is_edge_tight(edge_index)
    }

    /// `get_edge_slack` but in tuning phase
    fn get_edge_slack_tune(&self, edge_index: EdgeIndex) -> Rational {
        eprintln!("this dual_module does not implement tuning");
        self.get_edge_slack(edge_index)
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
    fn calculate_cluster_affinity(&mut self, _cluster: PrimalClusterPtr) -> Option<Affinity> {
        eprintln!("not implemented, skipping");
        Some(Affinity::from(100.0))
    }

    /// In the tuning phase, given the optimizer result and the dual node deltas, return the Obstacles that are caused by the current dual node deltas
    fn get_obstacles_tune(
        &self,
        optimizer_result: OptimizerResult,
        dual_node_deltas: FastIterMap<OrderedDualNodePtr, (Rational, NodeIndex)>,
    ) -> FastIterSet<Obstacle> {
        let mut obstacles = FastIterSet::new();
        match optimizer_result {
            OptimizerResult::EarlyReturned => {
                // if early returned, meaning optimizer didn't optimize, but simply should find current obstacles and return
                for (dual_node_ptr, (grow_rate, _)) in dual_node_deltas.into_iter() {
                    let node_ptr_read = dual_node_ptr.ptr.read_recursive();
                    if grow_rate.is_negative() && node_ptr_read.dual_variable_at_last_updated_time.is_zero() {
                        obstacles.insert(Obstacle::ShrinkToZero {
                            dual_node_ptr: dual_node_ptr.clone(),
                        });
                    }
                    for edge_index in node_ptr_read.invalid_subgraph.hair.iter() {
                        if grow_rate.is_positive() && self.is_edge_tight_tune(*edge_index) {
                            obstacles.insert(Obstacle::Conflict { edge_index: *edge_index });
                        }
                    }
                }
            }
            OptimizerResult::Skipped => {
                // if skipped, should check if is growable, if not return the obstacles that leads to that conclusion
                for (dual_node_ptr, (grow_rate, _cluster_index)) in dual_node_deltas.into_iter() {
                    // check if the single direction is growable
                    let mut actual_grow_rate = Rational::from_usize(std::usize::MAX).unwrap();
                    let node_ptr_read = dual_node_ptr.ptr.read_recursive();
                    for edge_index in node_ptr_read.invalid_subgraph.hair.iter() {
                        actual_grow_rate = std::cmp::min(actual_grow_rate, self.get_edge_slack_tune(*edge_index));
                    }
                    if actual_grow_rate.is_zero() {
                        // if not, return the current obstacles
                        for edge_index in node_ptr_read.invalid_subgraph.hair.iter() {
                            if grow_rate.is_positive() && self.is_edge_tight_tune(*edge_index) {
                                obstacles.insert(Obstacle::Conflict { edge_index: *edge_index });
                            }
                        }
                        if grow_rate.is_negative() && node_ptr_read.dual_variable_at_last_updated_time.is_zero() {
                            obstacles.insert(Obstacle::ShrinkToZero {
                                dual_node_ptr: dual_node_ptr.clone(),
                            });
                        }
                    } else {
                        // if yes, grow and return new obstacles
                        //      note: can grow directly here because this is guaranteed to only have a single direction
                        drop(node_ptr_read);
                        let mut node_ptr_write = dual_node_ptr.ptr.write();
                        for edge_index in node_ptr_write.invalid_subgraph.hair.iter() {
                            self.grow_edge(*edge_index, &actual_grow_rate);
                            #[cfg(feature = "incr_lp")]
                            self.update_edge_cluster_weights(*edge_index, _cluster_index, actual_grow_rate.clone()); // note: comment out if not using cluster-based
                            if actual_grow_rate.is_positive() && self.is_edge_tight_tune(*edge_index) {
                                obstacles.insert(Obstacle::Conflict { edge_index: *edge_index });
                            }
                        }
                        node_ptr_write.dual_variable_at_last_updated_time += actual_grow_rate.clone();
                        if actual_grow_rate.is_negative() && node_ptr_write.dual_variable_at_last_updated_time.is_zero() {
                            obstacles.insert(Obstacle::ShrinkToZero {
                                dual_node_ptr: dual_node_ptr.clone(),
                            });
                        }
                    }
                }
            }
            _ => {
                // in other cases, optimizer should have optimized, so we should apply the deltas and return the nwe obstacles

                // edge deltas needs to be applied at once for accurate obstacles calculation
                let mut edge_deltas = FastIterMap::new();
                for (dual_node_ptr, (grow_rate, _cluster_index)) in dual_node_deltas.into_iter() {
                    // update the dual node and check for obstacles
                    let mut node_ptr_write = dual_node_ptr.ptr.write();
                    node_ptr_write.dual_variable_at_last_updated_time += grow_rate.clone();
                    if grow_rate.is_negative() && node_ptr_write.dual_variable_at_last_updated_time.is_zero() {
                        obstacles.insert(Obstacle::ShrinkToZero {
                            dual_node_ptr: dual_node_ptr.clone(),
                        });
                    }

                    // calculate the total edge deltas
                    for edge_index in node_ptr_write.invalid_subgraph.hair.iter() {
                        match edge_deltas.entry(*edge_index) {
                            FastIterEntry::Vacant(v) => {
                                v.insert(grow_rate.clone());
                            }
                            FastIterEntry::Occupied(mut o) => {
                                o.insert(o.get() + grow_rate.clone());
                            }
                        }

                        #[cfg(feature = "incr_lp")]
                        self.update_edge_cluster_weights(*edge_index, _cluster_index, grow_rate.clone());
                        // note: comment out if not using cluster-based
                    }
                }

                // apply the edge deltas and check for obstacles
                for (edge_index, grow_rate) in edge_deltas.into_iter() {
                    if grow_rate.is_zero() {
                        continue;
                    }
                    self.grow_edge(edge_index, &grow_rate);
                    if grow_rate.is_positive() && self.is_edge_tight_tune(edge_index) {
                        obstacles.insert(Obstacle::Conflict { edge_index });
                    }
                }
            }
        }
        obstacles
    }

    /// get the edge free weight, for each edge what is the weight that are free to use by the given participating dual variables
    fn get_edge_free_weight(
        &self,
        edge_index: EdgeIndex,
        participating_dual_variables: &hashbrown::HashSet<usize>,
    ) -> Weight;

    fn get_edge_weight(&self, edge_index: EdgeIndex) -> Weight;

    fn get_subgraph_weight(&self, subgraph: &Subgraph) -> Weight {
        let mut weight = Weight::zero();
        for &edge_index in subgraph {
            weight += self.get_edge_weight(edge_index);
        }
        weight
    }

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights(&self, edge_index: EdgeIndex, cluster_index: NodeIndex, grow_rate: Rational);

    #[cfg(feature = "incr_lp")]
    fn get_edge_free_weight_cluster(&self, edge_index: EdgeIndex, cluster_index: NodeIndex) -> Rational;

    #[cfg(feature = "incr_lp")]
    fn update_edge_cluster_weights_union(
        &self,
        dual_node_ptr: &DualNodePtr,
        drained_cluster_index: NodeIndex,
        absorbing_cluster_index: NodeIndex,
    );

    fn adjust_weights_for_negative_edges(&mut self) {
        unimplemented!()
    }

    /// update weights of dual_module;
    /// the weight of the dual module is set to be `old_weight + mix_ratio * (new_weight - old_weight)`
    fn update_weights(&mut self, _new_weights: Vec<Weight>, _mix_ratio: Weight) {
        unimplemented!()
    }

    /// force set the weights given the indices and new weights
    fn set_weights(&mut self, _new_weights: FastIterMap<EdgeIndex, Weight>) {
        unimplemented!()
    }

    fn get_negative_weight_sum(&self) -> Rational {
        unimplemented!()
    }

    fn get_negative_edges(&self) -> HashSet<EdgeIndex> {
        unimplemented!()
    }

    fn get_flip_vertices(&self) -> HashSet<VertexIndex> {
        unimplemented!()
    }
}

#[derive(PartialEq, Eq, Debug, Clone, PartialOrd, Ord)]
pub enum Obstacle {
    Conflict { edge_index: EdgeIndex },
    ShrinkToZero { dual_node_ptr: OrderedDualNodePtr },
}

// implement hash for Obstacle
impl std::hash::Hash for Obstacle {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Obstacle::Conflict { edge_index } => {
                (0, *edge_index as u64).hash(state);
            }
            Obstacle::ShrinkToZero { dual_node_ptr } => {
                (1, dual_node_ptr.index).hash(state);
            }
        }
    }
}

impl DualReport {
    pub fn add_obstacle(&mut self, obstacle: Obstacle) {
        match self {
            Self::Unbounded | Self::ValidGrow(_) => {
                *self = Self::Obstacles(vec![obstacle]);
            }
            Self::Obstacles(obstacles) => {
                obstacles.push(obstacle);
            }
        }
    }

    pub fn is_unbounded(&self) -> bool {
        matches!(self, Self::Unbounded)
    }

    pub fn get_valid_growth(&self) -> Option<Rational> {
        match self {
            Self::Unbounded => {
                panic!("please call DualReport::is_unbounded to check if it's unbounded");
            }
            Self::ValidGrow(length) => Some(length.clone()),
            _ => None,
        }
    }

    pub fn pop(&mut self) -> Option<Obstacle> {
        match self {
            Self::Unbounded | Self::ValidGrow(_) => {
                panic!("please call DualReport::get_valid_growth to check if this group is none_zero_growth");
            }
            Self::Obstacles(obstacles) => obstacles.pop(),
        }
    }

    pub fn peek(&self) -> Option<&Obstacle> {
        match self {
            Self::Unbounded | Self::ValidGrow(_) => {
                panic!("please call DualReport::get_valid_growth to check if this group is none_zero_growth");
            }
            Self::Obstacles(obstacles) => obstacles.last(),
        }
    }

    pub fn iter(&self) -> Option<std::slice::Iter<Obstacle>> {
        match self {
            Self::Unbounded | Self::ValidGrow(_) => None,
            Self::Obstacles(obstacles) => Some(obstacles.iter()),
        }
    }
}

impl DualModuleInterfacePtr {
    pub fn new(model_graph: Arc<ModelHyperGraph>) -> Self {
        Self::new_value(DualModuleInterface {
            nodes: Vec::new(),
            hashmap: HashMap::new(),
            decoding_graph: DecodingHyperGraph::new(model_graph, Arc::new(SyndromePattern::new_empty())),
        })
    }

    /// a dual module interface MUST be created given a concrete implementation of the dual module
    pub fn new_load(decoding_graph: DecodingHyperGraph, dual_module_impl: &mut impl DualModuleImpl) -> Self {
        let interface_ptr = Self::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern, dual_module_impl);
        interface_ptr
    }

    pub fn load(&self, syndrome_pattern: Arc<SyndromePattern>, dual_module_impl: &mut impl DualModuleImpl) {
        self.write().decoding_graph.set_syndrome(syndrome_pattern.clone());
        for vertex_idx in syndrome_pattern.defect_vertices.iter() {
            self.create_defect_node(*vertex_idx, dual_module_impl);
        }
    }

    pub fn sum_dual_variables(&self) -> Rational {
        let interface = self.read_recursive();
        interface
            .nodes
            .iter()
            .map(|node_ptr| node_ptr.read_recursive().get_dual_variable())
            .sum()
    }

    pub fn clear(&self) {
        let mut interface = self.write();
        interface.nodes.clear();
        interface.hashmap.clear();
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn get_node(&self, node_index: NodeIndex) -> Option<DualNodePtr> {
        let interface = self.read_recursive();
        interface.nodes.get(node_index as usize).cloned()
    }

    /// make it private; use `load` instead
    fn create_defect_node(&self, vertex_idx: VertexIndex, dual_module: &mut impl DualModuleImpl) -> DualNodePtr {
        let mut interface = self.write();
        let invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(
            vec![vertex_idx].into_iter().collect(),
            FastIterSet::new(),
            &interface.decoding_graph,
        ));
        let node_index = interface.nodes.len() as NodeIndex;
        let node_ptr = DualNodePtr::new_value(DualNode {
            index: node_index,
            invalid_subgraph: invalid_subgraph.clone(),
            grow_rate: Rational::one(),
            dual_variable_at_last_updated_time: Rational::zero(),
            global_time: None,
            last_updated_time: Rational::zero(),
        });

        interface.nodes.push(node_ptr.clone());
        interface.hashmap.insert(invalid_subgraph, node_index);
        drop(interface);

        dual_module.add_defect_node(&node_ptr);
        node_ptr
    }

    /// find existing node
    #[allow(clippy::unnecessary_cast)]
    pub fn find_node(&self, invalid_subgraph: &Arc<InvalidSubgraph>) -> Option<DualNodePtr> {
        let interface = self.read_recursive();
        interface
            .hashmap
            .get(invalid_subgraph)
            .map(|index| interface.nodes[*index as usize].clone())
    }

    pub fn create_node(&self, invalid_subgraph: Arc<InvalidSubgraph>, dual_module: &mut impl DualModuleImpl) -> DualNodePtr {
        self.create_node_internal(invalid_subgraph, dual_module, Rational::one(), DualModuleImpl::add_dual_node)
    }

    /// `create_node` for tuning
    pub fn create_node_tune(
        &self,
        invalid_subgraph: Arc<InvalidSubgraph>,
        dual_module: &mut impl DualModuleImpl,
    ) -> DualNodePtr {
        self.create_node_internal(
            invalid_subgraph,
            dual_module,
            Rational::zero(),
            DualModuleImpl::add_dual_node_tune,
        )
    }

    /// return whether it's existing node or not
    pub fn find_or_create_node(
        &self,
        invalid_subgraph: &Arc<InvalidSubgraph>,
        dual_module: &mut impl DualModuleImpl,
    ) -> (bool, DualNodePtr) {
        match self.find_node(invalid_subgraph) {
            Some(node_ptr) => (true, node_ptr),
            None => (false, self.create_node(invalid_subgraph.clone(), dual_module)),
        }
    }

    /// `find_or_create_node` for tuning
    pub fn find_or_create_node_tune(
        &self,
        invalid_subgraph: &Arc<InvalidSubgraph>,
        dual_module: &mut impl DualModuleImpl,
    ) -> Option<(bool, DualNodePtr)> {
        match self.find_node(invalid_subgraph) {
            Some(node_ptr) => Some((true, node_ptr)),
            None => Some((false, self.create_node_tune(invalid_subgraph.clone(), dual_module))),
        }
    }

    /// internal function for creating a node, for D.R.Y.
    fn create_node_internal<D: DualModuleImpl>(
        &self,
        invalid_subgraph: Arc<InvalidSubgraph>,
        dual_module: &mut D,
        grow_rate: Rational,
        add_dual_node_fn: fn(&mut D, &DualNodePtr),
    ) -> DualNodePtr {
        debug_assert!(
            self.find_node(&invalid_subgraph).is_none(),
            "do not create the same node twice"
        );

        let mut interface = self.write();
        let node_index = interface.nodes.len() as NodeIndex;
        interface.hashmap.insert(invalid_subgraph.clone(), node_index);

        let node_ptr = DualNodePtr::new_value(DualNode {
            index: node_index,
            invalid_subgraph,
            grow_rate,
            dual_variable_at_last_updated_time: Rational::zero(),
            global_time: None,
            last_updated_time: Rational::zero(),
        });

        interface.nodes.push(node_ptr.clone());
        drop(interface);

        add_dual_node_fn(dual_module, &node_ptr);

        node_ptr
    }
}

// shortcuts for easier code writing at debugging
impl DualModuleInterfacePtr {
    pub fn create_node_vec(&self, edges: &[EdgeIndex], dual_module: &mut impl DualModuleImpl) -> DualNodePtr {
        let invalid_subgraph = Arc::new(InvalidSubgraph::new(
            edges.iter().cloned().collect(),
            &self.read_recursive().decoding_graph,
        ));
        self.create_node(invalid_subgraph, dual_module)
    }
    pub fn create_node_complete_vec(
        &self,
        vertices: &[VertexIndex],
        edges: &[EdgeIndex],
        dual_module: &mut impl DualModuleImpl,
    ) -> DualNodePtr {
        let invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(
            vertices.iter().cloned().collect(),
            edges.iter().cloned().collect(),
            &self.read_recursive().decoding_graph,
        ));
        self.create_node(invalid_subgraph, dual_module)
    }
}

impl MWPSVisualizer for DualModuleInterfacePtr {
    fn snapshot(&self, abbrev: bool) -> serde_json::Value {
        let interface = self.read_recursive();
        let mut dual_nodes = Vec::<serde_json::Value>::new();
        for dual_node_ptr in interface.nodes.iter() {
            let dual_node = dual_node_ptr.read_recursive();
            #[cfg(feature = "fast_ds")]
            dual_nodes.push(json!({
                if abbrev { "e" } else { "edges" }: dual_node.invalid_subgraph.edges.iter().copied().collect::<Vec<_>>(),
                if abbrev { "v" } else { "vertices" }: dual_node.invalid_subgraph.vertices.iter().copied().collect::<Vec<_>>(),
                if abbrev { "h" } else { "hair" }: dual_node.invalid_subgraph.hair.iter().copied().collect::<Vec<_>>(),
                if abbrev { "d" } else { "dual_variable" }: dual_node.get_dual_variable().to_f64(),
                if abbrev { "dn" } else { "dual_variable_numerator" }: numer_of(&dual_node.get_dual_variable()),
                if abbrev { "dd" } else { "dual_variable_denominator" }: denom_of(&dual_node.get_dual_variable()),
                if abbrev { "r" } else { "grow_rate" }: dual_node.grow_rate.to_f64(),
                if abbrev { "rn" } else { "grow_rate_numerator" }: numer_of(&dual_node.grow_rate),
                if abbrev { "rd" } else { "grow_rate_denominator" }: denom_of(&dual_node.grow_rate),
            }));
            #[cfg(not(feature = "fast_ds"))]
            dual_nodes.push(json!({
                if abbrev { "e" } else { "edges" }: dual_node.invalid_subgraph.edges,
                if abbrev { "v" } else { "vertices" }: dual_node.invalid_subgraph.vertices,
                if abbrev { "h" } else { "hair" }: dual_node.invalid_subgraph.hair,
                if abbrev { "d" } else { "dual_variable" }: dual_node.get_dual_variable().to_f64(),
                if abbrev { "dn" } else { "dual_variable_numerator" }: numer_of(&dual_node.get_dual_variable()),
                if abbrev { "dd" } else { "dual_variable_denominator" }: denom_of(&dual_node.get_dual_variable()),
                if abbrev { "r" } else { "grow_rate" }: dual_node.grow_rate.to_f64(),
                if abbrev { "rn" } else { "grow_rate_numerator" }: numer_of(&dual_node.grow_rate),
                if abbrev { "rd" } else { "grow_rate_denominator" }: denom_of(&dual_node.grow_rate),
            }));
        }
        let sum_dual = self.sum_dual_variables();
        json!({
            "interface": {
                "sum_dual": sum_dual.to_f64(),
                "sdn": numer_of(&sum_dual),
                "sdd": denom_of(&sum_dual),
            },
            "dual_nodes": dual_nodes,
        })
    }
}
