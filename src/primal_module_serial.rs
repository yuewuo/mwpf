//! Serial Primal Module
//!
//! This implementation targets to be an exact MWPF solver, although it's not yet sure whether it is actually one.
//!

use crate::decoding_hypergraph::*;
use crate::dual_module::*;
use crate::invalid_subgraph::*;
use crate::matrix::*;
use crate::num_traits::{One, Zero};
use crate::plugin::*;
use crate::pointers::*;
use crate::primal_module::*;
use crate::relaxer_optimizer::*;
use crate::util::*;
use crate::visualize::*;

use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

use crate::itertools::Itertools;
#[cfg(feature = "incr_lp")]
use parking_lot::Mutex;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct PrimalModuleSerial {
    /// dual nodes information
    pub nodes: Vec<PrimalModuleSerialNodePtr>,
    /// clusters of dual nodes
    pub clusters: Vec<PrimalClusterPtr>,
    /// pending dual variables to grow, when using SingleCluster growing strategy
    pending_nodes: VecDeque<PrimalModuleSerialNodeWeak>,
    /// plugins
    pub plugins: Arc<PluginVec>,
    /// how many plugins are actually executed for every cluster
    pub plugin_count: Arc<RwLock<usize>>,
    pub plugin_pending_clusters: Vec<usize>,
    /// configuration
    pub config: PrimalModuleSerialConfig,
    /// the time spent on resolving the obstacles
    pub time_resolve: f64,
    /// sorted clusters by affinity, only exist when needed
    pub sorted_clusters_aff: Option<FastIterSet<ClusterAffinity>>,
    #[cfg(feature = "incr_lp")]
    /// parameter indicating if the primal module has initialized states necessary for `incr_lp` slack calculation
    pub cluster_weights_initialized: bool,
}

#[derive(Eq, Debug, Clone, Default)]
pub struct ClusterAffinity {
    pub cluster_index: NodeIndex,
    pub affinity: Affinity,
}

impl std::hash::Hash for ClusterAffinity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.cluster_index.hash(state);
        self.affinity.hash(state);
    }
}

impl PartialEq for ClusterAffinity {
    fn eq(&self, other: &Self) -> bool {
        self.affinity == other.affinity && self.cluster_index == other.cluster_index
    }
}

// first sort by affinity in descending order, then by cluster_index in ascending order
impl Ord for ClusterAffinity {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // First, compare affinity in descending order
        match other.affinity.cmp(&self.affinity) {
            std::cmp::Ordering::Equal => {
                // If affinities are equal, compare cluster_index in ascending order
                self.cluster_index.cmp(&other.cluster_index)
            }
            other => other.reverse(),
        }
    }
}

impl PartialOrd for ClusterAffinity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub enum Unionable {
    Can,
    DoesNotNeed,
    Cannot,
}

#[cfg_attr(feature = "python_binding", pyclass(module = "mwpf", get_all, set_all))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimalModuleSerialConfig {
    /// timeout for the whole solving process
    #[serde(default = "primal_serial_default_configs::timeout")]
    pub timeout: f64,
    /// cluster size limit in tuning phase, possibly based on the code-distance
    ///     note: this is not monitored in the searching phase because we need to ensure at least one valid solution is generated
    #[serde(default = "primal_serial_default_configs::cluster_node_limit")]
    pub cluster_node_limit: usize,
    /// by default, we will constantly trying to solve primal problem given the tight matrix from a plugin; however, one
    ///     might want to speed it up by disabling the feature and instead only solve primal problem once at the end
    #[serde(default = "primal_serial_default_configs::only_solve_primal_once")]
    pub only_solve_primal_once: bool,
}

pub mod primal_serial_default_configs {
    pub fn timeout() -> f64 {
        f64::MAX
    }
    pub fn cluster_node_limit() -> usize {
        (2 << 53) - 1 // maximum integer that can be stored in JSON number without loss
    }
    pub fn only_solve_primal_once() -> bool {
        false
    }
}

pub struct PrimalModuleSerialNode {
    /// the dual node
    pub dual_node_ptr: DualNodePtr,
    /// the cluster that it belongs to
    pub cluster_weak: PrimalClusterWeak,
}

pub type PrimalModuleSerialNodePtr = ArcRwLock<PrimalModuleSerialNode>;
pub type PrimalModuleSerialNodeWeak = WeakRwLock<PrimalModuleSerialNode>;

pub struct PrimalCluster {
    /// the index in the cluster
    pub cluster_index: NodeIndex,
    /// the nodes that belongs to this cluster
    pub nodes: Vec<PrimalModuleSerialNodePtr>,
    /// all the edges ever exists in any hair
    pub edges: FastIterSet<EdgeIndex>,
    /// all the vertices ever touched by any tight edge
    pub vertices: FastIterSet<VertexIndex>,
    /// the parity matrix to determine whether it's a valid cluster and also find new ways to increase the dual
    pub matrix: EchelonMatrix,
    /// the parity subgraph result, only valid when it's solved
    pub subgraph: Option<Subgraph>,
    /// plugin manager helps to execute the plugin and find an executable relaxer
    pub plugin_manager: PluginManager,
    /// optimizing the direction of relaxers
    pub relaxer_optimizer: RelaxerOptimizer,
    /// HIHGS solution stored for incrmental lp
    #[cfg(feature = "incr_lp")] //note: really depends where we want the error to manifest
    pub incr_solution: Option<Arc<Mutex<IncrLPSolution>>>,
}

pub type PrimalClusterPtr = ArcRwLock<PrimalCluster>;
pub type PrimalClusterWeak = WeakRwLock<PrimalCluster>;

impl PrimalModuleImpl for PrimalModuleSerial {
    fn new_empty(_initializer: &Arc<SolverInitializer>) -> Self {
        Self {
            nodes: vec![],
            clusters: vec![],
            pending_nodes: VecDeque::new(),
            plugins: Arc::new(vec![]), // default to UF decoder, i.e., without any plugins
            plugin_count: Arc::new(RwLock::new(1)),
            plugin_pending_clusters: vec![],
            config: serde_json::from_value(json!({})).unwrap(),
            time_resolve: 0.,
            sorted_clusters_aff: None,
            #[cfg(feature = "incr_lp")]
            cluster_weights_initialized: false,
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.clusters.clear();
        self.pending_nodes.clear();
        *self.plugin_count.write() = 1;
        self.plugin_pending_clusters.clear();
        self.time_resolve = 0.;
        self.sorted_clusters_aff = None;
        #[cfg(feature = "incr_lp")]
        self.uninit_cluster_weight();
    }

    #[allow(clippy::unnecessary_cast)]
    fn load<D: DualModuleImpl>(&mut self, interface_ptr: &DualModuleInterfacePtr, _dual_module: &mut D) {
        let interface = interface_ptr.read_recursive();
        for index in 0..interface.nodes.len() as NodeIndex {
            let dual_node_ptr = &interface.nodes[index as usize];
            let node = dual_node_ptr.read_recursive();
            debug_assert!(
                node.invalid_subgraph.edges.is_empty(),
                "must load a fresh dual module interface, found a complex node; did you forget to call solver.clear()?"
            );
            debug_assert!(
                node.invalid_subgraph.vertices.len() == 1,
                "must load a fresh dual module interface, found invalid defect node; did you forget to call solver.clear()?"
            );
            debug_assert_eq!(
                node.index, index,
                "must load a fresh dual module interface, found index out of order; did you forget to call solver.clear()?"
            );
            assert_eq!(
                node.index as usize,
                self.nodes.len(),
                "must load defect nodes in order, did you forget to call solver.clear()?"
            );
            // construct cluster and its parity matrix (will be reused over all iterations)
            let primal_cluster_ptr = PrimalClusterPtr::new_value(PrimalCluster {
                cluster_index: self.clusters.len() as NodeIndex,
                nodes: vec![],
                edges: node.invalid_subgraph.hair.clone(),
                vertices: node.invalid_subgraph.vertices.clone(),
                matrix: node.invalid_subgraph.generate_matrix(&interface.decoding_graph),
                subgraph: None,
                plugin_manager: PluginManager::new(self.plugins.clone(), self.plugin_count.clone()),
                relaxer_optimizer: RelaxerOptimizer::new(),
                #[cfg(all(feature = "incr_lp", feature = "highs"))]
                incr_solution: None,
            });
            // create the primal node of this defect node and insert into cluster
            let primal_node_ptr = PrimalModuleSerialNodePtr::new_value(PrimalModuleSerialNode {
                dual_node_ptr: dual_node_ptr.clone(),
                cluster_weak: primal_cluster_ptr.downgrade(),
            });
            primal_cluster_ptr.write().nodes.push(primal_node_ptr.clone());
            // add to self
            self.nodes.push(primal_node_ptr);
            self.clusters.push(primal_cluster_ptr);
        }
    }

    fn resolve(
        &mut self,
        dual_report: DualReport,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        let begin = Instant::now();
        let res = self.resolve_core(dual_report, interface_ptr, dual_module);
        self.time_resolve += begin.elapsed().as_secs_f64();
        res
    }

    fn old_resolve(
        &mut self,
        dual_report: DualReport,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        let begin = Instant::now();
        let res = self.old_resolve_core(dual_report, interface_ptr, dual_module);
        self.time_resolve += begin.elapsed().as_secs_f64();
        res
    }

    fn resolve_tune(
        &mut self,
        dual_report: FastIterSet<Obstacle>,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> (FastIterSet<Obstacle>, bool) {
        let begin = Instant::now();
        let res = self.resolve_core_tune(dual_report, interface_ptr, dual_module);
        self.time_resolve += begin.elapsed().as_secs_f64();
        res
    }

    fn subgraph(&mut self, _interface: &DualModuleInterfacePtr, _dual_module: &mut impl DualModuleImpl) -> OutputSubgraph {
        let mut subgraph = vec![];
        for cluster_ptr in self.clusters.iter() {
            let cluster = cluster_ptr.read_recursive();
            if cluster.nodes.is_empty() {
                continue;
            }
            subgraph.extend(
                cluster
                    .subgraph
                    .clone()
                    .unwrap_or_else(|| {
                        panic!(
                            "cluster {:?} is unsolvable: V_S = {:?}, E_S = {:?}",
                            cluster.cluster_index, cluster.vertices, cluster.edges
                        )
                    })
                    .iter(),
            );
        }
        OutputSubgraph::new(subgraph, _dual_module.get_negative_edges())
    }

    /// check if there are more plugins to be applied
    ///     will return false if timeout has been reached, else consume a plugin
    fn has_more_plugins(&mut self) -> bool {
        if self.time_resolve > self.config.timeout {
            return false;
        }
        return if *self.plugin_count.read_recursive() < self.plugins.len() {
            // increment the plugin count
            *self.plugin_count.write() += 1;
            self.plugin_pending_clusters = (0..self.clusters.len()).collect();
            true
        } else {
            false
        };
    }

    /// get the pending clusters
    fn pending_clusters(&mut self) -> Vec<usize> {
        self.plugin_pending_clusters.clone()
    }

    // TODO: extract duplicate codes

    /// analyze a cluster and return whether there exists an optimal solution (depending on optimization levels)
    #[allow(clippy::unnecessary_cast)]
    fn resolve_cluster(
        &mut self,
        cluster_index: NodeIndex,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        let cluster_ptr = self.clusters[cluster_index as usize].clone();
        let mut cluster = cluster_ptr.write();
        if cluster.nodes.is_empty() {
            return true; // no longer a cluster, no need to handle
        }
        // set all nodes to stop growing in the cluster
        for primal_node_ptr in cluster.nodes.iter() {
            let dual_node_ptr = primal_node_ptr.read_recursive().dual_node_ptr.clone();
            dual_module.set_grow_rate(&dual_node_ptr, Rational::zero());
        }
        // update the matrix with new tight edges
        let cluster = &mut *cluster;
        for &edge_index in cluster.edges.iter() {
            cluster
                .matrix
                .update_edge_tightness(edge_index, dual_module.is_edge_tight(edge_index));
        }

        // find an executable relaxer from the plugin manager
        let relaxer = {
            let positive_dual_variables: Vec<DualNodePtr> = cluster
                .nodes
                .iter()
                .map(|p| p.read_recursive().dual_node_ptr.clone())
                .filter(|dual_node_ptr| !dual_node_ptr.read_recursive().get_dual_variable().is_zero())
                .collect();
            let decoding_graph = &interface_ptr.read_recursive().decoding_graph;
            let cluster_mut = &mut *cluster; // must first get mutable reference
            let plugin_manager = &mut cluster_mut.plugin_manager;
            let matrix = &mut cluster_mut.matrix;
            plugin_manager.find_relaxer(decoding_graph, matrix, &positive_dual_variables)
        };

        // if a relaxer is found, execute it and return
        if let Some(relaxer) = relaxer {
            for (invalid_subgraph, grow_rate) in relaxer.get_direction().iter() {
                let (existing, dual_node_ptr) = interface_ptr.find_or_create_node(invalid_subgraph, dual_module);
                if !existing {
                    // create the corresponding primal node and add it to cluster
                    let primal_node_ptr = PrimalModuleSerialNodePtr::new_value(PrimalModuleSerialNode {
                        dual_node_ptr: dual_node_ptr.clone(),
                        cluster_weak: cluster_ptr.downgrade(),
                    });
                    cluster.nodes.push(primal_node_ptr.clone());
                    self.nodes.push(primal_node_ptr);
                }

                dual_module.set_grow_rate(&dual_node_ptr, grow_rate.clone());
            }
            cluster.relaxer_optimizer.insert(relaxer);
            return false;
        }

        // TODO idea: plugins can suggest subgraph (ideally, a global maximum), if so, then it will adopt th
        // subgraph with minimum weight from all plugins as the starting point to do local minimum

        // find a local minimum (hopefully a global minimum)
        let weight_of = |edge_index: EdgeIndex| dual_module.get_edge_weight(edge_index);
        cluster.subgraph = Some(cluster.matrix.get_solution_local_minimum(weight_of).expect("satisfiable"));
        true
    }

    /// analyze a cluster and return whether there exists an optimal solution (depending on optimization levels)
    #[allow(clippy::unnecessary_cast)]
    fn resolve_cluster_tune(
        &mut self,
        cluster_index: NodeIndex,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
        // dual_node_deltas: &mut FastIterMap<OrderedDualNodePtr, Rational>,
        dual_node_deltas: &mut FastIterMap<OrderedDualNodePtr, (Rational, NodeIndex)>,
    ) -> (bool, OptimizerResult) {
        let mut optimizer_result = OptimizerResult::default();
        #[cfg(feature = "incr_lp")]
        let mut cluster_ptr = self.clusters[cluster_index as usize].clone();
        #[cfg(not(feature = "incr_lp"))]
        let cluster_ptr = self.clusters[cluster_index as usize].clone();
        let mut cluster_temp = cluster_ptr.write();
        if cluster_temp.nodes.is_empty() {
            return (true, optimizer_result); // no longer a cluster, no need to handle
        }
        if cluster_temp.nodes.len() >= self.config.cluster_node_limit {
            return (true, optimizer_result);
        }
        // update the matrix with new tight edges
        #[cfg(feature = "incr_lp")]
        let mut cluster = &mut *cluster_temp;
        #[cfg(not(feature = "incr_lp"))]
        let cluster = &mut *cluster_temp;

        for &edge_index in cluster.edges.iter() {
            cluster
                .matrix
                .update_edge_tightness(edge_index, dual_module.is_edge_tight_tune(edge_index));
        }

        // find an executable relaxer from the plugin manager
        let relaxer = {
            let positive_dual_variables: Vec<DualNodePtr> = cluster
                .nodes
                .iter()
                .map(|p| p.read_recursive().dual_node_ptr.clone())
                .filter(|dual_node_ptr| !dual_node_ptr.read_recursive().dual_variable_at_last_updated_time.is_zero())
                .collect();
            let decoding_graph = &interface_ptr.read_recursive().decoding_graph;
            let cluster_mut = &mut *cluster; // must first get mutable reference
            let plugin_manager = &mut cluster_mut.plugin_manager;
            let matrix = &mut cluster_mut.matrix;
            plugin_manager.find_relaxer(decoding_graph, matrix, &positive_dual_variables)
        };

        // Yue added 2025.1.31: also check for local minimum during the algorithm; otherwise when we increase
        // the value of cluster_node_limit, the logical error rate may not decrease monotonically because
        // more complicated dual solution does not necessarily mean better logical error rate. Rather, if we
        // keep looking for smaller weighted solutions in the middle, the result is hopefully better.
        if !self.config.only_solve_primal_once {
            let weight_of = |edge_index: EdgeIndex| dual_module.get_edge_weight(edge_index);
            if let Some(subgraph) = cluster.matrix.get_solution_local_minimum(weight_of) {
                if let Some(original_subgraph) = &cluster.subgraph {
                    let original_weight = dual_module.get_subgraph_weight(original_subgraph);
                    let weight = dual_module.get_subgraph_weight(&subgraph);
                    if weight < original_weight {
                        cluster.subgraph = Some(subgraph);
                    }
                } else {
                    cluster.subgraph = Some(subgraph);
                }
            }
        }

        // if a relaxer is found, execute it and return
        if let Some(mut relaxer) = relaxer {
            #[cfg(feature = "float_lp")]
            // float_lp is enabled, optimizer really plays a role
            if cluster.relaxer_optimizer.should_optimize(&relaxer) {
                #[cfg(not(feature = "incr_lp"))]
                {
                    let dual_variables: FastIterMap<Arc<InvalidSubgraph>, Rational> = cluster
                        .nodes
                        .iter()
                        .map(|primal_node_ptr| {
                            let primal_node = primal_node_ptr.read_recursive();
                            let dual_node = primal_node.dual_node_ptr.read_recursive();
                            (
                                dual_node.invalid_subgraph.clone(),
                                dual_node.dual_variable_at_last_updated_time.clone(),
                            )
                        })
                        .collect();
                    let edge_slacks: FastIterMap<EdgeIndex, Rational> = dual_variables
                        .keys()
                        .flat_map(|invalid_subgraph: &Arc<InvalidSubgraph>| invalid_subgraph.hair.iter().cloned())
                        .chain(
                            relaxer
                                .get_direction()
                                .keys()
                                .flat_map(|invalid_subgraph| invalid_subgraph.hair.iter().cloned()),
                        )
                        .unique()
                        .map(|edge_index| (edge_index, dual_module.get_edge_slack_tune(edge_index)))
                        .collect();
                    let (new_relaxer, early_returned) =
                        cluster.relaxer_optimizer.optimize(relaxer, edge_slacks, dual_variables);
                    relaxer = new_relaxer;
                    if early_returned {
                        optimizer_result = OptimizerResult::EarlyReturned;
                    } else {
                        optimizer_result = OptimizerResult::Optimized;
                    }
                }

                #[cfg(feature = "incr_lp")]
                {
                    if !self.is_cluster_weight_initialized() {
                        drop(cluster_temp);
                        drop(cluster_ptr);
                        self.calculate_edges_free_weight_clusters(dual_module);
                        cluster_ptr = self.clusters[cluster_index as usize].clone();
                        cluster_temp = cluster_ptr.write();
                        cluster = &mut *cluster_temp;
                    }
                    let mut dual_variables: FastIterMap<NodeIndex, (Arc<InvalidSubgraph>, Rational)> = FastIterMap::new();
                    let mut participating_dual_variable_indices = hashbrown::HashSet::new();
                    for primal_node_ptr in cluster.nodes.iter() {
                        let primal_node = primal_node_ptr.read_recursive();
                        let dual_node = primal_node.dual_node_ptr.read_recursive();
                        dual_variables.insert(
                            dual_node.index,
                            (
                                dual_node.invalid_subgraph.clone(),
                                dual_node.dual_variable_at_last_updated_time.clone(),
                            ),
                        );
                        participating_dual_variable_indices.insert(dual_node.index);
                    }

                    for (invalid_subgraph, _) in relaxer.get_direction().iter() {
                        if let Some((existing, dual_node_ptr)) =
                            interface_ptr.find_or_create_node_tune(invalid_subgraph, dual_module)
                        {
                            if !existing {
                                // create the corresponding primal node and add it to cluster
                                let primal_node_ptr = PrimalModuleSerialNodePtr::new_value(PrimalModuleSerialNode {
                                    dual_node_ptr: dual_node_ptr.clone(),
                                    cluster_weak: cluster_ptr.downgrade(),
                                });
                                cluster.nodes.push(primal_node_ptr.clone());
                                self.nodes.push(primal_node_ptr);
                                // participating_dual_variable_indices.insert(dual_node_ptr.read_recursive().index);

                                // maybe optimize here
                            }
                            match dual_variables.get_mut(&dual_node_ptr.read_recursive().index) {
                                Some(_) => {}
                                None => {
                                    dual_variables.insert(
                                        dual_node_ptr.read_recursive().index,
                                        (
                                            dual_node_ptr.read_recursive().invalid_subgraph.clone(),
                                            dual_node_ptr.read_recursive().dual_variable_at_last_updated_time.clone(),
                                        ),
                                    );
                                }
                            };
                        }
                    }
                    let edge_free_weights: FastIterMap<EdgeIndex, Rational> = dual_variables
                        .values()
                        .flat_map(|(invalid_subgraph, _)| invalid_subgraph.hair.iter().cloned())
                        .chain(
                            relaxer
                                .get_direction()
                                .keys()
                                .flat_map(|invalid_subgraph| invalid_subgraph.hair.iter().cloned()),
                        )
                        .unique()
                        .map(|edge_index| {
                            (
                                edge_index,
                                // dual_module.get_edge_free_weight(edge_index, &participating_dual_variable_indices),
                                dual_module.get_edge_free_weight_cluster(edge_index, cluster_index),
                            )
                        })
                        .collect();

                    let (new_relaxer, early_returned) = cluster.relaxer_optimizer.optimize_incr(
                        relaxer,
                        edge_free_weights,
                        dual_variables,
                        &mut cluster.incr_solution,
                    );
                    relaxer = new_relaxer;
                    if early_returned {
                        optimizer_result = OptimizerResult::EarlyReturned;
                    } else {
                        optimizer_result = OptimizerResult::Optimized;
                    }
                }
            } else {
                optimizer_result = OptimizerResult::Skipped;
            }

            #[cfg(not(feature = "float_lp"))]
            // with rationals, it is actually usually better when always optimized
            {
                let dual_variables: FastIterMap<Arc<InvalidSubgraph>, Rational> = cluster
                    .nodes
                    .iter()
                    .map(|primal_node_ptr| {
                        let primal_node = primal_node_ptr.read_recursive();
                        let dual_node = primal_node.dual_node_ptr.read_recursive();
                        (
                            dual_node.invalid_subgraph.clone(),
                            dual_node.dual_variable_at_last_updated_time.clone(),
                        )
                    })
                    .collect();
                let edge_slacks: FastIterMap<EdgeIndex, Rational> = dual_variables
                    .keys()
                    .flat_map(|invalid_subgraph: &Arc<InvalidSubgraph>| invalid_subgraph.hair.iter().cloned())
                    .chain(
                        relaxer
                            .get_direction()
                            .keys()
                            .flat_map(|invalid_subgraph| invalid_subgraph.hair.iter().cloned()),
                    )
                    .unique()
                    .map(|edge_index| (edge_index, dual_module.get_edge_slack_tune(edge_index)))
                    .collect();

                let (new_relaxer, early_returned) = cluster.relaxer_optimizer.optimize(relaxer, edge_slacks, dual_variables);
                relaxer = new_relaxer;
                if early_returned {
                    optimizer_result = OptimizerResult::EarlyReturned;
                } else {
                    optimizer_result = OptimizerResult::Optimized;
                }
            }

            for (invalid_subgraph, grow_rate) in relaxer.get_direction().iter() {
                if let Some((existing, dual_node_ptr)) =
                    interface_ptr.find_or_create_node_tune(invalid_subgraph, dual_module)
                {
                    if !existing {
                        // create the corresponding primal node and add it to cluster
                        let primal_node_ptr = PrimalModuleSerialNodePtr::new_value(PrimalModuleSerialNode {
                            dual_node_ptr: dual_node_ptr.clone(),
                            cluster_weak: cluster_ptr.downgrade(),
                        });
                        cluster.nodes.push(primal_node_ptr.clone());
                        self.nodes.push(primal_node_ptr);
                    }

                    // Document the desired deltas
                    let index = dual_node_ptr.read_recursive().index;
                    dual_node_deltas.insert(
                        OrderedDualNodePtr::new(index, dual_node_ptr),
                        (grow_rate.clone(), cluster_index),
                    );
                }
            }

            cluster.relaxer_optimizer.insert(relaxer);
            return (false, optimizer_result);
        }

        // find a local minimum (hopefully a global minimum)
        if self.config.only_solve_primal_once {
            let weight_of = |edge_index: EdgeIndex| dual_module.get_edge_weight(edge_index);
            cluster.subgraph = Some(cluster.matrix.get_solution_local_minimum(weight_of).expect("satisfiable"));
        }

        (true, optimizer_result)
    }

    /// update the sorted clusters_aff, should be None to start with
    fn update_sorted_clusters_aff<D: DualModuleImpl>(&mut self, dual_module: &mut D) {
        let pending_clusters = self.pending_clusters();
        let mut sorted_clusters_aff = FastIterSet::default();

        for cluster_index in pending_clusters.iter() {
            let cluster_ptr = self.clusters[*cluster_index].clone();
            let affinity = dual_module.calculate_cluster_affinity(cluster_ptr);
            if let Some(affinity) = affinity {
                sorted_clusters_aff.insert(ClusterAffinity {
                    cluster_index: *cluster_index,
                    affinity,
                });
            }
        }
        self.sorted_clusters_aff = Some(sorted_clusters_aff);
    }

    /// consume the sorted_clusters_aff
    fn get_sorted_clusters_aff(&mut self) -> FastIterSet<ClusterAffinity> {
        self.sorted_clusters_aff.take().unwrap()
    }

    #[cfg(feature = "incr_lp")]
    fn calculate_edges_free_weight_clusters(&mut self, dual_module: &mut impl DualModuleImpl) {
        for cluster in self.clusters.iter() {
            let cluster = cluster.read_recursive();
            for node in cluster.nodes.iter() {
                let dual_node = node.read_recursive();
                let dual_node_read = dual_node.dual_node_ptr.read_recursive();
                for edge_index in dual_node_read.invalid_subgraph.hair.iter() {
                    dual_module.update_edge_cluster_weights(
                        *edge_index,
                        cluster.cluster_index,
                        dual_node_read.dual_variable_at_last_updated_time.clone(),
                    );
                }
            }
        }
        self.cluster_weights_initialized = true;
    }

    #[cfg(feature = "incr_lp")]
    fn uninit_cluster_weight(&mut self) {
        self.cluster_weights_initialized = false;
    }

    #[cfg(feature = "incr_lp")]
    fn is_cluster_weight_initialized(&self) -> bool {
        self.cluster_weights_initialized
    }
}

impl PrimalModuleSerial {
    // union the cluster of two dual nodes
    #[allow(clippy::unnecessary_cast)]
    pub fn union(
        &self,
        dual_node_ptr_1: &DualNodePtr,
        dual_node_ptr_2: &DualNodePtr,
        decoding_graph: &DecodingHyperGraph,
        _dual_module: &mut impl DualModuleImpl, // note: remove if not for cluster-based
    ) {
        // cluster_1 will become the union of cluster_1 and cluster_2
        // and cluster_2 will be outdated
        let node_index_1 = dual_node_ptr_1.read_recursive().index;
        let node_index_2 = dual_node_ptr_2.read_recursive().index;
        if node_index_1 == node_index_2 {
            return; // already the same node
        }
        let primal_node_1 = self.nodes[node_index_1 as usize].read_recursive();
        let primal_node_2 = self.nodes[node_index_2 as usize].read_recursive();
        if primal_node_1.cluster_weak.ptr_eq(&primal_node_2.cluster_weak) {
            return; // already in the same cluster
        }
        let cluster_ptr_1 = primal_node_1.cluster_weak.upgrade_force();
        let cluster_ptr_2 = primal_node_2.cluster_weak.upgrade_force();
        drop(primal_node_1);
        drop(primal_node_2);
        let mut cluster_1 = cluster_ptr_1.write();
        let mut cluster_2 = cluster_ptr_2.write();

        #[cfg(feature = "incr_lp")]
        if self.is_cluster_weight_initialized() {
            let cluster_2_index = cluster_2.cluster_index;
            for primal_node_ptr in cluster_2.nodes.drain(..) {
                {
                    let primal_node = primal_node_ptr.read_recursive();
                    _dual_module.update_edge_cluster_weights_union(
                        &primal_node.dual_node_ptr,
                        cluster_2_index,
                        cluster_1.cluster_index,
                    );
                }

                primal_node_ptr.write().cluster_weak = cluster_ptr_1.downgrade();
                cluster_1.nodes.push(primal_node_ptr);
            }
            cluster_1.edges.append(&mut cluster_2.edges);
            // in the case that union fails later on, the previous solutions should be preserved
            match (cluster_1.subgraph.take(), cluster_2.subgraph.take()) {
                (Some(mut c1), Some(mut c2)) => {
                    c1.append(&mut c2);
                    cluster_1.subgraph = Some(c1);
                }
                (None, Some(c2)) => {
                    cluster_1.subgraph = Some(c2);
                }
                _ => {}
            }
            // cluster_1.subgraph = None; // mark as no subgraph

            match (&cluster_1.incr_solution, &cluster_2.incr_solution) {
                (None, Some(_)) => {
                    cluster_1.incr_solution = cluster_2.incr_solution.take();
                }
                (Some(c1), Some(c2)) => {
                    if c2.lock().constraints_len() > c1.lock().constraints_len() {
                        cluster_1.incr_solution = cluster_2.incr_solution.take();
                    }
                }

                // no need to changes
                (None, None) => {}
                (Some(_), None) => {}
            }
        } else {
            for primal_node_ptr in cluster_2.nodes.drain(..) {
                primal_node_ptr.write().cluster_weak = cluster_ptr_1.downgrade();
                cluster_1.nodes.push(primal_node_ptr);
            }
            cluster_1.edges.append(&mut cluster_2.edges);
            // in the case that union fails later on, the previous solutions should be preserved
            match (cluster_1.subgraph.take(), cluster_2.subgraph.take()) {
                (Some(mut c1), Some(mut c2)) => {
                    c1.append(&mut c2);
                    cluster_1.subgraph = Some(c1);
                }
                (None, Some(c2)) => {
                    cluster_1.subgraph = Some(c2);
                }
                _ => {}
            }
            // cluster_1.subgraph = None; // mark as no subgraph

            match (&cluster_1.incr_solution, &cluster_2.incr_solution) {
                (None, Some(_)) => {
                    cluster_1.incr_solution = cluster_2.incr_solution.take();
                }
                (Some(c1), Some(c2)) => {
                    if c2.lock().constraints_len() > c1.lock().constraints_len() {
                        cluster_1.incr_solution = cluster_2.incr_solution.take();
                    }
                }

                // no need to changes
                (None, None) => {}
                (Some(_), None) => {}
            }
        }

        #[cfg(not(feature = "incr_lp"))]
        {
            for primal_node_ptr in cluster_2.nodes.drain(..) {
                primal_node_ptr.write().cluster_weak = cluster_ptr_1.downgrade();
                cluster_1.nodes.push(primal_node_ptr);
            }
            cluster_1.edges.append(&mut cluster_2.edges);

            // in the case that union fails later on, the previous solutions should be preserved
            match (cluster_1.subgraph.take(), cluster_2.subgraph.take()) {
                (Some(mut c1), Some(mut c2)) => {
                    c1.append(&mut c2);
                    cluster_1.subgraph = Some(c1);
                }
                (None, Some(c2)) => {
                    cluster_1.subgraph = Some(c2);
                }
                _ => {}
            }
            // cluster_1.subgraph = None; // mark as no subgraph
        }

        for &vertex_index in cluster_2.vertices.iter() {
            if !cluster_1.vertices.contains(&vertex_index) {
                cluster_1.vertices.insert(vertex_index);
                let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
                let parity = decoding_graph.is_vertex_defect(vertex_index);
                cluster_1.matrix.add_constraint(vertex_index, incident_edges, parity);
            }
        }
        cluster_1.relaxer_optimizer.append(&mut cluster_2.relaxer_optimizer);
        cluster_2.vertices.clear();
    }

    #[allow(clippy::unnecessary_cast)]
    fn resolve_core(
        &mut self,
        mut dual_report: DualReport,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        debug_assert!(!dual_report.is_unbounded() && dual_report.get_valid_growth().is_none());
        let mut active_clusters = FastIterSet::<NodeIndex>::new();
        let interface = interface_ptr.read_recursive();
        let decoding_graph = &interface.decoding_graph;
        while let Some(obstacle) = dual_report.pop() {
            match obstacle {
                Obstacle::Conflict { edge_index } => {
                    // union all the dual nodes in the edge index and create new dual node by adding this edge to `internal_edges`
                    let dual_nodes = dual_module.get_edge_nodes(edge_index);
                    debug_assert!(
                        !dual_nodes.is_empty(),
                        "should not conflict if no dual nodes are contributing"
                    );
                    let dual_node_ptr_0 = &dual_nodes[0];
                    // first union all the dual nodes
                    for dual_node_ptr in dual_nodes.iter().skip(1) {
                        // self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph);
                        self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph, dual_module);
                    }
                    let cluster_ptr = self.nodes[dual_node_ptr_0.read_recursive().index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let mut cluster = cluster_ptr.write();
                    // then add new constraints because these edges may touch new vertices
                    let incident_vertices = decoding_graph.get_edge_neighbors(edge_index);
                    for &vertex_index in incident_vertices.iter() {
                        if !cluster.vertices.contains(&vertex_index) {
                            cluster.vertices.insert(vertex_index);
                            let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
                            let parity = decoding_graph.is_vertex_defect(vertex_index);
                            cluster.matrix.add_constraint(vertex_index, incident_edges, parity);
                        }
                    }
                    cluster.edges.insert(edge_index);
                    // add to active cluster so that it's processed later
                    active_clusters.insert(cluster.cluster_index);
                }
                Obstacle::ShrinkToZero { dual_node_ptr } => {
                    let cluster_ptr = self.nodes[dual_node_ptr.index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let cluster_index = cluster_ptr.read_recursive().cluster_index;
                    active_clusters.insert(cluster_index);
                }
            }
        }
        drop(interface);
        if *self.plugin_count.read_recursive() != 0 && self.time_resolve > self.config.timeout {
            *self.plugin_count.write() = 0; // force only the first plugin
        }
        let mut all_solved = true;
        for &cluster_index in active_clusters.iter() {
            let solved = self.resolve_cluster(cluster_index, interface_ptr, dual_module);
            all_solved &= solved;
        }
        if !all_solved {
            return false; // already give dual module something to do
        }

        true
    }

    #[allow(clippy::unnecessary_cast)]
    /// for backwards-compatibility
    fn old_resolve_core(
        &mut self,
        mut dual_report: DualReport,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        debug_assert!(!dual_report.is_unbounded() && dual_report.get_valid_growth().is_none());
        let mut active_clusters = FastIterSet::<NodeIndex>::new();
        let interface = interface_ptr.read_recursive();
        let decoding_graph = &interface.decoding_graph;
        while let Some(obstacle) = dual_report.pop() {
            match obstacle {
                Obstacle::Conflict { edge_index } => {
                    // union all the dual nodes in the edge index and create new dual node by adding this edge to `internal_edges`
                    let dual_nodes = dual_module.get_edge_nodes(edge_index);
                    debug_assert!(
                        !dual_nodes.is_empty(),
                        "should not conflict if no dual nodes are contributing"
                    );
                    let dual_node_ptr_0 = &dual_nodes[0];
                    // first union all the dual nodes
                    for dual_node_ptr in dual_nodes.iter().skip(1) {
                        // self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph);
                        self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph, dual_module);
                    }
                    let cluster_ptr = self.nodes[dual_node_ptr_0.read_recursive().index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let mut cluster = cluster_ptr.write();
                    // then add new constraints because these edges may touch new vertices
                    let incident_vertices = decoding_graph.get_edge_neighbors(edge_index);
                    for &vertex_index in incident_vertices.iter() {
                        if !cluster.vertices.contains(&vertex_index) {
                            cluster.vertices.insert(vertex_index);
                            let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
                            let parity = decoding_graph.is_vertex_defect(vertex_index);
                            cluster.matrix.add_constraint(vertex_index, incident_edges, parity);
                        }
                    }
                    cluster.edges.insert(edge_index);
                    // add to active cluster so that it's processed later
                    active_clusters.insert(cluster.cluster_index);
                }
                Obstacle::ShrinkToZero { dual_node_ptr } => {
                    let cluster_ptr = self.nodes[dual_node_ptr.index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let cluster_index = cluster_ptr.read_recursive().cluster_index;
                    active_clusters.insert(cluster_index);
                }
            }
        }
        drop(interface);
        if *self.plugin_count.read_recursive() != 0 && self.time_resolve > self.config.timeout {
            *self.plugin_count.write() = 0; // force only the first plugin
        }
        let mut all_solved = true;
        for &cluster_index in active_clusters.iter() {
            let solved = self.resolve_cluster(cluster_index, interface_ptr, dual_module);
            all_solved &= solved;
        }
        if !all_solved {
            return false; // already give dual module something to do
        }
        while !self.pending_nodes.is_empty() {
            let primal_node_weak = self.pending_nodes.pop_front().unwrap();
            let primal_node_ptr = primal_node_weak.upgrade_force();
            let primal_node = primal_node_ptr.read_recursive();
            let cluster_ptr = primal_node.cluster_weak.upgrade_force();
            if cluster_ptr.read_recursive().subgraph.is_none() {
                dual_module.set_grow_rate(&primal_node.dual_node_ptr, Rational::one());
                return false; // let the dual module to find more obstacles
            }
        }
        if *self.plugin_count.read_recursive() == 0 {
            return true;
        }
        // check that all clusters have passed the plugins
        loop {
            while let Some(cluster_index) = self.plugin_pending_clusters.pop() {
                let solved = self.resolve_cluster(cluster_index, interface_ptr, dual_module);
                if !solved {
                    return false; // let the dual module to handle one
                }
            }
            if *self.plugin_count.read_recursive() < self.plugins.len() {
                // increment the plugin count
                *self.plugin_count.write() += 1;
                self.plugin_pending_clusters = (0..self.clusters.len()).collect();
            } else {
                break; // nothing more to check
            }
        }
        true
    }

    #[allow(clippy::unnecessary_cast)]
    // returns (obstacles_needing_to_be_resolved, should_grow)
    fn resolve_core_tune(
        &mut self,
        dual_report: FastIterSet<Obstacle>,
        interface_ptr: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> (FastIterSet<Obstacle>, bool) {
        let mut active_clusters = FastIterSet::<NodeIndex>::new();
        let interface = interface_ptr.read_recursive();
        let decoding_graph = &interface.decoding_graph;

        for obstacle in dual_report.into_iter() {
            match obstacle {
                Obstacle::Conflict { edge_index } => {
                    // union all the dual nodes in the edge index and create new dual node by adding this edge to `internal_edges`
                    let dual_nodes = dual_module.get_edge_nodes(edge_index);
                    debug_assert!(
                        !dual_nodes.is_empty(),
                        "should not conflict if no dual nodes are contributing"
                    );

                    let dual_node_ptr_0 = &dual_nodes[0];
                    // first union all the dual nodes
                    for dual_node_ptr in dual_nodes.iter().skip(1) {
                        // self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph);
                        self.union(dual_node_ptr_0, dual_node_ptr, &interface.decoding_graph, dual_module);
                    }
                    let cluster_ptr = self.nodes[dual_node_ptr_0.read_recursive().index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let mut cluster = cluster_ptr.write();
                    // then add new constraints because these edges may touch new vertices
                    let incident_vertices = decoding_graph.get_edge_neighbors(edge_index);
                    for &vertex_index in incident_vertices.iter() {
                        if !cluster.vertices.contains(&vertex_index) {
                            cluster.vertices.insert(vertex_index);
                            let incident_edges = decoding_graph.get_vertex_neighbors(vertex_index);
                            let parity = decoding_graph.is_vertex_defect(vertex_index);
                            cluster.matrix.add_constraint(vertex_index, incident_edges, parity);
                        }
                    }
                    cluster.edges.insert(edge_index);
                    // add to active cluster so that it's processed later
                    active_clusters.insert(cluster.cluster_index);
                }
                Obstacle::ShrinkToZero { dual_node_ptr } => {
                    let cluster_ptr = self.nodes[dual_node_ptr.index as usize]
                        .read_recursive()
                        .cluster_weak
                        .upgrade_force();
                    let cluster_index = cluster_ptr.read_recursive().cluster_index;
                    active_clusters.insert(cluster_index);
                }
            }
        }

        drop(interface);
        if *self.plugin_count.read_recursive() != 0 && self.time_resolve > self.config.timeout {
            *self.plugin_count.write() = 0; // force only the first plugin
        }
        let mut all_solved = true;
        let mut dual_node_deltas = FastIterMap::new();
        let mut optimizer_result = OptimizerResult::default();
        for &cluster_index in active_clusters.iter() {
            let (solved, other) =
                self.resolve_cluster_tune(cluster_index, interface_ptr, dual_module, &mut dual_node_deltas);
            if !solved {
                // todo: investigate more
                return (dual_module.get_obstacles_tune(other, dual_node_deltas), false);
            }
            all_solved &= solved;
            optimizer_result.or(other);
        }

        let all_obstacles = dual_module.get_obstacles_tune(optimizer_result, dual_node_deltas);

        (all_obstacles, all_solved)
    }

    pub fn print_clusters(&self) {
        let mut vertices = FastIterSet::new();
        let mut edges = FastIterSet::new();
        let mut invalid_subgraphs: FastIterSet<Arc<InvalidSubgraph>> = FastIterSet::new();
        for cluster in self.clusters.iter() {
            let cluster = cluster.read_recursive();
            if cluster.nodes.is_empty() {
                continue;
            }
            println!("cluster: {}", cluster.cluster_index);
            println!(
                "nodes: {:?}",
                cluster
                    .nodes
                    .iter()
                    .map(|node| node.read_recursive().dual_node_ptr.read_recursive().index)
                    .collect::<Vec<_>>()
            );
            println!("vertices: {:?}", cluster.vertices);
            if !vertices.is_disjoint(&cluster.vertices) {
                println!("vertices overlap");
                // print the overlapping vertices
                println!("overlap: {:?}", vertices.intersection(&cluster.vertices).collect::<Vec<_>>());
            }
            vertices.extend(cluster.vertices.iter());
            // print edge overlaps
            if !edges.is_disjoint(&cluster.edges) {
                println!("edges overlap");
                println!("overlap: {:?}", edges.intersection(&cluster.edges).collect::<Vec<_>>());
            }

            // print the node and the invalid subgraph overlaps
            for node in cluster.nodes.iter() {
                let node = node.read_recursive();
                if invalid_subgraphs.contains(&node.dual_node_ptr.read_recursive().invalid_subgraph) {
                    println!("invalid subgraph overlap");
                    println!("overlap: {:?}", node.dual_node_ptr.read_recursive().invalid_subgraph);
                }
                invalid_subgraphs.insert(node.dual_node_ptr.read_recursive().invalid_subgraph.clone());
            }

            edges.extend(cluster.edges.iter());
            println!();
            println!("invalid subgraphs: {:?}", invalid_subgraphs.len());
        }
    }
}

impl MWPSVisualizer for PrimalModuleSerial {
    fn snapshot(&self, _abbrev: bool) -> serde_json::Value {
        json!({})
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::dual_module_pq::*;
    use super::super::example_codes::*;
    use super::*;
    use crate::plugin_single_hair::PluginSingleHair;
    use crate::plugin_union_find::PluginUnionFind;

    #[allow(clippy::too_many_arguments)]
    pub fn primal_module_serial_basic_standard_syndrome_optional_viz(
        _code: impl ExampleCode,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
        mut dual_module: impl DualModuleImpl + MWPSVisualizer,
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
        // primal_module.config = serde_json::from_value(json!({"timeout":1})).unwrap();
        // try to work on a simple syndrome
        let decoding_graph = DecodingHyperGraph::new_defects(model_graph, defect_vertices.clone());
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        primal_module.solve_visualizer(
            &interface_ptr,
            decoding_graph.syndrome_pattern.clone(),
            &mut dual_module,
            visualizer.as_mut(),
        );

        let (subgraph, weight_range) = primal_module.subgraph_range(&interface_ptr, &mut dual_module);
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
        assert!(
            rational_approx_eq(&final_dual, &weight_range.upper),
            "unmatched sum dual variables"
        );
        assert!(
            rational_approx_eq(&final_dual, &weight_range.lower),
            "unexpected final dual variable sum"
        );
        (interface_ptr, primal_module, dual_module)
    }

    pub fn primal_module_serial_basic_standard_syndrome(
        code: impl ExampleCode,
        visualize_filename: String,
        defect_vertices: Vec<VertexIndex>,
        final_dual: Weight,
        plugins: PluginVec,
    ) -> (
        DualModuleInterfacePtr,
        PrimalModuleSerial,
        impl DualModuleImpl + MWPSVisualizer,
    ) {
        println!("{defect_vertices:?}");
        let visualizer = Visualizer::new(
            Some(visualize_data_folder() + visualize_filename.as_str()),
            code.get_positions(),
            true,
        )
        .unwrap();
        // create dual module
        let model_graph = code.get_model_graph();
        primal_module_serial_basic_standard_syndrome_optional_viz(
            code,
            defect_vertices,
            final_dual,
            plugins,
            DualModulePQ::new_empty(&model_graph.initializer),
            model_graph,
            Some(visualizer),
        )
    }

    /// test a simple case
    #[test]
    fn primal_module_serial_basic_1() {
        // cargo test primal_module_serial_basic_1 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_1.json".to_string();
        let defect_vertices = vec![23, 24, 29, 30];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(4.59511985013459).unwrap(),
            vec![],
        );
    }

    #[test]
    fn primal_module_serial_basic_2() {
        // cargo test primal_module_serial_basic_2 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_2.json".to_string();
        let defect_vertices = vec![16, 17, 23, 25, 29, 30];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(9.19023970026918).unwrap(),
            vec![],
        );
    }

    #[test]
    fn primal_module_serial_basic_3() {
        // cargo test primal_module_serial_basic_3 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_3.json".to_string();
        let defect_vertices = vec![14, 15, 16, 17, 22, 25, 28, 31, 36, 37, 38, 39];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(22.97559925067295).unwrap(),
            vec![],
        );
    }

    #[test]
    fn primal_module_serial_basic_3_improved() {
        // cargo test primal_module_serial_basic_3_improved -- --nocapture
        let visualize_filename = "primal_module_serial_basic_3_improved.json".to_string();
        let defect_vertices = vec![14, 15, 16, 17, 22, 25, 28, 31, 36, 37, 38, 39];
        let code = CodeCapacityTailoredCode::new(7, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(22.97559925067295).unwrap(),
            vec![
                PluginUnionFind::entry(),
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once),
            ],
        );
    }

    /// this is a case where the union find version will deterministically fail to decode,
    /// because not all edges are fully grown and those fully grown edges lead to suboptimal result
    #[test]
    #[should_panic]
    fn primal_module_serial_basic_4() {
        // cargo test primal_module_serial_basic_4 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_4.json".to_string();
        let defect_vertices = vec![10, 11, 12, 15, 16, 17, 18];
        let code = CodeCapacityTailoredCode::new(5, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(4.).unwrap(), // fixme: ???
            vec![],
        );
    }

    #[test]
    fn primal_module_serial_basic_4_single_improved() {
        // cargo test primal_module_serial_basic_4_single_improved -- --nocapture
        let visualize_filename = "primal_module_serial_basic_4_single_improved.json".to_string();
        let defect_vertices = vec![10, 11, 12, 15, 16, 17, 18];
        let code = CodeCapacityTailoredCode::new(5, 0., 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(18.38047940053836).unwrap(),
            vec![
                PluginUnionFind::entry(),
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once),
            ],
        );
    }

    #[test]
    fn primal_module_serial_basic_5() {
        // cargo test primal_module_serial_basic_5 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_5.json".to_string();
        let defect_vertices = vec![32, 33, 37, 47, 86, 87, 72, 82];
        let code = CodeCapacityPlanarCode::new(11, 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(18.38047940053836).unwrap(),
            vec![],
        );
    }

    #[test]
    fn primal_module_serial_basic_6() {
        // cargo test primal_module_serial_basic_6 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_6.json".to_string();
        let defect_vertices = vec![12, 22, 23, 32, 17, 26, 27, 37, 62, 72, 73, 82, 67, 76, 77, 87];
        let code = CodeCapacityPlanarCode::new(11, 0.01);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(55.14143820161507).unwrap(),
            vec![
                PluginUnionFind::entry(),
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once),
            ],
        );
    }

    #[test]
    fn primal_module_serial_basic_7() {
        // cargo test primal_module_serial_basic_7 -- --nocapture
        let visualize_filename = "primal_module_serial_basic_7.json".to_string();
        let defect_vertices = vec![1, 2, 4, 5];
        let code = CodeCapacityTailoredCode::new(3, 0., 0.1);
        primal_module_serial_basic_standard_syndrome(
            code,
            visualize_filename,
            defect_vertices,
            Rational::from_float(6.591673732008658).unwrap(),
            vec![
                PluginUnionFind::entry(),
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Once),
                PluginSingleHair::entry_with_strategy(RepeatStrategy::Multiple {
                    max_repetition: usize::MAX,
                }),
            ],
        );
    }
}
