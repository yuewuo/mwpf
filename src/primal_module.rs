//! Primal Module
//!
//! Generics for primal modules, defining the necessary interfaces for a primal module
//!
#![cfg_attr(feature = "unsafe_pointer", allow(dropping_references))]

use std::collections::VecDeque;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::dual_module::*;
use crate::num_traits::Zero;
use crate::ordered_float::OrderedFloat;
use crate::pointers::*;
use crate::primal_module_serial::{ClusterAffinity, PrimalClusterPtr, PrimalClusterWeak};
use crate::relaxer_optimizer::OptimizerResult;
use crate::util::*;
use crate::visualize::*;

pub type Affinity = OrderedFloat;

const MAX_HISTORY: usize = 10;

/// common trait that must be implemented for each implementation of primal module
pub trait PrimalModuleImpl {
    /// create a primal module given the dual module
    fn new_empty(solver_initializer: &SolverInitializer) -> Self;

    /// clear all states; however this method is not necessarily called when load a new decoding problem, so you need to call it yourself
    fn clear(&mut self);

    /// load a new decoding problem given dual interface: note that all nodes MUST be defect node
    fn load<D: DualModuleImpl>(&mut self, interface_ptr: &DualModuleInterfacePtr, dual_module: &mut D);

    /// analyze the reason why dual module cannot further grow, update primal data structure (alternating tree, temporary matches, etc)
    /// and then tell dual module what to do to resolve these conflicts;
    /// note that this function doesn't necessarily resolve all the conflicts, but can return early if some major change is made.
    /// when implementing this function, it's recommended that you resolve as many conflicts as possible.
    ///
    /// note: this is only ran in the "search" mode
    fn resolve(
        &mut self,
        dual_report: DualReport,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> bool;

    /// kept in case of future need for this deprecated function (backwards compatibility for cases such as `SingleCluster` growing strategy)
    fn old_resolve(
        &mut self,
        _group_max_update_length: DualReport,
        _interface: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        false
    }

    /// resolve the conflicts in the "tune" mode
    fn resolve_tune(
        &mut self,
        _obstacles: BTreeSet<Obstacle>,
        _interface: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> (BTreeSet<Obstacle>, bool) {
        panic!("`resolve_tune` not implemented, this primal module does not work with tuning mode");
    }

    fn solve(
        &mut self,
        interface: &DualModuleInterfacePtr,
        syndrome_pattern: Arc<SyndromePattern>,
        dual_module: &mut impl DualModuleImpl,
    ) {
        interface.load(syndrome_pattern, dual_module);
        self.load(interface, dual_module);
        self.solve_step_callback_interface_loaded(interface, dual_module, |_, _, _, _| {})
    }

    fn visualizer_callback<D: DualModuleImpl + MWPSVisualizer>(
        visualizer: &mut Visualizer,
    ) -> impl FnMut(&DualModuleInterfacePtr, &mut D, &mut Self, &DualReport)
    where
        Self: MWPSVisualizer + Sized,
    {
        |interface: &DualModuleInterfacePtr, dual_module: &mut D, primal_module: &mut Self, dual_report: &DualReport| {
            if cfg!(debug_assertions) {
                // println!("dual_report: {:?}", dual_report);
                // dual_module.debug_print();
            }
            if dual_report.is_unbounded() {
                visualizer
                    .snapshot_combined("unbounded grow".to_string(), vec![interface, dual_module, primal_module])
                    .unwrap();
            } else if let Some(length) = dual_report.get_valid_growth() {
                visualizer
                    .snapshot_combined(format!("grow {length}"), vec![interface, dual_module, primal_module])
                    .unwrap();
            } else {
                let first_conflict = format!("{:?}", dual_report.peek().unwrap());
                visualizer
                    .snapshot_combined(
                        format!("resolve {first_conflict}"),
                        vec![interface, dual_module, primal_module],
                    )
                    .unwrap();
            };
        }
    }

    fn solve_visualizer<D: DualModuleImpl + MWPSVisualizer>(
        &mut self,
        interface: &DualModuleInterfacePtr,
        syndrome_pattern: Arc<SyndromePattern>,
        dual_module: &mut D,
        visualizer: Option<&mut Visualizer>,
    ) where
        Self: MWPSVisualizer + Sized,
    {
        if let Some(visualizer) = visualizer {
            let callback = Self::visualizer_callback(visualizer);
            interface.load(syndrome_pattern, dual_module);
            self.load(interface, dual_module);
            self.solve_step_callback_interface_loaded(interface, dual_module, callback);
            visualizer
                .snapshot_combined("solved".to_string(), vec![interface, dual_module, self])
                .unwrap();
        } else {
            self.solve(interface, syndrome_pattern, dual_module);
        }
    }

    fn solve_step_callback_interface_loaded<D: DualModuleImpl, F>(
        &mut self,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut D,
        mut callback: F,
    ) where
        F: FnMut(&DualModuleInterfacePtr, &mut D, &mut Self, &DualReport),
    {
        // println!("solve_step_callback_interface_loaded");
        // Search, this part is unchanged
        let mut dual_report = dual_module.report();

        while !dual_report.is_unbounded() {
            callback(interface, dual_module, self, &dual_report);
            match dual_report.get_valid_growth() {
                Some(length) => dual_module.grow(length),
                None => {
                    self.resolve(dual_report, interface, dual_module);
                }
            }
            dual_report = dual_module.report();
        }

        // from here, all states should be syncronized
        let mut start = true;

        // starting with unbounded state here: All edges and nodes are not growing as of now
        // Tune
        while self.has_more_plugins() {
            if start {
                start = false;
                dual_module.advance_mode();
            }
            self.update_sorted_clusters_aff(dual_module);
            let cluster_affs = self.get_sorted_clusters_aff();

            for cluster_affinity in cluster_affs.into_iter() {
                let cluster_ptr = cluster_affinity.cluster_ptr;
                let mut dual_node_deltas = BTreeMap::new();
                let (mut resolved, optimizer_result) =
                    self.resolve_cluster_tune(&cluster_ptr, interface, dual_module, &mut dual_node_deltas);

                let mut obstacles = dual_module.get_obstacles_tune(optimizer_result, dual_node_deltas);

                // for cycle resolution
                let mut order: VecDeque<BTreeSet<Obstacle>> = VecDeque::with_capacity(MAX_HISTORY); // fifo order of the obstacles sets seen
                let mut current_sequences: Vec<(usize, BTreeSet<Obstacle>)> = Vec::new(); // the indexes that are currently being processed

                '_resolving: while !resolved {
                    let (_obstacles, _resolved) = self.resolve_tune(obstacles.clone(), interface, dual_module);

                    // cycle resolution
                    let drained: Vec<(usize, BTreeSet<Obstacle>)> = std::mem::take(&mut current_sequences);
                    for (idx, start) in drained.into_iter() {
                        if _obstacles.eq(&start) {
                            dual_module.end_tuning();
                            break '_resolving;
                        }
                        if _obstacles.eq(order
                            .get(MAX_HISTORY - idx - 1)
                            .unwrap_or(order.get(order.len() - idx - 1).unwrap()))
                        {
                            current_sequences.push((idx + 1, start));
                        }
                    }

                    order.push_back(_obstacles.clone());
                    if order.len() > MAX_HISTORY {
                        order.pop_front();
                        current_sequences = current_sequences
                            .into_iter()
                            .filter_map(|(x, start)| if x >= MAX_HISTORY { None } else { Some((x + 1, start)) })
                            .collect();
                    }

                    for (idx, c) in order.iter().enumerate() {
                        if c.eq(&_obstacles) {
                            current_sequences.push((idx, c.clone()));
                        }
                    }

                    if _resolved {
                        dual_module.end_tuning();
                        break;
                    }

                    obstacles = _obstacles;
                    resolved = _resolved;
                }
            }
        }
    }

    fn subgraph(
        &mut self,
        interface: &DualModuleInterfacePtr,
        dual_module: &(impl DualModuleImpl + Send + Sync),
    ) -> OutputSubgraph;

    fn subgraph_range(
        &mut self,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut (impl DualModuleImpl + Send + Sync),
    ) -> (OutputSubgraph, WeightRange) {
        let output_subgraph = self.subgraph(interface, dual_module);
        let internal_subgraph = OutputSubgraph::get_internal_subgraph(&output_subgraph);
        let mut upper = Rational::zero();
        for edge_weak in internal_subgraph.iter() {
            upper += edge_weak.upgrade_force().read_recursive().weight.clone();
        }

        let weight_range = WeightRange::new(interface.sum_dual_variables() + dual_module.get_negative_weight_sum(), upper);
        (output_subgraph, weight_range)
    }

    /// performance profiler report
    fn generate_profiler_report(&self) -> serde_json::Value {
        json!({})
    }

    /* tune mode methods */
    /// check if there are more plugins to be applied, defaulted to having no plugins
    fn has_more_plugins(&mut self) -> bool {
        false
    }

    /// in "tune" mode, return the list of clusters that need to be resolved
    fn pending_clusters(&mut self) -> Vec<PrimalClusterWeak> {
        panic!("not implemented `pending_clusters`");
    }

    /// check if a cluster has been solved, if not then resolve it
    fn resolve_cluster(
        &mut self,
        _cluster_ptr: &PrimalClusterPtr,
        _interface_ptr: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        panic!("not implemented `resolve_cluster`");
    }

    /// `resolve_cluster` but in tuning mode, optimizer result denotes what the optimizer has accomplished
    fn resolve_cluster_tune(
        &mut self,
        _cluster_ptr: &PrimalClusterPtr,
        _interface_ptr: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
        // _dual_node_deltas: &mut BTreeMap<OrderedDualNodePtr, Rational>,
        _dual_node_deltas: &mut BTreeMap<OrderedDualNodePtr, (Rational, PrimalClusterPtr)>,
    ) -> (bool, OptimizerResult) {
        panic!("not implemented `resolve_cluster_tune`");
    }

    /* affinity */

    /// calculate the affinity map of clusters and maintain an decreasing order of priority
    fn update_sorted_clusters_aff<D: DualModuleImpl>(&mut self, _dual_module: &mut D) {
        panic!("not implemented `update_sorted_clusters_aff`");
    }

    /// get the sorted clusters by affinity
    fn get_sorted_clusters_aff(&mut self) -> BTreeSet<ClusterAffinity> {
        panic!("not implemented `get_sorted_clusters_aff`");
    }

    #[cfg(feature = "incr_lp")]
    /// calculate the edges free weight map by cluster
    fn calculate_edges_free_weight_clusters(&mut self, _dual_module: &mut impl DualModuleImpl) {
        panic!("not implemented `calculate_edges_free_weight_clusters`");
    }

    /// unset the cluster_weight parameter
    #[cfg(feature = "incr_lp")]
    fn uninit_cluster_weight(&mut self) {}

    /// get the cluster_weight parameter
    #[cfg(feature = "incr_lp")]
    fn is_cluster_weight_initialized(&self) -> bool {
        true
    }
}
