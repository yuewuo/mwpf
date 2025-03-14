//! Primal Module
//!
//! Generics for primal modules, defining the necessary interfaces for a primal module
//!

use std::collections::VecDeque;
use std::sync::Arc;

use crate::dual_module::*;
use crate::itertools::Itertools;
use crate::ordered_float::OrderedFloat;
use crate::primal_module_serial::ClusterAffinity;
use crate::relaxer_optimizer::OptimizerResult;
use crate::util::*;
use crate::visualize::*;
use hashbrown::HashSet;

pub type Affinity = OrderedFloat;

const MAX_HISTORY: usize = 10;

/// common trait that must be implemented for each implementation of primal module
pub trait PrimalModuleImpl {
    /// create a primal module given the dual module
    fn new_empty(solver_initializer: &Arc<SolverInitializer>) -> Self;

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
        _obstacles: FastIterSet<Obstacle>,
        _interface: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> (FastIterSet<Obstacle>, bool) {
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
                println!("dual_report: {:?}", dual_report);
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

    /// update the weights given the syndrome pattern; return a new syndrome pattern
    /// that has some of the vertices flipped due to negative weights
    fn weight_preprocessing<D: DualModuleImpl + MWPSVisualizer>(
        &mut self,
        syndrome_pattern: Arc<SyndromePattern>,
        dual_module: &mut D,
        initializer: &Arc<SolverInitializer>,
    ) -> Arc<SyndromePattern> {
        // update weights given the syndrome pattern
        if let Some((weights, mix_ratio, floor_weight)) = syndrome_pattern.override_weights.as_ref() {
            let mut weights = weights.clone();
            if let Some(floor_weight) = floor_weight {
                weights.iter_mut().for_each(|w| {
                    if *w < *floor_weight {
                        *w = floor_weight.clone();
                    }
                });
            }
            dual_module.update_weights(weights, mix_ratio.clone());
        } else {
            let mut weight_updates: FastIterMap<EdgeIndex, Weight> = FastIterMap::new();
            for &herald_index in syndrome_pattern.heralds.iter() {
                for (edge_index, weight) in initializer.heralds[herald_index].iter() {
                    let value = weight_updates.remove(edge_index);
                    let original_weight = value
                        .as_ref()
                        .unwrap_or_else(|| &initializer.weighted_edges[*edge_index].weight);
                    let mixed_weight = exclusive_weight_sum(original_weight, weight);
                    weight_updates.insert(*edge_index, mixed_weight);
                }
            }
            for &edge_index in syndrome_pattern.erasures.iter() {
                use crate::num_traits::Zero;
                weight_updates.insert(edge_index, Weight::zero());
            }
            dual_module.set_weights(weight_updates);
        }

        // after all the edge weights are set, adjust the negative weights and find the flipped vertices
        dual_module.adjust_weights_for_negative_edges();
        let flip_vertices = dual_module.get_flip_vertices();
        if flip_vertices.is_empty() {
            // we don't need to modify the syndrome pattern
            return syndrome_pattern;
        }

        // otherwise modify the syndrome
        let mut moved_out_set = syndrome_pattern
            .defect_vertices
            .iter()
            .cloned()
            .collect::<HashSet<VertexIndex>>();
        for to_flip in flip_vertices.iter() {
            if moved_out_set.contains(to_flip) {
                moved_out_set.remove(to_flip);
            } else {
                moved_out_set.insert(*to_flip);
            }
        }
        Arc::new(SyndromePattern::new_vertices(moved_out_set.into_iter().collect()))
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
        // then call the solver to
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
        // do not trigger on Python keyboard interrupt until 100ms later
        PYTHON_SIGNAL_CHECKER.skip_next();
        // Search, this part is unchanged
        let mut dual_report = dual_module.report();

        while !dual_report.is_unbounded() {
            PYTHON_SIGNAL_CHECKER.check().unwrap();
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
            PYTHON_SIGNAL_CHECKER.check().unwrap();
            if start {
                start = false;
                dual_module.advance_mode();
            }
            self.update_sorted_clusters_aff(dual_module);
            let cluster_affs = self.get_sorted_clusters_aff();

            for cluster_affinity in cluster_affs.into_iter().sorted() {
                let cluster_index = cluster_affinity.cluster_index;
                let mut dual_node_deltas = FastIterMap::new();
                let (mut resolved, optimizer_result) =
                    self.resolve_cluster_tune(cluster_index, interface, dual_module, &mut dual_node_deltas);

                let mut obstacles = dual_module.get_obstacles_tune(optimizer_result, dual_node_deltas);

                // for cycle resolution
                let mut order: VecDeque<FastIterSet<Obstacle>> = VecDeque::with_capacity(MAX_HISTORY); // fifo order of the obstacles sets seen
                let mut current_sequences: Vec<(usize, FastIterSet<Obstacle>)> = Vec::new(); // the indexes that are currently being processed

                '_resolving: while !resolved {
                    PYTHON_SIGNAL_CHECKER.check().unwrap();
                    let (_obstacles, _resolved) = self.resolve_tune(obstacles.clone(), interface, dual_module);

                    // cycle resolution
                    // process current sequences and try to extend them
                    let drained: Vec<(usize, FastIterSet<Obstacle>)> = std::mem::take(&mut current_sequences);
                    for (idx, start) in drained.into_iter() {
                        // if the current obstacles match a starting state, we have a cycle
                        if _obstacles.eq(&start) {
                            dual_module.end_tuning();
                            break '_resolving;
                        }

                        // determine which candidate in the history to compare with
                        let candidate_index = order.len() - idx - 1;

                        if let Some(candidate) = order.get(candidate_index) {
                            if _obstacles.eq(candidate) {
                                // extend this sequence by increasing its offset
                                current_sequences.push((idx + 1, start));
                            }
                        } else {
                            eprintln!(
                                "Warning: candidate index {} out of bounds, order length {}, idx {}, start {:?}, candidate {:?}",
                                candidate_index,
                                order.len(),
                                idx,
                                start,
                                order.get(candidate_index)
                            );
                        }
                    }

                    // add the current obstacles state to the history
                    order.push_back(_obstacles.clone());
                    if order.len() > MAX_HISTORY {
                        // remove the oldest state
                        order.pop_front();
                        // because the deque shifted left, update each sequence by subtracting 1 from its index
                        current_sequences = current_sequences
                            .into_iter()
                            .filter_map(|(x, start)| {
                                if x == 0 {
                                    // this element is popped
                                    None
                                } else {
                                    // the previous bug, everything should be shifted left/forward instead right/to the right
                                    Some((x - 1, start))
                                }
                            })
                            .collect();
                    }

                    // add new sequences from matching obstacles in the history
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

    fn subgraph(&mut self, interface: &DualModuleInterfacePtr, dual_module: &mut impl DualModuleImpl) -> OutputSubgraph;

    fn subgraph_range(
        &mut self,
        interface: &DualModuleInterfacePtr,
        dual_module: &mut impl DualModuleImpl,
    ) -> (OutputSubgraph, WeightRange) {
        let output_subgraph = self.subgraph(interface, dual_module);
        let weight_range = WeightRange::new(
            interface.sum_dual_variables() + dual_module.get_negative_weight_sum(),
            dual_module.get_subgraph_weight(&output_subgraph.subgraph) + dual_module.get_negative_weight_sum(),
        );
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
    fn pending_clusters(&mut self) -> Vec<usize> {
        panic!("not implemented `pending_clusters`");
    }

    /// check if a cluster has been solved, if not then resolve it
    fn resolve_cluster(
        &mut self,
        _cluster_index: NodeIndex,
        _interface_ptr: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
    ) -> bool {
        panic!("not implemented `resolve_cluster`");
    }

    /// `resolve_cluster` but in tuning mode, optimizer result denotes what the optimizer has accomplished
    fn resolve_cluster_tune(
        &mut self,
        _cluster_index: NodeIndex,
        _interface_ptr: &DualModuleInterfacePtr,
        _dual_module: &mut impl DualModuleImpl,
        // _dual_node_deltas: &mut FastIterMap<OrderedDualNodePtr, Rational>,
        _dual_node_deltas: &mut FastIterMap<OrderedDualNodePtr, (Rational, NodeIndex)>,
    ) -> (bool, OptimizerResult) {
        panic!("not implemented `resolve_cluster_tune`");
    }

    /* affinity */

    /// calculate the affinity map of clusters and maintain an decreasing order of priority
    fn update_sorted_clusters_aff<D: DualModuleImpl>(&mut self, _dual_module: &mut D) {
        panic!("not implemented `update_sorted_clusters_aff`");
    }

    /// get the sorted clusters by affinity
    fn get_sorted_clusters_aff(&mut self) -> FastIterSet<ClusterAffinity> {
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
