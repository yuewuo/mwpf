//! Plugin
//!
//! Generics for plugins, defining the necessary interfaces for a plugin
//!
//! A plugin must implement Clone trait, because it will be cloned multiple times for each cluster
//!

use crate::decoding_hypergraph::*;
use crate::derivative::Derivative;
use crate::dual_module::*;
use crate::matrix::*;
use crate::plugin_union_find::*;
use crate::relaxer::*;
use crate::relaxer_forest::*;
use crate::pointers::*;
use num_traits::Signed;
use parking_lot::RwLock;
use std::sync::Arc;

pub type EchelonMatrix = Echelon<Tail<Tight<BasicMatrix>>>;

/// common trait that must be implemented for each plugin
pub trait PluginImpl: std::fmt::Debug {
    /// given the tight edges and parity constraints, find relaxers
    fn find_relaxers(
        &self,
        decoding_graph: &DecodingHyperGraph,
        matrix: &mut EchelonMatrix,
        positive_dual_nodes: &[DualNodePtr],
    ) -> RelaxerVec;

    /// create a plugin entry with default settings
    fn entry() -> PluginEntry
    where
        Self: 'static + Send + Sync + Sized + Default,
    {
        PluginEntry {
            plugin: Arc::new(Self::default()),
            repeat_strategy: RepeatStrategy::default(),
        }
    }

    fn entry_with_strategy(strategy: RepeatStrategy) -> PluginEntry
    where
        Self: 'static + Send + Sync + Sized + Default,
    {
        PluginEntry {
            plugin: Arc::new(Self::default()),
            repeat_strategy: strategy,
        }
    }
}

/// configuration of how a plugin should be repeated
#[derive(Derivative, PartialEq, Eq, Clone)]
#[derivative(Debug, Default(new = "true"))]
pub enum RepeatStrategy {
    /// single execution
    #[derivative(Default)]
    Once,
    /// repeated execution
    Multiple {
        /// it stops after `max_repetition` repeats
        max_repetition: usize,
    },
}

/// describes what plugins to enable and also the recursive strategy
pub struct PluginEntry {
    /// the implementation of a plugin
    pub plugin: Arc<dyn PluginImpl + Send + Sync>,
    /// repetition strategy
    pub repeat_strategy: RepeatStrategy,
}

impl PluginEntry {
    pub fn execute(
        &self,
        decoding_graph: &DecodingHyperGraph,
        matrix: &mut EchelonMatrix,
        positive_dual_nodes: &[DualNodePtr],
        relaxer_forest: &mut RelaxerForest,
    ) -> Option<Relaxer> {
        let mut repeat = true;
        let mut repeat_count = 0;
        while repeat {
            // execute the plugin
            let relaxers = self.plugin.find_relaxers(decoding_graph, &mut *matrix, positive_dual_nodes);
            if relaxers.is_empty() {
                repeat = false;
            }
            for relaxer in relaxers.into_iter() {
                for edge_index in relaxer.get_untighten_edges().keys() {
                    matrix.update_edge_tightness(*edge_index, false);
                }
                let relaxer = Arc::new(relaxer);
                let sum_speed = relaxer.get_sum_speed();
                if sum_speed.is_positive() {
                    return Some(relaxer_forest.expand(&relaxer));
                } else {
                    relaxer_forest.add(relaxer);
                }
            }
            // determine whether repeat again
            match self.repeat_strategy {
                RepeatStrategy::Once => {
                    repeat = false;
                }
                RepeatStrategy::Multiple { max_repetition } => {
                    if repeat_count + 1 >= max_repetition {
                        repeat = false;
                    }
                }
            }
            repeat_count += 1;
        }
        None
    }
}

pub type PluginVec = Vec<PluginEntry>;

pub struct PluginManager {
    pub plugins: Arc<PluginVec>,
    /// the plugin manager will stop at this index; this is helpful when we want
    /// to execute the first plugin for all clusters, and then the second plugin for all, and so on.
    pub plugin_count: Arc<RwLock<usize>>,
}

impl PluginManager {
    pub fn new(plugins: Arc<PluginVec>, plugin_count: Arc<RwLock<usize>>) -> Self {
        Self { plugins, plugin_count }
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn find_relaxer(
        &mut self,
        decoding_graph: &DecodingHyperGraph,
        matrix: &mut EchelonMatrix,
        positive_dual_nodes: &[DualNodePtr],
    ) -> Option<Relaxer> {
        let mut relaxer_forest = RelaxerForest::new(
            matrix.get_view_edges().into_iter(),
            positive_dual_nodes
                .iter()
                .map(|ptr| ptr.read_recursive().invalid_subgraph.clone()),
        );
        for plugin_entry in self.plugins.iter().take(*self.plugin_count.read_recursive()) {
            if let Some(relaxer) = plugin_entry.execute(decoding_graph, matrix, positive_dual_nodes, &mut relaxer_forest) {
                return Some(relaxer);
            }
        }
        // add a union find relaxer finder as the last resort if nothing is reported
        PluginUnionFind::entry().execute(decoding_graph, matrix, positive_dual_nodes, &mut relaxer_forest)
    }
}
