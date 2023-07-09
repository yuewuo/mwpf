//! Plugin
//!
//! Generics for plugins, defining the necessary interfaces for a plugin
//!
//! A plugin must implement Clone trait, because it will be cloned multiple times for each cluster
//!

use crate::derivative::Derivative;
use crate::dual_module::*;
use crate::framework::*;
use crate::parity_matrix::*;
use crate::relaxer_pool::*;
use std::sync::Arc;

/// common trait that must be implemented for each plugin
pub trait PluginImpl {
    /// given the tight edges and parity constraints, find relaxers
    fn find_relaxers(
        &self,
        decoding_graph: &HyperDecodingGraph,
        matrix: &ParityMatrix,
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
}

/// configuration of how a plugin should be repeated
#[derive(Derivative, PartialEq, Eq, Clone)]
#[derivative(Debug, Default(new = "true"))]
pub enum RepeatStrategy {
    /// single execution
    #[derivative(Default)]
    Once,
    /// repeated execution
    Repeated {
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

pub type PluginVec = Vec<PluginEntry>;

pub struct PluginManager {
    pub plugins: Arc<PluginVec>,
}

impl PluginManager {
    pub fn new(plugins: Arc<PluginVec>) -> Self {
        Self { plugins }
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub fn find_relaxer(
        &self,
        decoding_graph: &HyperDecodingGraph,
        matrix: &ParityMatrix,
        positive_dual_nodes: &[DualNodePtr],
    ) -> Option<Relaxer> {
        None
    }
}
