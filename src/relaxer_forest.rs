//! Relaxer Forest
//!
//! Maintain several lists of relaxers
//!

use crate::invalid_subgraph::*;
use crate::num_traits::Zero;
use crate::relaxer::*;
use crate::util::*;
use num_traits::Signed;

use std::sync::Arc;

use crate::dual_module_pq::{EdgePtr, EdgeWeak};
#[cfg(feature = "unsafe_pointer")]
use crate::pointers::UnsafePtr;

pub type RelaxerVec = Vec<Relaxer>;

/// a forest of relaxers that possibly depend on each other
pub struct RelaxerForest {
    /// keep track of the remaining tight edges for quick validation:
    /// these edges cannot grow unless untightened by some relaxers
    tight_edges: BTreeSet<EdgePtr>,
    /// keep track of the subgraphs that are allowed to shrink:
    /// these should be all positive dual variables, all others are yS = 0
    shrinkable_subgraphs: FastIterSet<Arc<InvalidSubgraph>>,
    /// each untightened edge corresponds to a relaxer with speed:
    /// to untighten the edge for a unit length, how much should a relaxer be executed
    edge_untightener: BTreeMap<EdgePtr, (Arc<Relaxer>, Rational)>,
    /// expanded relaxer results, as part of the dynamic programming:
    /// the expanded relaxer is a valid relaxer only growing of initial un-tight edges,
    /// not any edges untightened by other relaxers
    expanded_relaxers: FastIterMap<Arc<Relaxer>, Relaxer>,
}

pub const FOREST_ERR_MSG_GROW_TIGHT_EDGE: &str = "invalid relaxer: try to grow a tight edge";
pub const FOREST_ERR_MSG_UNSHRINKABLE: &str = "invalid relaxer: try to shrink a unshrinkable subgraph";

impl RelaxerForest {
    pub fn new<IterEdge, IterSubgraph>(tight_edges: IterEdge, shrinkable_subgraphs: IterSubgraph) -> Self
    where
        IterEdge: Iterator<Item = EdgeWeak>,
        IterSubgraph: Iterator<Item = Arc<InvalidSubgraph>>,
    {
        Self {
            tight_edges: BTreeSet::from_iter(tight_edges.map(|e| e.upgrade_force())),
            shrinkable_subgraphs: BTreeSet::from_iter(shrinkable_subgraphs),
            edge_untightener: BTreeMap::new(),
            expanded_relaxers: BTreeMap::new(),
        }
    }

    /// check if the proposed relaxers are indeed relaxers given the edges
    /// untightened by existing relaxers
    pub fn validate(&self, relaxer: &Relaxer) -> Result<(), String> {
        // non-negative overall speed and effectiveness check
        relaxer.sanity_check()?;
        // a relaxer cannot grow any tight edge
        for (edge_ptr, _) in relaxer.get_growing_edges().iter() {
            if self.tight_edges.contains(edge_ptr) && !self.edge_untightener.contains_key(edge_ptr) {
                let edge_index = edge_ptr.read_recursive().edge_index;
                return Err(format!("{FOREST_ERR_MSG_GROW_TIGHT_EDGE}: {edge_index}"));
            }
        }
        // a relaxer cannot shrink any zero dual variable
        for (invalid_subgraph, grow_ratio) in relaxer.get_direction().iter() {
            if grow_ratio.is_negative() && !self.shrinkable_subgraphs.contains(invalid_subgraph) {
                return Err(format!("{FOREST_ERR_MSG_UNSHRINKABLE}: {invalid_subgraph:?}"));
            }
        }
        Ok(())
    }

    /// add a relaxer to the forest
    pub fn add(&mut self, relaxer: Arc<Relaxer>) {
        // validate only at debug mode to improve speed
        debug_assert_eq!(self.validate(&relaxer), Ok(()));
        // add this relaxer to the forest
        for (edge_ptr, speed) in relaxer.get_untighten_edges() {
            debug_assert!(speed.is_negative());
            if !self.edge_untightener.contains_key(edge_ptr) {
                self.edge_untightener
                    .insert(edge_ptr.clone(), (relaxer.clone(), -speed.recip()));
            }
        }
    }

    fn compute_expanded(&mut self, relaxer: &Arc<Relaxer>) {
        if self.expanded_relaxers.contains_key(relaxer) {
            return;
        }
        let mut untightened_edges: BTreeMap<EdgePtr, Rational> = BTreeMap::new();
        let mut directions: BTreeMap<Arc<InvalidSubgraph>, Rational> = relaxer.get_direction().clone();
        for (edge_ptr, speed) in relaxer.get_growing_edges() {
            debug_assert!(speed.is_positive());
            if self.tight_edges.contains(edge_ptr) {
                debug_assert!(self.edge_untightener.contains_key(edge_ptr));
                let require_speed = if let Some(existing_speed) = untightened_edges.get_mut(edge_ptr) {
                    if &*existing_speed >= speed {
                        *existing_speed -= speed;
                        Rational::zero()
                    } else {
                        let required_speed = speed - &*existing_speed;
                        existing_speed.set_zero();
                        required_speed
                    }
                } else {
                    speed.clone()
                };
                if require_speed.is_positive() {
                    // we need to invoke another relaxer to untighten this edge
                    let edge_relaxer = self.edge_untightener.get(edge_ptr).unwrap().0.clone();
                    self.compute_expanded(&edge_relaxer);
                    let (edge_relaxer, speed_ratio) = self.edge_untightener.get(edge_ptr).unwrap();
                    debug_assert!(speed_ratio.is_positive());
                    let expanded_edge_relaxer = self.expanded_relaxers.get(edge_relaxer).unwrap();
                    for (subgraph, original_speed) in expanded_edge_relaxer.get_direction().iter() {
                        let new_speed = original_speed * speed_ratio * require_speed.clone();
                        if let Some(mut speed) = directions.get_mut(subgraph) {
                            *speed += new_speed;
                            continue;
                        }
                        directions.insert(subgraph.clone(), new_speed);
                    }
                    for (edge_ptr, original_speed) in expanded_edge_relaxer.get_untighten_edges() {
                        debug_assert!(original_speed.is_negative());
                        let new_speed = -original_speed * speed_ratio;
                        if let Some(speed) = untightened_edges.get_mut(edge_ptr) {
                            *speed += new_speed;
                        } else {
                            untightened_edges.insert(edge_ptr.clone(), new_speed);
                        }
                        untightened_edges.insert(*edge_index, new_speed);
                    }
                    debug_assert_eq!(untightened_edges.get(edge_ptr), Some(&require_speed));
                    *untightened_edges.get_mut(edge_ptr).unwrap() -= require_speed;
                }
            }
        }
        let expanded = Relaxer::new(directions);
        // an expanded relaxer will not rely on any non-tight edges
        debug_assert!(expanded
            .get_growing_edges()
            .iter()
            .all(|(edge_index, _)| !self.tight_edges.contains(edge_index)));
        self.expanded_relaxers.insert(relaxer.clone(), expanded);
    }

    /// expand a relaxer
    pub fn expand(&mut self, relaxer: &Arc<Relaxer>) -> Relaxer {
        debug_assert_eq!(self.validate(relaxer), Ok(()));
        self.compute_expanded(relaxer);
        self.expanded_relaxers.get(relaxer).unwrap().clone()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::decoding_hypergraph::tests::color_code_5_decoding_graph;
    use crate::dual_module::DualModuleImpl;
    use crate::dual_module::DualModuleInterfacePtr;
    use crate::dual_module_pq::DualModulePQ;
    use num_traits::{FromPrimitive, One};

    #[test]
    fn relaxer_forest_example() {
        // cargo test relaxer_forest_example -- --nocapture
        // initialize an arbitrary decoding graph, this is required because invalid subgraph needs the vertex and edge pointers
        let visualize_filename = "relaxer_forest_example.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern.clone(), &mut dual_module, 0); // this is needed to load the defect vertices

        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [1, 2, 3].into(),
                &mut dual_module,
            )),
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [4, 5].into(),
                &mut dual_module,
            )),
        ];
        let tight_edges_weak = dual_module
            .get_edge_ptr_vec(&tight_edges)
            .into_iter()
            .map(|e| e.downgrade())
            .collect::<Vec<_>>();
        let mut relaxer_forest = RelaxerForest::new(tight_edges_weak.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [7, 8, 9].into(),
            &mut dual_module,
        ));
        let relaxer_1 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_1.clone(), Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
            ]
            .into(),
        ));
        let expanded_1 = relaxer_forest.expand(&relaxer_1);
        assert_eq!(expanded_1, *relaxer_1);
        relaxer_forest.add(relaxer_1);
        // now add a relaxer that is relying on relaxer_1
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [1, 2, 7].into(),
            &mut dual_module,
        ));
        let relaxer_2 = Arc::new(Relaxer::new_raw([(invalid_subgraph_2.clone(), Rational::one())].into()));
        let expanded_2 = relaxer_forest.expand(&relaxer_2);
        let expected_relaxer = Relaxer::new(
            [
                (invalid_subgraph_1, Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
                (invalid_subgraph_2, Rational::one()),
            ]
            .into(),
        );
        // println!("{expanded_2:#?}");
        // println!("{expected_relaxer:#?}");
        assert_eq!(expanded_2, expected_relaxer);
    }

    #[test]
    fn relaxer_forest_require_multiple() {
        // cargo test relaxer_forest_require_multiple -- --nocapture
        // initialize an arbitrary decoding graph, this is required because invalid subgraph needs the vertex and edge pointers
        let visualize_filename = "relaxer_forest_require_multiple.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer);
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern.clone(), &mut dual_module, 0); // this is needed to load the defect vertices

        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [1, 2].into(),
                &mut dual_module,
            )),
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [3].into(),
                &mut dual_module,
            )),
        ];
        let tight_edges_weak = dual_module
            .get_edge_ptr_vec(&tight_edges)
            .into_iter()
            .map(|e| e.downgrade())
            .collect::<Vec<_>>();
        let mut relaxer_forest = RelaxerForest::new(tight_edges_weak.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [7, 8, 9].into(),
            &mut dual_module,
        ));
        let relaxer_1 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_1.clone(), Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
            ]
            .into(),
        ));
        relaxer_forest.add(relaxer_1);
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [1, 2, 7].into(),
            &mut dual_module,
        ));
        let invalid_subgraph_3 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [2].into(),
            &mut dual_module,
        ));
        let relaxer_2 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_2.clone(), Rational::one()),
                (invalid_subgraph_3.clone(), Rational::one()),
            ]
            .into(),
        ));
        let expanded_2 = relaxer_forest.expand(&relaxer_2);
        assert_eq!(
            expanded_2,
            Relaxer::new(
                [
                    (invalid_subgraph_2, Rational::one()),
                    (invalid_subgraph_3, Rational::one()),
                    (invalid_subgraph_1, Rational::from_usize(2).unwrap()),
                    (shrinkable_subgraphs[0].clone(), -Rational::from_usize(2).unwrap()),
                ]
                .into()
            )
        );
        // println!("{expanded_2:#?}");
    }

    #[test]
    fn relaxer_forest_relaxing_same_edge() {
        // cargo test relaxer_forest_relaxing_same_edge -- --nocapture
        // initialize an arbitrary decoding graph, this is required because invalid subgraph needs the vertex and edge pointers
        let visualize_filename = "relaxer_forest_relaxing_same_edge.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer); // initialize vertex and edge pointers
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern.clone(), &mut dual_module, 0); // this is needed to load the defect vertices

        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [1, 2].into(),
                &mut dual_module,
            )),
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [2, 3].into(),
                &mut dual_module,
            )),
        ];
        let tight_edges_weak = dual_module
            .get_edge_ptr_vec(&tight_edges)
            .into_iter()
            .map(|e| e.downgrade())
            .collect::<Vec<_>>();
        let mut relaxer_forest = RelaxerForest::new(tight_edges_weak.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [7, 8, 9].into(),
            &mut dual_module,
        ));
        let relaxer_1 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_1.clone(), Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
            ]
            .into(),
        ));
        relaxer_forest.add(relaxer_1);
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw_from_indices(
            [].into(),
            [].into(),
            [10, 11].into(),
            &mut dual_module,
        ));
        let relaxer_2 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_2.clone(), Rational::one()),
                (shrinkable_subgraphs[1].clone(), -Rational::one()),
            ]
            .into(),
        ));
        relaxer_forest.add(relaxer_2);
    }

    #[test]
    fn relaxer_forest_validate() {
        // cargo test relaxer_forest_validate -- --nocapture
        // initialize an arbitrary decoding graph, this is required because invalid subgraph needs the vertex and edge pointers
        let visualize_filename = "relaxer_forest_validate.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let initializer = decoding_graph.model_graph.initializer.clone();
        let mut dual_module = DualModulePQ::new_empty(&initializer); // initialize vertex and edge pointers
        let interface_ptr = DualModuleInterfacePtr::new(decoding_graph.model_graph.clone());
        interface_ptr.load(decoding_graph.syndrome_pattern.clone(), &mut dual_module, 0); // this is needed to load the defect vertices

        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [1].into(),
                [].into(),
                [1, 2].into(),
                &mut dual_module,
            )),
            Arc::new(InvalidSubgraph::new_raw_from_indices(
                [].into(),
                [].into(),
                [].into(),
                &mut dual_module,
            )),
        ];
        let tight_edges_weak = dual_module
            .get_edge_ptr_vec(&tight_edges)
            .into_iter()
            .map(|e| e.downgrade())
            .collect::<Vec<_>>();
        let relaxer_forest = RelaxerForest::new(tight_edges_weak.into_iter(), shrinkable_subgraphs.iter().cloned());
        println!("relaxer_forest: {:?}", relaxer_forest.shrinkable_subgraphs);
        // invalid relaxer is forbidden
        let invalid_relaxer = Relaxer::new_raw(
            [(
                Arc::new(InvalidSubgraph::new_raw_from_indices(
                    [].into(),
                    [].into(),
                    [].into(),
                    &mut dual_module,
                )),
                -Rational::one(),
            )]
            .into(),
        );
        let error_message = relaxer_forest.validate(&invalid_relaxer).expect_err("should panic");
        assert_eq!(
            &error_message[..RELAXER_ERR_MSG_NEGATIVE_SUMMATION.len()],
            RELAXER_ERR_MSG_NEGATIVE_SUMMATION
        );
        // relaxer that increases a tight edge is forbidden
        let relaxer = Relaxer::new_raw(
            [(
                Arc::new(InvalidSubgraph::new_raw_from_indices(
                    [].into(),
                    [].into(),
                    [1].into(),
                    &mut dual_module,
                )),
                Rational::one(),
            )]
            .into(),
        );
        let error_message = relaxer_forest.validate(&relaxer).expect_err("should panic");
        assert_eq!(
            &error_message[..FOREST_ERR_MSG_GROW_TIGHT_EDGE.len()],
            FOREST_ERR_MSG_GROW_TIGHT_EDGE
        );
        // relaxer that shrinks a zero dual variable is forbidden
        let relaxer = Relaxer::new_raw(
            [
                (
                    Arc::new(InvalidSubgraph::new_raw_from_indices(
                        [].into(),
                        [].into(),
                        [9].into(),
                        &mut dual_module,
                    )),
                    Rational::one(),
                ),
                (
                    Arc::new(InvalidSubgraph::new_raw_from_indices(
                        [].into(),
                        [].into(),
                        [2, 3].into(),
                        &mut dual_module,
                    )),
                    -Rational::one(),
                ),
            ]
            .into(),
        );
        let error_message = relaxer_forest.validate(&relaxer).expect_err("should panic");
        assert_eq!(
            &error_message[..FOREST_ERR_MSG_UNSHRINKABLE.len()],
            FOREST_ERR_MSG_UNSHRINKABLE
        );
        // otherwise a relaxer is ok
        let relaxer = Relaxer::new_raw(
            [(
                Arc::new(InvalidSubgraph::new_raw_from_indices(
                    [].into(),
                    [].into(),
                    [9].into(),
                    &mut dual_module,
                )),
                Rational::one(),
            )]
            .into(),
        );
        relaxer_forest.validate(&relaxer).unwrap();
    }
}
