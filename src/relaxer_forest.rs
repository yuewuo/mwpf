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

pub type RelaxerVec = Vec<Relaxer>;

/// a forest of relaxers that possibly depend on each other
pub struct RelaxerForest {
    /// keep track of the remaining tight edges for quick validation:
    /// these edges cannot grow unless untightened by some relaxers
    tight_edges: FastIterSet<EdgeIndex>,
    /// keep track of the subgraphs that are allowed to shrink:
    /// these should be all positive dual variables, all others are yS = 0
    shrinkable_subgraphs: FastIterSet<Arc<InvalidSubgraph>>,
    /// each untightened edge corresponds to a relaxer with speed:
    /// to untighten the edge for a unit length, how much should a relaxer be executed
    edge_untightener: FastIterMap<EdgeIndex, (Arc<Relaxer>, Rational)>,
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
        IterEdge: Iterator<Item = EdgeIndex>,
        IterSubgraph: Iterator<Item = Arc<InvalidSubgraph>>,
    {
        Self {
            tight_edges: FastIterSet::from_iter(tight_edges),
            shrinkable_subgraphs: FastIterSet::from_iter(shrinkable_subgraphs),
            edge_untightener: FastIterMap::new(),
            expanded_relaxers: FastIterMap::new(),
        }
    }

    /// check if the proposed relaxers are indeed relaxers given the edges
    /// untightened by existing relaxers
    pub fn validate(&self, relaxer: &Relaxer) -> Result<(), String> {
        // non-negative overall speed and effectiveness check
        relaxer.sanity_check()?;
        // a relaxer cannot grow any tight edge
        for (edge_index, _) in relaxer.get_growing_edges().iter() {
            if self.tight_edges.contains(edge_index) && !self.edge_untightener.contains_key(edge_index) {
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
        for (edge_index, speed) in relaxer.get_untighten_edges().iter() {
            debug_assert!(speed.is_negative());
            if !self.edge_untightener.contains_key(edge_index) {
                self.edge_untightener.insert(*edge_index, (relaxer.clone(), -speed.recip()));
            }
        }
    }

    fn compute_expanded(&mut self, relaxer: &Arc<Relaxer>) {
        if self.expanded_relaxers.contains_key(relaxer) {
            return;
        }
        let mut untightened_edges: FastIterMap<EdgeIndex, Rational> = FastIterMap::new();
        let mut directions: FastIterMap<Arc<InvalidSubgraph>, Rational> = relaxer.get_direction().clone();
        for (edge_index, speed) in relaxer.get_growing_edges().iter() {
            debug_assert!(speed.is_positive());
            if self.tight_edges.contains(edge_index) {
                debug_assert!(
                    self.edge_untightener.contains_key(edge_index),
                    "edge {} is tight but no untightener presents, thus new relaxer cannot grow on it",
                    edge_index
                );
                let require_speed = if let Some(mut existing_speed) = untightened_edges.get_mut(edge_index) {
                    if *existing_speed >= *speed {
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
                    let edge_relaxer = self.edge_untightener.get(edge_index).unwrap().0.clone();
                    self.compute_expanded(&edge_relaxer);
                    let (edge_relaxer, speed_ratio) = self.edge_untightener.get(edge_index).unwrap();
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
                    for (edge_index, original_speed) in expanded_edge_relaxer.get_untighten_edges().iter() {
                        debug_assert!(original_speed.is_negative());
                        let new_speed = -original_speed * speed_ratio * require_speed.clone();
                        if let Some(mut speed) = untightened_edges.get_mut(edge_index) {
                            *speed += new_speed;
                            continue;
                        }
                        untightened_edges.insert(*edge_index, new_speed);
                    }
                    debug_assert_eq!(untightened_edges.get(edge_index), Some(&require_speed));
                    *untightened_edges.get_mut(edge_index).unwrap() -= require_speed;
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
    use num_traits::{FromPrimitive, One};

    #[test]
    fn relaxer_forest_example() {
        // cargo test relaxer_forest_example -- --nocapture
        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1, 2, 3].into())),
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [4, 5].into())),
        ];
        let mut relaxer_forest = RelaxerForest::new(tight_edges.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [7, 8, 9].into()));
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
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1, 2, 7].into()));
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
        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1, 2].into())),
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [3].into())),
        ];
        let mut relaxer_forest = RelaxerForest::new(tight_edges.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [7, 8, 9].into()));
        let relaxer_1 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_1.clone(), Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
            ]
            .into(),
        ));
        relaxer_forest.add(relaxer_1);
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1, 2, 7].into()));
        let invalid_subgraph_3 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [2].into()));
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
        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1, 2].into())),
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [2, 3].into())),
        ];
        let mut relaxer_forest = RelaxerForest::new(tight_edges.into_iter(), shrinkable_subgraphs.iter().cloned());
        let invalid_subgraph_1 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [7, 8, 9].into()));
        let relaxer_1 = Arc::new(Relaxer::new_raw(
            [
                (invalid_subgraph_1.clone(), Rational::one()),
                (shrinkable_subgraphs[0].clone(), -Rational::one()),
            ]
            .into(),
        ));
        relaxer_forest.add(relaxer_1);
        let invalid_subgraph_2 = Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [10, 11].into()));
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
        let tight_edges = [0, 1, 2, 3, 4, 5, 6];
        let shrinkable_subgraphs = [
            Arc::new(InvalidSubgraph::new_raw([1].into(), [].into(), [1, 2].into())),
            Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [].into())),
        ];
        let relaxer_forest = RelaxerForest::new(tight_edges.into_iter(), shrinkable_subgraphs.iter().cloned());
        println!("relaxer_forest: {:?}", relaxer_forest.shrinkable_subgraphs);
        // invalid relaxer is forbidden
        let invalid_relaxer = Relaxer::new_raw(
            [(
                Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [].into())),
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
                Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [1].into())),
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
                    Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [9].into())),
                    Rational::one(),
                ),
                (
                    Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [2, 3].into())),
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
                Arc::new(InvalidSubgraph::new_raw([].into(), [].into(), [9].into())),
                Rational::one(),
            )]
            .into(),
        );
        relaxer_forest.validate(&relaxer).unwrap();
    }
}
