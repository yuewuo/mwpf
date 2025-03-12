use crate::derivative::Derivative;
use crate::invalid_subgraph::*;
use crate::util::*;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;
// use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Eq, Derivative, Default)]
#[derivative(Debug)]
pub struct Relaxer {
    /// the hash value calculated by other fields
    #[derivative(Debug = "ignore")]
    hash_value: u64,
    /// the direction of invalid subgraphs
    direction: FastIterMap<Arc<InvalidSubgraph>, Rational>,
    /// the edges that will be untightened after growing along `direction`;
    /// basically all the edges that have negative `overall_growing_rate`
    untighten_edges: FastIterMap<EdgeIndex, Rational>,
    /// the edges that will grow
    growing_edges: FastIterMap<EdgeIndex, Rational>,
}

impl Hash for Relaxer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_value.hash(state);
    }
}

impl PartialEq for Relaxer {
    fn eq(&self, other: &Self) -> bool {
        self.hash_value == other.hash_value && self.direction == other.direction
    }
}

impl Ord for Relaxer {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.hash_value != other.hash_value {
            self.hash_value.cmp(&other.hash_value)
        } else if self == other {
            Ordering::Equal
        } else {
            // rare cases: same hash value but different state; `direction` uniquely determines the relaxer
            self.direction.cmp(&other.direction)
        }
    }
}

impl PartialOrd for Relaxer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub const RELAXER_ERR_MSG_NEGATIVE_SUMMATION: &str = "the summation of ΔyS is negative";
pub const RELAXER_ERR_MSG_USEFUL: &str = "a valid relaxer must either increase overall ΔyS or untighten some edges";

impl Relaxer {
    pub fn new(direction: FastIterMap<Arc<InvalidSubgraph>, Rational>) -> Self {
        let relaxer = Self::new_raw(direction);
        debug_assert_eq!(relaxer.sanity_check(), Ok(()));
        relaxer
    }

    pub fn clear(&mut self) {
        self.direction.clear();
    }

    pub fn new_raw(direction: FastIterMap<Arc<InvalidSubgraph>, Rational>) -> Self {
        let mut edges = FastIterMap::new();
        for (invalid_subgraph, speed) in direction.iter() {
            for &edge_index in invalid_subgraph.hair.iter() {
                if let Some(mut edge) = edges.get_mut(&edge_index) {
                    *edge += speed;
                    continue;
                }
                edges.insert(edge_index, speed.clone());
            }
        }
        let mut untighten_edges = FastIterMap::new();
        let mut growing_edges = FastIterMap::new();
        for (edge_index, speed) in edges {
            if speed.is_negative() {
                untighten_edges.insert(edge_index, speed);
            } else if speed.is_positive() {
                growing_edges.insert(edge_index, speed);
            }
        }
        let mut relaxer = Self {
            hash_value: 0,
            direction,
            untighten_edges,
            growing_edges,
        };
        relaxer.update_hash();
        relaxer
    }

    pub fn sanity_check(&self) -> Result<(), String> {
        // check summation of ΔyS >= 0
        let sum_speed = self.get_sum_speed();
        if sum_speed.is_negative() {
            return Err(format!("{RELAXER_ERR_MSG_NEGATIVE_SUMMATION}: {sum_speed:?}"));
        }
        if self.untighten_edges.is_empty() && sum_speed.is_zero() {
            return Err(RELAXER_ERR_MSG_USEFUL.to_string());
        }
        Ok(())
    }

    pub fn update_hash(&mut self) {
        let mut hasher = DefaultHasher::default();
        // only hash the direction since other field are derived from the direction
        self.direction.hash(&mut hasher);
        self.hash_value = hasher.finish();
    }

    pub fn get_sum_speed(&self) -> Rational {
        let mut sum_speed = Rational::zero();
        for (_, speed) in self.direction.iter() {
            sum_speed += speed;
        }
        sum_speed
    }

    pub fn get_direction(&self) -> &FastIterMap<Arc<InvalidSubgraph>, Rational> {
        &self.direction
    }

    pub fn get_growing_edges(&self) -> &FastIterMap<EdgeIndex, Rational> {
        &self.growing_edges
    }

    pub fn get_untighten_edges(&self) -> &FastIterMap<EdgeIndex, Rational> {
        &self.untighten_edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoding_hypergraph::tests::*;
    use crate::invalid_subgraph::tests::*;
    use num_traits::One;

    #[test]
    fn relaxer_good() {
        // cargo test relaxer_good -- --nocapture
        let visualize_filename = "relaxer_good.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(
            vec![7].into_iter().collect(),
            FastIterSet::new(),
            decoding_graph.as_ref(),
        ));
        use num_traits::One;
        let relaxer = Relaxer::new([(invalid_subgraph, Rational::one())].into());
        println!("relaxer: {relaxer:?}");
        assert!(relaxer.untighten_edges.is_empty());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic)]
    fn relaxer_bad() {
        // cargo test relaxer_bad -- --nocapture
        let visualize_filename = "relaxer_bad.json".to_string();
        let (decoding_graph, ..) = color_code_5_decoding_graph(vec![7, 1], visualize_filename);
        let invalid_subgraph = Arc::new(InvalidSubgraph::new_complete(
            vec![7].into_iter().collect(),
            FastIterSet::new(),
            decoding_graph.as_ref(),
        ));
        let relaxer: Relaxer = Relaxer::new([(invalid_subgraph, Rational::zero())].into());
        println!("relaxer: {relaxer:?}"); // should not print because it panics
    }

    #[test]
    fn relaxer_hash() {
        // cargo test relaxer_hash -- --nocapture
        let vertices: FastIterSet<VertexIndex> = [1, 2, 3].into();
        let edges: FastIterSet<EdgeIndex> = [4, 5].into();
        let hair: FastIterSet<EdgeIndex> = [6, 7, 8].into();
        let invalid_subgraph = InvalidSubgraph::new_raw(vertices.clone(), edges.clone(), hair.clone());
        let relaxer_1 = Relaxer::new([(Arc::new(invalid_subgraph.clone()), Rational::one())].into());
        let relaxer_2 = Relaxer::new([(Arc::new(invalid_subgraph), Rational::one())].into());
        assert_eq!(relaxer_1, relaxer_2);
        // they should have the same hash value
        assert_eq!(
            get_default_hash_value(&relaxer_1),
            get_default_hash_value(&relaxer_1.hash_value)
        );
        assert_eq!(get_default_hash_value(&relaxer_1), get_default_hash_value(&relaxer_2));
        // the pointer should also have the same hash value
        let ptr_1 = Arc::new(relaxer_1);
        let ptr_2 = Arc::new(relaxer_2);
        assert_eq!(get_default_hash_value(&ptr_1), get_default_hash_value(&ptr_1.hash_value));
        assert_eq!(get_default_hash_value(&ptr_1), get_default_hash_value(&ptr_2));
    }
}
