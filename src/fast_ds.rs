use std::mem;

use derivative::Derivative;
use hashbrown::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Map<K, V> {
    map: HashMap<K, (V, usize)>,
    iter: Vec<(K, V)>,
}

impl<K: Eq + std::hash::Hash + Clone, V: Clone> Map<K, V> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            iter: Vec::new(),
        }
    }

    pub fn insert(&mut self, key: K, mut value: V) -> Option<V> {
        // Note: what is faster, more cloning or entry?
        let mut previous = None;
        if let Some((curr_val, curr_pos)) = self.map.get_mut(&key) {
            // changing value of in existing iter
            self.iter[*curr_pos].1 = value.clone();
            // update value and prepare for insert api return
            std::mem::swap(&mut value, curr_val);

            // now value is to be returned
            previous = Some(value);
        } else {
            // adding to the end of iter, and add the position to map
            self.map.insert(key.clone(), (value.clone(), self.iter.len()));
            self.iter.push((key, value));
        }

        previous
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some((_, index)) = self.map.remove(key) {
            let last = self.iter.len() - 1;
            let (_, removed_value) = self.iter.swap_remove(index);

            // If we didn't remove the last element, update the moved element's index in the map
            if index != last {
                let (moved_key, _) = &self.iter[index];
                self.map.get_mut(moved_key).unwrap().1 = index;
            }

            Some(removed_value)
        } else {
            None
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.iter.clear();
    }
}
