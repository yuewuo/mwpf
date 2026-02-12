use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    hash::{Hash, Hasher},
};

use derivative::Derivative;
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "python_binding")]
use pyo3::prelude::*;
use std::ops::{Deref, DerefMut};

/* MAP implementation */
#[derive(Derivative, Clone)]
#[derivative(Debug)]
/// A `Map<K, V>` that provides Ord and fast Hash
pub struct Map<K, V> {
    map: HashMap<K, V>,
    #[derivative(Debug = "ignore")]
    combined_hash: u64,
}

/// A "guard" that holds a mutable reference to `value` in the `Map` along with
/// the associated `key`. On drop, it will re-hash the new value.
pub struct MutValueGuard<'a, K: Hash + Clone + Eq, V: Hash> {
    key: &'a K,
    value: &'a mut V,
    hash: &'a mut u64,
}

impl<'a, K: Hash + Clone + Eq, V: Hash> Deref for MutValueGuard<'a, K, V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        self.value
    }
}
impl<'a, K: Hash + Clone + Eq, V: Hash> DerefMut for MutValueGuard<'a, K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}

impl<'a, K: Hash + Clone + Eq, V: Hash> Drop for MutValueGuard<'a, K, V> {
    fn drop(&mut self) {
        let new_hash = Map::<K, V>::compute_hash(self.key, self.value);
        *self.hash = self.hash.wrapping_add(new_hash);
    }
}

// Basic methods need no bounds
impl<K, V> Map<K, V> {
    /// Creates a new empty map
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            combined_hash: 0,
        }
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
        self.combined_hash = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    #[inline]
    pub fn combined_hash(&self) -> u64 {
        self.combined_hash
    }
}

// Methods that require Hash/Eq but NOT Clone
impl<K: Eq + Hash, V: Hash> Map<K, V> {
    /// Computes the hash of a key-value pair
    fn compute_hash(key: &K, value: &V) -> u64 {
        let mut hasher = crate::util::DefaultHasher::default();
        key.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(old_val) = self.map.remove(key) {
            let old_hash = Self::compute_hash(key, &old_val);
            self.combined_hash = self.combined_hash.wrapping_sub(old_hash);
            Some(old_val)
        } else {
            None
        }
    }
}

impl<K: Eq + Hash, V> Map<K, V> {
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }
}

// Methods that require Clone (for inserting keys)
impl<K: Eq + Hash + Clone, V: Hash> Map<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = Self::compute_hash(&key, &value);
        match self.map.entry(key.clone()) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => {
                let old_value = entry.get_mut();
                let old_hash = Self::compute_hash(&key, old_value);
                self.combined_hash = self.combined_hash.wrapping_sub(old_hash).wrapping_add(hash);
                Some(std::mem::replace(old_value, value))
            }
            hashbrown::hash_map::Entry::Vacant(entry) => {
                self.combined_hash = self.combined_hash.wrapping_add(hash);
                entry.insert(value);
                None
            }
        }
    }

    pub fn get_mut<'a>(&'a mut self, key: &'a K) -> Option<MutValueGuard<'a, K, V>> {
        if let Some(value) = self.map.get_mut(key) {
            let old_hash = Self::compute_hash(key, value);
            self.combined_hash = self.combined_hash.wrapping_sub(old_hash);
            Some(MutValueGuard {
                key,
                value,
                hash: &mut self.combined_hash,
            })
        } else {
            None
        }
    }
    
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        match self.map.entry(key) {
            hashbrown::hash_map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry {
                entry,
                combined_hash: &mut self.combined_hash,
            }),
            hashbrown::hash_map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry {
                entry,
                combined_hash: &mut self.combined_hash,
            }),
        }
    }
}

// Implement Extend
impl<K: Eq + Hash + Clone, V: Hash> Extend<(K, V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        self.map.reserve(reserve);
        into_iter.for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }
}

impl<K: Eq + Hash, V> std::ops::Index<&K> for Map<K, V> {
    type Output = V;

    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("Key not found in Map")
    }
}

// -----------------------------------------------------------------------------
// IntoIterator for References (allows `&map` and `&set` to be used in loops/cmp)
// -----------------------------------------------------------------------------

impl<'a, K, V> IntoIterator for &'a Map<K, V> {
    type Item = (&'a K, &'a V);
    // Use the iterator type from the underlying hashbrown map
    type IntoIter = hashbrown::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl<'a, T> IntoIterator for &'a Set<T> {
    type Item = &'a T;
    type IntoIter = hashbrown::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.set.iter()
    }
}

impl<'a, K: Eq + Hash + Clone, V: Hash + Clone> Extend<(&'a K, &'a V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        self.map.reserve(reserve);
        into_iter.for_each(move |(k, v)| {
            self.insert(k.clone(), v.clone());
        });
    }
}

// Standard Trait Impls
impl<K: Eq + Hash, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.combined_hash.hash(state);
    }
}

impl<K, V> IntoIterator for Map<K, V> {
    type Item = (K, V);
    type IntoIter = hashbrown::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<K: Eq + Hash + Clone, V: Hash> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Map::new();
        map.extend(iter);
        map
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash, V: PartialEq + Hash> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.combined_hash != other.combined_hash {
            return false;
        }
        self.map == other.map
    }
}

impl<K: Eq + Hash, V: Eq + Hash> Eq for Map<K, V> {}

// -----------------------------------------------------------------------------
// Ord and PartialOrd for Map
// -----------------------------------------------------------------------------

impl<K: Ord + Hash, V: Ord + Hash> PartialOrd for Map<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord + Hash, V: Ord + Hash> Ord for Map<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Fast Path: Compare combined hashes first.
        let order = self.combined_hash.cmp(&other.combined_hash);
        if order != Ordering::Equal {
            return order;
        }

        // 2. Slow Path: Deterministic comparison.
        // We collect references into a BTreeMap to sort by Key.
        // We use references (&K, &V) so we don't need K: Clone or V: Clone.
        let self_sorted: BTreeMap<&K, &V> = self.map.iter().collect();
        let other_sorted: BTreeMap<&K, &V> = other.map.iter().collect();
        
        self_sorted.cmp(&other_sorted)
    }
}

// Entry Implementations
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V> {
    entry: hashbrown::hash_map::OccupiedEntry<'a, K, V>,
    combined_hash: &'a mut u64,
}

pub struct VacantEntry<'a, K, V> {
    entry: hashbrown::hash_map::VacantEntry<'a, K, V>,
    combined_hash: &'a mut u64,
}

impl<'a, K: Eq + Hash + Clone, V: Hash> OccupiedEntry<'a, K, V> {
    #[inline]
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    pub fn insert(&mut self, value: V) -> V {
        let key = self.entry.key();
        let old_value = self.entry.get();
        let old_hash = Map::<K, V>::compute_hash(key, old_value);
        let new_hash = Map::<K, V>::compute_hash(key, &value);

        *self.combined_hash = self.combined_hash.wrapping_sub(old_hash).wrapping_add(new_hash);
        self.entry.insert(value)
    }

    pub fn remove(self) -> V {
        let key = self.entry.key().clone();
        let value = self.entry.remove();
        let removed_hash = Map::<K, V>::compute_hash(&key, &value);
        *self.combined_hash = self.combined_hash.wrapping_sub(removed_hash);
        value
    }
}

impl<'a, K: Eq + Hash + Clone, V: Hash> VacantEntry<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        let key = self.entry.key();
        let hash = Map::<K, V>::compute_hash(key, &value);
        *self.combined_hash = self.combined_hash.wrapping_add(hash);
        self.entry.insert(value)
    }
}

/* SET implementation */

/// A `Set<T>` that provides Ord and fast Hash
#[derive(Debug, Clone, Derivative)]
pub struct Set<T> {
    set: HashSet<T>,
    combined_hash: u64,
}

#[cfg(feature = "python_binding")]
impl<'py, T: Hash + Clone + Eq + IntoPyObject<'py>> IntoPyObject<'py> for Set<T> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let set: std::collections::HashSet<T> = self.set.iter().cloned().collect();
        use crate::pyo3::IntoPyObjectExt;
        Ok(set.into_bound_py_any(py).unwrap())
    }
}

// Base Implementation (No Bounds)
impl<T> Set<T> {
    /// Creates a new empty set
    pub fn new() -> Self {
        Self {
            set: HashSet::new(),
            combined_hash: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.set.clear();
        self.combined_hash = 0;
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.set.iter()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }
}

// Logic Implementation (Requires Hash + Eq, but NO Clone/Debug needed for basic ops)
impl<T: Eq + Hash> Set<T> {
    pub fn compute_hash(value: &T) -> u64 {
        let mut hasher = crate::util::DefaultHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    pub fn insert(&mut self, value: T) -> bool {
        let hash = Self::compute_hash(&value);
        let inserted = self.set.insert(value);
        if inserted {
            self.combined_hash = self.combined_hash.wrapping_add(hash);
        }
        inserted
    }

    pub fn remove(&mut self, value: &T) -> bool {
        let hash = Self::compute_hash(value);
        let removed = self.set.remove(value);
        if removed {
            self.combined_hash = self.combined_hash.wrapping_sub(hash);
        }
        removed
    }

    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.set.contains(value)
    }

    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.set.is_disjoint(&other.set)
    }

    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.set.intersection(&other.set)
    }
    
    // Only append requires mutable access to other's hash, keeping it here is fine
    pub fn append(&mut self, other: &mut Self) {
        // Note: drain() requires T to be moved, so no extra bounds needed
        self.set.extend(other.set.drain());
        self.combined_hash = self.combined_hash.wrapping_add(other.combined_hash);
        other.combined_hash = 0;
    }
}

// Extend
impl<T: Eq + Hash> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        self.set.reserve(reserve);
        into_iter.for_each(move |t| {
            self.insert(t);
        });
    }
}

// Extend for References (Requires Clone)
impl<'a, T: Eq + Hash + Clone> Extend<&'a T> for Set<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        self.set.reserve(reserve);
        into_iter.for_each(move |t| {
            self.insert(t.clone());
        });
    }
}

impl<T> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = hashbrown::hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.set.into_iter()
    }
}

// Removed "Debug" and "Clone" requirements from FromIterator
impl<T: Eq + Hash> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Set::new();
        set.extend(iter);
        set
    }
}

impl<T: Eq + Hash> PartialEq for Set<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.combined_hash != other.combined_hash {
            return false;
        }
        self.set == other.set
    }
}
impl<T: Eq + Hash> Eq for Set<T> {}

impl<T: Ord + Hash> PartialOrd for Set<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Ord requires BTreeSet collection, so T must be Clone (to be collected) or we must iterate refs.
// Standard trick: collect references to sort.
impl<T: Ord + Hash> Ord for Set<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.combined_hash != other.combined_hash {
            return self.combined_hash.cmp(&other.combined_hash); 
        }
        // Collect references to avoid requiring T: Clone
        let self_sorted: BTreeSet<_> = self.set.iter().collect();
        let other_sorted: BTreeSet<_> = other.set.iter().collect();
        self_sorted.cmp(&other_sorted)
    }
}

impl<T> Default for Set<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Corrected Send/Sync: We don't need Hash/Eq to move the Set between threads.
// We only need T to be Send/Sync.
unsafe impl<T: Send> Send for Set<T> {}
unsafe impl<T: Sync> Sync for Set<T> {}

impl<T: Eq + Hash> Hash for Set<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.combined_hash.hash(state);
    }
}

impl<T: Eq + Hash + Clone, const N: usize> From<[T; N]> for Set<T> {
    fn from(array: [T; N]) -> Self {
        array.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_contains() {
        let mut set = Set::new();
        assert!(set.insert(1));
        assert!(set.contains(&1));
        assert!(!set.insert(1));
        assert_eq!(set.len(), 1);
    }
}