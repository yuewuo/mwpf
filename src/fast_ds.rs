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

/// The guard will implement Deref/DerefMut so it can be used like a &mut V
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

/// On drop, recalc new hash and update the map’s combined_hash
impl<'a, K: Hash + Clone + Eq, V: Hash> Drop for MutValueGuard<'a, K, V> {
    fn drop(&mut self) {
        let new_hash = Map::<K, V>::compute_hash(self.key, self.value);
        *self.hash = self.hash.wrapping_add(new_hash);
    }
}

impl<K: Eq + Hash + Clone, V: Hash> Map<K, V> {
    /// Creates a new empty map
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            combined_hash: 0,
        }
    }

    /// Computes the hash of a key-value pair
    fn compute_hash(key: &K, value: &V) -> u64 {
        let mut hasher = crate::util::DefaultHasher::default();
        key.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Inserts a key-value pair into the map, returning the old value if it exists
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

    /// Removes a key-value pair from the map, returning the value if it exists
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(old_val) = self.map.remove(key) {
            let old_hash = Self::compute_hash(key, &old_val);
            self.combined_hash = self.combined_hash.wrapping_sub(old_hash);
            Some(old_val)
        } else {
            None
        }
    }

    /// Checks if the map contains a key
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// clear
    #[inline]
    pub fn clear(&mut self) {
        self.map.clear();
        self.combined_hash = 0;
    }

    /// iter
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }

    /// len
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// is_empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// get
    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// The key method: returns a "guard" instead of `&mut V`.
    /// When that guard is dropped, it will populate the hash with the re-hashed new value
    pub fn get_mut<'a>(&'a mut self, key: &'a K) -> Option<MutValueGuard<'a, K, V>> {
        return if let Some(value) = self.map.get_mut(key) {
            let old_hash = Self::compute_hash(key, value);
            self.combined_hash = self.combined_hash.wrapping_sub(old_hash);
            Some(MutValueGuard {
                key,
                value,
                hash: &mut self.combined_hash,
            })
        } else {
            None
        };
    }

    /// Get keys in iterator form
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    /// Get combined hash value
    #[inline]
    pub fn combined_hash(&self) -> u64 {
        self.combined_hash
    }
}

// implement extend for owned values
impl<K: Eq + Hash + Clone, V: Hash> Extend<(K, V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            // consistent with std implementation
            (into_iter.size_hint().0 + 1) / 2
        };
        self.map.reserve(reserve);
        into_iter.for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }
}

// implement extend for references
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

impl<K: Eq + Hash + Clone, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.combined_hash.hash(state);
    }
}

impl<K: Eq + Hash + Clone, V: Default> IntoIterator for Map<K, V> {
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

impl<K: Eq + Hash + Clone, V: Hash> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone, V: PartialEq + Hash> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.combined_hash != other.combined_hash {
            return false;
        }
        self.map == other.map
    }
}

impl<K: Ord + Hash + Clone, V: Ord + Hash> PartialOrd for Map<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Eq + Hash + Clone, V: PartialEq + Hash> Eq for Map<K, V> {}

impl<K: Ord + Hash + Clone, V: Ord + Hash> Ord for Map<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        let order = self.combined_hash.cmp(&other.combined_hash);
        if !matches!(order, Ordering::Equal) {
            return order;
        }

        let self_sorted: BTreeMap<_, _> = self.map.iter().collect();
        let other_sorted: BTreeMap<_, _> = other.map.iter().collect();
        self_sorted.cmp(&other_sorted)
    }
}
impl<K: Eq + std::hash::Hash + Clone, V: Hash> std::ops::Index<&K> for Map<K, V> {
    type Output = V;

    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("Key not found in Map")
    }
}

impl<K: Eq + Hash + Clone, V: Hash, const N: usize> From<[(K, V); N]> for Map<K, V> {
    fn from(array: [(K, V); N]) -> Self {
        array.into_iter().collect()
    }
}

/// An enum representing either an occupied or vacant entry in the map, consisten with std
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
    /// Returns a reference to the key
    #[inline]
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    /// Replaces the value and returns the old value
    pub fn insert(&mut self, value: V) -> V {
        let key = self.entry.key();
        let old_value = self.entry.get();
        let old_hash = Map::<K, V>::compute_hash(key, old_value);
        let new_hash = Map::<K, V>::compute_hash(key, &value);

        *self.combined_hash = self.combined_hash.wrapping_sub(old_hash).wrapping_add(new_hash);

        self.entry.insert(value)
    }

    /// Removes the entry and returns the value
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

impl<K: Eq + Hash + Clone, V: Hash> Map<K, V> {
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

/* SET implementation */
/// A `Set<T>` that provides Ord and fast Hash
#[derive(Debug, Clone, Derivative)]
pub struct Set<T: Hash> {
    set: HashSet<T>,
    combined_hash: u64,
}

#[cfg(feature = "python_binding")]
impl<'py, T: Hash + Clone + Eq + IntoPyObject<'py>> IntoPyObject<'py> for Set<T> {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = std::convert::Infallible; // the conversion error type, has to be convertable to `PyErr`

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let set: std::collections::HashSet<T> = self.set.iter().cloned().collect();
        use crate::pyo3::IntoPyObjectExt;
        Ok(set.into_bound_py_any(py).unwrap())
    }
}

impl<T: Eq + Hash + Clone + Debug> Set<T> {
    /// Creates a new empty set
    pub fn new() -> Self {
        Self {
            set: HashSet::new(),
            combined_hash: 0,
        }
    }

    /// Computes the hash of a value
    pub fn compute_hash(value: &T) -> u64 {
        let mut hasher = crate::util::DefaultHasher::default();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Inserts an element, returning `true` if it was newly inserted
    pub fn insert(&mut self, value: T) -> bool {
        let hash = Self::compute_hash(&value);
        let inserted = self.set.insert(value);
        if inserted {
            self.combined_hash = self.combined_hash.wrapping_add(hash);
        }
        inserted
    }

    /// Removes an element, returning `true` if it was present
    pub fn remove(&mut self, value: &T) -> bool {
        let hash = Self::compute_hash(value);
        let removed = self.set.remove(value);
        if removed {
            self.combined_hash = self.combined_hash.wrapping_sub(hash);
        }
        removed
    }

    /// Checks if an element exists in the set
    #[inline]
    pub fn contains(&self, value: &T) -> bool {
        self.set.contains(value)
    }

    /// clear
    #[inline]
    pub fn clear(&mut self) {
        self.set.clear();
        self.combined_hash = 0;
    }

    /// iter
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.set.iter()
    }

    /// Appends elements from `other` into `self`, consuming `other`.
    pub fn append(&mut self, other: &mut Self) {
        self.set.extend(other.set.drain());
        self.combined_hash = self.combined_hash.wrapping_add(other.combined_hash);
        other.combined_hash = 0;
    }

    /// len
    #[inline]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// is_empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Checks if two sets have no elements in common
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.set.is_disjoint(&other.set)
    }

    /// Returns a new set containing only elements found in both sets
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a T> {
        self.set.intersection(&other.set)
    }
}

// implement extend
impl<T: Eq + Hash + Clone + Debug> Extend<T> for Set<T> {
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

// implement extend for references
impl<'a, T: Eq + Hash + Clone + Debug> Extend<&'a T> for Set<T> {
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

impl<T: Eq + Hash + Clone> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = hashbrown::hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.set.into_iter()
    }
}
impl<T: Eq + Hash + Clone + Debug> FromIterator<T> for Set<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Set::new();
        iter.into_iter().for_each(|x| {
            set.insert(x);
        });
        set
    }
}

// implement `PartialEq` and `Eq` for `Set<T>`
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
impl<T: Ord + Hash> Ord for Set<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.combined_hash != other.combined_hash {
            return self.combined_hash.cmp(&other.combined_hash); // ✅ Compare hash first
        }
        let self_sorted: BTreeSet<_> = self.set.iter().collect();
        let other_sorted: BTreeSet<_> = other.set.iter().collect();
        self_sorted.cmp(&other_sorted)
    }
}

impl<T: Eq + Hash + Clone + Debug> Default for Set<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Eq + Hash + Clone + Send> Send for Set<T> {}
unsafe impl<T: Eq + Hash + Clone + Sync> Sync for Set<T> {}

impl<T: Eq + Hash + Clone> Hash for Set<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.combined_hash.hash(state);
    }
}

impl<T: Eq + Hash + Clone + Debug, const N: usize> From<[T; N]> for Set<T> {
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
        // Inserting a new element should return true.
        assert!(set.insert(1));
        assert!(set.contains(&1));
        // Re-inserting the same element should return false.
        assert!(!set.insert(1));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_removal() {
        let mut set = Set::new();
        set.insert(2);
        set.insert(3);
        // Remove existing element.
        assert!(set.remove(&2));
        assert!(!set.contains(&2));
        assert_eq!(set.len(), 1);
        // Removing a non-existent element should return false.
        assert!(!set.remove(&2));
    }

    // #[test]
    fn _test_iteration_order() {
        let mut set = Set::new();
        set.insert(10);
        set.insert(20);
        set.insert(30);
        // Expect the iteration order to match insertion order.
        let elements: Vec<_> = set.iter().cloned().collect();
        assert_eq!(elements, vec![10, 20, 30]);
    }

    #[test]
    fn test_extend_and_append() {
        let mut set1 = Set::new();
        set1.insert(1);
        set1.insert(2);

        let mut set2 = Set::new();
        set2.insert(3);
        set2.insert(4);

        // Append set2 into set1.
        set1.append(&mut set2);
        assert_eq!(set1.len(), 4);
        assert!(set1.contains(&3));
        assert!(set1.contains(&4));
        // After appending, set2 should be empty.
        assert!(set2.is_empty());
    }

    #[test]
    fn test_intersection() {
        let mut set1 = Set::new();
        set1.insert(1);
        set1.insert(2);
        set1.insert(3);

        let mut set2 = Set::new();
        set2.insert(2);
        set2.insert(4);

        // The intersection should only contain the common element.
        let inter: Vec<_> = set1.intersection(&set2).cloned().collect();
        assert_eq!(inter, vec![2]);
    }

    #[test]
    fn test_into_iter() {
        let mut set = Set::new();
        set.insert(100);
        set.insert(200);
        // Collect the elements by consuming the set.
        let mut collected: Vec<_> = set.into_iter().collect();
        collected.sort();
        assert_eq!(collected, vec![100, 200]);
    }
}
