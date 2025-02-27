use std::{
    cell::UnsafeCell,
    cmp::Ordering,
    collections::{btree_map, btree_set, BTreeMap, BTreeSet},
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
};

use derivative::Derivative;
// use hashbrown::{BTreeMap, BTreeSet};

/* MAP implementation */
#[derive(Debug, Derivative)]
/// A `Map<K, V>` that provides Ord and fast Hash
pub struct Map<K, V> {
    map: BTreeMap<K, V>,
    combined_hash: UnsafeCell<u64>,
    dirty: UnsafeCell<bool>,
}

// implement Clone
impl<K: Eq + Hash + Clone, V: Hash + Clone> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            combined_hash: UnsafeCell::new(unsafe { *self.combined_hash.get() }),
            dirty: UnsafeCell::new(unsafe { *self.dirty.get() }),
        }
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash> Map<K, V> {
    /// Creates a new empty map
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            combined_hash: UnsafeCell::new(0),
            dirty: UnsafeCell::new(false),
        }
    }

    /// Update the combined hash if dirty
    pub fn update_hash(&self) {
        if unsafe { *self.dirty.get() } {
            let mut hash = 0u64;
            for (key, value) in self.map.iter() {
                hash = hash.wrapping_add(Self::compute_hash(key, value));
            }
            unsafe { *self.combined_hash.get() = hash };
            unsafe { *self.dirty.get() = false };
        }
    }

    /// Computes the hash of a key-value pair
    fn compute_hash(key: &K, value: &V) -> u64 {
        let mut hasher = DefaultHasher::default();
        key.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Inserts a key-value pair into the map, returning the old value if it exists
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = Self::compute_hash(&key, &value);
        match self.map.entry(key.clone()) {
            btree_map::Entry::Occupied(mut entry) => {
                let old_value = entry.get_mut();
                let old_hash = Self::compute_hash(&key, old_value);
                unsafe { *self.combined_hash.get() = (*self.combined_hash.get()).wrapping_sub(old_hash).wrapping_add(hash) };
                Some(std::mem::replace(old_value, value))
            }
            btree_map::Entry::Vacant(entry) => {
                unsafe { *self.combined_hash.get() = (*self.combined_hash.get()).wrapping_add(hash) };
                entry.insert(value);
                None
            }
        }
    }

    /// Removes a key-value pair from the map, returning the value if it exists
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(removed_val) = self.map.remove(key) {
            let removed_hash = Self::compute_hash(key, &removed_val);
            unsafe { *self.combined_hash.get() = (*self.combined_hash.get()).wrapping_sub(removed_hash) };
            Some(removed_val)
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
        self.combined_hash = UnsafeCell::new(0);
        self.dirty = UnsafeCell::new(false);
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

    /// Get mutable reference
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        unsafe { *self.dirty.get() = true };
        self.map.get_mut(key)
    }

    /// Get keys in iterator form
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    /// Get combined hash value
    #[inline]
    pub fn combined_hash(&self) -> u64 {
        unsafe { *self.combined_hash.get() }
    }
}

// implement extend for owned values
impl<K: Eq + Hash + Clone + Ord, V: Hash> Extend<(K, V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            // consistent with std implementation
            (into_iter.size_hint().0 + 1) / 2
        };
        // self.map.reserve(reserve);
        into_iter.for_each(move |(k, v)| {
            self.insert(k, v);
        });
    }
}

// implement extend for references
impl<'a, K: Eq + Hash + Clone + Ord, V: Hash + Clone> Extend<(&'a K, &'a V)> for Map<K, V> {
    fn extend<I: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        // self.map.reserve(reserve);
        into_iter.for_each(move |(k, v)| {
            self.insert(k.clone(), v.clone());
        });
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.update_hash();
        unsafe { *self.combined_hash.get() }.hash(state);
    }
}

impl<K: Eq + Hash + Clone, V: Default> IntoIterator for Map<K, V> {
    type Item = (K, V);
    type IntoIter = btree_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Map::new();
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone + Ord, V: PartialEq + Hash> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.update_hash();
        other.update_hash();
        if unsafe { *self.combined_hash.get() != *other.combined_hash.get() } {
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
impl<K: Eq + Hash + Clone + Ord, V: PartialEq + Hash> Eq for Map<K, V> {}

impl<K: Ord + Hash + Clone, V: Ord + Hash> Ord for Map<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.update_hash();
        other.update_hash();

        let order = unsafe { *self.combined_hash.get() }.cmp(&unsafe { *other.combined_hash.get() });
        if !matches!(order, Ordering::Equal) {
            return order;
        }

        let self_sorted: BTreeMap<_, _> = self.map.iter().collect();
        let other_sorted: BTreeMap<_, _> = other.map.iter().collect();
        self_sorted.cmp(&other_sorted)
    }
}
impl<K: Eq + std::hash::Hash + Clone + Ord, V: Hash> std::ops::Index<&K> for Map<K, V> {
    type Output = V;

    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("Key not found in Map")
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash, const N: usize> From<[(K, V); N]> for Map<K, V> {
    fn from(array: [(K, V); N]) -> Self {
        array.into_iter().collect()
    }
}

unsafe impl<K: Eq + Hash + Clone + Send, V: Send> Send for Map<K, V> {}
unsafe impl<K: Eq + Hash + Clone + Sync, V: Sync> Sync for Map<K, V> {}

/// An enum representing either an occupied or vacant entry in the map, consisten with std
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V> {
    entry: btree_map::OccupiedEntry<'a, K, V>,
    combined_hash: &'a mut u64,
}

pub struct VacantEntry<'a, K, V> {
    entry: btree_map::VacantEntry<'a, K, V>,
    combined_hash: &'a mut u64,
}

impl<'a, K: Eq + Hash + Clone + Ord, V: Hash> OccupiedEntry<'a, K, V> {
    /// Returns a reference to the key
    #[inline]
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    /// Returns a mutable reference to the value
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        self.entry.get_mut()
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

impl<'a, K: Eq + Hash + Clone + Ord, V: Hash> VacantEntry<'a, K, V> {
    pub fn insert(self, value: V) -> &'a mut V {
        let key = self.entry.key();
        let hash = Map::<K, V>::compute_hash(key, &value);
        *self.combined_hash = self.combined_hash.wrapping_add(hash);
        self.entry.insert(value)
    }
}

impl<K: Eq + Hash + Clone + Ord, V: Hash> Map<K, V> {
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        match self.map.entry(key) {
            btree_map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry {
                entry,
                combined_hash: unsafe { &mut *self.combined_hash.get() },
            }),
            btree_map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry {
                entry,
                combined_hash: unsafe { &mut *self.combined_hash.get() },
            }),
        }
    }
}

/* SET implementation */
/// A `Set<T>` that provides Ord and fast Hash
#[derive(Debug, Clone, Derivative)]
pub struct Set<T: Hash> {
    set: BTreeSet<T>,
    combined_hash: u64,
}

impl<T: Eq + Hash + Clone + Debug + Ord> Set<T> {
    /// Creates a new empty set
    pub fn new() -> Self {
        Self {
            set: BTreeSet::new(),
            combined_hash: 0,
        }
    }

    /// Computes the hash of a value
    pub fn compute_hash(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
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
        self.combined_hash = self.combined_hash.wrapping_add(other.combined_hash);
        self.set.append(&mut other.set);
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
impl<T: Eq + Hash + Clone + Debug + Ord> Extend<T> for Set<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        // self.set.reserve(reserve);
        into_iter.for_each(move |t| {
            self.insert(t);
        });
    }
}

// implement extend for references
impl<'a, T: Eq + Hash + Clone + Debug + Ord> Extend<&'a T> for Set<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let into_iter = iter.into_iter();
        let reserve = if self.is_empty() {
            into_iter.size_hint().0
        } else {
            (into_iter.size_hint().0 + 1) / 2
        };
        // self.set.reserve(reserve);
        into_iter.for_each(move |t| {
            self.insert(t.clone());
        });
    }
}

impl<T: Eq + Hash + Clone> IntoIterator for Set<T> {
    type Item = T;
    type IntoIter = btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.set.into_iter()
    }
}
impl<T: Eq + Hash + Clone + Debug + Ord> FromIterator<T> for Set<T> {
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

impl<T: Eq + Hash + Clone + Debug + Ord> Default for Set<T> {
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

impl<T: Eq + Hash + Clone + Debug + Ord, const N: usize> From<[T; N]> for Set<T> {
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
    fn test_iteration_order() {
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
