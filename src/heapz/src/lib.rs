#![deny(missing_docs)]
#![deny(rustdoc::missing_doc_code_examples)]

/*!
A collection of heap/priority queue implementations.

### Heap types that have been implemented
 - [Pairing Heap](https://en.wikipedia.org/wiki/Pairing_heap)
 - [Rank Paring Heap](https://skycocoo.github.io/Rank-Pairing-Heap/)
*/

mod utils;
use std::hash::Hash;

mod pairing_heap;
mod rank_pairing_heap;

pub use pairing_heap::*;
pub use rank_pairing_heap::*;

/// [`HeapType`] Represents whether a heap/queue is min ([`HeapType::Min`]) or max ([`HeapType::Max`]) priority
#[derive(PartialEq, Copy, Clone, Debug)]
enum HeapType {
    /// represents a heap type which prioritizes elements with the maximum value
    Max,
    /// represents a heap type which prioritizes elements with the minimum value
    Min,
}

/// [`Heap`] contains all the methods common to heaps/queues
pub trait Heap<K, V>
where
    K: Hash + Eq,
    V: PartialOrd,
{
    /// Indicates whether a [`Heap`] is empty or not
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     assert_eq!(heap.is_empty(), true);
    ///
    ///     heap.push("Hello".to_string(), 5);
    ///
    ///     assert_eq!(heap.is_empty(), false);
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn is_empty(&self) -> bool;

    /// Returns the amount of elements in the [`Heap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     assert_eq!(heap.size(), 0);
    ///
    ///     heap.push("Hello".to_string(), 5);
    ///
    ///     assert_eq!(heap.size(), 1);
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn size(&self) -> usize;

    /// Adds an element to the [`Heap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     let value = "Hello".to_string();
    ///
    ///     heap.push(value.clone(), 5);
    ///
    ///     assert_eq!(heap.top(), Some(&value));
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn push(&mut self, key: K, value: V);

    /// Returns the highest priority element of a [`Heap`] (or None)
    ///
    /// ```
    /// use mwpf::heapz::{PairingHeap, Heap};
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     let value = "Hello".to_string();
    ///
    ///     assert!(heap.top().is_none());
    ///
    ///     heap.push(value.clone(), 5);
    ///
    ///     assert_eq!(heap.top(), Some(&value));
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn top(&self) -> Option<&K>;

    /// Returns the highest priority element of a [`Heap`] (or None) as mutable
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     let value = "Hello".to_string();
    ///
    ///     assert!(heap.top_mut().is_none());
    ///
    ///     heap.push(value.clone(), 5);
    ///
    ///     assert_eq!(heap.top_mut(), Some(&mut value.clone()));
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn top_mut(&mut self) -> Option<&mut K>;

    /// Removes and Returns the highest priority element of a [`Heap`] (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// fn check_heap<T: Heap<String, u8>>(mut heap: T) {
    ///
    ///     let value = "Hello".to_string();
    ///
    ///     heap.push(value.clone(), 5);
    ///
    ///     assert_eq!(heap.pop(), Some(value.clone()));
    ///     assert_eq!(heap.pop(), None);
    /// }
    ///
    /// check_heap(PairingHeap::min());
    /// ```
    fn pop(&mut self) -> Option<K>;
}

/// [`DecreaseKey`] defines extra methods for a [`Heap`] that implement decrease-key and delete operations
pub trait DecreaseKey<K, V>: Heap<K, V>
where
    K: Hash + Eq,
    V: PartialOrd,
{
    /// Updates the priority of an element in the [`Heap`] (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{DecreaseKey, RankPairingHeap};
    ///
    /// fn check_heap<T: DecreaseKey<String, u8>>(mut heap: T) {
    ///     let hello = "Hello".to_string();
    ///     let world = "World".to_string();
    ///
    ///     heap.push(hello.clone(), 5);
    ///     heap.push(world.clone(), 2);
    ///
    ///     assert_eq!(heap.top(), Some(&world));
    ///
    ///     heap.update(&hello, 1);
    ///
    ///     assert_eq!(heap.top(), Some(&hello));
    /// }
    ///
    /// check_heap(RankPairingHeap::multi_pass_min2());
    /// ```
    fn update(&mut self, key: &K, value: V);

    ///  Deletes an element from the [`Heap`] and returns it (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{DecreaseKey, RankPairingHeap};
    ///
    /// fn check_heap<T: DecreaseKey<String, u8>>(mut  heap: T) {
    ///
    ///     let hello = "Hello".to_string();
    ///     let world = "World".to_string();
    ///
    ///     heap.push(hello.clone(), 5);
    ///     heap.push(world.clone(), 2);
    ///
    ///     assert_eq!(heap.top(), Some(&world));
    ///     assert_eq!(heap.delete(&hello), Some(hello.clone()));
    ///
    ///     heap.pop();
    ///
    ///     assert_eq!(heap.top(), None);
    ///     assert_eq!(heap.delete(&hello), None);
    /// }
    ///
    /// check_heap(RankPairingHeap::multi_pass_min2());
    /// ```
    fn delete(&mut self, key: &K) -> Option<K>;
}
