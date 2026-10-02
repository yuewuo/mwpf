use crate::heapz::{Heap, HeapType};
use std::hash::Hash;

type BoxedNode<K, V> = Box<Node<K, V>>;

#[derive(Debug)]
struct Node<K, V: PartialOrd> {
    pub value: V,
    pub key: K,
    left: Option<BoxedNode<K, V>>,
    next: Option<BoxedNode<K, V>>,
}

impl<K, V: PartialOrd> Node<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Node {
            key,
            value,
            left: None,
            next: None,
        }
    }
    pub fn set_left(&mut self, node: Option<BoxedNode<K, V>>) {
        self.left = node;
    }
    pub fn set_next(&mut self, node: Option<BoxedNode<K, V>>) {
        self.next = node;
    }
}

/**
[`PairingHeap`] is an implementation of a [pairing heap](https://en.wikipedia.org/wiki/Pairing_heap).

It can have either a min or max [`HeapType`] and is implemented using a pattern similar to [singly linked lists](https://en.wikipedia.org/wiki/Linked_list#Singly_linked_list)
 */
pub struct PairingHeap<K, V: PartialOrd> {
    root: Option<BoxedNode<K, V>>,
    heap_type: HeapType,
    size: usize,
}

impl<K, V: PartialOrd> PairingHeap<K, V> {
    /// Initializes a min priority ([`HeapType::Min`]) [`PairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::PairingHeap;
    ///
    /// let heap: PairingHeap<(usize, usize), i32> = PairingHeap::min();
    /// ```
    pub fn min() -> Self {
        Self::new(HeapType::Min)
    }

    /// Initializes a max priority ([`HeapType::Max`]) [`PairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::PairingHeap;
    ///
    /// let heap: PairingHeap<(usize, usize), i32> = PairingHeap::max();
    /// ```
    pub fn max() -> Self {
        Self::new(HeapType::Max)
    }

    fn new(heap_type: HeapType) -> Self {
        PairingHeap {
            root: None,
            heap_type,
            size: 0,
        }
    }

    fn compare(&self, a: &BoxedNode<K, V>, b: &BoxedNode<K, V>) -> bool {
        match self.heap_type {
            HeapType::Max => a.value >= b.value,
            HeapType::Min => a.value <= b.value,
        }
    }

    fn add_child(mut parent: BoxedNode<K, V>, mut child: BoxedNode<K, V>) -> BoxedNode<K, V> {
        if parent.left.is_some() {
            child.set_next(parent.left.take());
        }
        parent.set_left(Some(child));
        parent
    }

    fn merge(
        &mut self,
        node_a: Option<BoxedNode<K, V>>,
        node_b: Option<BoxedNode<K, V>>,
    ) -> Option<BoxedNode<K, V>> {
        match (node_a, node_b) {
            (Some(a), Some(b)) => Some(if self.compare(&a, &b) {
                Self::add_child(a, b)
            } else {
                Self::add_child(b, a)
            }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            _ => None,
        }
    }

    fn two_pass_merge(&mut self, node: Option<BoxedNode<K, V>>) -> Option<BoxedNode<K, V>> {
        let mut root = node;
        let mut merged: Option<BoxedNode<K, V>> = None;

        while let Some(mut parent) = root {
            if let Some(mut child) = parent.next.take() {
                root = child.next.take();
                let children = self.merge(Some(parent), Some(child));
                merged = self.merge(merged, children);
            } else {
                merged = self.merge(merged, Some(parent));
                root = None;
            }
        }
        merged
    }
}

impl<K: Hash + Eq, V: PartialOrd> Heap<K, V> for PairingHeap<K, V> {
    /// Indicates whether a [`PairingHeap`] is empty or not
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let mut heap = PairingHeap::min();
    ///
    /// assert_eq!(heap.is_empty(), true);
    ///
    /// heap.push("Hello".to_string(), 5);
    ///
    /// assert_eq!(heap.is_empty(), false);
    /// ```
    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Returns the amount of elements in the [`PairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let mut heap = PairingHeap::max();
    ///
    /// assert_eq!(heap.size(), 0);
    ///
    /// heap.push("Hello".to_string(), 5);
    ///
    /// assert_eq!(heap.size(), 1);
    /// ```
    fn size(&self) -> usize {
        self.size.clone()
    }

    /// Adds an element to the [`PairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let mut heap = PairingHeap::min();
    /// let value = "Hello".to_string();
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top(), Some(&value));
    /// ```
    fn push(&mut self, key: K, value: V) {
        self.root = if self.root.is_some() {
            let root = self.root.take();
            self.merge(root, Some(Box::new(Node::new(key, value))))
        } else {
            Some(Box::new(Node::new(key, value)))
        };
        self.size += 1;
    }

    /// Returns the highest priority element of a [`PairingHeap`] (or None)
    ///
    /// ```
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let value = "Hello".to_string();
    /// let mut heap = PairingHeap::max();
    ///
    /// assert!(heap.top().is_none());
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top(), Some(&value));
    /// ```
    fn top(&self) -> Option<&K> {
        self.root.as_ref().map(|node| &node.key)
    }

    /// Returns the highest priority element of a [`PairingHeap`] (or None) as mutable
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let value = "Hello".to_string();
    /// let mut heap = PairingHeap::min();
    ///
    /// assert!(heap.top_mut().is_none());
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top_mut(), Some(&mut value.clone()));
    /// ```
    fn top_mut(&mut self) -> Option<&mut K> {
        self.root.as_mut().map(|node| &mut node.key)
    }

    /// Removes and Returns the highest priority element of a [`PairingHeap`] (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{PairingHeap, Heap};
    ///
    /// let value1 = "Hello".to_string();
    /// let value2 = "World".to_string();
    /// let mut heap = PairingHeap::max();
    ///
    /// heap.push(value1.clone(), 5);
    /// heap.push(value2.clone(), 4);
    ///
    /// assert_eq!(heap.pop(), Some(value1.clone()));
    /// assert_eq!(heap.pop(), Some(value2.clone()));
    /// assert_eq!(heap.pop(), None);
    /// ```
    fn pop(&mut self) -> Option<K> {
        self.root.take().map(|mut node| {
            self.size -= 1;
            self.root = self.two_pass_merge(node.left.take());
            node.key
        })
    }
}
