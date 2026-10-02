use crate::heapz::utils::Bucket;
use crate::heapz::{DecreaseKey, Heap, HeapType};
use std::{
    cmp::{max, Eq},
    collections::HashMap,
    hash::Hash,
};

/**!
[`HeapRank`] represents which algorithm will be used to calculate the rank of a node/tree
*/
#[derive(PartialEq, Clone, Debug)]
enum HeapRank {
    /// [`HeapRank::One`] has larger constant factors in the time bounds than [`HeapRank::Two`] but is simpler
    One,
    /// [`HeapRank::Two`] has smaller constant factors in the time bounds than [`HeapRank::One`]
    Two,
}

/**!
[`HeapPasses`] represent how many passes will be made when restructuring a [`RankPairingHeap`]

[Rank pairing heaps]() use a list of trees that can be combined if they have identical size (rank).
Combining all trees of identical size (rank) takes multiple passes but is not required for the [`RankPairingHeap`] to work.
*/
#[derive(PartialEq, Clone, Debug)]
enum HeapPasses {
    /// A single pass will cause the heap to restructure the heap lazily, only iterating over each node a single time and combining any nodes with matching size/ranks.
    Single,

    /// Multiple passes restructure the heap eagerly, merging trees repeatedly until no two trees have matching size/rank.
    Multi,
}

type Position = Option<usize>;

#[derive(Clone, Debug)]
struct Node<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> {
    key: K,
    value: V,
    left: Position,
    next: Position,
    parent: Position,
    rank: usize,
    root: bool,
}

impl<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> Node<K, V> {
    pub fn new(key: K, value: V) -> Self {
        Node {
            key,
            value,
            left: None,
            next: None,
            parent: None,
            rank: 0,
            root: true,
        }
    }
}

/**
[`RankPairingHeap`] is an implementation of a [rank pairing heap](https://skycocoo.github.io/Rank-Pairing-Heap/)

Due to the [difficulty](https://rcoh.me/posts/rust-linked-list-basically-impossible/) in creating [doubly linked lists](https://en.wikipedia.org/wiki/Doubly_linked_list) using safe rust, this [rank pairing heap](https://skycocoo.github.io/Rank-Pairing-Heap/) implementation uses an array to store nodes and uses their indices as pointers.

[rank pairing heaps](https://skycocoo.github.io/Rank-Pairing-Heap/) have a few variations on how their ranks are calculated, how the heap is restructured and the order in which priority is determined.
To address these different options there are three properties that can be set in any combination for the [`RankPairingHeap`]: [`HeapType`], [`HeapRank`] and [`HeapPasses`]
 */
pub struct RankPairingHeap<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> {
    root: Position,
    heap_rank: HeapRank,
    heap_type: HeapType,
    passes: HeapPasses,
    list: Vec<Node<K, V>>,
    keys: HashMap<K, Position>,
}

// impelement clone
impl<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> Clone for RankPairingHeap<K, V> {
    fn clone(&self) -> Self {
        RankPairingHeap {
            root: self.root,
            heap_rank: self.heap_rank.clone(),
            heap_type: self.heap_type,
            passes: self.passes.clone(),
            list: self.list.clone(),
            keys: self.keys.clone(),
        }
    }
}

// implement Debug
impl<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> std::fmt::Debug
    for RankPairingHeap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankPairingHeap")
            .field("root", &self.root)
            .field("heap_rank", &self.heap_rank)
            .field("heap_type", &self.heap_type)
            .field("passes", &self.passes)
            .field("list", &self.list)
            .field("keys", &self.keys)
            .finish()
    }
}

// struct initialization
impl<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> RankPairingHeap<K, V> {
    fn new(heap_type: HeapType, heap_rank: HeapRank, passes: HeapPasses) -> Self {
        RankPairingHeap {
            root: None,
            heap_rank,
            heap_type,
            passes,
            list: vec![],
            keys: HashMap::new(),
        }
    }

    /// Initializes a max ([`HeapType::Max`]) heap using [`HeapRank::One`] and [`HeapPasses::Single`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::single_pass_max();
    /// ```
    pub fn single_pass_max() -> Self {
        Self::new(HeapType::Max, HeapRank::One, HeapPasses::Single)
    }

    /// Initializes a max ([`HeapType::Max`]) heap using [`HeapRank::Two`] and [`HeapPasses::Single`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::single_pass_max2();
    /// ```
    pub fn single_pass_max2() -> Self {
        Self::new(HeapType::Max, HeapRank::Two, HeapPasses::Single)
    }

    /// Initializes a min ([`HeapType::Min`]) heap using [`HeapRank::One`] and [`HeapPasses::Single`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::single_pass_min();
    /// ```
    pub fn single_pass_min() -> Self {
        Self::new(HeapType::Min, HeapRank::One, HeapPasses::Single)
    }

    /// Initializes a min ([`HeapType::Min`]) heap using [`HeapRank::Two`] and [`HeapPasses::Single`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::single_pass_min2();
    /// ```
    pub fn single_pass_min2() -> Self {
        Self::new(HeapType::Min, HeapRank::Two, HeapPasses::Single)
    }

    /// Initializes a min ([`HeapType::Max`]) heap using [`HeapRank::One`] and [`HeapPasses::Multi`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::multi_pass_max();
    /// ```
    pub fn multi_pass_max() -> Self {
        Self::new(HeapType::Max, HeapRank::One, HeapPasses::Multi)
    }

    /// Initializes a min ([`HeapType::Max`]) heap using [`HeapRank::Two`] and [`HeapPasses::Multi`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::multi_pass_max2();
    /// ```
    pub fn multi_pass_max2() -> Self {
        Self::new(HeapType::Max, HeapRank::Two, HeapPasses::Multi)
    }

    /// Initializes a min ([`HeapType::Min`]) heap using [`HeapRank::One`] and [`HeapPasses::Multi`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::multi_pass_min();
    /// ```
    pub fn multi_pass_min() -> Self {
        Self::new(HeapType::Min, HeapRank::One, HeapPasses::Multi)
    }

    /// Initializes a min ([`HeapType::Min`]) heap using [`HeapRank::Two`] and [`HeapPasses::Multi`]
    ///
    /// ```rust
    /// use mwpf::heapz::RankPairingHeap;
    ///
    /// let heap: RankPairingHeap<(usize, usize), i32> = RankPairingHeap::multi_pass_max2();
    /// ```
    pub fn multi_pass_min2() -> Self {
        Self::new(HeapType::Min, HeapRank::Two, HeapPasses::Multi)
    }
}

// Ranking
#[allow(dead_code)]
impl<K, V> RankPairingHeap<K, V>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: PartialOrd + Clone + std::fmt::Debug,
{
    fn rank1(left: i32, next: i32) -> i32 {
        if left != next {
            max(left, next)
        } else {
            left + 1
        }
    }

    fn rank2(left: i32, next: i32) -> i32 {
        max(left, next) + (if (&left as &i32 - &next as &i32).abs() > 1 { 0 } else { 1 })
    }

    fn rank(&self, left: i32, next: i32) -> usize {
        (match self.heap_rank {
            HeapRank::One => Self::rank1(left, next),
            HeapRank::Two => Self::rank2(left, next),
        }) as usize
    }

    fn rank_nodes(&self, left: Position, next: Position) -> usize {
        let left_rank = self.get_rank(left);
        let right_rank = self.get_rank(next);
        self.rank(left_rank, right_rank)
    }

    fn get_rank(&self, position: Position) -> i32 {
        if let Some(n) = self.get_node(position) {
            n.rank as i32
        } else {
            0 - 1
        }
    }
}

// storage interaction
impl<K, V> RankPairingHeap<K, V>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: PartialOrd + Clone + std::fmt::Debug,
{
    fn get_node(&self, position: Position) -> Option<&Node<K, V>> {
        position.map(|index| self.list.get(index)).unwrap_or(None)
    }

    fn get_node_mut(&mut self, position: Position) -> Option<&mut Node<K, V>> {
        if let Some(index) = position {
            self.list.get_mut(index)
        } else {
            None
        }
    }

    fn remove_array_node(&mut self, position: Position) -> Option<Node<K, V>> {
        self.get_node(self.last_position()).map(|node| node.key.clone()).map(|key| {
            self.keys.remove(&key);
            self.keys.insert(key, position);
        });
        position.map(|index| self.list.swap_remove(index))
    }

    fn add_node(&mut self, node: Node<K, V>) -> Position {
        let position = Some(self.list.len());
        self.keys.insert(node.key.clone(), position);
        self.list.push(node);
        position
    }

    fn get_position(&self, key: &K) -> Position {
        self.keys.get(key).cloned().unwrap_or(None)
    }
}

// utility functions
#[allow(dead_code)]
impl<K: Hash + Eq + Clone + std::fmt::Debug, V: PartialOrd + Clone + std::fmt::Debug> RankPairingHeap<K, V> {
    fn last_position(&self) -> Position {
        let size = self.size();
        if size > 0 {
            Some(size - 1)
        } else {
            None
        }
    }

    fn is_left(&self, position: Position, parent: Position) -> bool {
        self.get_node(parent).map(|parent| parent.left == position).unwrap_or(false)
    }

    fn is_root(&self, position: Position) -> bool {
        self.get_node(position).map(|node| node.root).unwrap_or(false)
    }

    fn get_value(&self, position: Position) -> Option<&V> {
        self.get_node(position).map(|node| &node.value)
    }

    fn get_key(&self, position: Position) -> Option<&K> {
        self.get_node(position).map(|node| &node.key)
    }

    fn get_index<F: Fn(&Node<K, V>) -> Position>(&self, index: Position, get_adjacent: F) -> Position {
        self.get_node(index).map(get_adjacent).unwrap_or(None)
    }

    fn get_left_index(&self, index: Position) -> Position {
        self.get_index(index, |node| node.left)
    }

    fn get_next_index(&self, index: Position) -> Position {
        self.get_index(index, |node| node.next)
    }

    fn get_parent_index(&self, index: Position) -> Position {
        self.get_index(index, |node| node.parent)
    }

    fn get_links(&self, position: Position) -> Option<(Position, Position, Position)> {
        self.get_node(position).map(|node| (node.parent, node.left, node.next))
    }

    fn get_siblings(&self, position: Position) -> Option<(Position, Position)> {
        self.get_links(position).map(|(parent, _, next)| (parent, next))
    }

    fn set_next(&mut self, parent: Position, next: Position) {
        self.get_node_mut(parent).map(|node| {
            node.next = next;
        });
    }

    fn set_left(&mut self, parent: Position, left: Position) {
        self.get_node_mut(parent).map(|node| {
            node.left = left;
        });
    }

    fn set_parent(&mut self, child: Position, parent: Position) {
        self.get_node_mut(child).map(|node| {
            node.parent = parent;
        });
    }

    fn link_next(&mut self, parent: Position, next: Position) {
        self.set_next(parent, next);
        self.set_parent(next, parent);
    }

    fn link_left(&mut self, parent: Position, left: Position) {
        self.set_left(parent, left);
        self.set_parent(left, parent);
    }

    fn compare_values<T: PartialOrd>(&self, value_a: T, value_b: T) -> bool {
        if self.heap_type == HeapType::Max {
            value_a > value_b
        } else {
            value_a < value_b
        }
    }

    fn compare(&self, a: Position, b: Position) -> bool {
        self.get_value(a)
            .zip(self.get_value(b))
            .map_or(false, |(value_a, value_b)| self.compare_values(value_a, value_b))
    }

    fn merge_trees(&mut self, node_a: Position, node_b: Position) -> Position {
        assert_ne!(node_a, node_b);
        let a = self.get_node_mut(node_a).unwrap() as *mut Node<K, V>;
        let b = self.get_node_mut(node_b).unwrap() as *mut Node<K, V>;
        let parent: Position;
        let child: Position;
        unsafe {
            let parent_node: *mut Node<K, V>;
            let child_node: *mut Node<K, V>;
            let node_a_is_parent = if self.heap_type == HeapType::Max {
                (*a).value > (*b).value
            } else {
                (*a).value < (*b).value
            };
            if node_a_is_parent {
                parent = node_a;
                child = node_b;
                parent_node = a;
                child_node = b;
            } else {
                parent = node_b;
                child = node_a;
                parent_node = b;
                child_node = a;
            }
            let left_of_parent = (*parent_node).left;
            (*parent_node).left = child;
            (*parent_node).rank = (*child_node).rank + 1;
            (*child_node).parent = parent;
            (*child_node).next = left_of_parent;
            (*child_node).root = false;
            self.set_parent(left_of_parent, child);
        }
        parent
    }

    fn link(&mut self, node_a: Position, node_b: Position) -> Position {
        if node_b != node_a {
            match (node_a, node_b) {
                (Some(_), Some(_)) => self.merge_trees(node_a, node_b),
                (Some(_), None) => node_a,
                (None, Some(_)) => node_b,
                _ => None,
            }
        } else {
            node_a.or(node_b)
        }
    }

    fn calculate_swapped_positions(position: Position, parent: Position, next: Position, removed: Position) -> Position {
        if parent == position {
            if next == position {
                position
            } else {
                removed
            }
        } else {
            parent
        }
    }

    fn swap_remove_with_tree(&mut self, position: Position) -> Option<Node<K, V>> {
        let last = self.last_position();
        self.get_links(last)
            .map(|(parent_of_last, left_of_last, next_of_last)| {
                self.remove_array_node(position).map(|removed| {
                    if removed.next != position {
                        self.link_next(removed.parent, removed.next);
                        if last != position {
                            let parent =
                                Self::calculate_swapped_positions(position, parent_of_last, next_of_last, removed.parent);
                            let next =
                                Self::calculate_swapped_positions(position, next_of_last, parent_of_last, removed.next);
                            self.get_node_mut(position).map(|node| {
                                node.parent = parent;
                                node.next = next;
                                node.left = left_of_last;
                            });
                            self.set_next(parent, position);
                            vec![next, left_of_last]
                                .into_iter()
                                .for_each(|sibling| self.set_parent(sibling, position));
                        } else {
                            self.link_left(position, left_of_last);
                        }
                    }
                    removed
                })
            })
            .unwrap_or(None)
    }

    fn get_next_root(&mut self, position: Position) -> Position {
        let last = self.last_position();
        if let Some((linked_to_self, next)) = self.get_node(position).map(|node| (node.next == position, node.next)) {
            if linked_to_self {
                None
            } else if next == last {
                position
            } else {
                next
            }
        } else {
            None
        }
    }

    fn swap_remove_with_branch(&mut self, position: Position) -> Option<Node<K, V>> {
        let last = self.last_position();
        self.get_links(last)
            .map(|(parent, left, next)| {
                let is_left = self.is_left(last, parent);
                self.remove_array_node(position).map(|mut removed| {
                    self.link_next(removed.parent, removed.next);
                    let parent_of_last = if removed.left == last {
                        removed.left = position;
                        last
                    } else {
                        parent
                    };
                    self.get_node_mut(position).map(|node| {
                        node.left = left;
                        node.next = next;
                        node.parent = parent_of_last;
                    });
                    self.set_parent(left, position);
                    self.set_parent(next, position);
                    self.get_node_mut(parent_of_last).map(|node| {
                        if is_left {
                            node.left = position;
                        } else {
                            node.next = position;
                        }
                    });
                    removed
                })
            })
            .unwrap_or(None)
    }

    fn remove(&mut self, position: Position) -> Option<Node<K, V>> {
        if self.is_root(self.last_position()) {
            self.swap_remove_with_tree(position)
        } else {
            self.swap_remove_with_branch(position)
        }
    }

    fn single_pass(&mut self, mut node: Position) -> Position {
        let mut bucket = Bucket::new(self.size());
        let mut root = None;
        while node.is_some() {
            let (rank, next, parent) = self
                .get_node_mut(node)
                .map(|n| {
                    let parent = n.parent;
                    let next = n.next;
                    n.parent = None;
                    n.next = None;
                    (n.rank as usize, next, parent)
                })
                .unwrap();
            self.link_next(parent, next);
            if let Some(matched) = bucket.remove(rank) {
                let linked = self.link(node, matched);
                root = self.add_root_to_list(linked, root);
            } else {
                bucket.insert(rank, node);
            }
            node = next;
        }
        bucket.drain().fold(root, |list, node| self.add_root_to_list(node, list))
    }

    fn multi_pass(&mut self, mut node: Position) -> Position {
        let mut bucket: Bucket<Position> = Bucket::new(self.size());
        let mut root = None;
        while node.is_some() {
            let (mut rank, next, parent) = self
                .get_node_mut(node)
                .map(|n| {
                    let parent = n.parent;
                    let next = n.next;
                    n.parent = None;
                    n.next = None;
                    (n.rank as usize, next, parent)
                })
                .unwrap();
            self.link_next(parent, next);
            if let Some(matched) = bucket.remove(rank) {
                let (parent, next) = self
                    .get_node_mut(matched)
                    .map(|n| {
                        let parent = n.parent;
                        let next = n.next;
                        if root == matched {
                            root = if next == matched && parent == matched { None } else { next }
                        }
                        n.next = None;
                        n.parent = None;
                        (parent, next)
                    })
                    .unwrap();
                self.link_next(parent, next);
                node = self.link(node, matched);
                rank += 1;
            }
            if bucket.contains_key(rank) {
                self.link_next(node, next);
            } else {
                bucket.insert(rank, node);
                root = self.add_root_to_list(node, root);
                node = next;
            }
        }
        root
    }

    fn combine_ranks(&mut self, node: Position) -> Position {
        if self.passes == HeapPasses::Single {
            self.single_pass(node)
        } else {
            self.multi_pass(node)
        }
    }

    fn add_root_to_list(&mut self, root: Position, list: Position) -> Position {
        if list.is_some() && root.is_some() {
            let root_node = self.get_node_mut(root).unwrap() as *mut Node<K, V>;
            let list_node = self.get_node_mut(list).unwrap() as *mut Node<K, V>;
            unsafe {
                let is_new_root = if self.heap_type == HeapType::Max {
                    (*root_node).value > (*list_node).value
                } else {
                    (*root_node).value < (*list_node).value
                };
                let mut parent = (*list_node).parent;
                let mut next = (*list_node).next;
                parent = if is_new_root { parent } else { list };
                next = if is_new_root { list } else { next };
                self.set_next(parent, root);
                (*root_node).root = true;
                (*root_node).next = next;
                (*root_node).parent = parent;
                self.set_parent(next, root);
                if is_new_root {
                    root
                } else {
                    list
                }
            }
        } else {
            self.get_node_mut(root).map(|node| {
                node.root = true;
                node.next = root;
                node.parent = root;
            });
            root
        }
    }

    fn concatenate_lists(&mut self, head_list: Position, tail_list: Position) -> Position {
        let tail = self
            .get_node_mut(head_list)
            .map(|node| {
                let parent = node.parent;
                node.parent = None;
                parent
            })
            .unwrap_or(None);
        self.link_next(tail, tail_list);
        head_list.or(tail_list)
    }

    fn unlink_tree(&mut self, position: Position, mut parent: Position, next: Position) {
        let mut rank = self
            .get_node_mut(next)
            .map(|node| {
                node.parent = parent;
                node.rank + 1
            })
            .unwrap_or(0);

        parent = self
            .get_node_mut(parent)
            .map(|node| {
                if node.left == position {
                    node.left = next;
                } else {
                    node.next = next;
                }
                node.rank = rank;
                if node.root {
                    None
                } else {
                    node.parent
                }
            })
            .unwrap_or(None);

        while parent.is_some() {
            rank += 1;
            parent = self
                .get_node_mut(parent)
                .map(|node| {
                    node.rank = rank;
                    if node.root {
                        None
                    } else {
                        node.parent
                    }
                })
                .unwrap_or(None);
        }
    }
}

impl<K, V> Heap<K, V> for RankPairingHeap<K, V>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: PartialOrd + Clone + std::fmt::Debug,
{
    /// Indicates whether a [`RankPairingHeap`] is empty or not
    ///
    /// ```rust
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let mut heap = RankPairingHeap::multi_pass_min();
    ///
    /// assert_eq!(heap.is_empty(), true);
    ///
    /// heap.push("Hello".to_string(), 5);
    ///
    /// assert_eq!(heap.is_empty(), false);
    /// ```
    fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Returns the amount of elements in the [`RankPairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let mut heap = RankPairingHeap::multi_pass_max2();
    ///
    /// assert_eq!(heap.size(), 0);
    ///
    /// heap.push("Hello".to_string(), 5);
    ///
    /// assert_eq!(heap.size(), 1);
    /// ```
    fn size(&self) -> usize {
        self.list.len()
    }

    /// Adds an element to the [`RankPairingHeap`]
    ///
    /// ```rust
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let mut heap = RankPairingHeap::multi_pass_min();
    /// let value = "Hello".to_string();
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top(), Some(&value));
    /// ```
    fn push(&mut self, key: K, value: V) {
        let node = Node::new(key, value);
        let position = self.add_node(node);
        self.root = self.add_root_to_list(position, self.root);
    }

    /// Returns the highest priority element of a [`RankPairingHeap`] (or None)
    ///
    /// ```
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let value = "Hello".to_string();
    /// let mut heap = RankPairingHeap::multi_pass_min2();
    ///
    /// assert!(heap.top().is_none());
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top(), Some(&value));
    /// ```
    fn top(&self) -> Option<&K> {
        self.get_key(self.root)
    }

    /// Returns the highest priority element of a [`RankPairingHeap`] (or None) as mutable
    ///
    /// ```rust
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let value = "Hello".to_string();
    /// let mut heap = RankPairingHeap::single_pass_min();
    ///
    /// assert!(heap.top_mut().is_none());
    ///
    /// heap.push(value.clone(), 5);
    ///
    /// assert_eq!(heap.top_mut(), Some(&mut value.clone()));
    /// ```
    fn top_mut(&mut self) -> Option<&mut K> {
        self.get_node_mut(self.root).map(|node| &mut node.key)
    }

    /// Removes and Returns the highest priority element of a [`RankPairingHeap`] (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{RankPairingHeap, Heap};
    ///
    /// let value1 = "Hello".to_string();
    /// let value2 = "World".to_string();
    /// let mut heap = RankPairingHeap::single_pass_min2();
    ///
    /// heap.push(value1.clone(), 4);
    /// heap.push(value2.clone(), 5);
    ///
    /// assert_eq!(heap.pop(), Some(value1.clone()));
    /// assert_eq!(heap.pop(), Some(value2.clone()));
    /// assert_eq!(heap.pop(), None);
    /// ```
    fn pop(&mut self) -> Option<K> {
        let root = self.root;
        if root.is_some() {
            let next_root = self.get_next_root(root);
            self.remove(root).map(|removed| {
                let head = self.concatenate_lists(next_root, removed.left);
                self.root = self.combine_ranks(head);
                removed.key
            })
        } else {
            None
        }
    }
}

impl<K, V> DecreaseKey<K, V> for RankPairingHeap<K, V>
where
    K: Hash + Eq + Clone + std::fmt::Debug,
    V: PartialOrd + Clone + std::fmt::Debug,
{
    /// Updates the priority of an element in the [`RankPairingHeap`] (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{DecreaseKey, Heap, RankPairingHeap};
    ///
    /// let mut heap = RankPairingHeap::single_pass_max();
    /// let hello = "Hello".to_string();
    /// let world = "World".to_string();
    ///
    /// heap.push(hello.clone(), 2);
    /// heap.push(world.clone(), 5);
    ///
    /// assert_eq!(heap.top(), Some(&world));
    ///
    /// heap.update(&hello, 6);
    ///
    /// assert_eq!(heap.top(), Some(&hello));
    /// ```
    fn update(&mut self, key: &K, value: V) {
        let position = self.get_position(key);
        let heap_type = self.heap_type;
        self.get_node_mut(position)
            .map(|node| {
                let can_update = if heap_type == HeapType::Max {
                    value > node.value
                } else {
                    value < node.value
                };
                if can_update {
                    node.value = value;
                }
                (node.root, can_update, node.left, node.parent, node.next)
            })
            .map(|(is_root, can_update, left, parent, next)| {
                if can_update {
                    if is_root {
                        if self.compare(position, self.root) {
                            self.root = position;
                        }
                    } else {
                        let rank = (self.get_rank(left) + 1) as usize;
                        self.get_node_mut(position).map(|node| {
                            node.rank = rank;
                        });
                        self.unlink_tree(position, parent, next);
                        self.root = self.add_root_to_list(position, self.root);
                    }
                }
            });
    }

    ///  Deletes an element from the [`RankPairingHeap`] and returns it (or None)
    ///
    /// ```rust
    /// use mwpf::heapz::{DecreaseKey, Heap, RankPairingHeap};
    ///
    /// let mut heap = RankPairingHeap::single_pass_max2();
    /// let hello = "Hello".to_string();
    /// let world = "World".to_string();
    ///
    /// heap.push(hello.clone(), 2);
    /// heap.push(world.clone(), 6);
    ///
    /// assert_eq!(heap.top(), Some(&world));
    /// assert_eq!(heap.delete(&hello), Some(hello.clone()));
    ///
    /// heap.pop();
    ///
    /// assert_eq!(heap.top(), None);
    /// assert_eq!(heap.delete(&hello), None);
    /// ```
    fn delete(&mut self, key: &K) -> Option<K> {
        let position = self.get_position(key);
        self.get_node(position)
            .map(|node| (node.root, node.parent, node.next))
            .map(|(is_root, parent, next)| {
                if !is_root {
                    self.unlink_tree(position, parent, next);
                    self.add_root_to_list(position, self.root);
                }
            });
        self.root = position;
        self.pop()
    }
}
