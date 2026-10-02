use std::{
    collections::HashMap,
    fs::File,
    io::{LineWriter, Write},
    path::Path,
};

use std::ops::AddAssign;

use num_traits::{Bounded, Num, Zero};

use crate::pheap::{ph::HeapElmt, PairingHeap};

/// A simple and undirected graph.
///
/// A simple graph assumes that the node indexing starts from ```0``` and is not equipped with a hash map
/// for a mapping from external complex objects to internal graph indices. As a result, [`SimpleGraph`]
/// doesn't have no runtime overhead for such object storage and mapping.
///
/// # Examples
/// The following example shows how to construct a graph and find the shortest path between node 1 and 5.
/// The data is taken from the illustration in Wikipedia's page for [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra's_algorithm).
///
/// Here, the numbering is adjusted so that the node indexing starts from ```0```.
/// ```
/// use mwpf::pheap::graph::SimpleGraph;
///
/// let mut g = SimpleGraph::<u32>::with_capacity(6);
///
/// g.add_weighted_edges(0, 1, 7);
/// g.add_weighted_edges(0, 2, 9);
/// g.add_weighted_edges(0, 5, 14);
/// g.add_weighted_edges(1, 2, 10);
/// g.add_weighted_edges(1, 3, 15);
/// g.add_weighted_edges(2, 5, 2);
/// g.add_weighted_edges(2, 3, 11);
/// g.add_weighted_edges(3, 4, 6);
/// g.add_weighted_edges(4, 5, 9);
///
/// // Finds an SSSP from 0 to 4.
/// let mut sp = g.sssp_dijkstra(0, &[4]);
/// assert_eq!(1, sp.len());
///
/// let sp = sp.pop().unwrap();
/// assert_eq!(20, sp.dist());
/// assert_eq!(&[0, 2, 5, 4], sp.path().as_slice());
///
/// // Adds a disconnected component to the graph.
/// g.add_weighted_edges(6, 7, 2);
/// g.add_weighted_edges(6, 8, 3);
///
/// // Finds an SSSP starting from 0. The result can be used for later query.
/// let lsp = g.sssp_dijkstra_lazy(0);
/// let lsp = g.sssp_dijkstra_lazy(0);
/// let sp = lsp.get(7);
/// assert_eq!(false, sp.is_feasible());
///
/// let sp = lsp.get(4);
/// assert_eq!(true, sp.is_feasible());
/// assert_eq!(20, sp.dist());
/// assert_eq!(&[0, 2, 5, 4], sp.path().as_slice());
///
/// ```
///
#[derive(Debug, Default)]
pub struct SimpleGraph<W> {
    n_edges: usize,
    weights: HashMap<usize, Vec<(usize, W)>>,
}

impl<W> SimpleGraph<W> {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self {
            n_edges: 0,
            weights: HashMap::new(),
        }
    }

    /// Creates an empty graph with the given capacitiy of nodes.
    pub fn with_capacity(n_nodes: usize) -> Self {
        Self {
            n_edges: 0,
            weights: HashMap::with_capacity(n_nodes),
        }
    }

    /// Returns the number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of edges in the graph.
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Adds a weighted edge to the graph.
    ///
    /// If the edge already exists in the graph, the weight will be updated.
    pub fn add_weighted_edges(&mut self, node1: usize, node2: usize, weight: W)
    where
        W: Clone + Copy,
    {
        if node1 != node2 {
            self.insert_weight(node1, node2, weight);
            self.insert_weight(node2, node1, weight);
        }

        self.n_edges += 2;
    }

    /// Returns the neighbours of a node.
    #[inline]
    pub(crate) fn neighbours(&self, node: &usize) -> Option<&Vec<(usize, W)>> {
        self.weights.get(&node)
    }

    /// Finds the shortest paths from a source node to destination nodes.
    ///
    /// If you want to keep the result for later usage and/or want to save memory, consider using
    /// the lazy version [`SimpleGraph::sssp_dijkstra_lazy`], which returns the intermediate result
    /// from Dijkstra's algorithm.
    pub fn sssp_dijkstra(&self, src: usize, dest: &[usize]) -> Vec<ShortestPath<W>>
    where
        W: Bounded + Num + Zero + PartialOrd + Copy,
    {
        let nodes = self.dijkstra(src);
        let mut result = Vec::with_capacity(dest.len());

        for ii in dest {
            result.push(traverse_path(src, *ii, &nodes));
        }

        result
    }

    /// Finds the shortest paths from a source node to all nodes and returns the intermediate result
    /// for later usage.
    pub fn sssp_dijkstra_lazy(&self, src: usize) -> LazyShortestPaths<W>
    where
        W: Bounded + Num + Zero + PartialOrd + Copy,
    {
        LazyShortestPaths {
            src,
            paths: self.dijkstra(src),
        }
    }

    #[inline]
    fn dijkstra(&self, src: usize) -> Vec<DijNode<W>>
    where
        W: Bounded + Num + Zero + PartialOrd + Copy,
    {
        let mut pq = PairingHeap::<usize, W>::new();
        pq.insert(src, W::zero());

        let mut nodes = vec![DijNode::<W>::new(); self.weights.len()];
        nodes[src].dist = W::zero();
        let mut len = pq.len();

        while len != 0 {
            let (node, prio) = pq.delete_min().unwrap();
            let count = nodes[node].len + 1;

            if let Some(nb) = self.neighbours(&node) {
                for (u, dist) in nb {
                    let dijnode = &mut nodes[*u];
                    let alt = prio + *dist;
                    if !dijnode.visited && alt < dijnode.dist {
                        dijnode.dist = alt;
                        dijnode.pred = node;
                        dijnode.len = count;
                        dijnode.feasible = true;
                        pq.insert(*u, alt);
                    }
                }
            }

            let dijnode = nodes.get_mut(node).unwrap();
            dijnode.visited = true;
            len = pq.len();
        }

        nodes
    }

    fn insert_weight(&mut self, node1: usize, node2: usize, weight: W) {
        match self.weights.get_mut(&node1) {
            Some(v) => {
                v.push((node2, weight));
            }
            None => {
                let v = vec![(node2, weight)];
                self.weights.insert(node1, v);
            }
        }
    }

    /// Write graph as a list of edges.
    ///
    /// Each line contains one edge, following [networkx](https://networkx.org/)'s format:
    /// ```index 1 index 2 {'weight': {}}```.
    pub fn write_edgelist<P>(&self, filepath: P) -> std::io::Result<()>
    where
        P: AsRef<Path>,
        W: std::fmt::Display,
    {
        let file = File::create(filepath)?;
        let mut file = LineWriter::new(file);

        for (node_idx, nb) in &self.weights {
            for (vtx_idx, w) in nb {
                file.write_all(
                    format!("{} {} {{'weight': {}}}\n", node_idx, vtx_idx, w).as_bytes(),
                )?;
            }
        }

        file.flush()?;

        Ok(())
    }
}

/// The shortest path from a source node to a destination node.
#[derive(Debug)]
pub struct ShortestPath<W> {
    src: usize,
    dest: usize,
    feasible: bool,
    dist: W,
    path: Vec<usize>,
}

impl<W> ShortestPath<W> {
    /// Returns the index of the source node in the shortest path.
    pub fn src(&self) -> usize {
        self.src
    }

    /// Returns the index of the destination node in the shortest path.
    pub fn dest(&self) -> usize {
        self.dest
    }

    /// Returns the shortest path's distance
    pub fn dist(&self) -> W
    where
        W: Copy,
    {
        self.dist
    }

    /// Returns whether a path from the source node to the destination node is feasible.
    pub fn is_feasible(&self) -> bool {
        self.feasible
    }

    /// Returns the path from the source node to destination node.
    ///
    /// The first element of the vector is the source node, the last the destination node.
    pub fn path(&self) -> &Vec<usize> {
        &self.path
    }
}

/// A struct representing the intermediate output of Dijkstra's algorithm.
#[derive(Debug)]
pub struct LazyShortestPaths<W> {
    src: usize,
    paths: Vec<DijNode<W>>,
}

impl<W> LazyShortestPaths<W> {
    /// Returns the shortest path for a given node.
    pub fn get(&self, node_index: usize) -> ShortestPath<W>
    where
        W: Zero + Copy,
    {
        traverse_path(self.src, node_index, &self.paths)
    }

    /// Returns the shortest paths for all nodes.
    pub fn get_all(&self) -> Vec<ShortestPath<W>>
    where
        W: Zero + Copy,
    {
        let n_nodes = self.paths.len();
        let mut result: Vec<ShortestPath<W>> = Vec::with_capacity(n_nodes);

        for ii in 0..n_nodes {
            let end_node = &self.paths[ii];
            let expected = end_node.len + 1;

            let sp = if end_node.feasible {
                let mut len = 0;
                let mut chain = Vec::with_capacity(expected);
                let mut next = end_node.pred;

                while len < expected {
                    if next < ii {
                        let mut sp = result[next].path.clone();
                        if ii < self.src {
                            sp.reverse();
                        }

                        sp.append(&mut chain);
                        chain = sp;
                        break;
                    }

                    chain.insert(0, next);
                    next = self.paths[next].pred;
                    len = chain.len();
                }

                ShortestPath {
                    src: self.src,
                    dest: ii,
                    dist: end_node.dist,
                    path: chain,
                    feasible: true,
                }
            } else {
                ShortestPath {
                    src: self.src,
                    dest: ii,
                    dist: <W as Zero>::zero(),
                    path: Vec::with_capacity(0),
                    feasible: false,
                }
            };

            result.push(sp);
        }

        result
    }

    /// Returns the shortest paths for a given list of node indices.
    pub fn get_list(&self, node_indices: &[usize]) -> Vec<ShortestPath<W>>
    where
        W: Zero + Copy,
    {
        let mut result = Vec::with_capacity(node_indices.len());

        for ii in node_indices {
            result.push(traverse_path(self.src, *ii, &self.paths));
        }

        result
    }
}

#[derive(Clone, Debug)]
struct DijNode<W> {
    /// Id of the predecessor's node in SSSP solution from Dijkstra's algorithm.
    pred: usize,
    /// Flag whether a node is visisted or not.
    len: usize,
    /// Flag indicating whether the node is already visisted or not.
    visited: bool,
    /// Flag indicating whether a path from source node is feasible.
    feasible: bool,
    /// Distance to the predecessor.
    dist: W,
}

impl<W> DijNode<W> {
    pub fn new() -> Self
    where
        W: Bounded,
    {
        Self {
            pred: 0,
            dist: <W as Bounded>::max_value(),
            visited: false,
            len: 0,
            feasible: false,
        }
    }
}

#[inline(always)]
fn traverse_path<W>(src: usize, dest: usize, paths: &[DijNode<W>]) -> ShortestPath<W>
where
    W: Zero + Copy,
{
    let end_node = &paths[dest];
    if end_node.feasible {
        let expected = end_node.len + 1;

        let mut len = 0;
        let mut path = Vec::with_capacity(expected);
        path.push(dest);
        let mut next = end_node.pred;

        while len < expected {
            path.insert(0, next);
            next = paths[next].pred;
            len = path.len();
        }

        ShortestPath {
            src,
            dest,
            dist: end_node.dist,
            path,
            feasible: true,
        }
    } else {
        ShortestPath {
            src,
            dest,
            dist: <W as Zero>::zero(),
            path: Vec::with_capacity(0),
            feasible: false,
        }
    }
}

/// Find the minimum spanning tree (MST) in a graph using Prim's algorithm.
///
/// The function returns a tuple of the total weight of the MST and a simple graph, whose edges are the MST's edges.
///
/// # Examples
/// ```rust
/// use mwpf::pheap::graph::{mst_prim, SimpleGraph};
///
/// let mut g = SimpleGraph::<u32>::new();
///
/// g.add_weighted_edges(0, 1, 4);
/// g.add_weighted_edges(0, 7, 8);
/// g.add_weighted_edges(1, 2, 8);
/// g.add_weighted_edges(1, 7, 11);
/// g.add_weighted_edges(2, 3, 7);
/// g.add_weighted_edges(2, 5, 4);
/// g.add_weighted_edges(2, 8, 2);
/// g.add_weighted_edges(3, 4, 9);
/// g.add_weighted_edges(3, 5, 14);
/// g.add_weighted_edges(4, 5, 10);
/// g.add_weighted_edges(5, 6, 2);
/// g.add_weighted_edges(6, 7, 1);
/// g.add_weighted_edges(6, 8, 6);
/// g.add_weighted_edges(7, 8, 7);
///
/// // gx is the new graph containing the MST's edges and dx is the total weight.
/// let (g0, d0) = mst_prim(&g, 0);
/// let (g4, d4) = mst_prim(&g, 4);
///
/// assert_eq!(d0, d4);
/// assert_eq!(g0.n_nodes(), g4.n_nodes());
/// assert_eq!(g0.n_edges(), g4.n_edges());
/// ```
pub fn mst_prim<W>(graph: &SimpleGraph<W>, src: usize) -> (SimpleGraph<W>, W)
where
    W: Copy + PartialOrd + Bounded + Zero + AddAssign,
{
    let mut pq = PairingHeap::<usize, W>::new();
    let mut nodes: Vec<_> = (0..graph.n_nodes())
        .map(|ii| {
            let mut node = PrimNode::<W>::new();
            node.dist = if ii == src {
                <W as Zero>::zero()
            } else {
                <W as Bounded>::max_value()
            };
            node.idx = ii;
            node.heap = pq.insert2(ii, node.dist);
            node
        })
        .collect();

    let mut len = pq.len();

    while len != 0 {
        let (node, _) = pq.delete_min().unwrap();
        nodes[node].heap.none();

        if let Some(nb) = graph.neighbours(&node) {
            for (u, dist) in nb {
                let primnode = &mut nodes[*u];
                if !primnode.heap.is_none() && *dist < primnode.dist {
                    primnode.dist = *dist;
                    primnode.parent = Some(node);
                    pq.update_prio(&primnode.heap, primnode.dist);
                }
            }
        }

        len = pq.len();
    }

    let mut rg = SimpleGraph::<W>::with_capacity(graph.n_nodes());
    let mut dist = <W as Zero>::zero();
    for node in nodes {
        if let Some(p) = node.parent {
            rg.add_weighted_edges(p, node.idx, node.dist);
            dist += node.dist;
        }
    }

    (rg, dist)
}

#[derive(Clone, Debug)]
struct PrimNode<W> {
    idx: usize,
    parent: Option<usize>,
    heap: HeapElmt<usize, W>,
    dist: W,
}

impl<W> PrimNode<W> {
    pub fn new() -> Self
    where
        W: Bounded,
    {
        Self {
            idx: 0,
            parent: None,
            heap: HeapElmt::<usize, W>::default(),
            dist: <W as Bounded>::max_value(),
        }
    }
}
