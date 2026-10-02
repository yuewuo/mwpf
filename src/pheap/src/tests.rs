#![cfg(test)]
use super::PairingHeap;
use crate::pheap::graph::{mst_prim, SimpleGraph};
use crate::pheap::ph::HeapElmt;

#[cfg(test)]
fn create_heap(start: i32, end: i32) -> (PairingHeap<i32, i32>, Vec<HeapElmt<i32, i32>>) {
    let mut ph = PairingHeap::<i32, i32>::new();
    let elmts: Vec<_> = (start..end).map(|ii| ph.insert2(ii, ii)).collect();
    (ph, elmts)
}

#[test]
fn create_insert() {
    let mut ph = PairingHeap::<i32, i32>::new();
    assert_eq!(0, ph.len());
    assert!(ph.is_empty());

    for ii in 1..=10 {
        ph.insert(ii, ii);
    }

    assert_eq!(10, ph.len());
}

#[test]
fn find_min() {
    let (ph, _) = create_heap(0, 0);
    assert!(ph.find_min().is_none());

    let (ph, _) = create_heap(1, 11);
    let min = ph.find_min();
    assert!(min.is_some());
    let (k, _) = min.unwrap();
    assert_eq!(1, *k);

    let min_prio = ph.find_min();
    assert!(min_prio.is_some());
    let (k, p) = min_prio.unwrap();
    assert_eq!(1, *k);
    assert_eq!(1, *p);
}

#[test]
fn merge() {
    let ph1 = create_heap(1, 11).0;
    let len1 = ph1.len();
    let ph2 = create_heap(11, 21).0;
    let len2 = ph2.len();

    let ph = ph2.merge(ph1);
    println!("Len: {}", ph.len());
    assert_eq!(len1 + len2, ph.len());
    let min_prio = ph.find_min();
    assert!(min_prio.is_some());
    let (k, p) = min_prio.unwrap();
    assert_eq!(1, *k);
    assert_eq!(1, *p);
}

#[test]
fn delete_min() {
    let (mut ph, _) = create_heap(1, 11);
    let mut len = ph.len();
    let mut tracker = 1;

    while len != 0 {
        let min_prio = ph.find_min();
        assert!(min_prio.is_some());
        let (k, p) = min_prio.unwrap();
        let (k, p) = (*k, *p);
        assert_eq!(tracker, p);
        tracker += 1;

        let del_prio = ph.delete_min();
        assert!(del_prio.is_some());
        let (kt, pt) = del_prio.unwrap();
        assert_eq!(k, kt);
        assert_eq!(p, pt);

        len = ph.len();
    }

    assert!(ph.find_min().is_none());
    assert_eq!(0, ph.len());
}

#[test]
fn decrease_prio() {
    let (mut ph, _) = create_heap(1, 11);

    ph.delete_min();
    ph.decrease_prio(&8, 4);
    ph.decrease_prio(&6, 3);
    ph.decrease_prio(&9, 3);
    ph.decrease_prio(&10, 2);

    let mut len = ph.len();
    let mut count = 0;

    let key_exp = vec![2, 6, 3, 8, 4, 5, 9, 7, 10];
    let prio_exp = vec![2, 3, 3, 4, 4, 5, 6, 7, 8];

    while len != 0 {
        let del_prio = ph.delete_min();
        assert!(del_prio.is_some());
        let (k, p) = del_prio.unwrap();
        assert_eq!(
            key_exp[count], k,
            "Check key: Expected: {} | Got: {}",
            key_exp[count], k
        );
        assert_eq!(
            prio_exp[count], p,
            "Check prio for key {}: Expected: {} | Got: {}",
            k, prio_exp[count], p
        );

        len = ph.len();
        count += 1;
    }
}

#[test]
fn update_prio() {
    let (mut ph, v) = create_heap(1, 11);

    ph.delete_min();

    ph.update_prio(&v[7], 4);
    ph.update_prio(&v[5], 3);
    ph.update_prio(&v[8], 6);
    ph.update_prio(&v[9], 8);

    let key_exp = vec![2, 6, 3, 8, 4, 5, 9, 7, 10];
    let prio_exp = vec![2, 3, 3, 4, 4, 5, 6, 7, 8];

    let mut len = ph.len();
    let mut count = 0;

    while len != 0 {
        let del_prio = ph.delete_min();
        assert!(del_prio.is_some());
        let (k, p) = del_prio.unwrap();
        assert_eq!(
            key_exp[count], k,
            "Check key: Expected: {} | Got: {}",
            key_exp[count], k
        );
        assert_eq!(
            prio_exp[count], p,
            "Check prio for key {}: Expected: {} | Got: {}",
            k, prio_exp[count], p
        );

        len = ph.len();
        count += 1;
    }
}

#[test]
fn test_dijkstra() {
    let mut g = SimpleGraph::<u32>::with_capacity(6);

    g.add_weighted_edges(0, 1, 7);
    g.add_weighted_edges(0, 2, 9);
    g.add_weighted_edges(0, 5, 14);
    g.add_weighted_edges(1, 2, 10);
    g.add_weighted_edges(1, 3, 15);
    g.add_weighted_edges(2, 5, 2);
    g.add_weighted_edges(2, 3, 11);
    g.add_weighted_edges(3, 4, 6);
    g.add_weighted_edges(4, 5, 9);

    let mut sp = g.sssp_dijkstra(0, &[4]);
    assert_eq!(1, sp.len());

    let sp = sp.pop().unwrap();
    assert_eq!(true, sp.is_feasible());
    assert_eq!(20, sp.dist());
    assert_eq!(&[0, 2, 5, 4], sp.path().as_slice());

    g.add_weighted_edges(6, 7, 2);
    g.add_weighted_edges(6, 8, 3);

    let lsp = g.sssp_dijkstra_lazy(0);
    let sp = lsp.get(7);
    assert_eq!(false, sp.is_feasible());

    let sp = lsp.get(4);
    assert_eq!(true, sp.is_feasible());
    assert_eq!(20, sp.dist());
    assert_eq!(&[0, 2, 5, 4], sp.path().as_slice());
}

#[test]
fn test_mst_prim() {
    let mut g = SimpleGraph::<u32>::new();

    g.add_weighted_edges(0, 1, 4);
    g.add_weighted_edges(0, 7, 8);
    g.add_weighted_edges(1, 2, 8);
    g.add_weighted_edges(1, 7, 11);
    g.add_weighted_edges(2, 3, 7);
    g.add_weighted_edges(2, 5, 4);
    g.add_weighted_edges(2, 8, 2);
    g.add_weighted_edges(3, 4, 9);
    g.add_weighted_edges(3, 5, 14);
    g.add_weighted_edges(4, 5, 10);
    g.add_weighted_edges(5, 6, 2);
    g.add_weighted_edges(6, 7, 1);
    g.add_weighted_edges(6, 8, 6);
    g.add_weighted_edges(7, 8, 7);

    let (g0, d0) = mst_prim(&g, 0);
    let (g4, d4) = mst_prim(&g, 4);

    assert_eq!(d0, d4);
    assert_eq!(g0.n_nodes(), g4.n_nodes());
    assert_eq!(g0.n_edges(), g4.n_edges());
}
