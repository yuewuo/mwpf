use keyed_priority_queue::KeyedPriorityQueue;
use mwpf::pheap::PairingHeap;
use priority_queue::PriorityQueue;

fn create_ph(n: i32) {
    let mut ph = PairingHeap::<i32, i32>::new();

    for ii in 0..n {
        ph.insert(ii, ii);
    }
}

fn create_pq(n: i32) {
    let mut pq = PriorityQueue::<i32, i32>::new();

    for ii in 0..n {
        pq.push(ii, ii);
    }
}

fn create_kpq(n: i32) {
    let mut pq = KeyedPriorityQueue::<i32, i32>::new();

    for ii in 0..n {
        pq.push(ii, ii);
    }
}

fn create_aph(n: i32) {
    let mut aph = addressable_pairing_heap::PairingHeap::<i32, i32>::new();

    for ii in 0..n {
        aph.push(ii, ii);
    }
}

fn create_a422ph(n: i32) {
    let mut aph = pairing_heap::PairingHeap::<i32>::new();

    for ii in 0..n {
        aph.push(ii);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        help();
        std::process::exit(1);
    }

    let n = args[2].to_string().parse::<i32>().unwrap();

    match args[1].as_str() {
        "pairing_heap" => create_ph(n),
        "priority_queue" => create_pq(n),
        "keyed_priority_queue" => create_kpq(n),
        "addressable_pairing_heap" => create_aph(n),
        "ap422_pairing_heap" => create_a422ph(n),
        _ => {
            help();
            std::process::exit(1)
        }
    }
}

fn help() {
    println!("Usage: ./stress <data structure name> <number of elements>");
    println!("The following data structures are available for testing: ");
    println!("> pairing_heap (this library)");
    println!("> addressable_pairing_heap (https://crates.io/crates/addressable-pairing-heap)");
    println!("> ap422_pairing_heap (https://crates.io/crates/pairing-heap)");
    println!("> priority_queue (from crate: https://crates.io/crates/priority-queue)");
    println!("> keyed_priority_queue (from crate: https://crates.io/crates/keyed_priority_queue)");
}
