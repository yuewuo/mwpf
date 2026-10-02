use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

use clap_2::{App, Arg};
use mwpf::pheap::graph::SimpleGraph;
use pathfinding::prelude::dijkstra_all;

fn main() {
    let matches = App::new("Single source shortest path benchmark")
        .arg(
            Arg::with_name("file")
                .short("f")
                .long("file")
                .takes_value(true)
                .required(true)
                .help("Path to a DIMACS file."),
        )
        .arg(
            Arg::with_name("lib")
                .long("lib")
                .takes_value(true)
                .required(true)
                .help("The library to be used to solve the shortest path problem. Options: pheap | fast_paths."),
        )
        .arg(
            Arg::with_name("runs")
                .long("runs")
                .takes_value(true)
                .default_value("5")
                .help("Number of runs for search query."),
        )
        .get_matches();

    let filepath = match matches.value_of("file") {
        Some(fp) => fp,
        None => std::process::exit(1),
    };

    let runs = matches.value_of("runs").unwrap().to_string().parse::<usize>().unwrap();

    match matches.value_of("lib") {
        Some(lib) => match lib {
            "pheap" => graph(filepath, runs),
            "pathfinding" => pathfinding(filepath, runs),
            _ => std::process::exit(1),
        },
        None => std::process::exit(1),
    };
}

macro_rules! run_exp {
    ($runs:expr, $exe:stmt) => {
        let mut durations = Vec::with_capacity($runs);

        for ii in 0..$runs {
            println!("> Run {}/{}", ii + 1, $runs);
            let start = std::time::Instant::now();
            $exe
            let end = std::time::Instant::now() - start;
            println!(
                "> Time taken to solve the problem: {} (ms)",
                end.as_millis()
            );
            durations.push(end.as_millis());
        }

        let avg = durations.iter().sum::<u128>() as usize;
        println!("Average time: {} (ms)", avg / $runs);
    };
}

fn graph(filepath: &str, runs: usize) {
    println!("> Load file: {}", filepath);

    let file = File::open(filepath).unwrap();
    let mut reader = BufReader::new(file);

    let mut n_nodes = 0;
    let mut _n_edges = 0;

    for _ in 0..7 {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();

        if !line.is_empty() && line.starts_with('p') {
            let s = line.trim().split_whitespace().collect::<Vec<&str>>();
            n_nodes = s[2].parse::<usize>().unwrap();
            _n_edges = s[3].parse::<usize>().unwrap();
        }
    }

    let mut g = SimpleGraph::<u32>::with_capacity(n_nodes);

    for line in reader.lines() {
        let (node1, node2, weight) = parse_line(&line.unwrap());
        g.add_weighted_edges(node1, node2, weight);
    }

    println!("> Graph created.");

    run_exp!(runs, let _ = g.sssp_dijkstra_lazy(10_000));
}

fn pathfinding(filepath: &str, runs: usize) {
    println!("> Load file: {}", filepath);

    let file = File::open(filepath).unwrap();
    let mut reader = BufReader::new(file);

    for _ in 0..7 {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
    }

    fn insert_weight(hm: &mut HashMap<usize, Vec<(usize, u32)>>, node1: usize, node2: usize, weight: u32) {
        match hm.get_mut(&node1) {
            Some(v) => {
                v.push((node2, weight));
            }
            None => {
                let v = vec![(node2, weight)];
                hm.insert(node1, v);
            }
        }
    }

    let mut hm = HashMap::<usize, Vec<(usize, u32)>>::new();

    for line in reader.lines() {
        let (node1, node2, weight) = parse_line(&line.unwrap());
        insert_weight(&mut hm, node1, node2, weight);
        insert_weight(&mut hm, node2, node1, weight);
    }

    run_exp!(runs, let _ = dijkstra_all(&0, |x| {
        let nbs = hm.get(x).unwrap();
        nbs.iter().map(|(idx, w)| (*idx, *w))
    }));
}

fn parse_line(line: &str) -> (usize, usize, u32) {
    let s = line.trim().split_whitespace().collect::<Vec<&str>>();
    let node1 = s[1].parse::<usize>().unwrap() - 1;
    let node2 = s[2].parse::<usize>().unwrap() - 1;
    let weight = s[3].parse::<u32>().unwrap();
    (node1, node2, weight)
}
