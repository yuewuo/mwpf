import { Vector3, type Intersection, Quaternion } from 'three'
import { assert, parse_rust_bigint } from '@/util'

export const zero_vector = new Vector3(0, 0, 0)
export const unit_up_vector = new Vector3(0, 1, 0)
export const renderer_params = {
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
    precision: 'highp',
    stencil: true,
}

export interface Position {
    t: number
    i: number
    j: number
}

export function load_position (mesh_position: Vector3, data_position: Position) {
    mesh_position.z = data_position.i
    mesh_position.x = data_position.j
    mesh_position.y = data_position.t
}

export function compute_vector3 (data_position: Position): Vector3 {
    const vector = new Vector3(0, 0, 0)
    load_position(vector, data_position)
    return vector
}

export interface DualNode {
    // V_S: vertex list
    v: number[]
    // E_S: edge list
    e: number[]
    // delta(S): hair edge list
    h: number[]
    // dual variable (d = dn/dd)
    d: number
    dn: bigint | number
    dd: bigint | number
    // grow rate (r = rn/rd)
    r: number
    rn: bigint | number
    rd: bigint | number
}

export interface Edge {
    // weight
    w: number
    // vertices
    v: number[]
    // grown (g = gn/gd)
    g?: number
    gn: bigint | number
    gd: bigint | number
    // un-grown (u = un/ud = w_e - g)
    u: number
    un: bigint | number
    ud: bigint | number
}

export interface Interface {
    // sum of dual variables (sum_dual = sdn/sdd)
    sum_dual: number
    sdn: bigint | number
    sdd: bigint | number
}

export interface Vertex {
    // is defect
    s: boolean
}

export type Subgraph = number[]

export interface WeightRange {
    // lower (l = ln/ld)
    lower: number
    ld: bigint | number
    ln: bigint | number
    // upper (u = un/ud)
    upper: number
    ud: bigint | number
    un: bigint | number
}

export interface Snapshot {
    dual_nodes?: DualNode[]
    edges: Edge[]
    interface: Interface
    vertices: Vertex[]
    subgraph?: Subgraph
    weight_range: WeightRange
}

export type SnapshotTuple = (string | Snapshot)[]

export interface VisualizerData {
    format: string
    version: string
    positions: Position[]
    // [name, snapshot]
    snapshots: SnapshotTuple[]
}

export function fix_visualizer_data (visualizer: VisualizerData) {
    for (const entry of visualizer.snapshots) {
        assert(entry.length == 2)
        const snapshot = entry[1] as Snapshot
        if (snapshot.dual_nodes != undefined) {
            for (const dual_node of snapshot.dual_nodes) {
                if (dual_node.dn != undefined) dual_node.dn = parse_rust_bigint(dual_node.dn)
                if (dual_node.dd != undefined) dual_node.dd = parse_rust_bigint(dual_node.dd)
                if (dual_node.rn != undefined) dual_node.rn = parse_rust_bigint(dual_node.rn)
                if (dual_node.rd != undefined) dual_node.rd = parse_rust_bigint(dual_node.rd)
            }
        }
        if (snapshot.edges != undefined) {
            for (const edge of snapshot.edges) {
                if (edge.gn != undefined) edge.gn = parse_rust_bigint(edge.gn)
                if (edge.gd != undefined) edge.gd = parse_rust_bigint(edge.gd)
                if (edge.un != undefined) edge.un = parse_rust_bigint(edge.un)
                if (edge.ud != undefined) edge.ud = parse_rust_bigint(edge.ud)
            }
        }
        if (snapshot.interface != undefined) {
            if (snapshot.interface.sdn != undefined) snapshot.interface.sdn = parse_rust_bigint(snapshot.interface.sdn)
            if (snapshot.interface.sdd != undefined) snapshot.interface.sdd = parse_rust_bigint(snapshot.interface.sdd)
        }
        if (snapshot.weight_range != undefined) {
            if (snapshot.weight_range.ld != undefined) snapshot.weight_range.ld = parse_rust_bigint(snapshot.weight_range.ld)
            if (snapshot.weight_range.ln != undefined) snapshot.weight_range.ln = parse_rust_bigint(snapshot.weight_range.ln)
            if (snapshot.weight_range.ud != undefined) snapshot.weight_range.ud = parse_rust_bigint(snapshot.weight_range.ud)
            if (snapshot.weight_range.un != undefined) snapshot.weight_range.un = parse_rust_bigint(snapshot.weight_range.un)
        }
    }
}

/* runtime data */
export class RuntimeData {
    visualizer: VisualizerData
    hovered: any = undefined
    selected: any = undefined

    constructor (visualizer: VisualizerData) {
        // first fix the visualizer data (primarily the BigInts)
        fix_visualizer_data(visualizer)
        this.visualizer = visualizer
    }
}

export function clickable_of (intersect: Intersection | undefined): any {
    if (intersect == undefined) {
        return
    }
    if (intersect.instanceId != undefined) {
        const instance_state = intersect?.object?.userData?.vecData?.[intersect.instanceId]
        if (instance_state != undefined) {
            if (instance_state.type == 'vertex') {
                return {
                    type: 'vertex',
                    vi: instance_state.vi,
                }
            }
            if (instance_state.type == 'edge') {
                return {
                    type: 'edge',
                    ei: instance_state.ei,
                }
            }
        }
    }
}

export class ConfigProps {
    show_config: boolean = true
    show_info: boolean = true
    full_screen: boolean = false
    segments: number = 32
    visualizer_config: any = undefined
    initial_aspect_ratio?: number = undefined
    snapshot_index?: number = undefined
    config_setters?: object = undefined
}

// https://stackoverflow.com/questions/8817394/javascript-get-deep-value-from-object-by-passing-path-to-it-as-string/8817473
export function set_deep_value (obj: any, path: string, value: any) {
    const paths = path.split('.')
    for (let i = 0; i < paths.length - 1; i++) {
        obj = obj[paths[i]]
    }
    obj[paths[paths.length - 1]] = value
}

/*
The following are visualization specific states
*/

export class VertexState {
    type: string = 'vertex'
    vi: number
    color?: string = undefined
    position: Position

    constructor (vi: number, position: Position) {
        this.vi = vi
        this.position = position
    }
}

export class VertexStates {
    normal_vertices: Array<VertexState> = []
    defect_vertices: Array<VertexState> = []
    all_outlines: Array<VertexState> = []
}

export class EdgeRingState {
    type: string = 'edge'
    ei: number
    position: Vector3
    color: string = 'black'
    inner: number = 0
    outer: number = 1

    constructor (ei: number, position: Vector3) {
        this.ei = ei
        this.position = position
    }
}

export class EdgeTubeState {
    type: string = 'edge'
    ei: number
    position: Vector3
    color: string = 'black'
    length: number
    degree: number
    quaternion: Quaternion

    constructor (ei: number, position: Vector3, length: number, degree: number, quaternion: Quaternion) {
        this.ei = ei
        this.position = position
        this.length = length
        this.degree = degree
        this.quaternion = quaternion
    }
}

export const ring_resolution = 100
export class EdgeStates {
    // rings (degree-1 edges)
    ungrown_edge_rings: Array<Array<EdgeRingState>> = Array.from({ length: ring_resolution + 1 }, () => [])
    grown_edge_rings: Array<Array<EdgeRingState>> = Array.from({ length: ring_resolution + 1 }, () => [])
    tight_edge_rings: Array<Array<EdgeRingState>> = Array.from({ length: ring_resolution + 1 }, () => [])
    // tubes (higher degree edges)
    ungrown_edge_tubes: Array<EdgeTubeState> = []
    grown_edge_tubes: Array<EdgeTubeState> = []
    tight_edge_tubes: Array<EdgeTubeState> = []
}

export function compute_edge_to_dual_indices (snapshot: Snapshot): Array<Array<number>> {
    const dual_indices: Array<Array<number>> = Array.from({ length: snapshot.edges.length }, () => [])
    if (snapshot.dual_nodes != null) {
        for (const [node_index, node] of snapshot.dual_nodes.entries()) {
            for (const edge_index of node.h) {
                dual_indices[edge_index].push(node_index)
            }
        }
    }
    return dual_indices
}

export function compute_vertex_to_dual_indices (snapshot: Snapshot): Array<Array<number>> {
    const dual_indices: Array<Array<number>> = Array.from({ length: snapshot.vertices.length }, () => [])
    if (snapshot.dual_nodes != null) {
        for (const [node_index, node] of snapshot.dual_nodes.entries()) {
            for (const vertex_index of node.v) {
                dual_indices[vertex_index].push(node_index)
            }
        }
    }
    return dual_indices
}

export function compute_vertex_incident_edges (snapshot: Snapshot): Array<Array<number>> {
    const incident_edges: Array<Array<number>> = Array.from({ length: snapshot.vertices.length }, () => [])
    for (const [edge_index, edge] of snapshot.edges.entries()) {
        for (const vertex_index of edge.v) {
            incident_edges[vertex_index].push(edge_index)
        }
    }
    return incident_edges
}
