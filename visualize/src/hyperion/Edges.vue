<script setup lang="ts">
/**
 * The hyperedge is displayed based on its degree.
 * 1. degree-1 edges are displayed as a 2D ring with a single vertex at the center.
 * 2. degree-(2+) edges are displayed as tubes from all its incident vertices to the center.
 *      The tube is colored based on the Cover of the dual variables, starting from vertices $\delta(S) \cap e$
 *      and gradually growing towards the center and further to other vertices $e \setminus \delta(S)$.
 *
 * InstancedMesh cannot be used across objects with different opacities, and thus we need to create different objects
 * for different opacities. For simplicity, we only have 3 opacities: 0.1 (empty), 0.2 (partial grown), and 1 (tight).
 */

import { type Ref, computed, watchEffect, useTemplateRef, onMounted } from 'vue'
import { compute_vector3, unit_up_vector, EdgeRingState, EdgeTubeState, EdgeStates, ring_resolution, compute_edge_to_dual_indices } from './hyperion'
import { type Config } from './config_pane'
import { PhysicalMaterial, CylinderGeometry, RingGeometry } from 'troisjs'
import MyInstancedMesh from '@/misc/MyInstancedMesh.vue'
import { Object3D, BackSide, DoubleSide, Color, Vector3, Quaternion } from 'three'
import { assert_inject } from '@/misc/util'

const config: Ref<Config> = assert_inject('config')
const ring_index_of_ratio = (ratio: number) => Math.max(0, Math.min(ring_resolution, Math.round(ratio * ring_resolution)))

const edge_states = computed(() => {
    const edge_states = new EdgeStates()
    const snapshot = config.value.snapshot
    for (const [i, edge] of snapshot.edges.entries()) {
        if (edge == null) {
            continue
        }
        // calculate the center point of all vertices
        let sum_position = new Vector3(0, 0, 0)
        for (let j = 0; j < edge.v.length; ++j) {
            const vertex_index = edge.v[j]
            const vertex_position = config.value.data.visualizer.positions[vertex_index]
            sum_position = sum_position.add(compute_vector3(vertex_position))
        }
        const center_position = sum_position.multiplyScalar(1 / edge.v.length)
        const is_tight = edge.u == 0
        for (let j = 0; j < edge.v.length; ++j) {
            const vertex_index = edge.v[j]
            const vertex_position = config.value.data.visualizer.positions[vertex_index]
            const relative = center_position.clone().add(compute_vector3(vertex_position).multiplyScalar(-1))
            const direction = relative.clone().normalize()
            const quaternion = new Quaternion()
            quaternion.setFromUnitVectors(unit_up_vector, direction)
            let start = edge_offset.value
            const distance = relative.length()
            let edge_length = distance - edge_offset.value
            if (edge_length < 0) {
                // edge length should be non-negative
                start = distance
                edge_length = 0
            }
            const end = start + edge_length
            let start_position = compute_vector3(vertex_position).add(relative.clone().multiplyScalar(start / distance))
            let end_position = compute_vector3(vertex_position).add(relative.clone().multiplyScalar(end / distance))
            if (edge.v.length == 1) {
                start_position = compute_vector3(vertex_position)
                end_position = compute_vector3(vertex_position)
            }
            const segment_position_of = (ratio: number) => {
                // 0: start, 1: end
                return start_position
                    .clone()
                    .multiplyScalar(1 - ratio)
                    .add(end_position.clone().multiplyScalar(ratio))
            }
            const edge_branch_segmented_data = edge_branch_segments.value[i]!
            const grown_end = edge_branch_segmented_data.grown_end[j]
            const grown_center = edge_branch_segmented_data.grown_center[j]
            const segments_center = edge_branch_segmented_data.contributor_center[j]
            const segments_end = edge_branch_segmented_data.contributor_end[j]
            // calculate the segments of this edge branch
            let accumulated_ratio = 0
            const branch_weight = Math.abs(edge.w) / edge.v.length
            interface Segment {
                ni: number | null
                accumulated_ratio: number
                segment_ratio: number
            }
            const segments: Array<Segment> = []
            //     growing from end vertices
            for (const [ni, length] of segments_end) {
                const segment_ratio = length / branch_weight
                segments.push({ ni, accumulated_ratio, segment_ratio })
                accumulated_ratio += segment_ratio
            }
            //     the middle empty segment
            if (grown_end + grown_center < branch_weight) {
                const segment_ratio = (branch_weight - grown_end - grown_center) / branch_weight
                segments.push({ ni: null, accumulated_ratio, segment_ratio })
                accumulated_ratio += segment_ratio
            }
            //     growing from center vertices
            for (let index = segments_center.length - 1; index >= 0; index--) {
                const [ni, length] = segments_center[index]
                const segment_ratio = length / branch_weight
                segments.push({ ni, accumulated_ratio, segment_ratio })
                accumulated_ratio += segment_ratio
            }
            // create the segments
            for (const { ni, accumulated_ratio, segment_ratio } of segments) {
                if (edge.v.length != 1) {
                    const state = new EdgeTubeState(i, segment_position_of(accumulated_ratio), edge_length * segment_ratio, edge.v.length, quaternion)
                    if (subgraph_set.value[i]) {
                        // display the solid blue edge as subgraph
                        state.color = config.value.edge.color_palette.subgraph
                        edge_states.tight_edge_tubes.push(state)
                    } else if (snapshot.subgraph != null) {
                        // do not display anything if the intention of this snapshot is to display the subgraph
                        state.color = config.value.edge.color_palette.ungrown
                        edge_states.ungrown_edge_tubes.push(state)
                    } else if (ni != null) {
                        state.color = config.value.edge.color_palette.get(ni)
                        if (is_tight) {
                            edge_states.tight_edge_tubes.push(state)
                        } else {
                            edge_states.grown_edge_tubes.push(state)
                        }
                    } else {
                        state.color = config.value.edge.color_palette.ungrown
                        edge_states.ungrown_edge_tubes.push(state)
                    }
                } else {
                    const outline_ratio = config.value.vertex.outline_ratio
                    const func = (ratio: number) => outline_ratio + (config.value.edge.deg_1_ratio - 1) * outline_ratio * ratio
                    const state = new EdgeRingState(i, segment_position_of(accumulated_ratio))
                    state.inner = func(accumulated_ratio)
                    state.outer = func(segment_ratio + accumulated_ratio)
                    const ring_index = ring_index_of_ratio(state.inner / state.outer)
                    if (subgraph_set.value[i]) {
                        // display the solid blue edge as subgraph
                        state.color = config.value.edge.color_palette.subgraph
                        edge_states.tight_edge_rings[ring_index].push(state)
                    } else if (snapshot.subgraph != null) {
                        // do not display anything if the intention of this snapshot is to display the subgraph
                        state.color = config.value.edge.color_palette.ungrown
                        edge_states.ungrown_edge_rings[ring_index].push(state)
                    } else if (ni != null) {
                        state.color = config.value.edge.color_palette.get(ni)
                        if (is_tight) {
                            edge_states.tight_edge_rings[ring_index].push(state)
                        } else {
                            edge_states.grown_edge_rings[ring_index].push(state)
                        }
                    } else {
                        state.color = config.value.edge.color_palette.ungrown
                        edge_states.ungrown_edge_rings[ring_index].push(state)
                    }
                }
            }
        }
    }
    return edge_states
})

function applyMeshEdgeRings(singular_edges: Array<Array<EdgeRingState>>, refs: any) {
    updateEdgeRingsMatrix(singular_edges, refs)
    updateEdgeRingsColor(singular_edges, refs)
    for (let i = 0; i < refs.length; i++) {
        const mesh = refs[i].mesh
        mesh.userData.vecData = singular_edges[i]
    }
}

function updateEdgeRingsMatrix(singular_edges: Array<Array<EdgeRingState>>, refs: any) {
    for (let i = 0; i < refs.length; i++) {
        const mesh = refs[i].mesh
        const edges = singular_edges[i]
        for (let j = 0; j < edges.length; j++) {
            const state = edges[j]
            const dummy = new Object3D()
            dummy.rotateX(Math.PI / 2)
            const radius_ratio = config.value.vertex.radius * state.outer
            dummy.scale.set(radius_ratio, radius_ratio, 1)
            dummy.position.copy(state.position.clone())
            dummy.updateMatrix()
            mesh.setMatrixAt(j, dummy.matrix)
        }
        mesh.instanceMatrix.needsUpdate = true
    }
}

function updateEdgeRingsColor(singular_edges: Array<Array<EdgeRingState>>, refs: any) {
    for (let i = 0; i < refs.length; i++) {
        const mesh = refs[i].mesh
        const edges = singular_edges[i]
        for (let j = 0; j < edges.length; j++) {
            const state = edges[j]
            if (selected_eis?.has(state.ei)) {
                mesh.setColorAt(j, new Color(config.value.basic.selected_color))
            } else if (hovered_eis?.has(state.ei)) {
                mesh.setColorAt(j, new Color(config.value.basic.hovered_color))
            } else {
                mesh.setColorAt(j, new Color(state.color))
            }
        }
        if (mesh.instanceColor) {
            mesh.instanceColor.needsUpdate = true
        }
    }
}

function applyMeshEdgeTubes(edges: Array<EdgeTubeState>, mesh: any) {
    updateEdgeTubesMatrix(edges, mesh)
    updateEdgeTubesColor(edges, mesh)
    mesh.userData.vecData = edges
}

function updateEdgeTubesMatrix(edges: Array<EdgeTubeState>, mesh: any) {
    for (let i = 0; i < edges.length; i++) {
        const state = edges[i]
        const dummy = new Object3D()
        const radius_ratio = config.value.edge.ratio_of_deg(state.degree)
        dummy.scale.set(radius_ratio, state.length, radius_ratio)
        dummy.setRotationFromQuaternion(state.quaternion)
        dummy.position.copy(state.position.clone())
        dummy.translateY(state.length / 2)
        dummy.updateMatrix()
        mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
}

function updateEdgeTubesColor(edges: Array<EdgeTubeState>, mesh: any) {
    for (let i = 0; i < edges.length; i++) {
        const state = edges[i]
        if (selected_eis?.has(state.ei)) {
            mesh.setColorAt(i, new Color(config.value.basic.selected_color))
        } else if (hovered_eis?.has(state.ei)) {
            mesh.setColorAt(i, new Color(config.value.basic.hovered_color))
        } else {
            mesh.setColorAt(i, new Color(state.color))
        }
    }
    if (mesh.instanceColor) {
        mesh.instanceColor.needsUpdate = true
    }
}

const ungrown_edge_rings_ref = useTemplateRef('ungrown_edge_rings')
const grown_edge_rings_ref = useTemplateRef('grown_edge_rings')
const tight_edge_rings_ref = useTemplateRef('tight_edge_rings')
const ungrown_edge_tubes_ref = useTemplateRef('ungrown_edge_tubes')
const grown_edge_tubes_ref = useTemplateRef('grown_edge_tubes')
const tight_edge_tubes_ref = useTemplateRef('tight_edge_tubes')

const update_ungrown_edge_rings = () => applyMeshEdgeRings(edge_states.value.ungrown_edge_rings, ungrown_edge_rings_ref.value)
const update_grown_edge_rings = () => applyMeshEdgeRings(edge_states.value.grown_edge_rings, grown_edge_rings_ref.value)
const update_tight_edge_rings = () => applyMeshEdgeRings(edge_states.value.tight_edge_rings, tight_edge_rings_ref.value)
const update_ungrown_edge_tubes = () => applyMeshEdgeTubes(edge_states.value.ungrown_edge_tubes, (ungrown_edge_tubes_ref.value as any).mesh)
const update_grown_edge_tubes = () => applyMeshEdgeTubes(edge_states.value.grown_edge_tubes, (grown_edge_tubes_ref.value as any).mesh)
const update_tight_edge_tubes = () => applyMeshEdgeTubes(edge_states.value.tight_edge_tubes, (tight_edge_tubes_ref.value as any).mesh)
function update() {
    update_ungrown_edge_rings()
    update_grown_edge_rings()
    update_tight_edge_rings()
    update_ungrown_edge_tubes()
    update_grown_edge_tubes()
    update_tight_edge_tubes()
}
function update_intersect_color() {
    updateEdgeRingsColor(edge_states.value.ungrown_edge_rings, ungrown_edge_rings_ref.value)
    updateEdgeRingsColor(edge_states.value.grown_edge_rings, grown_edge_rings_ref.value)
    updateEdgeRingsColor(edge_states.value.tight_edge_rings, tight_edge_rings_ref.value)
    updateEdgeTubesColor(edge_states.value.ungrown_edge_tubes, (ungrown_edge_tubes_ref.value as any).mesh)
    updateEdgeTubesColor(edge_states.value.grown_edge_tubes, (grown_edge_tubes_ref.value as any).mesh)
    updateEdgeTubesColor(edge_states.value.tight_edge_tubes, (tight_edge_tubes_ref.value as any).mesh)
}

let hovered_eis: Set<number> | undefined = undefined
let selected_eis: Set<number> | undefined = undefined
function get_eis(info: any): Set<number> | undefined {
    if (info?.type == 'edge') {
        return new Set([info.ei])
    }
    if (info?.type == 'edges') {
        return new Set(info.eis)
    }
}

onMounted(() => {
    // when anything changes, update the mesh
    watchEffect(() => update())
    // update color
    watchEffect(() => {
        // update the color when the hovered or selected state changes
        hovered_eis = get_eis(config.value.data.hovered)
        selected_eis = get_eis(config.value.data.selected)
        update_intersect_color()
    })
})

const subgraph_set = computed(() => {
    const snapshot = config.value.snapshot
    let subgraph_set: { [edge_index: number]: boolean } = {}
    if (snapshot.subgraph != null) {
        for (let edge_index of snapshot.subgraph) {
            subgraph_set[edge_index] = true
        }
    }
    return subgraph_set
})

// calculate the edge offset because of the vertex radius
const edge_offset = computed(() => {
    if (config.value.edge.radius < config.value.vertex.outline_radius) {
        return Math.sqrt(Math.pow(config.value.vertex.outline_radius, 2) - Math.pow(config.value.edge.radius, 2))
    }
    return 0
})

// calculate the dual variable indices for each edge
const edge_to_dual_indices = computed(() => {
    const snapshot = config.value.snapshot
    return compute_edge_to_dual_indices(snapshot)
})

/**
 * Idea from Katie 2024.10.2: draw each edge branch differently to show which vertices the dual variables contribute
 *
 * The previous visualization is to display all each hypergraph in deg_v branches, and each branch is identical.
 *     Then for each branch, we print the contribution of all the dual variables. This method is very simple, and essentially
 *     convert the hyperedge printing problem to deg_v number of simple edge printing. However, this method will not
 *     convey the information of which vertices are the dual variables "flooding" from. For example, a single defect vertex
 *     grows over its adjacent hyperedges, however this method does not allow readers to get the information of which vertex
 *     is growing by looking at the edge along.
 *
 * We then found a new method to display it better: Since we know the subset of vertices that a dual variable contributes,
 *     namely $e \cap V_S$, we can grow from these vertices and then show the direction of the dual variable. This method
 *     is stable, in a sense that a small change of dual variables corresponds to a small change of the visualization effect,
 *     given that the dual variables have a consistent order (by their indices).
 *
 * This function outputs an object describing the segments on each edge branch.
 */

interface EdgeBranchSegments {
    grown_end: Array<number>
    grown_center: Array<number>
    contributor_end: Array<Array<[number, number]>>
    contributor_center: Array<Array<[number, number]>>
}

const edge_branch_segments = computed(() => {
    const snapshot = config.value.snapshot
    let edge_branch_segments: Array<EdgeBranchSegments | null> = []
    for (let edge_index = 0; edge_index < snapshot.edges.length; ++edge_index) {
        edge_branch_segments.push(calculate_edge_branch_segmented(edge_index))
    }
    return edge_branch_segments
})

function calculate_edge_branch_segmented(edge_index: number): EdgeBranchSegments | null {
    const snapshot = config.value.snapshot
    // calculate the list of contributing dual variables
    let dual_indices = []
    let edge = snapshot.edges[edge_index]
    if (edge == null) {
        return null
    }
    if (snapshot.dual_nodes != null) {
        // check the non-zero contributing dual variables
        for (let node_index of edge_to_dual_indices.value[edge_index]) {
            if (snapshot.dual_nodes[node_index].d != 0) {
                dual_indices.push(node_index)
            }
        }
    }
    // the grown value for each edge branch
    let grown_end = Array(edge.v.length).fill(0)
    let grown_center = Array(edge.v.length).fill(0)
    // the contributing dual variables from the end vertex and the center vertex, respectively
    let contributor_end: Array<Array<[number, number]>> = Array.from({ length: edge.v.length }, () => [])
    let contributor_center: Array<Array<[number, number]>> = Array.from({ length: edge.v.length }, () => [])
    if (snapshot.dual_nodes == null || dual_indices.length == 0) {
        return { grown_end, grown_center, contributor_end, contributor_center }
    }
    // iterate over all dual variables and put them on the edge branches
    let branch_weight = Math.abs(edge.w) / edge.v.length
    for (let ni of dual_indices) {
        const node = snapshot.dual_nodes[ni]
        // calculate the contributing vertices of this dual variable: $e \cap V_S$
        let vertices = []
        let v_s = new Set(snapshot.dual_nodes[ni].v)
        for (let [v_eid, v] of edge.v.entries()) {
            if (v_s.has(v)) {
                vertices.push(v_eid)
            }
        }
        if (vertices.length == 0) {
            // this doesn't make sense, but we should not crash the program
            for (let [v_eid, _v] of edge.v.entries()) {
                vertices.push(v_eid)
            }
        }
        let center_grow = 0 // the amount of growth that must happen at the center because some edge branch is already tight
        let branch_growth = node.d / vertices.length
        // first, grow from end vertices, each with `branch_growth`
        for (let v_eid of vertices) {
            let remain = branch_weight - grown_end[v_eid] - grown_center[v_eid]
            if (branch_growth <= remain) {
                grown_end[v_eid] += branch_growth
                contributor_end[v_eid].push([ni, branch_growth])
            } else {
                grown_end[v_eid] += remain
                contributor_end[v_eid].push([ni, remain])
                center_grow += branch_growth - remain
            }
        }
        // then, grow from center vertices
        while (center_grow > 0) {
            let available_vertices = []
            let min_positive_remain = null
            for (let [v_eid] of edge.v.entries()) {
                let remain = branch_weight - grown_end[v_eid] - grown_center[v_eid]
                if (remain > 0) {
                    available_vertices.push(v_eid)
                    if (min_positive_remain == null) {
                        min_positive_remain = remain
                    } else if (remain < min_positive_remain) {
                        min_positive_remain = remain
                    }
                }
            }
            if (min_positive_remain == null) {
                if (center_grow > 1e-6) {
                    // tolerance of error
                    console.error(`need to grow from center of ${center_grow} but all branches are fully grown`)
                }
                break
            }
            // in this round, we can only grow `min_positive_remain` on the branches of `available_vertices`
            if (min_positive_remain > center_grow / available_vertices.length) {
                min_positive_remain = center_grow / available_vertices.length
            }
            center_grow -= min_positive_remain * available_vertices.length
            for (let v_eid of available_vertices) {
                grown_center[v_eid] += min_positive_remain
                const center = contributor_center[v_eid]
                // grow from center, potentially merging with existing segments
                if (center.length > 0 && center[center.length - 1][0] == ni) {
                    center[center.length - 1][1] += min_positive_remain
                } else {
                    center.push([ni, min_positive_remain])
                }
            }
        }
    }
    return { grown_end, grown_center, contributor_end, contributor_center }
}
</script>

<template>
    <!-- ungrown edge rings -->
    <MyInstancedMesh
        v-for="(_, idx) in ring_resolution + 1"
        :key="idx"
        ref="ungrown_edge_rings"
        :count="edge_states.ungrown_edge_rings[idx].length"
        @reinstantiated="update_ungrown_edge_rings"
    >
        <RingGeometry :inner-radius="idx / ring_resolution" :outer-radius="1" :theta-segments="config.basic.segments"> </RingGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.ungrown_opacity, side: DoubleSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- partially grown edge rings -->
    <MyInstancedMesh
        v-for="(_, idx) in ring_resolution + 1"
        :key="idx"
        ref="grown_edge_rings"
        :count="edge_states.grown_edge_rings[idx].length"
        @reinstantiated="update_grown_edge_rings"
    >
        <RingGeometry :inner-radius="idx / ring_resolution" :outer-radius="1" :theta-segments="config.basic.segments"> </RingGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.grown_opacity, side: DoubleSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- tight edge rings -->
    <MyInstancedMesh
        v-for="(_, idx) in ring_resolution + 1"
        :key="idx"
        ref="tight_edge_rings"
        :count="edge_states.tight_edge_rings[idx].length"
        @reinstantiated="update_tight_edge_rings"
    >
        <RingGeometry :inner-radius="idx / ring_resolution" :outer-radius="1" :theta-segments="config.basic.segments"> </RingGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.tight_opacity, side: DoubleSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- ungrown edge tubes -->
    <MyInstancedMesh ref="ungrown_edge_tubes" :count="edge_states.ungrown_edge_tubes.length" @reinstantiated="update_ungrown_edge_tubes">
        <CylinderGeometry
            :radius-top="config.edge.radius"
            :radius-bottom="config.edge.radius"
            :height="1"
            :height-segments="1"
            :radial-segments="config.basic.segments"
            :open-ended="true"
        >
        </CylinderGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.ungrown_opacity, side: BackSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- partially grown edge tubes -->
    <MyInstancedMesh ref="grown_edge_tubes" :count="edge_states.grown_edge_tubes.length" @reinstantiated="update_grown_edge_tubes">
        <CylinderGeometry
            :radius-top="config.edge.radius"
            :radius-bottom="config.edge.radius"
            :height="1"
            :height-segments="1"
            :radial-segments="config.basic.segments"
            :open-ended="true"
        >
        </CylinderGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.grown_opacity, side: BackSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!--tight edge tubes -->
    <MyInstancedMesh ref="tight_edge_tubes" :count="edge_states.tight_edge_tubes.length" @reinstantiated="update_tight_edge_tubes">
        <CylinderGeometry
            :radius-top="config.edge.radius"
            :radius-bottom="config.edge.radius"
            :height="1"
            :height-segments="1"
            :radial-segments="config.basic.segments"
            :open-ended="true"
        >
        </CylinderGeometry>
        <PhysicalMaterial :props="{ transparent: true, opacity: config.edge.tight_opacity, side: DoubleSide }"></PhysicalMaterial>
    </MyInstancedMesh>
</template>
