<script setup lang="ts">
import { type Ref, computed, watchEffect, useTemplateRef, onMounted } from 'vue'
import { VertexState, VertexStates, load_position } from './hyperion'
import { type Config } from './config_pane'
import { PhysicalMaterial, SphereGeometry } from 'troisjs'
import MyInstancedMesh from '@/misc/MyInstancedMesh.vue'
import { Object3D, FrontSide, BackSide, Color } from 'three'
import { assert_inject } from '@/misc/util'

const config: Ref<Config> = assert_inject('config')

const vertex_states = computed(() => {
    const vertex_states = new VertexStates()
    const snapshot = config.value.snapshot
    for (const [i, vertex] of snapshot.vertices.entries()) {
        if (vertex == null) {
            continue
        }
        const state = new VertexState(i, config.value.data.visualizer.positions[i])
        vertex_states.all_outlines.push({ ...state })
        if (vertex.s) {
            vertex_states.defect_vertices.push({ ...state, color: config.value.vertex.defect_color })
        } else {
            vertex_states.normal_vertices.push({ ...state, color: config.value.vertex.normal_color })
        }
    }
    return vertex_states
})

function applyMeshVertices(vertex_states: Array<VertexState>, mesh: any) {
    updateVerticesMatrix(vertex_states, mesh)
    updateVerticesColor(vertex_states, mesh)
    mesh.userData.vecData = vertex_states
}

function updateVerticesMatrix(vertex_states: Array<VertexState>, mesh: any) {
    for (let i = 0; i < vertex_states.length; i++) {
        const state = vertex_states[i]
        const dummy = new Object3D()
        load_position(dummy.position, state.position)
        dummy.updateMatrix()
        mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
}

function updateVerticesColor(vertex_states: Array<VertexState>, mesh: any) {
    for (let i = 0; i < vertex_states.length; i++) {
        const state = vertex_states[i]
        if (selected_vis?.has(state.vi)) {
            mesh.setColorAt(i, new Color(config.value.basic.selected_color))
        } else if (hovered_vis?.has(state.vi)) {
            mesh.setColorAt(i, new Color(config.value.basic.hovered_color))
        } else {
            mesh.setColorAt(i, new Color(state.color))
        }
    }
    if (mesh.instanceColor) {
        mesh.instanceColor.needsUpdate = true
    }
}

let hovered_vis: Set<number> | undefined = undefined
let selected_vis: Set<number> | undefined = undefined
function get_vis(info: any): Set<number> | undefined {
    if (info?.type == 'vertex') {
        return new Set([info.vi])
    }
    if (info?.type == 'vertices') {
        return new Set(info.vis)
    }
}

const normal_vertices_ref = useTemplateRef('normal_vertices')
const defect_vertices_ref = useTemplateRef('defect_vertices')
const vertices_outlines_ref = useTemplateRef('vertices_outlines')

const update_normal_vertices = () => applyMeshVertices(vertex_states.value.normal_vertices, (normal_vertices_ref.value as any).mesh)
const update_defect_vertices = () => applyMeshVertices(vertex_states.value.defect_vertices, (defect_vertices_ref.value as any).mesh)
const update_vertices_outlines = () => applyMeshVertices(vertex_states.value.all_outlines, (vertices_outlines_ref.value as any).mesh)
function update() {
    update_normal_vertices()
    update_defect_vertices()
    update_vertices_outlines()
}
function update_intersect_color() {
    updateVerticesColor(vertex_states.value.normal_vertices, (normal_vertices_ref.value as any).mesh)
    updateVerticesColor(vertex_states.value.defect_vertices, (defect_vertices_ref.value as any).mesh)
    updateVerticesColor(vertex_states.value.all_outlines, (vertices_outlines_ref.value as any).mesh)
}

onMounted(() => {
    // when anything changes, update the mesh
    watchEffect(() => update())
    // update color
    watchEffect(() => {
        // update the color when the hovered or selected state changes
        hovered_vis = get_vis(config.value.data.hovered)
        selected_vis = get_vis(config.value.data.selected)
        update_intersect_color()
    })
})
</script>

<template>
    <!-- note: color must be set individually, because otherwise ThreeJS will multiply the color values and create strange visual effects -->

    <!-- normal vertices -->
    <MyInstancedMesh ref="normal_vertices" :count="vertex_states.normal_vertices.length" @reinstantiated="update_normal_vertices">
        <SphereGeometry :radius="config.vertex.radius" :height-segments="config.basic.segments" :width-segments="config.basic.segments"> </SphereGeometry>
        <PhysicalMaterial :props="{ transparent: false, side: FrontSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- defect vertices -->
    <MyInstancedMesh ref="defect_vertices" :count="vertex_states.defect_vertices.length" @reinstantiated="update_defect_vertices">
        <SphereGeometry :radius="config.vertex.radius" :height-segments="config.basic.segments" :width-segments="config.basic.segments"> </SphereGeometry>
        <PhysicalMaterial :props="{ transparent: false, side: FrontSide }"></PhysicalMaterial>
    </MyInstancedMesh>

    <!-- all vertices outlines -->
    <MyInstancedMesh ref="vertices_outlines" :count="vertex_states.all_outlines.length" @reinstantiated="update_vertices_outlines">
        <SphereGeometry :radius="config.vertex.outline_radius" :height-segments="config.basic.segments" :width-segments="config.basic.segments"> </SphereGeometry>
        <PhysicalMaterial :props="{ transparent: false, side: BackSide, color: config.vertex.outline_color }"></PhysicalMaterial>
    </MyInstancedMesh>
</template>
