<script setup lang="ts">
import { computed } from 'vue'
import { Info } from '../info_pane'
import { compute_edge_to_dual_indices, compute_vertex_to_dual_indices, compute_vertex_incident_edges } from '../hyperion'
import Indices from './Indices.vue'
import EdgeDualSum from '../equations/EdgeDualSum.vue'
import { display_nominator } from '@/util'
import DualNodes from './DualNodes.vue'

interface Props {
    info: Info
}

const props = defineProps<Props>()
const config = props.info.config

const vi = computed(() => {
    const selected = config.data.selected
    if (selected?.type == 'vertex') {
        return selected.vi as number
    }
    return undefined
})

const vertex = computed(() => {
    if (vi.value == undefined) return
    return config.snapshot.vertices[vi.value]
})

const vertex_to_dual_indices = computed(() => {
    const snapshot = config.snapshot
    return compute_vertex_to_dual_indices(snapshot)
})

const vertex_incident_edges = computed(() => {
    const snapshot = config.snapshot
    return compute_vertex_incident_edges(snapshot)
})

const vertex_involving_nodes = computed(() => {
    if (vi.value == undefined) return
    let dual_indices = []
    if (config.snapshot.dual_nodes != null) {
        // check the non-zero contributing dual variables
        for (let node_index of vertex_to_dual_indices.value[vi.value]) {
            if (config.snapshot.dual_nodes[node_index].d != 0 || props.info.display_zero_dual_variables) {
                dual_indices.push(node_index)
            }
        }
    }
    return dual_indices
})

const ei = computed(() => {
    const selected = config.data.selected
    if (selected?.type == 'edge') {
        return selected.ei as number
    }
    return undefined
})

const edge = computed(() => {
    if (ei.value == undefined) return
    return config.snapshot.edges[ei.value]
})

const edge_to_dual_indices = computed(() => {
    const snapshot = config.snapshot
    return compute_edge_to_dual_indices(snapshot)
})

const edge_contributing_nodes = computed(() => {
    if (ei.value == undefined) return
    let dual_indices = []
    if (config.snapshot.dual_nodes != null) {
        // check the non-zero contributing dual variables
        for (let node_index of edge_to_dual_indices.value[ei.value]) {
            if (config.snapshot.dual_nodes[node_index].d != 0 || props.info.display_zero_dual_variables) {
                dual_indices.push(node_index)
            }
        }
    }
    return dual_indices
})
</script>

<template>
    <div class="div">
        <div v-if="vi != undefined">
            <div class="title">{{ vertex!.s ? 'Defect' : 'Normal' }} Vertex {{ vi }}</div>
            <div style="margin-top: 10px">
                <math display="inline-block" style="font-size: 120%; math-style: normal; position: relative; top: 3px">
                    <mrow>
                        <mi>δ</mi>
                        <mn>(</mn>
                        <mi>v</mi>
                        <mn>)</mn>
                    </mrow>
                    <mo>=</mo>
                </math>
                <Indices :titleWidth="0" :width="345" :indices="vertex_incident_edges[vi]" :info="props.info" index-type="edge"></Indices>
            </div>
            <div class="title" style="margin-top: 10px">Dual Variables involving Vertex {{ vi }}</div>
            <div style="margin-top: 10px"></div>
            <DualNodes :info="info" :dual_indices="vertex_involving_nodes"></DualNodes>
        </div>
        <div v-if="ei != undefined">
            <div class="title">{{ edge!.g == null ? '' : edge!.g >= edge!.w ? 'Tight' : 'Loose' }} Edge {{ ei }}</div>
            <div style="margin-top: 10px">
                <math display="inline" style="font-size: 120%; position: relative; top: 3px">
                    <mi>V</mi>
                    <mn>(</mn>
                    <mi>e</mi>
                    <mn>)</mn>
                    <mo>=</mo>
                </math>
                <Indices :titleWidth="0" :width="335" :indices="edge!.v" :info="props.info" index-type="vertex"></Indices>
            </div>
            <div style="margin-top: 10px">
                <math display="inline-block" style="font-size: 150%; math-style: normal">
                    <msub>
                        <mi>w</mi>
                        <mi>e</mi>
                    </msub>
                    <mo>=</mo>
                    <mn>{{ edge!.w }}</mn>
                </math>
            </div>
            <div class="title" style="margin-top: 10px">Dual Variables contributing to Edge {{ ei }}</div>
            <div style="margin-top: 0px">
                <math display="inline-block" style="font-size: 120%; math-style: normal; position: relative; top: 3px">
                    <mrow>
                        <mo>{</mo>
                        <mi>S</mi>
                        <mo>|</mo>
                        <mi>S</mi>
                        <mn>∈</mn>
                        <mi>O</mi>
                        <mn>,</mn>
                        <mi>e</mi>
                        <mn>∈</mn>
                        <mi>δ</mi>
                        <mn>(</mn>
                        <mi>S</mi>
                        <mn>)</mn>
                        <mo>}</mo>
                    </mrow>
                    <mo>=</mo>
                </math>
                <Indices :titleWidth="0" :width="267" :indices="edge_contributing_nodes!" style="margin-top: 10px"></Indices>
            </div>
            <div style="margin-top: 10px">
                <math display="inline-block" style="font-size: 150%; math-style: compact; position: relative; top: -12px">
                    <EdgeDualSum />
                    <mo>=</mo>
                </math>
                <div class="rational">
                    <div class="nominator limited-width">{{ display_nominator(edge!.gn) }}</div>
                    <div class="rational-divider"></div>
                    <div class="denominator limited-width">{{ edge!.gd }}</div>
                </div>
                <math display="inline-block" style="font-size: 150%; math-style: compact; position: relative; top: -12px">
                    <mo>=</mo>
                    <mn class="limited-width" style="width: 170px">{{ edge!.g }}</mn>
                </math>
            </div>
            <div style="margin-top: 0px">
                <math display="inline-block" style="font-size: 150%; math-style: compact; position: relative; top: -12px">
                    <msub>
                        <mi>w</mi>
                        <mi>e</mi>
                    </msub>
                    <mo>-</mo>
                    <EdgeDualSum />
                    <mo>=</mo>
                </math>
                <div class="rational">
                    <div class="nominator limited-width">{{ display_nominator(edge!.un) }}</div>
                    <div class="rational-divider"></div>
                    <div class="denominator limited-width">{{ edge!.ud }}</div>
                </div>
                <math display="inline-block" style="font-size: 150%; math-style: compact; position: relative; top: -12px">
                    <mo>=</mo>
                    <mn class="limited-width" style="width: 140px">{{ edge!.u }}</mn>
                </math>
            </div>
            <DualNodes :info="info" :dual_indices="edge_contributing_nodes"></DualNodes>
        </div>
    </div>
</template>

<style scoped>
.div {
    color: #bbbcc3;
}
.title {
    font-size: 120%;
    text-align: center;
}
.rational {
    display: inline-block;
    height: 34px;
    width: 80px;
}
.rational-divider {
    height: 0.5px;
    background-color: white;
}
.nominator {
    height: 13px;
    padding-top: 3px;
}
.denominator {
    height: 14px;
    padding-top: 2px;
}
.limited-width {
    text-align: center;
    overflow-y: scroll;
    scrollbar-width: none;
    scroll-padding: 0;
}
</style>
