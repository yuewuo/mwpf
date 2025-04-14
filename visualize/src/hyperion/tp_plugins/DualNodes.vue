<script setup lang="ts">
import Indices from './Indices.vue'
import { Info } from '../info_pane'
import { display_nominator } from '@/util'
import { computed, type ComputedRef } from 'vue'
import { type DualNode } from '../hyperion'

interface Props {
    info: Info
    dual_indices?: number[]
}

const props = defineProps<Props>()
const config = props.info.config

interface IndexedDualNode {
    ni: number
    dual_node: DualNode
}

const dual_nodes: ComputedRef<IndexedDualNode[]> = computed(() => {
    const snapshot_dual_nodes = config.snapshot.dual_nodes
    if (snapshot_dual_nodes == undefined) {
        return []
    }
    if (props.dual_indices) {
        return props.dual_indices.map(ni => {
            return { ni, dual_node: snapshot_dual_nodes[ni] }
        })
    } else {
        const dual_nodes = []
        for (const [ni, dual_node] of snapshot_dual_nodes.entries()) {
            dual_nodes.push({ ni, dual_node })
        }
        return dual_nodes
    }
})
</script>

<template>
    <div v-for="{ ni, dual_node } of dual_nodes" :key="ni">
        <div class="entry" v-if="props.info.display_zero_dual_variables || dual_node.d != 0">
            <div class="left">
                <div class="color-indicator" :style="{ 'background-color': config.edge.color_palette.get(ni) }"></div>
                <i style="font-size: 20px; position: relative; top: -13px">y</i>
                <i style="font-size: 13px; position: relative; top: -10px">S</i>
                <span style="display: inline-block; font-size: 10px; position: relative; top: -5px; width: 16px">{{ ni }}</span>
                <i style="font-size: 16px; position: relative; top: -11px">&nbsp;=&nbsp;</i>
                <div class="rational">
                    <div class="nominator limited-width">{{ display_nominator(dual_node.dn) }}</div>
                    <div class="rational-divider"></div>
                    <div class="denominator limited-width">{{ dual_node.dd }}</div>
                </div>
            </div>
            <div class="right">
                <Indices title="<i>V<i/>&nbsp;=" :width="148" :indices="dual_node.v" :info="props.info" index-type="vertex"></Indices>
                <div style="width: 4px; display: inline-block"></div>
                <Indices title="<i>E<i/>&nbsp;=" :width="148" :indices="dual_node.e" :info="props.info" index-type="edge"></Indices>
                <div style="height: 2px"></div>
                <Indices title="𝛅(<i>S</i>)&nbsp;=" :title-width="40" :indices="dual_node.h" :info="props.info" index-type="edge"></Indices>
            </div>
        </div>
    </div>
</template>

<style scoped>
.entry {
    font-family: 'Times New Roman', Times, serif;
    color: #bbbcc3;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    margin-right: 3px;
    margin-bottom: 3px;
}
.left {
    position: relative;
    padding-left: 5px;
    width: 100px;
    /* background-color: red; */
}
.right {
    width: 300px;
    /* background-color: blue; */
}
.rational {
    display: inline-block;
    height: 34px;
    width: 36px;
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
.color-indicator {
    position: absolute;
    width: 20px;
    height: 8px;
    border-radius: 5px;
}
</style>
