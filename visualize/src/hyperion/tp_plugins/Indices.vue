<script setup lang="ts">
import { Info } from '../info_pane'
import { computed } from 'vue'

interface Props {
    info?: Info
    indexType?: string
    title?: string
    indices: Array<number>
    titleWidth?: number
    width?: number
}
const props = withDefaults(defineProps<Props>(), {
    titleWidth: 30,
    width: 300,
})

const clickable_types = new Set(['vertex', 'edge'])
const is_clickable = computed(() => {
    return props.info != undefined && props.indexType != undefined && clickable_types.has(props.indexType)
})

function enter_title() {
    if (!is_clickable.value) return
    if (props.indexType == 'vertex') {
        props.info!.config.data.hovered = { type: 'vertices', vis: props.indices }
    } else if (props.indexType == 'edge') {
        props.info!.config.data.hovered = { type: 'edges', eis: props.indices }
    }
}
function leave_title() {
    if (!is_clickable.value) return
    props.info!.config.data.hovered = undefined
}

function enter_index(idx: number) {
    if (!is_clickable.value) return
    if (props.indexType == 'vertex') {
        props.info!.config.data.hovered = { type: 'vertex', vi: idx }
    } else if (props.indexType == 'edge') {
        props.info!.config.data.hovered = { type: 'edge', ei: idx }
    }
}
function leave_index() {
    if (!is_clickable.value) return
    props.info!.config.data.hovered = undefined
}
function click_index(idx: number) {
    if (!is_clickable.value) return
    if (props.indexType == 'vertex') {
        props.info!.config.data.selected = { type: 'vertex', vi: idx }
    } else if (props.indexType == 'edge') {
        props.info!.config.data.selected = { type: 'edge', ei: idx }
    }
}
</script>

<template>
    <div class="div" :style="{ width: props.width + 'px' }">
        <div class="flex-div">
            <span
                class="title"
                :class="{ 'title-clickable': is_clickable }"
                v-html="props.title"
                :style="{ width: props.titleWidth + 'px' }"
                @mouseenter="enter_title"
                @mouseleave="leave_title"
            ></span>
            <div class="indices-div" :style="{ width: props.width - props.titleWidth + 'px' }">
                <div class="flex-div">
                    <span class="bracket">{</span>
                    <button
                        v-for="(idx, i) of props.indices"
                        :key="i"
                        class="idx-button"
                        :class="{ 'idx-button-clickable': is_clickable }"
                        @mouseenter="enter_index(idx)"
                        @mouseleave="leave_index"
                        @mousedown="click_index(idx)"
                    >
                        {{ idx }}
                    </button>
                    <span class="bracket">}</span>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.div {
    display: inline-block;
}
.title {
    text-align: center;
    display: inline-block;
    user-select: none;
}
.title-clickable:hover {
    color: #6fdfdf;
}
.indices-div {
    display: inline-block;
    background-color: #adafb7;
    overflow-x: scroll;
    scrollbar-width: none;
    color: #29292e;
    border-radius: 5px;
    padding: 1px 0 1px 0;
    height: 14px;
}
.bracket {
    line-height: 14px;
    position: relative;
    top: -1px;
}
.flex-div {
    display: flex;
    flex-direction: row;
    align-items: center;
}
.idx-button {
    border: solid black 1px;
    border-radius: 3px;
    height: 14px;
    display: inline;
    font-size: 10px;
    line-height: 10px;
    padding: 0 1px 1px 2px;
    margin: 0 1px 0 1px;
}
.idx-button-clickable:hover {
    color: #6fdfdf;
}
.idx-button-clickable:active {
    color: #4b7be5;
}
</style>
