<script setup lang="ts">
import { computed, useTemplateRef, ref } from 'vue'
import { NButton, NIcon, NTooltip } from 'naive-ui'
import { ParityMatrixData } from './parser'
import { SaveSharp as SaveIcon, Open as OpenIcon, LogoHtml5, AlertCircle } from '@vicons/ionicons5'

const props = defineProps({
    object: {
        type: Object,
        required: true,
    },
})

const data = computed(() => {
    return new ParityMatrixData(props.object)
})
const show_echelon_info = ref(true)
function toggle_echelon_info() {
    show_echelon_info.value = !show_echelon_info.value
}
const num_rows = computed(() => {
    if (!show_echelon_info.value && data.value.is_echelon_form) {
        return data.value.table.rows.length - 1
    }
    return data.value.table.rows.length
})
const num_columns = computed(() => {
    if (!show_echelon_info.value && data.value.is_echelon_form) {
        return data.value.table.rows[0].length - 1
    }
    return data.value.table.rows[0].length
})
const h = 22

const show_svg = ref(true)
function toggle_html_svg() {
    show_svg.value = !show_svg.value
}

function is_data_block(ri: number, ci: number): boolean {
    const [rows, columns] = data.value.table.dimension
    if (!data.value.is_echelon_form) {
        return ri > 0 && ri < rows && ci > 0 && ci < columns
    } else {
        return ri > 0 && ri < rows - 1 && ci > 0 && ci < columns - 1
    }
}

function is_rhs(ri: number, ci: number): boolean {
    return is_data_block(ri, ci) && !is_data_block(ri, ci + 1)
}

function is_line_head(ri: number, ci: number): boolean {
    const [rows] = data.value.table.dimension
    return ri != 0 && ci == 0 && !(data.value.is_echelon_form && ri == rows - 1)
}

function is_echelon_info_row(ri: number, _ci: number): boolean {
    const [rows] = data.value.table.dimension
    return data.value.is_echelon_form && ri == rows - 1
}

function is_echelon_info_col(_ri: number, ci: number): boolean {
    const [, columns] = data.value.table.dimension
    return data.value.is_echelon_form && ci == columns - 1
}

function block_content(ri: number, ci: number): string {
    const element = data.value.table.rows[ri].elements[ci]
    if (is_rhs(ri, ci)) {
        // the default output includes extra space to have a width of three, so to distinguish with others
        // we don't really need it here because the HTML can have different colors
        return element.trim()
    }
    return element
}

function is_corner_block(ri: number, ci: number): boolean {
    if (!is_data_block(ri, ci) || is_rhs(ri, ci)) {
        return false
    }
    const tail_start_index = data.value.tail_start_index
    const corner_row_index = data.value.corner_row_index
    if (tail_start_index != null && corner_row_index != null && ri > corner_row_index && ci > tail_start_index) {
        return true
    }
    return false
}

function is_tail_columns(ri: number, ci: number): boolean {
    if (!is_data_block(ri, ci) || is_rhs(ri, ci) || is_corner_block(ri, ci)) {
        return false
    }
    const tail_start_index = data.value.tail_start_index
    if (tail_start_index != null && ci > tail_start_index) {
        return true
    }
    return false
}

const svg = useTemplateRef('svg')

function open_svg_in_new_tab() {
    const w = window.open('', '')
    if (w == null) {
        alert('cannot open new window')
        return
    }
    w.document.title = 'party matrix'
    w.document.body.style.backgroundColor = 'white'
    w.document.body.style.margin = '0'
    w.document.body.appendChild(svg!.value!.cloneNode(true))
}

function download_svg() {
    const svg_data = svg!.value!.outerHTML
    const svg_blob = new Blob([svg_data], { type: 'image/svg+xml;charset=utf-8' })
    const data_url = URL.createObjectURL(svg_blob)
    const a = document.createElement('a')
    a.href = data_url
    a.download = 'matrix.svg'
    a.click()
}

function svg_color_of_text(ri: number, ci: number): string | undefined {
    if (is_echelon_info_row(ri, ci)) {
        return 'lightgrey'
    }
    if (is_echelon_info_col(ri, ci)) {
        return 'lightgrey'
    }
}
function svg_background(ri: number, ci: number): string | undefined {
    if (ri == 0 && !is_echelon_info_col(ri, ci)) {
        return 'lightblue'
    }
    if (is_line_head(ri, ci)) {
        return 'lightpink'
    }
    if (is_rhs(ri, ci)) {
        return 'lightsalmon'
    }
    if (is_tail_columns(ri, ci)) {
        return 'lightcyan'
    }
    if (is_corner_block(ri, ci)) {
        return 'lightgreen'
    }
}
</script>

<template>
    <div class="box">
        <div class="toolbox">
            <n-tooltip placement="bottom" trigger="hover">
                <template #trigger>
                    <n-button strong secondary circle @click="toggle_html_svg" class="button">
                        <template #icon>
                            <n-icon color="grey"><logo-html5 /></n-icon>
                        </template>
                    </n-button>
                </template>
                <span>Toggle HTML/SVG</span>
            </n-tooltip>
            <n-tooltip v-if="show_svg" placement="bottom" trigger="hover">
                <template #trigger>
                    <n-button strong secondary circle @click="toggle_echelon_info" class="button">
                        <template #icon>
                            <n-icon color="grey"><alert-circle /></n-icon>
                        </template>
                    </n-button>
                </template>
                <span>Toggle Echelon Info</span>
            </n-tooltip>
            <n-tooltip v-if="show_svg" placement="bottom" trigger="hover">
                <template #trigger>
                    <n-button strong secondary circle @click="open_svg_in_new_tab" class="button">
                        <template #icon>
                            <n-icon color="blue"><open-icon /></n-icon>
                        </template>
                    </n-button>
                </template>
                <span>Open in new Tab</span>
            </n-tooltip>
            <n-tooltip v-if="show_svg" placement="bottom" trigger="hover">
                <template #trigger>
                    <n-button strong secondary circle @click="download_svg" class="button">
                        <template #icon>
                            <n-icon color="orange"><save-icon /></n-icon>
                        </template>
                    </n-button>
                </template>
                <span>Download</span>
            </n-tooltip>
        </div>
        <table v-show="!show_svg">
            <tr v-for="(row, ri) in data.table.rows" :key="ri">
                <th
                    v-for="(element, ci) of row.elements"
                    :key="ci"
                    :class="{
                        title: ri == 0 && !is_echelon_info_col(ri, ci),
                        square: is_data_block(ri, ci),
                        'line-head': is_line_head(ri, ci),
                        rhs: is_rhs(ri, ci),
                        'echelon-info-row': is_echelon_info_row(ri, ci),
                        'echelon-info-col': is_echelon_info_col(ri, ci),
                        'tail-columns': is_tail_columns(ri, ci),
                        'corner-block': is_corner_block(ri, ci),
                    }"
                >
                    {{ block_content(ri, ci) }}
                </th>
            </tr>
        </table>
        <svg v-show="show_svg" :width="2 + num_columns * (h + 1)" :height="2 + num_rows * (h + 1)" xmlns="http://www.w3.org/2000/svg" ref="svg">
            <g font-size="14" font-family="arial" font-weight="bold" stroke-width="0" stroke="#666" text-anchor="middle">
                <g v-for="(row, ri) in data.table.rows" :key="ri">
                    <g v-for="(element, ci) of row.elements" :key="ci">
                        <rect v-if="svg_background(ri, ci) != undefined" :width="h" :height="h" :x="1 + (h + 1) * ci" :y="1 + (h + 1) * ri" :fill="svg_background(ri, ci)" />
                        <text :width="h" :height="h" :x="1 + h / 2 + (h + 1) * ci" :y="6 + h / 2 + (h + 1) * ri" :textLength="h - 5" :fill="svg_color_of_text(ri, ci)">
                            {{ element }}
                        </text>
                    </g>
                </g>
                <g stroke-width="1">
                    <line
                        v-for="ri in num_rows + 1"
                        :key="ri"
                        x1="0"
                        :y1="0.5 + (h + 1) * (ri - 1)"
                        :x2="1 + (h + 1) * num_columns"
                        :y2="0.5 + (h + 1) * (ri - 1)"
                        :stroke-width="data.corner_row_index != null && data.corner_row_index + 2 == ri ? 3 : undefined"
                        :stroke="data.corner_row_index != null && data.corner_row_index + 2 == ri ? 'red' : undefined"
                    />
                    <line
                        v-for="ci in num_columns + 1"
                        :key="ci"
                        :x1="0.5 + (h + 1) * (ci - 1)"
                        :y1="0"
                        :x2="0.5 + (h + 1) * (ci - 1)"
                        :y2="1 + (h + 1) * num_rows"
                        :stroke-width="data.tail_start_index != null && data.tail_start_index + 2 == ci ? 3 : undefined"
                        :stroke="data.tail_start_index != null && data.tail_start_index + 2 == ci ? 'red' : undefined"
                    />
                </g>
            </g>
        </svg>
    </div>
</template>

<style scoped>
.box {
    display: inline-block;
    margin: 10px;
    border: 0;
    min-width: 50px;
    min-height: 50px;
}

th {
    border: 1px solid grey;
    min-width: 22px;
    text-align: center !important;
}

table,
th,
tr {
    text-align: center;
    border-collapse: collapse;
    font-size: 14px;
    padding: 0;
    white-space: pre;
    line-height: 14px;
    color: black;
    background-color: white;
}

.square {
    font-size: 18px;
    font-family: monospace;
    width: 22px;
    height: 22px;
}

.title {
    /* color: red; */
    background-color: lightblue;
}

.line-head {
    background-color: lightpink;
}

.rhs {
    background-color: lightsalmon;
}

.toolbox {
    display: flex;
    justify-content: right;
    align-items: right;
    margin-bottom: 3px;
}

.button {
    margin: 3px;
}

.echelon-info-row {
    color: lightgrey;
    font-size: 80%;
}

.echelon-info-col {
    color: lightgrey;
    font-size: 80%;
}

.tail-columns {
    background-color: lightcyan;
}

.corner-block {
    background-color: lightgreen;
}
</style>
