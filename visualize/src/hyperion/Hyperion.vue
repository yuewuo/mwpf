<script setup lang="ts">
import { onMounted, ref, computed, provide, watchEffect, onBeforeUnmount, useTemplateRef, onUnmounted, defineExpose, getCurrentInstance } from 'vue'
import { OrthographicCamera, Scene, AmbientLight } from 'troisjs'
// import { Renderer } from 'troisjs' // use individual renderer for each instance
import Renderer from '@/misc/SharedRenderer.vue' // optimization: share a single WebGL renderer across all the instances
import { type VisualizerData, RuntimeData, ConfigProps, renderer_params, clickable_of } from './hyperion'
import { Config } from './config_pane'
import { Info } from './info_pane'
import Vertices from './Vertices.vue'
import Edges from './Edges.vue'
import { WebGLRenderer, OrthographicCamera as ThreeOrthographicCamera, Raycaster, Vector2 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import iconString from './icon.svg?raw'
// @ts-expect-error the Stats module does not have a declaration file
import Stats from 'troisjs/src/components/misc/Stats'

interface Props {
    visualizer: VisualizerData
    config: ConfigProps
}

const props = withDefaults(defineProps<Props>(), {
    config: () => new ConfigProps(),
})

const config = ref(new Config(new RuntimeData(props.visualizer), props.config))
const info = ref(new Info(config as any))
provide('config', config) // prop drilling to all children components

// update the icon of the web page if full screen is enabled, and also customize save
if (config.value.config_prop.full_screen) {
    const link = document.createElement('link')
    link.rel = 'icon'
    link.href = 'data:image/svg+xml;base64,' + btoa(iconString)
    document.head.appendChild(link)
    document.addEventListener('keydown', function (event) {
        if (isSaving(event)) {
            config.value.download_html()
            event.preventDefault()
        }
    })
}

const container = useTemplateRef('container_ref')
const config_pane = useTemplateRef('container_config_ref')
const info_pane = useTemplateRef('container_info_ref')
const show_config = ref(props.config.show_config)
const show_info = ref(props.config.show_info)
const renderer = useTemplateRef('renderer_ref')
const width = ref(400)
const height = computed(() => width.value / config.value.basic.aspect_ratio)
const orthographic_camera = useTemplateRef('orthographic_camera_ref')
const raycaster = new Raycaster()
const stats = useTemplateRef('stats_ref')

defineExpose({
    config,
    info,
    width,
    raycaster,
})

onUnmounted(() => {
    console.log('Hyperion.vue unmounted')
})
onMounted(() => {
    console.log('Hyperion.vue mounted')

    // pass camera object
    const camera: ThreeOrthographicCamera = (orthographic_camera.value as any).camera
    const orbit_controls: OrbitControls = (renderer.value as any).three.cameraCtrl
    config.value.camera.orthographic_camera = camera
    config.value.camera.orbit_control = orbit_controls

    // initialize controller pane
    config.value.create_pane(config_pane.value as HTMLElement, renderer.value as any)
    info.value.create_pane(info_pane.value as HTMLElement)

    // make the renderer selected in HTML: https://stackoverflow.com/a/12887221, to react to key events
    const canvas: HTMLElement = (renderer.value as any).canvas
    canvas.setAttribute('tabindex', '1')
    canvas.style.setProperty('outline-style', 'none') // remove select border

    // listen to orbit control events and mouse over events, and focus on the canvas so that the key listener works
    orbit_controls.addEventListener('change', () => {
        config.value.camera.position = camera.position.clone()
        // @ts-expect-error _scale is a private property
        const orbit_control_scale: number = orbit_controls._scale
        config.value.camera.zoom = camera.zoom * orbit_control_scale
        config.value.pane.refresh()
    })
    canvas.addEventListener('mouseenter', () => {
        if (config.value.config_prop.full_screen) {
            canvas.focus()
        }
    })

    // hover and click handlers
    let mousedown_clientX: number | undefined = undefined
    let mousedown_clientY: number | undefined = undefined
    let is_mouse_currently_down = false
    canvas.addEventListener('mousedown', event => {
        mousedown_clientX = event.clientX
        mousedown_clientY = event.clientY
        is_mouse_currently_down = true
    })
    canvas.addEventListener('mouseup', event => {
        if (mousedown_clientX == event.clientX && mousedown_clientY == event.clientY) {
            onMouseChange(event, true)
        }
        is_mouse_currently_down = false
    })
    canvas.addEventListener('mousemove', event => {
        // to prevent triggering hover while moving camera
        if (!is_mouse_currently_down) {
            onMouseChange(event, false)
        }
    })

    // update renderer if width or height changes
    watchEffect(() => {
        if (renderer.value != undefined) {
            const webgl_renderer: WebGLRenderer = (renderer.value as any).renderer
            webgl_renderer.setSize(width.value, height.value)
            webgl_renderer.setPixelRatio(window.devicePixelRatio)
        }
    })

    // observe container size change and update the width and height values
    const container_resize_observer = new ResizeObserver(entries => {
        for (const entry of entries) {
            const container_width = entry.contentRect.width
            width.value = container_width
            if (props.config.full_screen) {
                config.value.basic.aspect_ratio = (document.documentElement.clientWidth / document.documentElement.clientHeight) * 1.02
                config.value.pane.refresh()
            }
        }
    })
    container_resize_observer.observe(container.value as any)

    // expose variables to global scope and the app instance (by `dom.__vue_app__.exposed`)
    const instance = getCurrentInstance()!
    ;(globalThis as any).hyperion_exposed = instance.exposed
    ;(instance.appContext.app as any).exposed = instance.exposed
})

watchEffect(() => {
    if (stats.value != undefined) {
        // set up stats
        stats.value.stats.showPanel(0) // 0: fps, 1: ms, 2: mb, 3+: custom
        stats.value.stats.dom.style.position = 'absolute'
        const renderer_component = renderer.value as any
        renderer_component.onBeforeRender(stats.value.begin)
        renderer_component.onAfterRender(stats.value.end)
        container.value?.appendChild(stats.value.stats.dom)
    }
})

onBeforeUnmount(() => {
    config.value.pane?.dispose()
})

function isSaving(event: KeyboardEvent): boolean {
    return (event.ctrlKey && event.key === 's') || (event.metaKey && event.key === 's')
}

function onKeyDown(event: KeyboardEvent) {
    if (!config.value.config_prop.full_screen && isSaving(event)) {
        config.value.download_html()
        event.preventDefault()
        return
    }
    if (!event.metaKey && !config.value.user_is_typing) {
        if (event.key == 't' || event.key == 'T') {
            config.value.camera.set_position('Top')
        } else if (event.key == 'l' || event.key == 'L') {
            config.value.camera.set_position('Left')
        } else if (event.key == 'f' || event.key == 'F') {
            config.value.camera.set_position('Front')
        } else if (event.key == 'c' || event.key == 'C') {
            show_config.value = !show_config.value
            if (show_config.value) {
                // automatically unfold if using keyboard shortcut to display it
                config.value.pane.expanded = true
            }
        } else if (event.key == 'i' || event.key == 'I') {
            show_info.value = !show_info.value
            if (show_info.value) {
                // automatically unfold if using keyboard shortcut to display it
                info.value.pane.expanded = true
            }
        } else if (event.key == 's' || event.key == 'S') {
            config.value.basic.show_stats = !config.value.basic.show_stats
        } else if (event.key == 'd' || event.key == 'D') {
            const current_showing = show_info.value && info.value.pane.expanded && info.value.dual_folder.expanded
            if (current_showing) {
                info.value.dual_folder.expanded = false
            } else {
                show_info.value = true
                info.value.pane.expanded = true
                info.value.dual_folder.expanded = true
            }
        } else if (event.key == 'a' || event.key == 'A') {
            const current_showing = show_info.value && info.value.pane.expanded && info.value.selection_folder.expanded
            if (current_showing) {
                info.value.selection_folder.expanded = false
            } else {
                show_info.value = true
                info.value.pane.expanded = true
                info.value.selection_folder.expanded = true
            }
        } else if (event.key == 'ArrowRight') {
            if (config.value.snapshot_index < config.value.snapshot_num - 1) {
                config.value.snapshot_index += 1
            }
        } else if (event.key == 'ArrowLeft') {
            if (config.value.snapshot_index > 0) {
                config.value.snapshot_index -= 1
            }
        } else {
            return // unrecognized, propagate to other listeners
        }
        config.value.pane.refresh()
        event.preventDefault()
        event.stopPropagation()
    }
}

function onMouseChange(event: MouseEvent, is_click: boolean = true) {
    const canvas: HTMLElement = (renderer.value as any).canvas
    const rect = canvas.getBoundingClientRect()
    const position = new Vector2(event.clientX - rect.left, event.clientY - rect.top)
    const positionN = new Vector2((position.x / rect.width) * 2 - 1, -(position.y / rect.height) * 2 + 1)
    raycaster.setFromCamera(positionN, (orthographic_camera.value as any).camera)
    const intersects = raycaster.intersectObjects((renderer.value as any).scene.children, false)
    for (let intersect of intersects) {
        if (!intersect.object.visible) continue // don't select invisible object
        // swap back to the original material
        if (is_click) {
            config.value.data.selected = clickable_of(intersect)
        } else {
            config.value.data.hovered = clickable_of(intersect)
        }
        return
    }
    if (is_click) {
        config.value.data.selected = undefined
    } else {
        config.value.data.hovered = undefined
    }
}
</script>

<template>
    <div ref="container_ref" class="hyperion-container" @keydown="onKeyDown">
        <!-- placeholder for controller pane container -->
        <div v-show="show_info" ref="container_info_ref" class="info-container"></div>
        <div v-show="show_config" ref="container_config_ref" class="config-container"></div>

        <Renderer ref="renderer_ref" :width="width + 'px'" :height="height + 'px'" :orbit-ctrl="true" :params="renderer_params">
            <OrthographicCamera
                :left="-config.basic.aspect_ratio"
                :right="config.basic.aspect_ratio"
                :zoom="config.camera.zoom"
                :position="config.camera.position"
                :near="0.1"
                :far="100000"
                ref="orthographic_camera_ref"
            >
            </OrthographicCamera>
            <Stats v-if="config.basic.show_stats" :noSetup="true" ref="stats_ref"></Stats>
            <Scene :background="config.basic.background">
                <AmbientLight color="#FFFFFF" :intensity="config.basic.light_intensity"></AmbientLight>
                <Vertices></Vertices>
                <Edges></Edges>
            </Scene>
        </Renderer>
    </div>
</template>

<style scoped>
.hyperion-container {
    margin: 10px;
    position: relative;
}

.config-container {
    position: absolute;
    top: 0;
    right: 0;
    width: 300px;
    padding: 0;
    margin: 0;
}

.info-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 400px;
    padding: 0;
    margin: 0;
}
</style>
