import { computed } from 'vue'
import { Pane, FolderApi } from 'tweakpane'
import * as EssentialsPlugin from '@tweakpane/plugin-essentials'
import { assert } from '@/util'
import { Vector3, OrthographicCamera, WebGLRenderer, Vector2 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RuntimeData, ConfigProps, renderer_params, type Snapshot } from './hyperion'

interface KeyShortcutDescription {
    key: string
    description: string
}

export const key_shortcuts: Array<KeyShortcutDescription> = [
    { key: 'T', description: 'top view' },
    { key: 'L', description: 'left view' },
    { key: 'F', description: 'front view' },
    { key: 'C', description: 'toggle config' },
    { key: 'I', description: 'toggle info' },
    { key: 'S', description: 'toggle stats' },
    { key: '⬅', description: 'last snapshot' },
    { key: '⮕', description: 'next snapshot' }
]

/* configuration helper class given the runtime data */
export class Config {
    data: RuntimeData
    config_prop: ConfigProps
    basic: BasicConfig
    snapshot_config: SnapshotConfig = new SnapshotConfig()
    camera: CameraConfig = new CameraConfig()
    vertex: VertexConfig = new VertexConfig()
    edge: EdgeConfig = new EdgeConfig()
    pane?: Pane

    constructor (data: RuntimeData, config_prop: ConfigProps) {
        this.data = data
        this.config_prop = config_prop
        this.basic = new BasicConfig(config_prop)
    }

    export_visualizer_parameters () {
        // first clear existing parameters to avoid being included
        this.parameters = ''
        // @ts-expect-error exportState is not in the type definition
        this.parameters = JSON.stringify(this.pane.exportState())
        this.refresh_pane()
    }

    import_visualizer_parameters () {
        const parameters = this.parameters
        // @ts-expect-error exportState is not in the type definition
        this.pane.importState(JSON.parse(this.parameters))
        this.parameters = parameters
        this.refresh_pane()
    }

    create_pane (container: HTMLElement, renderer: HTMLElement) {
        assert(this.pane == null, 'cannot create pane twice')
        this.renderer = renderer
        this.pane = new Pane({
            title: this.title,
            container: container,
            expanded: false
        })
        this.pane.registerPlugin(EssentialsPlugin)
        const pane: FolderApi = this.pane
        const snapshot_names = []
        for (const [name] of this.data.visualizer.snapshots) {
            snapshot_names.push(name as string)
        }
        // add export/import buttons
        this.add_import_export(pane.addFolder({ title: 'Import/Export', expanded: true }))
        // add everything else
        this.snapshot_config.add_to(pane.addFolder({ title: 'Snapshot', expanded: true }), snapshot_names)
        this.camera.add_to(pane.addFolder({ title: 'Camera', expanded: false }))
        this.basic.add_to(pane.addFolder({ title: 'Basic', expanded: false }))
        this.vertex.add_to(pane.addFolder({ title: 'Vertex', expanded: false }))
        this.edge.add_to(pane.addFolder({ title: 'Edge', expanded: false }))
        // add shortcut guide
        pane.addBlade({ view: 'separator' })
        this.add_shortcut_guide(pane.addFolder({ title: 'Key Shortcuts', expanded: true }))
        // if the config is passed from props, import it (must execute after all elements are created)
        if (this.config_prop.visualizer_config != undefined) {
            this.parameters = JSON.stringify(this.config_prop.visualizer_config)
            this.import_visualizer_parameters()
        }
    }

    parameters: string = '' // export or import parameters of the tweak pane
    renderer: any = undefined
    png_scale: number = 1
    add_import_export (pane: FolderApi): void {
        // add parameter import/export
        pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['export parameters', 'import parameters'][x]
            })
        }).on('click', (event: any) => {
            if (event.index[0] == 0) {
                this.export_visualizer_parameters()
            } else {
                this.import_visualizer_parameters()
            }
        })
        pane.addBinding(this, 'parameters')
        // add figure import/export
        pane.addBinding(this, 'png_scale', { min: 0.5, max: 2 })
        pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['Open PNG', 'Download PNG'][x]
            })
        }).on('click', (event: any) => {
            const data_url = this.generate_png()
            if (data_url == undefined) {
                return
            }
            if (event.index[0] == 0) {
                this.open_png(data_url)
            } else {
                this.download_png(data_url)
            }
        })
    }

    generate_png (): string | undefined {
        if (this.renderer == undefined) {
            alert('renderer is not initialized, please wait')
            return undefined
        }
        const renderer = new WebGLRenderer({ ...renderer_params, preserveDrawingBuffer: true })
        const old_renderer: WebGLRenderer = (this.renderer as any).renderer
        const size = old_renderer.getSize(new Vector2())
        renderer.setSize(size.x * this.png_scale, size.y * this.png_scale, false)
        renderer.setPixelRatio(window.devicePixelRatio * this.png_scale)
        renderer.render((this.renderer as any).scene, (this.renderer as any).camera)
        return renderer.domElement.toDataURL()
    }

    open_png (data_url: string) {
        const w = window.open('', '')
        if (w == null) {
            alert('cannot open new window')
            return
        }
        w.document.title = 'rendered image'
        w.document.body.style.backgroundColor = 'white'
        w.document.body.style.margin = '0'
        const img = new Image()
        img.src = data_url
        img.setAttribute('style', 'width: 100%; height: 100%; object-fit: contain;')
        w.document.body.appendChild(img)
    }

    download_png (data_url: string) {
        const a = document.createElement('a')
        a.href = data_url.replace('image/png', 'image/octet-stream')
        a.download = 'rendered.png'
        a.click()
    }

    add_shortcut_guide (pane: FolderApi): void {
        for (const key_shortcut of key_shortcuts) {
            pane.addBlade({
                view: 'text',
                label: key_shortcut.description,
                parse: (v: string) => v,
                value: key_shortcut.key,
                disabled: true
            })
        }
    }

    refresh_pane () {
        const pane: FolderApi = this.pane
        pane.refresh()
    }

    public get title (): string {
        return `MWPF Visualizer (${this.snapshot_index + 1}/${this.snapshot_num})`
    }

    public set aspect_ratio (aspect_ratio: number) {
        this.basic.aspect_ratio = aspect_ratio
        this.refresh_pane()
    }

    public set snapshot_index (index: number) {
        this.snapshot_config.index = index
        this.snapshot_config.name = index
        const pane: FolderApi = this.pane
        pane.title = this.title
        this.refresh_pane()
    }

    public get snapshot_index (): number {
        return this.snapshot_config.index
    }

    public get snapshot_count (): number {
        return this.data.visualizer.snapshots.length
    }

    public get_snapshot (snapshot_index: number): Snapshot {
        return this.data.visualizer.snapshots[snapshot_index][1] as Snapshot
    }

    public get snapshot (): Snapshot {
        // @ts-expect-error force type conversion
        return computed<Snapshot>(() => {
            return this.get_snapshot(this.snapshot_index)
        })
    }

    public get snapshot_num (): number {
        return this.data.visualizer.snapshots.length
    }
}

/* controls basic elements like background and aspect ratio */
export class BasicConfig {
    aspect_ratio: number = 1
    background: string = '#ffffff'
    hovered_color: string = '#6FDFDF'
    selected_color: string = '#4B7BE5'
    light_intensity: number = 3
    segments: number
    show_stats: boolean = false
    config_props: ConfigProps

    constructor (config_props: ConfigProps) {
        this.config_props = config_props
        this.segments = config_props.segments
    }

    add_to (pane: FolderApi): void {
        if (!this.config_props.full_screen) {
            // in full screen mode, user cannot adjust aspect ratio manually
            pane.addBinding(this, 'aspect_ratio', { min: 0.1, max: 3 })
        }
        pane.addBinding(this, 'background')
        pane.addBinding(this, 'hovered_color')
        pane.addBinding(this, 'selected_color')
        pane.addBinding(this, 'light_intensity', { min: 0.1, max: 10 })
        pane.addBinding(this, 'show_stats')
        pane.addBinding(this, 'segments', { step: 1, min: 3, max: 128 })
    }
}

export class SnapshotConfig {
    index: number = 0
    name: number = 0

    add_to (pane: FolderApi, snapshot_names: string[]): void {
        pane.addBinding(this, 'index', { step: 1, min: 0, max: snapshot_names.length - 1 }).on('change', () => {
            this.name = this.index
            pane.refresh()
        })
        const options: { [Name: string]: number } = {}
        for (const [index, name] of snapshot_names.entries()) {
            options[name] = index
        }
        pane.addBinding(this, 'name', { options }).on('change', () => {
            this.index = this.name
            pane.refresh()
        })
    }
}

const names = ['Top', 'Left', 'Front']
const positions = [new Vector3(0, 1000, 0), new Vector3(-1000, 0, 0), new Vector3(0, 0, 1000)]
export class CameraConfig {
    zoom: number = 0.2
    position: Vector3 = positions[0].clone()
    orthographic_camera?: OrthographicCamera
    orbit_control?: OrbitControls

    add_to (pane: FolderApi): void {
        pane.addBlade({
            view: 'buttongrid',
            size: [3, 1],
            cells: (x: number) => ({
                title: names[x]
            }),
            label: 'reset view'
        }).on('click', (event: any) => {
            const i: number = event.index[0]
            this.set_position(names[i])
        })
        this.zoom = this.zoom * 0.999 // trigger camera zoom
        pane.addBinding(this, 'zoom', { min: 0.001, max: 1000 })
        if (this.orthographic_camera != null) {
            pane.addBinding(this, 'position')
        }
    }

    set_position (name: string) {
        const index = names.indexOf(name)
        if (index == -1) {
            console.error(`position name "${name}" is not recognized`)
            return
        }
        this.position = positions[index].clone()
        if (this.orbit_control != undefined) {
            this.orbit_control.target = new Vector3()
        }
    }
}

export class VertexConfig {
    radius: number = 0.15
    outline_ratio: number = 1.2
    normal_color: string = '#FFFFFF'
    defect_color: string = '#FF0000'

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'radius', { min: 0, max: 10, step: 0.001 })
        pane.addBinding(this, 'outline_ratio', { min: 0, max: 10, step: 0.001 })
        pane.addBinding(this, 'normal_color')
        pane.addBinding(this, 'defect_color')
    }

    public get outline_radius (): number {
        return this.radius * this.outline_ratio
    }
}

export class ColorPaletteConfig {
    c0: string = '#44C03F' // green
    c1: string = '#F6C231' // yellow
    c2: string = '#4DCCFB' // light blue
    c3: string = '#F17B24' // orange
    c4: string = '#7C1DD8' // purple
    c5: string = '#8C4515' // brown
    c6: string = '#E14CB6' // pink
    c7: string = '#44C03F' // green
    c8: string = '#F6C231' // yellow
    c9: string = '#4DCCFB' // light blue
    c10: string = '#F17B24' // orange
    c11: string = '#7C1DD8' // purple
    c12: string = '#8C4515' // brown
    c13: string = '#E14CB6' // pink

    ungrown: string = '#1A1A1A' // dark grey
    subgraph: string = '#0000FF' // standard blue

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'ungrown')
        pane.addBinding(this, 'subgraph')
        for (let i = 0; i < 14; ++i) {
            pane.addBinding(this, `c${i}`)
        }
    }

    get (index: number): string {
        // @ts-expect-error string is not indexable
        return this[`c${index % 14}`]
    }
}

export class EdgeConfig {
    radius: number = 0.03
    ungrown_opacity: number = 0.1
    grown_opacity: number = 0.3
    tight_opacity: number = 1
    color_palette: ColorPaletteConfig = new ColorPaletteConfig()

    deg_1_ratio: number = 1.6
    deg_3_ratio: number = 1.5
    deg_4_ratio: number = 2
    deg_5_ratio: number = 2.5
    deg_10_ratio: number = 3

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'radius', { min: 0, max: 1, step: 0.001 })
        pane.addBinding(this, 'ungrown_opacity', { min: 0, max: 1, step: 0.01 })
        pane.addBinding(this, 'grown_opacity', { min: 0, max: 1, step: 0.01 })
        pane.addBinding(this, 'tight_opacity', { min: 0, max: 1, step: 0.01 })
        // add color palette
        const color_palette = pane.addFolder({ title: 'Color Palette', expanded: false })
        this.color_palette.add_to(color_palette)
        // add edge radius fine tuning
        const deg_ratios = pane.addFolder({ title: 'Edge Radius Ratios', expanded: true })
        deg_ratios.addBinding(this, 'deg_1_ratio', { min: 0, max: 10, step: 0.01 })
        deg_ratios.addBinding(this, 'deg_3_ratio', { min: 0, max: 10, step: 0.01 })
        deg_ratios.addBinding(this, 'deg_4_ratio', { min: 0, max: 10, step: 0.01 })
        deg_ratios.addBinding(this, 'deg_5_ratio', { min: 0, max: 10, step: 0.01 })
        deg_ratios.addBinding(this, 'deg_10_ratio', { min: 0, max: 10, step: 0.01 })
    }

    ratio_of_deg (deg: number): number {
        assert(deg >= 1, 'degree must be at least 1')
        switch (deg) {
            case 1:
                return this.deg_1_ratio
            case 2:
                return 1
            case 3:
                return this.deg_3_ratio
            case 4:
                return this.deg_4_ratio
            case 5:
                return this.deg_5_ratio
            default:
                if (deg <= 10) {
                    return this.deg_5_ratio + ((deg - 5) * (this.deg_10_ratio - this.deg_5_ratio)) / 5
                }
                return this.deg_10_ratio
        }
    }
}
