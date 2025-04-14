import { computed } from 'vue'
import { Pane, FolderApi } from 'tweakpane'
import * as EssentialsPlugin from '@tweakpane/plugin-essentials'
import { type ButtonGridApi } from '@tweakpane/plugin-essentials'
import { assert, bigInt, tweakpane_find_value } from '@/util'
import * as HTMLExport from './html_export'
import { Vector3, OrthographicCamera, WebGLRenderer, Vector2 } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RuntimeData, ConfigProps, renderer_params, type Snapshot, set_deep_value } from './hyperion'
import { Prism } from 'prism-esm'
import { loader as JsonLoader } from 'prism-esm/components/prism-json.js'
import prismCSS from 'prism-esm/themes/prism.min.css?raw'
import * as TextareaPlugin from '@pangenerator/tweakpane-textarea-plugin'
import { default as Sizzle } from 'sizzle'

interface KeyShortcutDescription {
    key: string
    description: string
}

export const key_shortcuts: Array<KeyShortcutDescription> = [
    { key: 'T/L/F', description: 'top/left/front view' },
    { key: 'C/I/S', description: 'toggle config/info/stat' },
    { key: 'D/A', description: 'toggle dual/active info' },
    { key: '⬅/⮕', description: 'previous/next snapshot' },
]

/* configuration helper class given the runtime data */
export class Config {
    user_is_typing: boolean = false
    data: RuntimeData
    config_prop: ConfigProps
    basic: BasicConfig
    snapshot_config: SnapshotConfig = new SnapshotConfig()
    user_note: string = ''
    camera: CameraConfig = new CameraConfig()
    vertex: VertexConfig = new VertexConfig()
    edge: EdgeConfig = new EdgeConfig()
    note_folder?: FolderApi
    // @ts-expect-error we will not use pane before it's initialized, ignore for simplicity
    pane: Pane

    constructor (data: RuntimeData, config_prop: ConfigProps) {
        this.data = data
        this.config_prop = config_prop
        this.basic = new BasicConfig(config_prop)
    }

    export_visualizer_parameters () {
        // first clear existing parameters to avoid being included
        this.parameters = ''
        this.parameters = JSON.stringify(this.pane.exportState())
        this.pane.refresh()
    }

    import_visualizer_parameters () {
        const parameters = this.parameters
        this.pane.importState(JSON.parse(this.parameters))
        // bug fix: tweakpane does not import textarea data correctly
        this.user_note = tweakpane_find_value(JSON.parse(this.parameters), 'user_note')
        this.parameters = parameters
        if (this.user_note != '') {
            this.note_folder!.expanded = true
        }
        this.pane.refresh()
    }

    create_pane (container: HTMLElement, renderer: HTMLElement) {
        assert(this.pane == null, 'cannot create pane twice')
        this.renderer = renderer
        this.pane = new Pane({
            title: this.title,
            container: container,
            expanded: false,
        })
        this.pane.registerPlugin(EssentialsPlugin)
        this.pane.registerPlugin(TextareaPlugin)
        const pane: FolderApi = this.pane
        const snapshot_names = []
        for (const [name] of this.data.visualizer.snapshots) {
            snapshot_names.push(name as string)
        }
        // add everything else
        this.camera.add_to(pane.addFolder({ title: 'Camera', expanded: false }))
        this.snapshot_config.add_to(pane.addFolder({ title: 'Snapshot', expanded: true }), snapshot_names)
        this.note_folder = pane.addFolder({ title: 'Note', expanded: false })
        const user_note = this.note_folder.addBinding(this, 'user_note', {
            view: 'textarea',
            rows: 10,
            placeholder: 'Type here...',
            label: undefined,
        })
        Sizzle('textarea', user_note.element).forEach((element: Element) => {
            element.addEventListener('focusin', () => (this.user_is_typing = true))
            element.addEventListener('focusout', () => (this.user_is_typing = false))
        })
        this.basic.add_to(pane.addFolder({ title: 'Basic', expanded: false }))
        this.vertex.add_to(pane.addFolder({ title: 'Vertex', expanded: false }))
        this.edge.add_to(pane.addFolder({ title: 'Edge', expanded: false }))
        // add shortcut guide
        pane.addBlade({ view: 'separator' })
        this.add_shortcut_guide(pane.addFolder({ title: 'Key Shortcuts', expanded: true }))
        // add export/import buttons
        this.add_import_export(pane.addFolder({ title: 'Import/Export', expanded: true }))
        // if the config is passed from props, import it (must execute after all elements are created)
        if (this.config_prop.visualizer_config != undefined) {
            this.parameters = JSON.stringify(this.config_prop.visualizer_config)
            this.import_visualizer_parameters()
        }
        // if config_setters is specified, apply then
        if (this.config_prop.config_setters != undefined) {
            const config_setters = JSON.parse(JSON.stringify(this.config_prop.config_setters))
            for (const [key, value] of Object.entries(config_setters)) {
                set_deep_value(this, key, value)
            }
        }
        // by default showing the most recent snapshot; user can move back if they want
        if (this.config_prop.snapshot_index != undefined) {
            this.snapshot_index = this.config_prop.snapshot_index
        } else {
            this.snapshot_index = Math.max(this.data.visualizer.snapshots.length - 1, 0)
        }
    }

    parameters: string = '' // export or import parameters of the tweak pane
    renderer: any = undefined
    png_scale: number = 1
    html_include_parameters: boolean = true
    html_use_visualizer_data: string = ''
    html_compress_data: boolean = true
    html_show_info: boolean = true
    html_show_config: boolean = true
    add_import_export (pane: FolderApi): void {
        // add parameter import/export
        const parameter_buttons: ButtonGridApi = pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['export parameters', 'import parameters'][x],
            }),
        }) as any
        parameter_buttons.on('click', (event: any) => {
            if (event.index[0] == 0) {
                this.export_visualizer_parameters()
            } else {
                this.import_visualizer_parameters()
            }
        })
        const parameters = pane.addBinding(this, 'parameters')
        Sizzle('input', parameters.element).forEach((element: Element) => {
            element.addEventListener('focusin', () => (this.user_is_typing = true))
            element.addEventListener('focusout', () => (this.user_is_typing = false))
        })
        // add figure export
        pane.addBinding(this, 'png_scale', { min: 0.2, max: 4 })
        const png_buttons: ButtonGridApi = pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['Open PNG', 'Download PNG'][x],
            }),
        }) as any
        png_buttons.on('click', (event: any) => {
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
        // add visualizer data export
        const data_buttons: ButtonGridApi = pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['Open JSON', 'Download JSON'][x],
            }),
        }) as any
        data_buttons.on('click', (event: any) => {
            if (event.index[0] == 0) {
                this.open_visualizer_data()
            } else {
                this.download_visualizer_data()
            }
        })
        // add html page export
        const html_buttons: ButtonGridApi = pane.addBlade({
            view: 'buttongrid',
            size: [2, 1],
            cells: (x: number) => ({
                title: ['Open HTML', 'Download HTML', 'Download '][x],
            }),
        }) as any
        const html_export_folder = pane.addFolder({ title: 'HTML Export Config', expanded: false })
        html_export_folder.addBinding(this, 'html_include_parameters', { label: 'save parameters' })
        html_export_folder.addBinding(this, 'html_compress_data', { label: 'compress data' })
        html_export_folder.addBinding(this, 'html_show_info', { label: 'show info' })
        html_export_folder.addBinding(this, 'html_show_config', { label: 'show config' })
        const html_use_visualizer_data = html_export_folder.addBinding(this, 'html_use_visualizer_data', {
            view: 'textarea',
            rows: 3,
            label: 'use alternative visualizer data (paste here)',
            placeholder: 'Type here...',
        })
        Sizzle('textarea', html_use_visualizer_data.element).forEach((element: Element) => {
            element.addEventListener('focusin', () => (this.user_is_typing = true))
            element.addEventListener('focusout', () => (this.user_is_typing = false))
        })
        if (HTMLExport.available) {
            html_buttons.on('click', (event: any) => {
                if (event.index[0] == 0) {
                    this.open_html()
                } else {
                    this.download_html()
                }
            })
        } else {
            html_buttons.disabled = true
            html_export_folder.disabled = true
            console.warn('Open/Download HTML only available in release build (which has compressed js library)')
        }
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
        renderer.setPixelRatio(window.devicePixelRatio)
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

    open_visualizer_data () {
        const w = window.open('', '')
        if (w == null) {
            alert('cannot open new window')
            return
        }
        w.document.title = 'visualizer data'
        w.document.body.style.backgroundColor = 'white'
        // add style for code highlight
        const css = w.document.createElement('style')
        css.textContent = prismCSS
        w.document.head.appendChild(css)
        // create div that contains the highlighted code
        const div = w.document.createElement('code')
        div.setAttribute('style', 'white-space: pre;')
        w.document.body.appendChild(div)
        // use prism-js to highlight the code
        const prism = new Prism()
        JsonLoader(prism)
        div.innerHTML = prism.highlight(bigInt.PrettyJSONStringify(this.data.visualizer, { maxLength: 160, indent: 4 }), prism.languages.json, 'json')
    }

    download_visualizer_data () {
        const a = document.createElement('a')
        a.href = 'data:text/json;base64,' + btoa(bigInt.JSONStringify(this.data.visualizer))
        a.download = 'mwpf-vis.json'
        a.click()
    }

    async generate_html (): Promise<string> {
        let visualizer_data = this.data.visualizer
        if (this.html_use_visualizer_data != '') {
            try {
                visualizer_data = bigInt.JSONParse(this.html_use_visualizer_data)
            } catch (e) {
                alert('failed to parse visualizer data')
                throw e
            }
            this.html_use_visualizer_data = '' // clear the field if no problem
        }
        const config_props = new ConfigProps()
        config_props.show_info = this.html_show_info
        config_props.show_config = this.html_show_config
        if (this.html_include_parameters) {
            config_props.visualizer_config = this.pane.exportState()
        }
        return HTMLExport.generate_html(visualizer_data, this.html_compress_data, config_props)
    }

    open_html () {
        this.generate_html().then(html => {
            const w = window.open('', '')
            if (w == null) {
                alert('cannot open new window')
                return
            }
            w.document.write(html)
            w.document.close()
        })
    }

    download_html () {
        this.generate_html().then(html => {
            const a = document.createElement('a')
            a.href = 'data:text/html;charset=utf-8,' + encodeURIComponent(html)
            a.download = 'mwpf-vis.html'
            a.click()
        })
    }

    add_shortcut_guide (pane: FolderApi): void {
        for (const key_shortcut of key_shortcuts) {
            pane.addBlade({
                view: 'text',
                label: key_shortcut.key,
                parse: (v: string) => v,
                value: key_shortcut.description,
                disabled: true,
            })
        }
    }

    public get title (): string {
        return `MWPF Visualizer (${this.snapshot_index + 1}/${this.snapshot_num})`
    }

    public set snapshot_index (index: number) {
        this.snapshot_config.index = index
        this.snapshot_config.name = index
        const pane: FolderApi = this.pane
        pane.title = this.title
        this.pane.refresh()
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
        if (config_props.initial_aspect_ratio != undefined && !isNaN(config_props.initial_aspect_ratio)) {
            this.aspect_ratio = config_props.initial_aspect_ratio
        }
        this.segments = config_props.segments
    }

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'aspect_ratio', { min: 0.3, max: 3, disabled: this.config_props.full_screen })
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
            options[`[${index}] ${name}`] = index
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
        const camera_position_buttons: ButtonGridApi = pane.addBlade({
            view: 'buttongrid',
            size: [3, 1],
            cells: (x: number) => ({
                title: names[x],
            }),
            label: 'reset view',
        }) as any
        camera_position_buttons.on('click', (event: any) => {
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
    outline_color: string = '#000000'

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'radius', { min: 0, max: 10, step: 0.001 })
        pane.addBinding(this, 'outline_ratio', { min: 0, max: 10, step: 0.001 })
        pane.addBinding(this, 'normal_color')
        pane.addBinding(this, 'defect_color')
        pane.addBinding(this, 'outline_color')
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
    c7: string = '#FF2600' // red (PPT apple)
    c8: string = '#18BDB0' // mint (PPT developer)
    c9: string = '#00FDFF' // cyan (PPT apple)
    c10: string = '#AA7942' // brown (PPT apple)
    c11: string = '#00F900' // green (PPT apple)
    c12: string = '#FFFB00' // yellow (PPT apple)

    ungrown: string = '#1A1A1A' // dark grey
    subgraph: string = '#0000FF' // standard blue

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'ungrown')
        pane.addBinding(this, 'subgraph')
        for (let i = 0; i < 13; ++i) {
            // @ts-expect-error cannot guarantee key exists
            pane.addBinding(this, `c${i}`)
        }
    }

    get (index: number): string {
        // @ts-expect-error string is not indexable
        return this[`c${index % 13}`]
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
