import { type Ref, ref, type ComputedRef, computed } from 'vue'
import { Pane, FolderApi } from 'tweakpane'
import { assert } from '@/util'
import { Vector3, type OrthographicCamera, type Intersection } from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export interface Position {
    t: number
    i: number
    j: number
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
    dn: number
    dd: number
    // grow rate (r = rn/rd)
    r: number
    rn: number
    rd: number
}

export interface Edge {
    // weight
    w: number
    // vertices
    v: number[]
    // grown (g = gn/gd)
    g: number
    gn: number
    gd: number
    // un-grown (u = un/ud = w_e - g)
    u: number
    un: number
    ud: number
}

export interface Interface {
    // sum of dual variables (sum_dual = sdn/sdd)
    sum_dual: number
    sdn: number
    sdd: number
}

export interface Vertex {
    // is defect
    s: boolean
}

export type Subgraph = number[]

export interface WeightRange {
    // lower (l = ln/ld)
    lower: number
    ld: number
    ln: number
    // upper (u = un/ud)
    upper: number
    ud: number
    un: number
}

export interface Snapshot {
    dual_nodes: DualNode[]
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

/* runtime data */
export class RuntimeData {
    visualizer: VisualizerData
    hovered: Ref<any> = ref(undefined)
    selected: Ref<any> = ref(undefined)

    constructor (visualizer: VisualizerData) {
        this.visualizer = visualizer
    }

    getPointerObject (event: PointerIntersectEventInterface) {
        if (event.intersect == undefined) {
            return undefined
        }
        const instanceId = event.intersect.instanceId
        const component = event.intersect.object.userData.component
        return { instanceId, ...component.userData }
    }

    onPointerEnter (event: PointerIntersectEventInterface) {
        this.hovered.value = this.getPointerObject(event)
    }

    onPointerLeave (event: PointerIntersectEventInterface) {
        this.hovered.value = undefined
    }
}

export class ConfigProps {
    show_config: boolean = true
    full_screen: boolean = true
    segments: number = 32
}

/* configuration helper class given the runtime data */
export class Config {
    data: RuntimeData
    config_prop: ConfigProps
    basic: BasicConfig
    snapshot_config: SnapshotConfig = new SnapshotConfig()
    camera: CameraConfig = new CameraConfig()
    pane?: Pane

    constructor (data: RuntimeData, config_prop: ConfigProps) {
        this.data = data
        this.config_prop = config_prop
        this.basic = new BasicConfig(config_prop.segments)
    }

    create_pane (container: any) {
        assert(this.pane == null, 'cannot create pane twice')
        this.pane = new Pane({
            title: this.title,
            container: container,
            expanded: false
        })
        const pane: FolderApi = this.pane
        const snapshot_names = []
        for (const [name, _] of this.data.visualizer.snapshots) {
            snapshot_names.push(name as string)
        }
        this.snapshot_config.add_to(pane.addFolder({ title: 'Camera', expanded: true }), snapshot_names)
        this.camera.add_to(pane.addFolder({ title: 'Camera', expanded: true }))
        this.basic.add_to(pane.addFolder({ title: 'Basic', expanded: true }))
    }

    refresh_pane () {
        const pane: FolderApi = this.pane
        pane.refresh()
    }

    public get title (): string {
        return `MWPF (${this.snapshot_index + 1}/${this.snapshot_num})`
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
        const that = this
        return computed<Snapshot>(() => {
            return that.get_snapshot(that.snapshot_index)
        }) as any
    }

    public get snapshot_num (): number {
        return this.data.visualizer.snapshots.length
    }
}

/* controls basic elements like background and aspect ratio */
export class BasicConfig {
    aspect_ratio: number = 1
    background: string = '#ffffff'
    segments: number
    show_stats: boolean = true

    constructor (segments: number) {
        this.segments = segments
    }

    add_to (pane: FolderApi): void {
        pane.addBinding(this, 'aspect_ratio', { min: 0.1, max: 3 })
        pane.addBinding(this, 'background')
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
        for (let i = 0; i < 3; ++i) {
            pane.addButton({ title: names[i] }).on('click', () => {
                this.set_position(names[i])
            })
        }
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

export interface PointerIntersectEventInterface {
    type: 'pointerenter' | 'pointerover' | 'pointermove' | 'pointerleave' | 'click'
    component: any
    over?: boolean
    intersect?: Intersection
}
