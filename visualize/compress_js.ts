import path from 'path'
import fs from 'fs'
import { PluginOption } from 'vite'
import { compress_content, decompress_content, assert_buffer_equal, assert } from './src/util'

export class PluginConfig {
    enabled: boolean = true
    folder: string = path.resolve(__dirname, 'dist/')
    js_filename: string
    gzip_filename: string

    constructor (js_filename: string, gzip_filename: string) {
        this.js_filename = js_filename
        this.gzip_filename = gzip_filename
    }
}

async function do_compress (folder: string, js_filename: string, gzip_filename: string) {
    const js_filepath = path.join(folder, js_filename)
    const js_content = fs.readFileSync(js_filepath)
    const base64_string = await compress_content(js_content)
    // now test decompress (to be used in browser)
    const decompressed = await decompress_content(base64_string)
    assert_buffer_equal(js_content, decompressed)
    // write to file
    const zip_filepath = path.join(folder, gzip_filename)
    if (fs.existsSync(zip_filepath)) {
        fs.unlinkSync(zip_filepath)
    }
    fs.writeFileSync(zip_filepath, base64_string)
}

async function generate_self_contained_html (folder: string, gzip_filename: string) {
    const html_filepath = path.resolve(__dirname, 'index.html')
    const html_content = fs.readFileSync(html_filepath).toString()
    const start_flag = '/* HYPERION_VISUAL_MODULE_LOADER_BEGIN */'
    const end_flag = '/* HYPERION_VISUAL_MODULE_LOADER_END */'
    const start_index = html_content.indexOf(start_flag)
    assert(start_index != -1, 'start flag not found in index.html')
    const end_index = html_content.indexOf(end_flag)
    assert(end_index != -1, 'end flag not found in index.html')
    assert(start_index + start_flag.length < end_index, 'start and end flag misplaced in index.html')
    const prefix = html_content.slice(0, start_index + start_flag.length)
    const suffix = html_content.slice(end_index)
    // build the inline library code
    const zip_filepath = path.join(folder, gzip_filename)
    const base64_string = fs.readFileSync(zip_filepath).toString()
    const js_code = `

const module_base64 =
/* HYPERION_VISUAL_GZIP_B64_BEGIN */
"${base64_string}"
/* HYPERION_VISUAL_GZIP_B64_END */
function uint8_to_array_buffer(array) {
    return array.buffer.slice(array.byteOffset, array.byteLength + array.byteOffset)
}
function base64_to_array_buffer(base64_str) {
    return uint8_to_array_buffer(Uint8Array.from(atob(base64_str), c => c.charCodeAt(0)))
}
async function decompress_content(base64_str) {
    const base64_binary = base64_to_array_buffer(base64_str)
    const blob = new Blob([base64_binary])
    const decompressed_stream = blob.stream()
        .pipeThrough(new DecompressionStream('gzip'))
    const decompressed = await new Response(decompressed_stream).arrayBuffer()
    return decompressed
}
async function load_module() {
    const decompressed = await decompress_content(module_base64)
    const text_decoder = new TextDecoder("utf-8")
    let module_code = text_decoder.decode(decompressed)
    /* HYPERION_VISUAL_MODULE_CODE_DECODED */
    // add script to html root
    if (window.hyperion_visual != undefined) {
        console.warn("window.hyperion_visual already loaded; skip loading")
    } else {
        console.log("window.hyperion_visual not loaded; loading")
        const script = document.createElement('script')
        script.type = "module"
        script.innerHTML = module_code
        document.body.appendChild(script)
    }
}
load_module()
`
    const standalone_html_content = prefix + js_code + suffix
    // write to file
    const standalone_html_filepath = path.join(folder, 'standalone.html')
    if (fs.existsSync(standalone_html_filepath)) {
        fs.unlinkSync(standalone_html_filepath)
    }
    fs.writeFileSync(standalone_html_filepath, standalone_html_content)
    // // also generate a compressed javascript file and its corresponding html
    // const compressed_js_filename = 'hyperion-visual.compressed.js'
    // const compressed_js_filepath = path.join(folder, compressed_js_filename)
    // if (fs.existsSync(compressed_js_filepath)) {
    //     fs.unlinkSync(compressed_js_filepath)
    // }
    // fs.writeFileSync(compressed_js_filepath, js_code)
    // // generate a html file that invokes the above javascript
    // const invoke_compressed_js_code = `
    // const script = document.createElement('script')
    // script.id = "hyperion_visual_compressed_js_library"
    // script.src = "./${compressed_js_filename}"
    // document.body.appendChild(script)
    // `
    // const invoke_compressed_html_content = prefix + invoke_compressed_js_code + suffix
    // const invoke_compressed_html_filepath = path.join(folder, 'invoke-compressed.html')
    // if (fs.existsSync(invoke_compressed_html_filepath)) {
    //     fs.unlinkSync(invoke_compressed_html_filepath)
    // }
    // fs.writeFileSync(invoke_compressed_html_filepath, invoke_compressed_html_content)
}

export function compress_js (user_config: PluginConfig): PluginOption {
    const { enabled, folder, js_filename, gzip_filename } = user_config
    return {
        name: 'vite-plugin-zip-file',
        apply: 'build',
        async closeBundle () {
            if (!enabled) {
                return
            }
            await do_compress(folder, js_filename, gzip_filename)
            await generate_self_contained_html(folder, gzip_filename)
        },
    }
}
