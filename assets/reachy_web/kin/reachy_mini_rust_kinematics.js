let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
  if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
    cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
  }
  return cachedUint8ArrayMemory0;
}

let cachedTextDecoder =
  typeof TextDecoder !== 'undefined'
    ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true })
    : {
        decode: () => {
          throw Error('TextDecoder not available');
        },
      };

if (typeof TextDecoder !== 'undefined') {
  cachedTextDecoder.decode();
}

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
  numBytesDecoded += len;
  if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
    cachedTextDecoder =
      typeof TextDecoder !== 'undefined'
        ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true })
        : {
            decode: () => {
              throw Error('TextDecoder not available');
            },
          };
    cachedTextDecoder.decode();
    numBytesDecoded = len;
  }
  return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return decodeText(ptr, len);
}

let cachedFloat64ArrayMemory0 = null;

function getFloat64ArrayMemory0() {
  if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
    cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
  }
  return cachedFloat64ArrayMemory0;
}

let WASM_VECTOR_LEN = 0;

function passArrayF64ToWasm0(arg, malloc) {
  const ptr = malloc(arg.length * 8, 8) >>> 0;
  getFloat64ArrayMemory0().set(arg, ptr / 8);
  WASM_VECTOR_LEN = arg.length;
  return ptr;
}
/**
 * Reset forward kinematics state
 * @param {Float64Array} pose
 */
export function reset_forward_kinematics(pose) {
  const ptr0 = passArrayF64ToWasm0(pose, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  wasm.reset_forward_kinematics(ptr0, len0);
}

function getArrayF64FromWasm0(ptr, len) {
  ptr = ptr >>> 0;
  return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}
/**
 * Calculate passive joints
 * @param {Float64Array} head_joints
 * @param {Float64Array} head_pose
 * @returns {Float64Array}
 */
export function calculate_passive_joints_wasm(head_joints, head_pose) {
  const ptr0 = passArrayF64ToWasm0(head_joints, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ptr1 = passArrayF64ToWasm0(head_pose, wasm.__wbindgen_malloc);
  const len1 = WASM_VECTOR_LEN;
  const ret = wasm.calculate_passive_joints_wasm(ptr0, len0, ptr1, len1);
  const v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v3;
}

const cachedTextEncoder =
  typeof TextEncoder !== 'undefined'
    ? new TextEncoder('utf-8')
    : {
        encode: () => {
          throw Error('TextEncoder not available');
        },
      };

const encodeString =
  typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
        return cachedTextEncoder.encodeInto(arg, view);
      }
    : function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
          read: arg.length,
          written: buf.length,
        };
      };

function passStringToWasm0(arg, malloc, realloc) {
  if (realloc === undefined) {
    const buf = cachedTextEncoder.encode(arg);
    const ptr = malloc(buf.length, 1) >>> 0;
    getUint8ArrayMemory0()
      .subarray(ptr, ptr + buf.length)
      .set(buf);
    WASM_VECTOR_LEN = buf.length;
    return ptr;
  }

  let len = arg.length;
  let ptr = malloc(len, 1) >>> 0;

  const mem = getUint8ArrayMemory0();

  let offset = 0;

  for (; offset < len; offset++) {
    const code = arg.charCodeAt(offset);
    if (code > 0x7f) break;
    mem[ptr + offset] = code;
  }

  if (offset !== len) {
    if (offset !== 0) {
      arg = arg.slice(offset);
    }
    ptr = realloc(ptr, len, (len = offset + arg.length * 3), 1) >>> 0;
    const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
    const ret = encodeString(arg, view);

    offset += ret.written;
    ptr = realloc(ptr, len, offset, 1) >>> 0;
  }

  WASM_VECTOR_LEN = offset;
  return ptr;
}

function takeFromExternrefTable0(idx) {
  const value = wasm.__wbindgen_export_0.get(idx);
  wasm.__externref_table_dealloc(idx);
  return value;
}
/**
 * Initialize the kinematics module with motor data (JSON string)
 * @param {string} kinematics_data_json
 */
export function init_kinematics(kinematics_data_json) {
  const ptr0 = passStringToWasm0(
    kinematics_data_json,
    wasm.__wbindgen_malloc,
    wasm.__wbindgen_realloc
  );
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.init_kinematics(ptr0, len0);
  if (ret[1]) {
    throw takeFromExternrefTable0(ret[0]);
  }
}

function isLikeNone(x) {
  return x === undefined || x === null;
}
/**
 * Forward kinematics: joint angles (6 floats) → pose matrix (16 floats)
 * @param {Float64Array} joint_angles
 * @param {number | null} [body_yaw]
 * @returns {Float64Array}
 */
export function forward_kinematics(joint_angles, body_yaw) {
  const ptr0 = passArrayF64ToWasm0(joint_angles, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.forward_kinematics(
    ptr0,
    len0,
    !isLikeNone(body_yaw),
    isLikeNone(body_yaw) ? 0 : body_yaw
  );
  const v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Clamp joint angles to limits
 * @param {Float64Array} angles
 * @returns {Float64Array}
 */
export function clamp_joint_angles(angles) {
  const ptr0 = passArrayF64ToWasm0(angles, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.clamp_joint_angles(ptr0, len0);
  const v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Initialize WASM module
 */
export function init() {
  wasm.init();
}

/**
 * Inverse kinematics: pose matrix (16 floats) → joint angles (6 floats)
 * @param {Float64Array} pose
 * @param {number | null} [body_yaw]
 * @returns {Float64Array}
 */
export function inverse_kinematics(pose, body_yaw) {
  const ptr0 = passArrayF64ToWasm0(pose, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.inverse_kinematics(
    ptr0,
    len0,
    !isLikeNone(body_yaw),
    isLikeNone(body_yaw) ? 0 : body_yaw
  );
  const v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Safe inverse kinematics with limits
 * Returns [body_yaw, stewart_1, ..., stewart_6] (7 floats)
 * @param {Float64Array} pose
 * @param {number | null} [body_yaw]
 * @param {number | null} [max_relative_yaw]
 * @param {number | null} [max_body_yaw]
 * @returns {Float64Array}
 */
export function inverse_kinematics_safe(pose, body_yaw, max_relative_yaw, max_body_yaw) {
  const ptr0 = passArrayF64ToWasm0(pose, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.inverse_kinematics_safe(
    ptr0,
    len0,
    !isLikeNone(body_yaw),
    isLikeNone(body_yaw) ? 0 : body_yaw,
    !isLikeNone(max_relative_yaw),
    isLikeNone(max_relative_yaw) ? 0 : max_relative_yaw,
    !isLikeNone(max_body_yaw),
    isLikeNone(max_body_yaw) ? 0 : max_body_yaw
  );
  const v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
  wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
  return v2;
}

/**
 * Validate joint angles
 * @param {Float64Array} angles
 * @returns {boolean}
 */
export function is_valid_angles(angles) {
  const ptr0 = passArrayF64ToWasm0(angles, wasm.__wbindgen_malloc);
  const len0 = WASM_VECTOR_LEN;
  const ret = wasm.is_valid_angles(ptr0, len0);
  return ret !== 0;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
  if (typeof Response === 'function' && module instanceof Response) {
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      try {
        return await WebAssembly.instantiateStreaming(module, imports);
      } catch (e) {
        const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

        if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
          console.warn(
            '`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n',
            e
          );
        } else {
          throw e;
        }
      }
    }

    const bytes = await module.arrayBuffer();
    return await WebAssembly.instantiate(bytes, imports);
  } else {
    const instance = await WebAssembly.instantiate(module, imports);

    if (instance instanceof WebAssembly.Instance) {
      return { instance, module };
    } else {
      return instance;
    }
  }
}

function __wbg_get_imports() {
  const imports = {};
  imports.wbg = {};
  imports.wbg.__wbindgen_init_externref_table = function () {
    const table = wasm.__wbindgen_export_0;
    const offset = table.grow(4);
    table.set(0, undefined);
    table.set(offset + 0, undefined);
    table.set(offset + 1, null);
    table.set(offset + 2, true);
    table.set(offset + 3, false);
  };
  imports.wbg.__wbindgen_string_new = function (arg0, arg1) {
    const ret = getStringFromWasm0(arg0, arg1);
    return ret;
  };

  return imports;
}

function __wbg_init_memory(imports, memory) {}

function __wbg_finalize_init(instance, module) {
  wasm = instance.exports;
  __wbg_init.__wbindgen_wasm_module = module;
  cachedFloat64ArrayMemory0 = null;
  cachedUint8ArrayMemory0 = null;

  wasm.__wbindgen_start();
  return wasm;
}

function initSync(module) {
  if (wasm !== undefined) return wasm;

  if (typeof module !== 'undefined') {
    if (Object.getPrototypeOf(module) === Object.prototype) {
      ({ module } = module);
    } else {
      console.warn('using deprecated parameters for `initSync()`; pass a single object instead');
    }
  }

  const imports = __wbg_get_imports();

  __wbg_init_memory(imports);

  if (!(module instanceof WebAssembly.Module)) {
    module = new WebAssembly.Module(module);
  }

  const instance = new WebAssembly.Instance(module, imports);

  return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
  if (wasm !== undefined) return wasm;

  if (typeof module_or_path !== 'undefined') {
    if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
      ({ module_or_path } = module_or_path);
    } else {
      console.warn(
        'using deprecated parameters for the initialization function; pass a single object instead'
      );
    }
  }

  if (typeof module_or_path === 'undefined') {
    module_or_path = new URL('reachy_mini_rust_kinematics_bg.wasm', import.meta.url);
  }
  const imports = __wbg_get_imports();

  if (
    typeof module_or_path === 'string' ||
    (typeof Request === 'function' && module_or_path instanceof Request) ||
    (typeof URL === 'function' && module_or_path instanceof URL)
  ) {
    module_or_path = fetch(module_or_path);
  }

  __wbg_init_memory(imports);

  const { instance, module } = await __wbg_load(await module_or_path, imports);

  return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
