"""

This example tests if a triple nested loop works correctly.

On a mac pro 2019 with wgpu '0.13.2' it fails (not all neighbors visited and
loop variable ii is incorrect).
AMD Radeon Pro W5700X 16 GB, macOS 13.4

On a macbook air M2 it passes.

On windows and linux it seems to work.


Code logic should work like this:

id = numpy.array([1,1,1])
dimensions = numpy.array([3,3,3])
neighborsVisited = 0
for kk in range(-1,2):
    for jj in range(-1,2):
        for ii in range(-1,2):
            if not (kk == 0 and jj == 0 and ii == 0) \
                  and (id[0] + kk) >= 0 and (id[0] + kk) < dimensions[0] \
                  and (id[1] + jj) >= 0 and (id[1] + jj) < dimensions[1] \
                  and (id[2] + ii) >= 0 and (id[2] + ii) < dimensions[2]:
                    neighborsVisited += 1
print(ii, jj, kk, neighborsVisited)

which results in: 1 1 1 26
because the python loops end at 1 not 2 like in C style for loops.
"""


import numpy
import time

import wgpu
import wgpu.backends.rs  # Select backend
import wgpu.utils

shape = [3,3,3]
volumeArray = numpy.random.normal(1000*numpy.ones(shape), 100)
volumeIntArray = volumeArray.astype('int32')
displacementsArray = numpy.zeros(shape=[3,3,3,4], dtype="float32")

# wgsl Shader code
shader_broken = """

@group(0) @binding(0)
var<storage,read> density: array<i32>;

@group(0) @binding(1)
var<storage,read_write> displacements: array<vec4<f32>>;

@compute
@workgroup_size(1)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let idi32 : vec3<i32> = vec3<i32>(id);
    var dimensions : vec3<i32> = vec3<i32>(3,3,3);
    var kk : i32; var jj : i32; var ii : i32;
    var neighborsVisited : i32 = 0;
    for (kk = -1; kk < 2; kk += 1) {
        for (jj = -1; jj < 2; jj += 1) {
            for (ii = -1; ii < 2; ii += 1) {
                if ( !(kk == 0 && jj == 0 && ii == 0)
                      && ((idi32.z + kk) >= 0 && (idi32.z + kk) < dimensions.z)
                      && ((idi32.y + jj) >= 0 && (idi32.y + jj) < dimensions.y)
                      && ((idi32.x + ii) >= 0 && (idi32.x + ii) < dimensions.x) ) {
                    neighborsVisited += 1;
                }
            }
        }
    }
    displacements[idi32.x + idi32.y * 3 + idi32.z * 9] = vec4<f32>(f32(kk), f32(jj), f32(ii), f32(neighborsVisited));
}
"""

shader = """

@group(0) @binding(0)
var<storage,read> density: array<i32>;

@group(0) @binding(1)
var<storage,read_write> displacements: array<vec4<f32>>;

@compute
@workgroup_size(1)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let idi32 : vec3<i32> = vec3<i32>(id);
    var dimensions : vec3<i32> = vec3<i32>(3,3,3);
    var kk : i32; var jj : i32; var ii : i32;
    var neighborsVisited : i32 = 0;
    for (kk = -1; kk < 2; kk += 1) {
        for (jj = -1; jj < 2; jj += 1) {
            for (ii = -1; ii < 2; ii += 1) {
                if (kk == 0 && jj == 0 && ii == 0) {
                    break;
                }
                if (idi32.z + kk < 0 || idi32.z + kk > dimensions.z - 1) {
                    break;
                }
                if (idi32.y + jj < 0 || idi32.y + jj > dimensions.y - 1) {
                    break;
                }
                if (idi32.x + ii < 0 || idi32.x + ii > dimensions.x - 1) {
                    break;
                }
                neighborsVisited += 1;
            }
        }
    }
    displacements[idi32.x + idi32.y * 3 + idi32.z * 9] = vec4<f32>(f32(kk), f32(jj), f32(ii), f32(neighborsVisited));
}
"""

# Create a device with max memory and compile the shader
adapter = wgpu.gpu.request_adapter(canvas=None, power_preference="high-performance")
required_limits={
    'max_storage_buffer_binding_size': adapter.limits['max_storage_buffer_binding_size'],
    'max_bind_groups': adapter.limits['max_bind_groups']
}
device = adapter.request_device(required_limits=required_limits)
cshader = device.create_shader_module(code=shader)

# Create buffers
buffers = {}
storageUsage= wgpu.BufferUsage.STORAGE
storageCopyUsage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
uniformUsage= wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
buffers[0] = device.create_buffer_with_data(data=volumeIntArray, usage=storageUsage)
buffers[1] = device.create_buffer_with_data(data=displacementsArray, usage=storageCopyUsage)

# Create bindings and binding layouts
binding_layouts = [
    { "binding": 0,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.read_only_storage, "has_dynamic_offset": False}
    },
    { "binding": 1,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False}
    },
]
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)

bindings = [
    { "binding": 0,
      "resource": {"buffer": buffers[0], "offset": 0, "size": buffers[0].size}
    },
    { "binding": 1,
      "resource": {"buffer": buffers[1], "offset": 0, "size": buffers[1].size}
    },
]
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create a pipeline and run it
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
compute_pass.dispatch_workgroups(*volumeIntArray.shape)
compute_pass.end()
device.queue.submit([command_encoder.finish()])
displacementsMemory = device.queue.read_buffer(buffers[1])  # slow, should be done async
displacementsArray = numpy.array(displacementsMemory.cast("f", displacementsArray.shape))

print(displacementsArray[1,1,1])

if displacementsArray[0,0,0][2] != 2:
    print(f"Error: Inner loop variable not incrementing: {displacementsArray[0,0,0]}")

if displacementsArray.max() != 26:
    print(f"Error: incorrect neighbor count {displacementsArray.max()} != 26")
