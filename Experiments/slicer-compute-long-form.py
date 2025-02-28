"""

Install wgpu as described here: https://github.com/pygfx/wgpu-py

Tested with Slicer 5.8.0 and wgpu 0.19.3

"""

import numpy

import wgpu
import wgpu.utils

try:
    mrHead = slicer.util.getNode("MRHead")
except slicer.util.MRMLNodeNotFoundException:
    import SampleData
    mrHead = SampleData.SampleDataLogic().downloadMRHead()

headArray = slicer.util.arrayFromVolume(mrHead)
sliceSize =  headArray.shape[1] * headArray.shape[2]
headIntArray = headArray.astype('int32')
bufferSize = headArray.flatten().shape[0]

shaderCode = """

@group(0) @binding(0)
var<storage,read> data1: array<i32>;

@group(0) @binding(1)
var<storage,read_write> data2: array<i32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x * @@SLICE_SIZE@@ + index.y * @@ROW_SIZE@@ + index.z;
    data2[i] = -1 * data1[i];
}

"""
shaderCode = shaderCode.replace("@@SLICE_SIZE@@", str(sliceSize)+"u")
shaderCode = shaderCode.replace("@@ROW_SIZE@@", str(headArray.shape[2])+"u")

print("computing...")

# Create a device with max memory and compile the shaderCode
adapters = wgpu.gpu.enumerate_adapters_sync()
for a in adapters:
    print(a.summary)
adapter = adapters[0]
required_limits={
    'max-storage-buffer-binding-size': adapter.limits['max-storage-buffer-binding-size']
}
device = adapter.request_device_sync(required_limits=required_limits)
shaderModule = device.create_shader_module(code=shaderCode)


# Create buffers
print("buffers")
buffers = {}
usage= wgpu.BufferUsage.STORAGE
buffers[0] = device.create_buffer_with_data(data=headIntArray, usage=usage)
usage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
buffers[1] = device.create_buffer(size=headIntArray.data.nbytes, usage=usage)

# Create bindings and binding layouts
bindings = [
    { "binding": 0,
      "resource": {"buffer": buffers[0], "offset": 0, "size": buffers[0].size}
    },
    { "binding": 1,
      "resource": {"buffer": buffers[1], "offset": 0, "size": buffers[1].size}
    }
]
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

# Put buffers together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create a pipeline and run it
print("pipeline")
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shaderModule, "entry_point": "main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
compute_pass.dispatch_workgroups(*headIntArray.shape)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# Read the current data of the output buffer
print("readback")
memory = device.queue.read_buffer(buffers[1])  # slow, can also be done async
resultArray = numpy.array(memory.cast("i", headIntArray.shape))

assert resultArray.mean() == -1 * headArray.mean()

print("drawing")

headArray[:] = resultArray.astype('int16').reshape(headArray.shape)
slicer.util.arrayFromVolumeModified(mrHead)
mrHead.GetDisplayNode().SetAutoWindowLevel(False)
mrHead.GetDisplayNode().SetAutoWindowLevel(True)

print("done")
