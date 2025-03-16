"""
NOT QUITE WORKING

Tested with Slicer 5.9.2 and wgpu 0.20.1

Cut and paste this file into the python console

Note that wgsl only support 32 bit ints, but the short data could be
packed: https://github.com/gpuweb/gpuweb/issues/2429

# TODO: for(var i: i32 = 0; i < 4; i++) broken in wgpu?
# https://www.w3.org/TR/WGSL/#for-statement

"""

import numpy

try:
    import wgpu
except ModuleNotFoundError:
    pip_install("wgpu")
    import wgpu
import wgpu.utils

import SampleData

def infoPrint(message):
    print(message)
    slicer.util.showStatusMessage(message)

# Load data
try:
    volumeNode = slicer.util.getNode("MRHead")
    #volumeNode = slicer.util.getNode("CTACardio")
except slicer.util.MRMLNodeNotFoundException:
    import SampleData
    volumeNode = SampleData.SampleDataLogic().downloadMRHead()
    #volumeNode = SampleData.SampleDataLogic().downloadCTACardio()

volumeArray = slicer.util.arrayFromVolume(volumeNode)
sliceSize =  volumeArray.shape[1] * volumeArray.shape[2]
volumeIntArray = volumeArray.astype('int32')

# wgsl Shader code
shaderCode = """

@group(0) @binding(0)
var<storage,read> background: array<i32>;

@group(0) @binding(1)
var<storage,read_write> label0: array<i32>;

@group(0) @binding(2)
var<storage,read_write> label1: array<i32>;

@group(0) @binding(3)
var<storage,read_write> strength0: array<f32>;

@group(0) @binding(4)
var<storage,read_write> strength1: array<f32>;

struct Parameters {
    iterations : u32,
};
/*
@group(0) @binding(5)
var<uniform> parameters : Parameters;
*/

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idi32 : vec3<i32> = vec3<i32>(id);
    let dimensions = vec3<i32>(@@SLICES@@,@@ROWS@@,@@COLUMNS@@);
    let index: i32 = idi32.x * @@SLICE_SIZE@@ + idi32.y * @@ROW_SIZE@@ + idi32.z;
    var iteration : u32 = 0u;
    var parameters : Parameters;
    parameters.iterations = 90u;
    loop {
        if (iteration > parameters.iterations) {
            break;
        }
        let indexBackground : i32 = background[index];
        var label : i32;
        var strength : f32;
        if (iteration % 2u == 0u) {
            label = label0[index];
            strength = strength0[index];
        } else {
            label = label1[index];
            strength = strength1[index];
        }
        for (var k : i32 = -1; k < 2; k += 1) {
            for (var j : i32 = -1; j < 2; j += 1) {
                for (var i : i32 = -1; i < 2; i += 1) {
                    if ((k != 0 && j != 0 && i != 0)
                          && idi32.x + k > 0 && idi32.x + k < dimensions.x
                          && idi32.y + j > 0 && idi32.y + j < dimensions.y
                          && idi32.z + i > 0 && idi32.z + i < dimensions.z ) {
                        let offset : i32 = k * @@SLICE_SIZE@@ + j * @@ROW_SIZE@@ + i;
                        let neighborBackground : i32 = background[index + offset];
                        var neighborLabel : i32;
                        var neighborStrength : f32;
                        if (iteration % 2u == 0u) {
                            neighborLabel = label0[index + offset];
                            neighborStrength = strength0[index + offset];
                        } else {
                            neighborLabel = label1[index + offset];
                            neighborStrength = strength1[index + offset];
                        }
                        var strengthCost : f32 = f32(abs(neighborBackground - indexBackground));
                        var takeoverStrength : f32 = neighborStrength - strengthCost;
                        if (takeoverStrength > strength) {
                            if (iteration % 2u == 0u) {
                                label1[index] = neighborLabel;
                                strength1[index] = takeoverStrength;
                            } else {
                                label0[index] = neighborLabel;
                                strength0[index] = takeoverStrength;
                            }
                        }
                    }
                }
            }
        }
        iteration += 1u;
    }
}


"""
shaderCode = shaderCode.replace("@@SLICES@@", str(volumeArray.shape[0]))
shaderCode = shaderCode.replace("@@ROWS@@", str(volumeArray.shape[1]))
shaderCode = shaderCode.replace("@@COLUMNS@@", str(volumeArray.shape[2]))
shaderCode = shaderCode.replace("@@SLICE_SIZE@@", str(sliceSize))
shaderCode = shaderCode.replace("@@ROW_SIZE@@", str(volumeArray.shape[2]))

infoPrint("computing...")

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
infoPrint("buffers 1")
buffers = {}
usage= wgpu.BufferUsage.STORAGE
buffers[0] = device.create_buffer_with_data(data=volumeIntArray, usage=usage)
infoPrint("buffers 2")
usage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
buffers[1] = device.create_buffer(size=volumeIntArray.data.nbytes, usage=usage)
infoPrint("buffers 3")
usage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
buffers[2] = device.create_buffer(size=volumeIntArray.data.nbytes, usage=usage)
infoPrint("buffers 4")
usage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
buffers[3] = device.create_buffer(size=volumeIntArray.data.nbytes, usage=usage)
infoPrint("buffers 5")
usage= wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
buffers[4] = device.create_buffer(size=volumeIntArray.data.nbytes, usage=usage)

infoPrint("buffers 6")
uniform_data = numpy.array([51], dtype='uint32')
usage= wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
buffers[5] = device.create_buffer_with_data(data=uniform_data, usage=usage)

# Create bindings and binding layouts
bindings = [
    { "binding": 0,
      "resource": {"buffer": buffers[0], "offset": 0, "size": buffers[0].size}
    },
    { "binding": 1,
      "resource": {"buffer": buffers[1], "offset": 0, "size": buffers[1].size}
    },
    { "binding": 2,
      "resource": {"buffer": buffers[2], "offset": 0, "size": buffers[1].size}
    },
    { "binding": 3,
      "resource": {"buffer": buffers[3], "offset": 0, "size": buffers[1].size}
    },
    { "binding": 4,
      "resource": {"buffer": buffers[4], "offset": 0, "size": buffers[1].size}
    },
#    { "binding": 5,
#      "resource": {"buffer": buffers[5], "offset": 0, "size": buffers[1].size}
#    },
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
    { "binding": 2,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False}
    },
    { "binding": 3,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False}
    },
    { "binding": 4,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False}
    },
#    { "binding": 5,
#      "visibility": wgpu.ShaderStage.COMPUTE,
#      "buffer": {"type": wgpu.BufferBindingType.uniform, "has_dynamic_offset": False}
#    },
]

# Put buffers together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create a pipeline and run it
infoPrint("pipeline")
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shaderModule, "entry_point": "main"},
)
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
compute_pass.dispatch_workgroups(*volumeIntArray.shape)
compute_pass.end()
device.queue.submit([command_encoder.finish()])

# Read the current data of the output buffer
infoPrint("readback")
memory = device.queue.read_buffer(buffers[1])  # slow, can also be done async
resultArray = numpy.array(memory.cast("i", volumeIntArray.shape))

# assert resultArray.mean() == -1 * volumeArray.mean()

infoPrint("drawing")

volumeArray[:] = resultArray.astype('int16').reshape(volumeArray.shape)
slicer.util.arrayFromVolumeModified(volumeNode)
volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
volumeNode.GetDisplayNode().SetAutoWindowLevel(True)
slicer.app.processEvents()
infoPrint("done")
qt.QTimer.singleShot(1000, lambda : infoPrint("done"))
