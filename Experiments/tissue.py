
"""
NOT WORKING

Install wgpu as described here: https://github.com/pygfx/wgpu-py

filePath = "c:/pieper/SlicerWGPU/Experiments/tissue.py"
filePath = "/Users/pieper/slicer/latest/SlicerWGPU/Experiments/tissue.py"
exec(open(filePath).read())

Note that wgsl only support 32 bit ints, but the short data could be
packed: https://github.com/gpuweb/gpuweb/issues/2429


# TODO: for(var i: i32 = 0; i < 4; i++) broken in wgpu?
# https://www.w3.org/TR/WGSL/#for-statement

"""

import numpy

import wgpu
import wgpu.backends.rs  # Select backend
import wgpu.utils

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
volumeSize =  sliceSize  * volumeArray.shape[0]
volumeIntArray = volumeArray.astype('int32')
displacementsArray = numpy.zeros((2,*volumeArray.shape,3),dtype="float32")
velocitiesArray = numpy.zeros((2,*volumeArray.shape,3),dtype="float32")

# wgsl Shader code
shader = """

@group(0) @binding(0)
var<storage,read> density: array<i32>;

@group(0) @binding(1)
var<storage,read_write> displacements: array<vec3<f32>>;

@group(0) @binding(2)
var<storage,read_write> velocities: array<vec3<f32>>;

struct Parameters {
    iterations : u32,
    timeStep : f32,
    gravity : vec3<f32>,
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
    parameters.iterations = 1u;
    parameters.timeStep = 0.01;
    parameters.gravity = vec3<f32>(0.0, 0.0, -0.9);
    let timeStepSquared = parameters.timeStep * parameters.timeStep;
    loop {
        if (iteration > parameters.iterations) {
            break;
        }
        let iterationOffset : i32 = @@VOLUME_SIZE@@i * (i32(iteration) % 2i);
        let indexDensity : i32 = density[index];
        var mass : f32 = 1.0;
        var stiffness : f32 = 1.0;
        if (indexDensity < 0i) {
            mass = 0.0; // air
            stiffness = 0.1;
        } else {
            stiffness = f32(indexDensity) / 1000.0;
        }
        let position : vec3<f32> = vec3<f32>(id) + displacements[iterationOffset + index];
        var force : vec3<f32> = vec3<f32>(0.0);
        for (var k : i32 = -1; k < 2; k += 1) {
            for (var j : i32 = -1; j < 2; j += 1) {
                for (var i : i32 = -1; i < 2; i += 1) {
                    if ((k != 0 && j != 0 && i != 0)
                          && idi32.x + k > 0 && idi32.x + k < dimensions.x
                          && idi32.y + j > 0 && idi32.y + j < dimensions.y
                          && idi32.z + i > 0 && idi32.z + i < dimensions.z ) {
                        let neighborOffset : i32 = k * @@SLICE_SIZE@@ + j * @@ROW_SIZE@@ + i;
                        let neighborPosition : vec3<f32> = vec3<f32>(id) + displacements[iterationOffset + index + neighborOffset];
                        let originalLength : f32 = length(vec3<f32>(id) - vec3<f32>(vec3<i32>(idi32.x + k, idi32.y + j, idi32.z+i)));
                        let strain : f32 = (length(position - neighborPosition) - originalLength) / originalLength;
                        var lineOfForce : vec3<f32> = normalize(neighborPosition - position);
                        if (strain < 1.0) {
                            lineOfForce *= -1.0;
                        }
                        let neighborForce : vec3<f32> = stiffness * strain * lineOfForce;
                        force += neighborForce;
                    }
                }
            }
        }
        let acceleration : vec3<f32> = force / mass;
        velocities[iterationOffset + index] += 0.5 * acceleration * timeStepSquared;
        displacements[iterationOffset + index] += velocities[iterationOffset + index] * parameters.timeStep;

        velocities[index] = vec3<f32>(0.1);
        displacements[index] = vec3<f32>(0.2);
        velocities[@@VOLUME_SIZE@@ + index] = vec3<f32>(0.3);
        displacements[@@VOLUME_SIZE@@ + index] = vec3<f32>(0.4);

        iteration += 1u;
    }
}


"""
shader = shader.replace("@@SLICES@@", str(volumeArray.shape[0]))
shader = shader.replace("@@ROWS@@", str(volumeArray.shape[1]))
shader = shader.replace("@@COLUMNS@@", str(volumeArray.shape[2]))
shader = shader.replace("@@SLICE_SIZE@@", str(sliceSize))
shader = shader.replace("@@VOLUME_SIZE@@", str(volumeSize))
shader = shader.replace("@@ROW_SIZE@@", str(volumeArray.shape[2]))

# Create a device with max memory and compile the shader
adapter = wgpu.request_adapter(canvas=None, power_preference="high-performance")
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
buffers[0] = device.create_buffer_with_data(data=volumeIntArray, usage=storageUsage)
buffers[1] = device.create_buffer(size=displacementsArray.data.nbytes, usage=storageCopyUsage)
buffers[2] = device.create_buffer(size=velocitiesArray.data.nbytes, usage=storageCopyUsage)

# Create bindings and binding layouts
bindings = [
    { "binding": 0,
      "resource": {"buffer": buffers[0], "offset": 0, "size": buffers[0].size}
    },
    { "binding": 1,
      "resource": {"buffer": buffers[1], "offset": 0, "size": buffers[1].size}
    },
    { "binding": 2,
      "resource": {"buffer": buffers[2], "offset": 0, "size": buffers[2].size}
    },
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
]

# Put buffers together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)
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

# Read the current data of the output buffer
displacementsMemory = device.queue.read_buffer(buffers[1])  # slow, can also be done async
displacementsArray[:] = numpy.array(displacementsMemory.cast("f", displacementsArray.shape))
velocitiesMemory = device.queue.read_buffer(buffers[2])  # slow, can also be done async
velocitiesArray[:] = numpy.array(velocitiesMemory.cast("f", velocitiesArray.shape))

stats = []
for buffer in range(2):
    stats.append(velocitiesArray[buffer].max())
    stats.append(displacementsArray[buffer].max())
print(stats)


"""

volumeArray[:] = resultArray.astype('int16').reshape(volumeArray.shape)
slicer.util.arrayFromVolumeModified(volumeNode)
volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
volumeNode.GetDisplayNode().SetAutoWindowLevel(True)
slicer.app.processEvents()
slicer.util.delayDisplay("done")
"""
