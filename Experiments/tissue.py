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

DEBUGGING = True

import numpy
import scipy
import time

try:
    import wgpu
except ModuleNotFoundError:
    pip_install("wgpu")
    import wgpu

import wgpu.backends.rs  # Select backend
import wgpu.utils

iterations = 1000
iterationInterval = 10

# Load data
sampleScenarios = ["MRHead", "CTACardio"]
scenario = sampleScenarios[0]
scenario = "smallRandom"

if scenario in sampleScenarios:
    try:
        volumeNode = slicer.util.getNode(scenario)
    except slicer.util.MRMLNodeNotFoundException:
        import SampleData
        volumeNode = SampleData.SampleDataLogic().downloadSample(scenario)
elif scenario == "smallRandom":
    iterations = 300
    iterationInterval = 1
    try:
        volumeNode = slicer.util.getNode("smallRandom")
    except slicer.util.MRMLNodeNotFoundException:
        #shape = [256,256,256]
        #shape = [32,32,32]
        #shape = [8,8,8]
        shape = [5,5,5]
        volumeArray = numpy.random.normal(1000*numpy.ones(shape), 100)
        volumeArray = scipy.ndimage.gaussian_filter(volumeArray, shape[0]/16.)
        volumeArray[volumeArray < 0] = 0
        ijkToRAS = numpy.diag([50,50,50,1])
        ijkToRAS[0:3,3] = -25
        volumeNode = slicer.util.addVolumeFromArray(volumeArray, ijkToRAS, "smallRandom")
        slicer.util.setSliceViewerLayers(volumeNode, fit=True)

slicer.util.setSliceViewerLayers(volumeNode)
volumeArray = slicer.util.arrayFromVolume(volumeNode)
sliceSize =  volumeArray.shape[1] * volumeArray.shape[2]
volumeSize =  sliceSize  * volumeArray.shape[0]
volumeIntArray = volumeArray.astype('int32')
displacementsArray = numpy.zeros((2,*volumeArray.shape,4),dtype="float32")
velocitiesArray = numpy.zeros((2,*volumeArray.shape,4),dtype="float32")
debugArray = numpy.zeros((2,*volumeArray.shape,4),dtype="float32")
ijkToRAS = vtk.vtkMatrix4x4()
volumeNode.GetIJKToRASMatrix(ijkToRAS)
debug0Volume = slicer.util.addVolumeFromArray(debugArray[0], ijkToRAS=ijkToRAS, name="Debug0", nodeClassName="vtkMRMLVectorVolumeNode")
debug1Volume = slicer.util.addVolumeFromArray(debugArray[1], ijkToRAS=ijkToRAS, name="Debug1", nodeClassName="vtkMRMLVectorVolumeNode")
debug0Array = slicer.util.arrayFromVolume(debug0Volume)
debug1Array = slicer.util.arrayFromVolume(debug1Volume)

def addGridTransformFromArray(narray, referenceVolume=None, name=None):
    """Create a new grid transform node from content of a numpy array and add it to the scene.

    Displacement values are deep-copied, therefore if the numpy array
    is modified after calling this method, voxel values in the volume node will not change.

    :param narray: numpy array containing grid transform vectors (shape should be [Nk, Nj, Ni, 3], i.e. one displacement vector for slice, row, column location).
    :param referenceVolume: a vtkMRMLVolumeNode or subclass to define the directions, origin, and spacing
    :param name: grid transform node name
    :return: created new grid transform node

    Example::

      # create an identity grid transform
      import numpy
      gridTransformNode = slicer.util.addGridTransformFromArray(numpy.zeros((30, 40, 50, 3)))
    """
    import slicer
    from vtk import vtkMatrix4x4
    from vtk.util import numpy_support

    gridTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLGridTransformNode")
    if name is not None:
        gridTransformNode.SetName(name)
    gridTransform = gridTransformNode.GetTransformFromParent()
    gridImage = vtk.vtkImageData()
    gridImage.SetDimensions(tuple(reversed(narray.shape[:3])))
    gridType = numpy_support.get_vtk_array_type(narray.dtype)
    gridImage.AllocateScalars(gridType, 3)
    if referenceVolume is not None:
        gridDirectionMatrix = vtk.vtkMatrix4x4()
        referenceVolume.GetIJKToRASDirectionMatrix(gridDirectionMatrix)
        gridTransform.SetGridDirectionMatrix(gridDirectionMatrix)
        gridImage.SetOrigin(referenceVolume.GetOrigin())
        gridImage.SetSpacing(referenceVolume.GetSpacing())
    gridTransform.SetDisplacementGridData(gridImage)
    transformArray = slicer.util.arrayFromGridTransform(gridTransformNode)
    transformArray[:] = narray
    slicer.util.arrayFromGridTransformModified(gridTransformNode)

    return gridTransformNode

def addVolumeFromGridTransform(gridTransformNode, name=None):
    """Create a new vector volume from grid transform node from content.

    :param gridTransformNode: source transform
    :param name: created volume node name
    :return: created new volume
    """
    displacements = arrayFromGridTransform(gridTransformNode)
    gridTransform = gridTransformNode.GetTransformFromParent()
    gridDirectionMatrix = gridTransform.GetGridDirectionMatrix()
    displacementGrid = gridTransform.GetDisplacementGrid()
    scratchVolume = slicer.vtkMRMLScalarVolumeNode()
    scratchVolume.SetIJKToRASDirectionMatrix(gridDirectionMatrix)
    scratchVolume.SetSpacing(displacementGrid.GetSpacing())
    scratchVolume.SetOrigin(displacementGrid.GetOrigin())
    ijkToRAS = vtk.vtkMatrix4x4()
    scratchVolume.GetIJKToRASMatrix(ijkToRAS)
    return addVolumeFromArray(displacements, ijkToRAS=ijkToRAS, name=name, nodeClassName="vtkMRMLVectorVolumeNode")

gridTransformNode = addGridTransformFromArray(displacementsArray[0][:,:,:,0:3], referenceVolume=volumeNode, name="Displacements")
volumeNode.SetAndObserveTransformNodeID(gridTransformNode.GetID())
gridTransformArray = slicer.util.arrayFromGridTransform(gridTransformNode)

# wgsl Shader code
shader = """

struct Parameters {
    iteration : f32,
    timeStep : f32,
    gravity : vec3<f32>
};

@group(0) @binding(0)
var<storage,read> density: array<i32>;

@group(0) @binding(1)
var<storage,read_write> displacements: array<vec4<f32>>;

@group(0) @binding(2)
var<storage,read_write> velocities: array<vec4<f32>>;

@group(0) @binding(3)
var<uniform> parameters : Parameters;

@group(0) @binding(4)
var<storage,read_write> debugBuffer : array<vec4<f32>>;

@compute
@workgroup_size(1)

fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions : vec3<i32> = vec3<i32>(@@SLICES@@,@@ROWS@@,@@COLUMNS@@);
    let idi32 : vec3<i32> = vec3<i32>(id);
    let pointIndex : i32 = idi32.z * @@SLICE_SIZE@@ + idi32.y * @@ROW_SIZE@@ + idi32.x;
    let iteration : i32 = i32(parameters.iteration);
    let currentOffset : i32 = (iteration % 2) * @@VOLUME_SIZE@@;
    let nextOffset : i32 = ((iteration + 1) % 2) * @@VOLUME_SIZE@@;
    let timeStepSquared : f32 = parameters.timeStep * parameters.timeStep;
    let indexDensity : i32 = density[pointIndex];
    var mass : f32 = 5.25086 / 4039135.0; // kg/voxel from MRHead assuming 1 kg/liter
    var stiffness : f32 = 30.; // MPa
    if (indexDensity < 100i) {
        mass /= 1000.; // air
        stiffness = 0.1;
    } else {
        stiffness = f32(indexDensity) / 100.0;
    }
    //mass *= f32(indexDensity);

    let displacement : vec3<f32> = displacements[currentOffset + pointIndex].xyz;
    let position : vec3<f32> = vec3<f32>(f32(id.z), f32(id.y), f32(id.x));
    let displacedPosition : vec3<f32> = position + displacement;
    var force : vec3<f32> = mass * parameters.gravity;
    var lineOfForce : vec3<f32> = vec3<f32>(0.0);
    var neighborOffset : i32 = 0;
    var neighborPosition : vec3<f32> = vec3<f32>(0.0);
    var neighborDisplacement : vec3<f32> = vec3<f32>(0.0);
    var displacedNeighbor : vec3<f32> = vec3<f32>(0.0);
    var originalLength : f32 = 0.0;
    var currentLength : f32 = 0.0;
    var strain : f32 = 0.0;
    var kk : i32; var jj : i32; var ii : i32;
    var neighborsVisited : i32 = 0;
    for (kk = -1; kk < 2; kk += 1) {
        if (idi32.z + kk < 0 || idi32.z + kk > dimensions.z - 1) {
            continue;
        }
        for (jj = -1; jj < 2; jj += 1) {
            if (idi32.y + jj < 0 || idi32.y + jj > dimensions.y - 1) {
                continue;
            }
            for (ii = -1; ii < 2; ii += 1) {
                if (idi32.x + ii < 0 || idi32.x + ii > dimensions.x - 1) {
                    continue;
                }
                if ( kk == 0 && jj == 0 && ii == 0 ) {
                    continue;
                }
                neighborOffset = kk * @@SLICE_SIZE@@ + jj * @@ROW_SIZE@@ + ii;
                //neighborDisplacement = displacements[currentOffset + pointIndex + neighborOffset].xyz;
                neighborPosition = vec3<f32>( f32(idi32.z + kk), f32(idi32.y + jj), f32(idi32.x + ii) );
                displacedNeighbor = neighborPosition + neighborDisplacement;
                originalLength = length(vec3<f32>(vec3<i32>(kk, jj, ii)));
                currentLength = length(displacedPosition - displacedNeighbor);
                strain = abs(currentLength - originalLength) / originalLength;
                lineOfForce = normalize(displacedNeighbor - displacedPosition);
                if (currentLength > originalLength) {
                    lineOfForce *= -1.0;
                }
                let neighborForce : vec3<f32> = stiffness * strain * lineOfForce;
                force += neighborForce;
                neighborsVisited += 1;
            }
        }
    }

    // boundary conditions
    let acceleration : vec3<f32> = force / mass;
    if (idi32.z == 0) {
        velocities[nextOffset + pointIndex] = vec4<f32>(0.0);
        displacements[nextOffset + pointIndex] = vec4<f32>(0.0);
    } else {
        let maxVelocity = vec3<f32>(1.0);
        let integratedVelocity = 0.5 * acceleration * timeStepSquared;
        let velocity = vec4<f32>(min(maxVelocity, integratedVelocity), 0.0);
        velocities[nextOffset + pointIndex] += velocity;
        displacements[nextOffset + pointIndex] += velocities[currentOffset + pointIndex] * parameters.timeStep;
    }

    // for testing
    //displacements[nextOffset + pointIndex] = vec4<f32>(0.001 * f32(pointIndex) * parameters.iteration);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(id), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(mass), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(force, 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(neighborForce, 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(dimensions), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(f32(k), f32(j), f32(i), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(currentLength), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(currentLength, originalLength, strain), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(f32(neighborsVisited), 0., 0., 0.);
    //displacements[nextOffset + pointIndex] = vec4<f32>(f32(kk), f32(jj), f32(ii), 0.0);
    //displacements[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(position), 0.0);

    //debugBuffer[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(force), 1.0);
    //debugBuffer[nextOffset + pointIndex] = vec4<f32>(strain, stiffness, 0.0, 1.0);
    //debugBuffer[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(position), 1.0);
    debugBuffer[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(neighborPosition), 1.0);
    //debugBuffer[nextOffset + pointIndex] = vec4<f32>(vec3<f32>(vec3<i32>(kk, jj, ii)), 1.0);
}
"""
shader = shader.replace("@@SLICES@@", str(volumeArray.shape[0]))
shader = shader.replace("@@ROWS@@", str(volumeArray.shape[1]))
shader = shader.replace("@@COLUMNS@@", str(volumeArray.shape[2]))
shader = shader.replace("@@SLICE_SIZE@@", str(sliceSize))
shader = shader.replace("@@VOLUME_SIZE@@", str(volumeSize))
shader = shader.replace("@@ROW_SIZE@@", str(volumeArray.shape[2]))

parametersArray = numpy.array([
    0., # iteration
    0.0001, # timeStep
    #0.0, 0.0, -9.8, # gravity
    0.0, 0.0, 0.0, # gravity
    0.0, 0.0, 0.0 # dummies
    ], dtype="float32");


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
buffers[2] = device.create_buffer_with_data(data=velocitiesArray, usage=storageCopyUsage)
buffers[3] = device.create_buffer_with_data(data=parametersArray, usage=uniformUsage)
buffers[4] = device.create_buffer_with_data(data=debugArray, usage=storageCopyUsage)

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
    { "binding": 2,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.storage, "has_dynamic_offset": False}
    },
    { "binding": 3,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {"type": wgpu.BufferBindingType.uniform, "has_dynamic_offset": False}
    },
    { "binding": 4,
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
    { "binding": 2,
      "resource": {"buffer": buffers[2], "offset": 0, "size": buffers[2].size}
    },
    { "binding": 3,
      "resource": {"buffer": buffers[3], "offset": 0, "size": buffers[3].size}
    },
    { "binding": 4,
      "resource": {"buffer": buffers[4], "offset": 0, "size": buffers[4].size}
    },
]
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

# Create a pipeline and run it
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

slicer.app.processEvents()

startTime = time.time()
stop = False
def stopSimulation():
    global stop
    stop = True
try:
    stopButton
except NameError:
    toolbar = slicer.util.findChildren(name="MainToolBar")[0]
    stopText = "Stop Simulation"
    if slicer.util.findChildren(toolbar, text=stopText) == []:
        stopButton = qt.QPushButton(stopText)
        stopButton.connect("clicked()", stopSimulation)
        stopAction = qt.QWidgetAction(toolbar)
        stopAction.setDefaultWidget(stopButton)
        toolbar.addAction(stopAction)

for iteration in range(iterations):
    parametersArray[0] = float(iteration)
    parametersBuffer = device.create_buffer_with_data(data=parametersArray, usage=wgpu.BufferUsage.COPY_SRC)
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(parametersBuffer, 0, buffers[3], 0, parametersArray.nbytes)
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 args not used
    compute_pass.dispatch_workgroups(*volumeIntArray.shape)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    # Read the current data of the output buffer
    if iteration % iterationInterval == 0 or iteration == iterations - 1 or stop:
        displacementsMemory = device.queue.read_buffer(buffers[1])  # slow, should be done async
        displacementsArray[:] = numpy.array(displacementsMemory.cast("f", displacementsArray.shape))
        gridTransformArray[:] = displacementsArray[(iteration + 1) % 2][:,:,:,0:3]
        slicer.util.arrayFromGridTransformModified(gridTransformNode)
        slicer.util.arrayFromVolumeModified(volumeNode)
        print(f"{iteration} of {iterations}")
        slicer.app.processEvents()

    if DEBUGGING:
        velocitiesMemory = device.queue.read_buffer(buffers[2])
        velocitiesArray[:] = numpy.array(velocitiesMemory.cast("f", velocitiesArray.shape))
        debugMemory = device.queue.read_buffer(buffers[4])
        debugArray[:] = numpy.array(debugMemory.cast("f", debugArray.shape))
        debug0Array = slicer.util.arrayFromVolume(debug0Volume)
        debug1Array = slicer.util.arrayFromVolume(debug1Volume)
        debug0Array[:] = debugArray[0]
        debug1Array[:] = debugArray[1]
        slicer.util.arrayFromVolumeModified(debug0Volume)
        slicer.util.arrayFromVolumeModified(debug1Volume)

    if stop == True:
        break

endTime = time.time()

displacementVolume = addVolumeFromGridTransform(gridTransformNode, name="Displacement Volume")
displacementVolume.GetDisplayNode().SetInterpolate(False)
slicer.util.setSliceViewerLayers(displacementVolume, fit=True)

print(f"Finished at {iterations / (endTime - startTime)} iterations/second, ")

"""

volumeArray[:] = resultArray.astype('int16').reshape(volumeArray.shape)
slicer.util.arrayFromVolumeModified(volumeNode)
volumeNode.GetDisplayNode().SetAutoWindowLevel(False)
volumeNode.GetDisplayNode().SetAutoWindowLevel(True)
slicer.app.processEvents()
slicer.util.delayDisplay("done")
"""
