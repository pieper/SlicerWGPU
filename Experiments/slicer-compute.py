"""

Install wgpu as described here: https://github.com/pygfx/wgpu-py

Tested with Slicer 5.0.2 and wgpu 537c3eab68e9eef77681fc5545532380df26d8cc (basically 0.8.1)

filePath = "/Users/pieper/slicer/latest/SlicerWGPU/Experiments/slicer-compute.py"
filePath = "c:/pieper/SlicerWGPU/Experiments/slicer-compute.py"

exec(open(filePath).read())

"""

import numpy

import wgpu
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function

try:
    mrHead = slicer.util.getNode("MRHead")
except slicer.util.MRMLNodeNotFoundException:
    import SampleData
    mrHead = SampleData.SampleDataLogic().downloadMRHead()

headArray = slicer.util.arrayFromVolume(mrHead)
sliceSize =  headArray.shape[1] * headArray.shape[2]
headIntArray = headArray.astype('int32')
bufferSize = headArray.flatten().shape[0]

shader = """

@group(0) @binding(0)
var<storage,read> inputData: array<i32>;

@group(0) @binding(1)
var<storage,read_write> outputData: array<i32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) index: vec3<u32>) {
    let i: u32 = index.x * @@SLICE_SIZE@@ + index.y * @@ROW_SIZE@@ + index.z;
    outputData[i] = -1 * inputData[i];
}

"""
shader = shader.replace("@@SLICE_SIZE@@", str(sliceSize)+"u")
shader = shader.replace("@@ROW_SIZE@@", str(headArray.shape[2])+"u")

print("computing...")
out = compute_with_buffers( input_arrays={0: headIntArray},
                            output_arrays={1: (bufferSize, "i")},
                            shader=shader,
                            n=headArray.shape )
print("done")

# `out` is a dict matching the output types
# Select data from buffer at binding 1
resultArray = numpy.array(out[1])
assert resultArray.mean() == -1 * headArray.mean()

headArray[:] = resultArray.astype('int16').reshape(headArray.shape)
slicer.util.arrayFromVolumeModified(mrHead)
mrHead.GetDisplayNode().SetAutoWindowLevel(False)
mrHead.GetDisplayNode().SetAutoWindowLevel(True)
