# SlicerWGPU
Use wgpu, a rust-python WebGPU implementation, in 3D Slicer

## Goals

* Provide a high-level and cross-platform way to do use GPU resources for computing and rendering.
  * Minimize the amount of low-level and platform-specific coding required to implement useful features.
  * Reduce maintainable burden by using a consistent code base across platforms.
  * Rely on the web standard and mulitiple widely used implementations to provide stability.
* Leverage the Python and WebGPU ecosystems in Slicer.
* Share WebGPU experience and possibly code with web systems such as VTK, vtk.js, cornerstone3D, etc.
* Support GPU computing without proprietary code or complex installations.
* Allow editing the GPU code on-the-fly using metaprogramming to adapt to the data.

## Status

This is currently an experiment that works for a few test cases.  It is not clear yet what performance gains will be possible.

### What currently works
* [Compute shaders](Experiments/slicer-compute.py) can be used to operate on numpy arrays (e.g. arrays from Slicer volumes).
* [Vertex and fragment shaders](Experiments/slicer-render.py) can be used to render into off-screen framebuffers.  Results of rendering are available as numpy arrays that can be rendered in the Slicer interface via conventional VTK or Qt methods.

### What's not currently avalable or not known
* wgpu operations are synchronous in Slicer so you cannot easily overlap wgpu rendering with other calculations.
* Rendered content must be copied from the GPU back to the CPU because VTK/Qt in Slicer do not have access to the wgpu GPU memory.  wgpu does provide hooks for PyQt and PySide so it may be possible to port these techniques for use in PythonQt.
* Compute shader operations that last over 2 seconds on windows can trigger windows to kill the process.
* Because wgpu is an abstraction, it will in some ways be a least-common-denominator of functionality compared to native solutions like Metal.  It's not clear yet if this will be a significant issue in terms of performance or availability of features.
* Using an abstraction and cross-compilation layer may introduce feature skew or implementation bugs across platforms and that difficult or impossible to track down and fix.  This is less of a concern for wgpu because it is not one-person or small team effort, but a coordinated project implementing a well defined web standard for a widely-used browser.

## Architecture
The wgpu Python module provides an api to access GPU features like you would have via JavaScript in a browser.  Note that wgpu does not call JavaScipt code but instead calls the same underlying implementation that is used by the browser itself (that is, the python calls machine code that was generated from rust source code).  Shaders are cross-compiled on the fly from [W3C standard WGSL](https://www.w3.org/TR/WGSL/) to the platform's shader language, e.g. to [Metal Shading Language](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) on Apple machines.

![image](https://user-images.githubusercontent.com/126077/175791989-4f3fdcdb-6e80-4d0c-b199-1ff7b51c2b6a.png)


## Background

* WebGL and WebGL2 are widely available, but strongly tied to OpenGL ES.
* WebGPU is a open community standards-based effort to replace WebGL with something that has less of the fixed-function assumptions of OpenGL and more closely matches the architecture of modern GPUs.
* WebGPU has participation from all major browser vendors (Apple, Google, & Mozilla) 
* WebGPU is still being finalized, but is pbeginning to converge and be useful](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status).
* There are (at least) two major open source efforts to implement the WebGPU standard.
   * [Dawn](https://dawn.googlesource.com/dawn) is in C++ by Google for use in Chromium / Chrome
   * [WGPU](https://github.com/gfx-rs/wgpu) is in Rust for use in Firefox
* While both implementation should ultimately be compatible with Slicer the WGPU offers a convenient pip-installable package compatible with Slicer and ready to use today.
* Longer-term there has [been discussion](https://discourse.vtk.org/t/vulkan-development/3307/22) of using WebGPU in VTK via Dawn, which would make it available via C++ or Python in Slicer.  Since Dawn and WGPU implement WGPU, porting code and shaders between the two should be managable.
