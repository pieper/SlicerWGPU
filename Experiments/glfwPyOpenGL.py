"""
filePath = "/Users/pieper/slicer/latest/SlicerWGPU/Experiments/glfwPyOpenGL.py"

exec(open(filePath).read())
"""
try:
    import glfw
    import OpenGL
    import OpenGL.GL
except ModuleNotFoundException:
    pip_install("glfw")
    pip_install("PyOpenGL")
    import glfw
    import OpenGL
    import OpenGL.GL

glfw.init()

w = glfw.create_window(500,500,"hoot", None, None)

glfw.make_context_current(w)
OpenGL.GL.glClearColor(1., .5, .2, 1.)
OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
glfw.swap_buffers(w)
