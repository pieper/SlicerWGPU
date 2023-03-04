"""
import sys
if sys.platform == 'darwin':
  filePath = "/Users/pieper/slicer/latest/SlicerWGPU/Experiments/glfwPyOpenGL.py"
if sys.platform == 'win32':
  filePath = "c:/pieper/SlicerWGPU/Experiments/glfwPyOpenGL.py"
if sys.platform == 'linux2':
  filePath = "/home/ubuntu/SlicerWGPU/Experiments/glfwPyOpenGL.py"

exec(open(filePath).read())
"""

import sys

try:
    import glfw
    import OpenGL
    import OpenGL.GL
except ModuleNotFoundError:
    pip_install("glfw")
    pip_install("PyOpenGL")
    import glfw
    import OpenGL
    import OpenGL.GL

glfw.init()

glfwWindow = glfw.create_window(500,500,"hoot", None, None)

def render(window):
    glfw.make_context_current(window)
    OpenGL.GL.glClearColor(1., .5, .2, 1.)
    OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
    glfw.swap_buffers(window)

widget = qt.QWidget()
layout = qt.QVBoxLayout(widget)

if sys.platform == 'linux2':
    qwindow = qt.QWindow.fromWinId(glfw.get_x11_window(glfwWindow))
elif sys.platform == 'darwin':
    qwindow = qt.QWindow.fromWinId(glfw.get_cocoa_window(glfwWindow))
elif sys.platform == 'win32':
    qwindow = qt.QWindow.fromWinId(glfw.get_win32_window(glfwWindow))

qcontainer = qt.QWidget.createWindowContainer(qwindow)
layout.addWidget(qcontainer)


renderButton = qt.QPushButton("Render")
renderButton.connect("clicked()", lambda window=glfwWindow : render(window))

layout.addWidget(renderButton)

widget.show()

