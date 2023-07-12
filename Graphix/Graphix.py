import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# Graphix
#

class Graphix(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Graphix"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of using Graphix within Slicer to make custom rendering code that uses WebGPU via wgpu.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""


#
# GraphixParameterNode
#

@parameterNodeWrapper
class GraphixParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# GraphixWidget
#

class GraphixWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Graphix.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = GraphixLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[GraphixParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# GraphixLogic
#

class GraphixLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return GraphixParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# GraphixTest
#

class GraphixTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        try:
            import Graphix
        except ModuleNotFoundError:
            slicer.util.pip_install("pygfx")
            import pygfx

        #self.setUp()
        #self.test_GraphixVolume()
        #self.setUp()
        #self.test_GraphixModel()
        self.setUp()
        self.test_GraphixTotalSeg()

    def test_GraphixVolume(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        """
         /c/Users/piepe/AppData/Local/slicer.org/Slicer\ 5.3.0-2023-07-01/Slicer.exe
        """

        self.delayDisplay("Starting the test", 100)

        from wgpu.gui.qt import QWgpuCanvas
        import pygfx

        canvas = QWgpuCanvas()
        renderer = pygfx.renderers.WgpuRenderer(canvas)
        scene = pygfx.Scene()

        try:
            mrHead = slicer.util.getNode("MRHead")
        except slicer.util.MRMLNodeNotFoundException:
            import SampleData
            mrHead = SampleData.SampleDataLogic().downloadMRHead()
        voldata = slicer.util.array("MRHead")

        geometry = pygfx.Geometry(grid=voldata)
        material = pygfx.VolumeRayMaterial(clim=(voldata.min(), voldata.max()))

        vol1 = pygfx.Volume(geometry, material)
        ijkToRAS = vtk.vtkMatrix4x4()
        mrHead.GetIJKToRASMatrix(ijkToRAS)
        vol1.local.matrix = slicer.util.arrayFromVTKMatrix(ijkToRAS)
        scene.add(vol1)

        camera = pygfx.PerspectiveCamera(70, 16 / 9)
        camera.show_object(scene, view_dir=(0, -500, 0), up=(0, 0, 1))
        controller = pygfx.OrbitController(camera, register_events=renderer)

        def animate():
            renderer.render(scene, camera)
            canvas.request_draw()
            
        canvas.request_draw(animate)

        self.delayDisplay('Test passed', 100)


    def test_GraphixModel(self):
        """ 
        Create a pygfx geometry using a vtkPolyData
        """

        self.delayDisplay("Starting the test", 100)

        import numpy
        import vtk
        import vtk.util.numpy_support

        from wgpu.gui.qt import QWgpuCanvas
        import pygfx

        canvas = QWgpuCanvas()
        renderer = pygfx.renderers.WgpuRenderer(canvas)
        scene = pygfx.Scene()

        source = vtk.vtkPlatonicSolidSource()
        source.SetSolidTypeToDodecahedron()
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputConnection(source.GetOutputPort())
        triangleFilter.Update()
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputConnection(triangleFilter.GetOutputPort())
        normalsFilter.Update()
        polyData = normalsFilter.GetOutput()

        triangleIndices = polyData.GetPolys().GetData()
        triangleIndexNumpyArray = vtk.util.numpy_support.vtk_to_numpy(triangleIndices).astype('uint32')
        # vtk stores the vertext count per triangle (so delete the 3 at every 4th entry)
        triangleIndexNumpyArray = numpy.delete(triangleIndexNumpyArray, slice(None,None,4))
        triangleIndexArray = numpy.asarray(triangleIndexNumpyArray, order='C')
        triangleIndexArray = triangleIndexArray.reshape(triangleIndexArray.shape[0]//3,3)

        trianglePoints = polyData.GetPoints().GetData()
        trianglePointArray = vtk.util.numpy_support.vtk_to_numpy(trianglePoints)

        triangleNormals = polyData.GetPointData().GetArray('Normals')
        triangleNormalsArray = vtk.util.numpy_support.vtk_to_numpy(triangleNormals)

        import random
        random.seed(42)
        for i in range(50):
            geometry = pygfx.Geometry(indices=triangleIndexArray, positions=trianglePointArray, normals=triangleNormalsArray)
            material = pygfx.MeshPhongMaterial()
            material = pygfx.MeshPhongMaterial(color=[0.75*random.random(),0.75*random.random(),0.75*random.random()])
            mesh = pygfx.Mesh(geometry,material)
            mesh.local.x = random.random() * 10 - 5
            mesh.local.y = random.random() * 10 - 5
            mesh.local.z = random.random() * 10 - 5
            mesh.local.scale_x *= random.random() * 2
            mesh.local.scale_y *= random.random() * 2
            mesh.local.scale_z *= random.random() * 2
            mesh.cast_shadow = True
            mesh.receive_shadow = True
            scene.add(mesh)

        ambient = pygfx.AmbientLight()
        scene.add(ambient)

        camera = pygfx.PerspectiveCamera(70, 16 / 9, depth_range=(0.1, 2000))
        camera.show_object(scene, view_dir=(0, -50, 0), up=(0, 0, 1))
        controller = pygfx.OrbitController(camera, register_events=renderer)

        shadowBias = 0.05

        light = pygfx.PointLight("#4040ff", 500, decay=2)
        light.local.x = 15
        light.local.y = 20
        light.cast_shadow = True
        light.bias = shadowBias
        scene.add(light)

        light2 = pygfx.DirectionalLight("#aaaa55", 5)
        light2.local.position = (-150, 100, 100)
        light2.cast_shadow = True
        light2.bias = shadowBias
        scene.add(light2)

        """
        light3 = pygfx.SpotLight("#ffffff", 500, angle=0.3, penumbra=0.2, decay=1)
        light3.local.position = (0, 0, 100)
        light3.cast_shadow = True
        light3.bias = shadowBias
        scene.add(light3)
        """
        
        def animate():
            renderer.render(scene, camera)
            canvas.request_draw()
            
        canvas.request_draw(animate)

       

        self.delayDisplay('Test passed', 100)

    def loadTotalSeg(self):
        filePath = f"{slicer.util.tempDirectory()}/ctacardio-totalseg-models-2023-05-22-Scene.mrb"
        print(f"Downloading to {filePath}")
        demoDataURL = "https://github.com/pieper/SlicerWeb/releases/download/v0.1/ctacardio-totalseg-models-2023-05-22-Scene.mrb"
        slicer.util.downloadFile(demoDataURL, filePath)
        slicer.util.loadScene(filePath)

    def test_GraphixTotalSeg(self):
        """ 
        Create a pygfx geometry using a vtkPolyData
        """

        self.delayDisplay("Starting the test", 100)

        self.loadTotalSeg()

        import numpy
        import vtk
        import vtk.util.numpy_support

        from wgpu.gui.qt import QWgpuCanvas
        import pygfx

        canvas = QWgpuCanvas()
        renderer = pygfx.renderers.WgpuRenderer(canvas)
        scene = pygfx.Scene()

        meshesNodes = []

        for model in slicer.util.getNodes("vtkMRMLModelNode*").values():
            if model.GetDisplayNode().GetVisibility() == 0:
                continue
            polyData = model.GetPolyData()
            triangleIndices = polyData.GetPolys().GetData()
            triangleIndexNumpyArray = vtk.util.numpy_support.vtk_to_numpy(triangleIndices).astype('uint32')
            # vtk stores the vertext count per triangle (so delete the 3 at every 4th entry)
            triangleIndexNumpyArray = numpy.delete(triangleIndexNumpyArray, slice(None,None,4))
            triangleIndexArray = numpy.asarray(triangleIndexNumpyArray, order='C')
            triangleIndexArray = triangleIndexArray.reshape(triangleIndexArray.shape[0]//3,3)

            trianglePoints = polyData.GetPoints().GetData()
            trianglePointArray = vtk.util.numpy_support.vtk_to_numpy(trianglePoints)

            triangleNormals = polyData.GetPointData().GetArray('Normals')
            triangleNormalsArray = vtk.util.numpy_support.vtk_to_numpy(triangleNormals)

            geometry = pygfx.Geometry(indices=triangleIndexArray, positions=trianglePointArray, normals=triangleNormalsArray)
            color = model.GetDisplayNode().GetColor()
            material = pygfx.MeshPhongMaterial(color=color, specular=[.6,.6,.6], shininess=100)
            mesh = pygfx.Mesh(geometry,material)
            mesh.cast_shadow = True
            mesh.receive_shadow = True
            scene.add(mesh)
            meshesNodes.append((mesh, model))

        ambient = pygfx.AmbientLight()
        scene.add(ambient)

        camera = pygfx.PerspectiveCamera(70, 16 / 9, depth_range=(0.1, 2000))
        camera.show_object(scene, view_dir=(0, -50, 0), up=(0, 0, 1))
        controller = pygfx.OrbitController(camera, register_events=renderer)

        shadowBias = 0.05

        light = pygfx.PointLight("#a0a0ff", 500, decay=2)
        light.local.x = 15
        light.local.y = 20
        light.cast_shadow = True
        light.bias = shadowBias
        scene.add(light)

        light2 = pygfx.DirectionalLight("#aaaa88", 5)
        light2.local.position = (-150, 100, 100)
        light2.cast_shadow = True
        light2.bias = shadowBias
        scene.add(light2)

        """
        light3 = pygfx.SpotLight("#ffffff", 500, angle=0.3, penumbra=0.2, decay=1)
        light3.local.position = (0, 0, 100)
        light3.cast_shadow = True
        light3.bias = shadowBias
        scene.add(light3)
        """
        
        def animate():
            for mesh,node in meshesNodes:
                mesh.visible = node.GetDisplayNode().GetVisibility()
            renderer.render(scene, camera)
            canvas.request_draw()
            
        canvas.request_draw(animate)

       

        self.delayDisplay('Test passed', 100)


"""
lm = slicer.app.layoutManager()
tdw = lm.threeDWidget(0)
tdw.threeDView().hide()
tdw.threeDController().hide()
canvas = slicer.modules._wgpuwidgets[1]
qwindow = qt.QWindow.fromWinId(canvas.get_window_id())
qwidget = qt.QWidget.createWindowContainer(qwindow)
tdw.layout().addWidget(qwidget)
"""
