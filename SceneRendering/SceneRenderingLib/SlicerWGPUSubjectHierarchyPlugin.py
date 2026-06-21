"""Subject hierarchy plugin contributing two right-click actions on
volume and segmentation nodes:

  - "Toggle SlicerWGPU rendering" : flips the renderEnabled flag on
    the per-data-node SlicerWGPU state node (a vtkMRMLScriptedModuleNode
    managed by SceneRenderingLib.wgpu_state) and applies it via the
    wgpu bridge if installed (one segmentation at a time; volumes via
    VRDN).
  - "Edit SlicerWGPU options..." : selects the SceneRendering module
    and pre-selects the right-clicked item in its edit-panel combo.

Registration pattern follows the precedent at
slicer-extensions/SlicerHeart/ValveAnnulusAnalysis/HeartValveLib/
HeartValvesSubjectHierarchyPlugin.py.
"""

import qt
import slicer
from AbstractScriptedSubjectHierarchyPlugin import (
    AbstractScriptedSubjectHierarchyPlugin,
)

from SceneRenderingLib import wgpu_state


class SlicerWGPUSubjectHierarchyPlugin(AbstractScriptedSubjectHierarchyPlugin):

    # Required by qSlicerSubjectHierarchyScriptedPlugin.setPythonSource().
    filePath = __file__

    def __init__(self, scriptedPlugin):
        scriptedPlugin.name = "SlicerWGPU"
        AbstractScriptedSubjectHierarchyPlugin.__init__(self, scriptedPlugin)

        self.toggleAction = qt.QAction(
            "Toggle SlicerWGPU rendering", scriptedPlugin)
        self.toggleAction.setCheckable(True)
        self.toggleAction.triggered.connect(self.onToggleRendering)

        self.editAction = qt.QAction(
            "Edit SlicerWGPU options...", scriptedPlugin)
        self.editAction.triggered.connect(self.onEditOptions)

    # ---- plugin API ----

    def itemContextMenuActions(self):
        return [self.toggleAction, self.editAction]

    def sceneContextMenuActions(self):
        return []

    def showContextMenuActionsForItem(self, itemID):
        # Default state: both actions hidden until we determine the
        # item is a volume or segmentation. The plugin framework
        # resets visibility before each call.
        node = self._nodeForItem(itemID)
        if node is None:
            return
        if not self._isSupportedNode(node):
            return
        self.toggleAction.visible = True
        self.editAction.visible = True
        self.toggleAction.setChecked(wgpu_state.is_render_enabled(node))

    # ---- action handlers ----

    def onToggleRendering(self):
        node = self._currentNode()
        if node is None:
            return
        new_state = not wgpu_state.is_render_enabled(node)
        wgpu_state.set_render_enabled(node, new_state)
        self._applyToBridge(node, new_state)

    def onEditOptions(self):
        handler = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
        currentItemID = handler.currentItem()
        slicer.util.selectModule("SceneRendering")
        widget = self._sceneRenderingWidget()
        if widget is not None and hasattr(widget, "selectItemForEditing"):
            widget.selectItemForEditing(currentItemID)

    # ---- helpers ----

    @staticmethod
    def _isSupportedNode(node):
        return node.IsA("vtkMRMLScalarVolumeNode") or node.IsA(
            "vtkMRMLSegmentationNode")

    @staticmethod
    def _nodeForItem(itemID):
        invalid = slicer.vtkMRMLSubjectHierarchyNode.GetInvalidItemID()
        if itemID == invalid:
            return None
        handler = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
        shNode = handler.subjectHierarchyNode()
        if shNode is None:
            return None
        return shNode.GetItemDataNode(itemID)

    @classmethod
    def _currentNode(cls):
        handler = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
        return cls._nodeForItem(handler.currentItem())

    @staticmethod
    def _sceneRenderingWidget():
        try:
            return slicer.modules.scenerendering.widgetRepresentation().self()
        except Exception:
            return None

    def _applyToBridge(self, node, rendered):
        # No-op direct bridge calls: the bridge observes wgpu_state
        # node Modified events and reconciles on its own (claims/
        # unclaims VRDNs, adds/removes segmentations, backfills VRDNs
        # for state-enabled volumes that lack one). Just surface a
        # status hint when no bridge is installed so the user knows
        # to open SceneRendering once.
        bridge = getattr(slicer.modules, "wgpuVtkBridge", None)
        if bridge is None:
            slicer.util.showStatusMessage(
                "SlicerWGPU bridge not installed. Open SceneRendering "
                "module to install it.", 3000)
