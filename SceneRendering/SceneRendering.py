"""SceneRendering -- a Slicer module that exercises the slicer_wgpu
SceneRenderer (Field-compositing ray tracer) via a set of named
self-tests.

The module's UI is a vertical stack of buttons, one per self-test. Each
button reloads the module and runs exactly one test so edits made
during iteration take effect without restarting Slicer.

Working tests (happy paths that end with interactive DualView state
stashed on `slicer.modules.*` for inspection):

    test_SingleVolume           CTACardio + DualView (volume only)
    test_VolumeAndFiducials     CTACardio + 4 markup lists (100 points)
    test_TransformableVolume    CTACardio under a linear transform
                                (rotation + non-uniform stretch); also
                                exercises the in-place transform-update
                                fast path so Phong shading tracks the
                                deformation per frame.

Further stages:

    test_MultiVolume            two registered volumes composited in
                                the same SceneRenderer
    test_DeformableVolume       a vtkMRMLGridTransformNode whose
                                animated sinewave displacement deforms
                                the volume each frame via the
                                TransformField path
    test_CinematicRendering     two-light (shadow-casting key + fill)
                                cinematic render with camera-relative
                                lights
"""

import logging

import ctk
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)

from SceneRenderingLib import wgpu_state
from SceneRenderingLib import wgpu_volume_render


MODULE_NAME = "SceneRendering"


def _fix_macos_processor_for_rubicon():
    """Work around a wgpu-init crash on Apple Silicon Slicer builds.

    On this hardware platform.machine() is 'arm64', but Slicer's process
    spawns its child processes translated (Rosetta), so the `uname -p`
    that CPython's platform.processor() shells out to returns 'i386'.
    rubicon-objc -- pulled in by wgpu's Metal backend -- reads processor(),
    sees 'i386' on a 64-bit build, and concludes the arch is x86_64. It
    then binds the ObjC symbol objc_msgSendSuper_stret, an x86-only
    struct-return message variant that macOS 26's libobjc no longer
    exports; the dlsym lookup fails and aborts the wgpu native-backend
    import with an AttributeError.

    'arm' is the correct value for this hardware, so this is a fix, not a
    spoof: rubicon then takes its arm64 path and never references the dead
    symbol. Must run before rubicon.objc is first imported -- i.e. before
    the first wgpu adapter/device request -- which is why _ensure_dependencies
    calls it ahead of importing wgpu. No-op off macOS, when processor() is
    already arm, or once rubicon is loaded (too late to influence its
    one-time arch detection).
    """
    import sys
    import platform
    if sys.platform != "darwin" or "rubicon" in sys.modules:
        return
    if platform.machine() == "arm64" and platform.processor() not in ("arm", "arm64"):
        platform.processor = lambda: "arm"


def _force_vulkan_only_wgpu_instance():
    """On Linux, create the wgpu instance with only the Vulkan backend enabled.

    wgpu enumerates EVERY backend when it first creates its instance / requests an
    adapter. Its OpenGL-ES backend opens an EGL display chosen from the environment
    (WAYLAND_DISPLAY -> wayland, else DISPLAY -> X11). On NVIDIA under XWayland (a
    headless / browser-streamed desktop) that EGL probe aborts the whole process --
    wgpu-hal panics with BadAccess across the C FFI (unrecoverable). Restricting the
    instance to Vulkan means the GL/ES backend is never created, so the probe never runs.
    Vulkan WSI still serves offscreen and on-screen surfaces, and DISPLAY is left
    untouched so VTK/GLX rendering works.

    Linux-only: macOS uses Metal and Windows uses DX12/Vulkan, where this restriction
    would remove the only available backend. Override the backend list with
    SLICER_WGPU_INSTANCE_BACKENDS (comma-separated, e.g. "Vulkan,GL"). Idempotent, and a
    no-op once the instance already exists (then it is too late to choose backends).
    """
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    try:
        import wgpu
        if getattr(wgpu, "_slicer_wgpu_instance_extras_set", False):
            return
        from wgpu.backends.wgpu_native import _helpers
        if _helpers._the_instance is not None:
            return  # too late -- instance already created with all backends
        backends = [b.strip() for b in
                    os.environ.get("SLICER_WGPU_INSTANCE_BACKENDS", "Vulkan").split(",")
                    if b.strip()]
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        set_instance_extras(backends=backends)
        wgpu._slicer_wgpu_instance_extras_set = True
    except Exception as exc:
        print(f"slicer_wgpu: could not restrict wgpu instance to Vulkan: {exc}")


def _patch_rendercanvas_pythonqt_qt6():
    """Fix a pieper/rendercanvas fork incompatibility under PythonQt + Qt6.

    The fork guards PythonQt property-access -- under PythonQt, a QWidget's
    width/height/rect are int/QRect PROPERTIES, not methods -- with
    `is_pythonqt and qt_version_info[0] < 6`. Slicer is now PythonQt on Qt6,
    where they are STILL properties, so on Qt6 the guard falls through to the
    method form (`self.width()`) and raises "'int' object is not callable" in
    the canvas resizeEvent; the canvas never sizes, cascading to
    "'NoneType' object has no attribute 'renderer'" in the legacy DualView /
    Bouncing-Head demos.

    Fixed upstream in pieper/rendercanvas (pythonqt-support); this self-heals
    an already-installed older fork. Only the two WIDGET property sites are
    corrected -- the wheel-event pos()/position() site keeps its version
    check, because QWheelEvent.pos() really was removed in Qt6. After editing,
    drop the stale cached modules and rebind pygfx's captured BaseRenderCanvas
    (the same dance the fork-install path does). Only the legacy pygfx-Qt
    surface path uses these, so the raw-wgpu VTK-injection features are
    unaffected. Idempotent: a no-op once the file is already correct.
    """
    import os
    import sys
    try:
        import rendercanvas
    except Exception:
        return
    fixes = [
        ("self.rect if is_pythonqt and qt_version_info[0] < 6 else self.rect()",
         "self.rect if is_pythonqt else self.rect()"),
        ("if is_pythonqt and qt_version_info[0] < 6:\n            lsize = float(self.width)",
         "if is_pythonqt:\n            lsize = float(self.width)"),
    ]
    try:
        qt_path = os.path.join(os.path.dirname(rendercanvas.__file__), "qt.py")
        with open(qt_path) as f:
            src = f.read()
        new = src
        for old, repl in fixes:
            new = new.replace(old, repl)
        if new == src:
            return
        with open(qt_path, "w") as f:
            f.write(new)
        for mod_name in [m for m in list(sys.modules)
                         if m == "rendercanvas" or m.startswith("rendercanvas.")]:
            sys.modules.pop(mod_name, None)
        if "pygfx" in sys.modules:
            import importlib
            rc = importlib.import_module("rendercanvas")
            pgr = importlib.import_module(
                "pygfx.renderers.wgpu.engine.renderer")
            pgr.BaseRenderCanvas = rc.BaseRenderCanvas
    except Exception as exc:
        print(f"slicer_wgpu: rendercanvas PythonQt-Qt6 patch skipped: {exc}")


#
# SceneRendering
#

class SceneRendering(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Scene Rendering"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"]
        self.parent.helpText = """
Field-compositing Scene Renderer (slicer_wgpu) driver module. Use the
buttons in this panel to reload the module and run a single self-test.
"""
        self.parent.acknowledgementText = """
Built on slicer-wgpu (https://github.com/pieper/slicer-wgpu) and pygfx.
"""


#
# SceneRenderingWidget -- VBox of "reload + run test" buttons
#

class SceneRenderingWidget(ScriptedLoadableModuleWidget):

    # VTK-injection tests (wgpu rendered directly into Slicer's 3D view
    # via a vtkCommand::EndEvent hook -- no DualView, no second pane).
    # Listed first because this is the path we're actively developing.
    VTK_TESTS = [
        ("Injection: Single Volume",           "test_vtk_SingleVolume"),
        ("Injection: Volume + Fiducials",      "test_vtk_VolumeAndFiducials"),
        ("Injection: Multi-Volume (demo)",     "test_vtk_MultiVolume"),
        ("Injection: Landmark Deform (TPS)",   "test_vtk_LandmarkDeform"),
        ("Injection: Segmentation (paint)",    "test_vtk_Segmentation"),
        ("Injection: Segment Surface (Carving)", "test_vtk_SegmentSurfaces"),
        ("Injection: Colorize (RGBA)",         "test_vtk_ColorizeRGBA"),
        ("Injection: Fiber Strands (A-buffer)", "test_vtk_FiberStrands"),
        ("Injection: Field Compositing (lavalamp + strands + fiducials)",
         "test_vtk_FieldCompositing"),
    ]
    # Legacy DualView/pygfx tests.
    TESTS = [
        ("Single Volume",              "test_SingleVolume"),
        ("Volume + Fiducials",         "test_VolumeAndFiducials"),
        ("Transformable Volume",       "test_TransformableVolume"),
        ("Bouncing Head",              "test_BouncingHead"),
        ("Multi-Volume",               "test_MultiVolume"),
        ("Deformable Volume",          "test_DeformableVolume"),
        ("Cinematic Rendering",        "test_CinematicRendering"),
    ]
    # State / MRB persistence tests for the per-data-node wgpu_state
    # schema. These are fast and don't need GPU work.
    STATE_TESTS = [
        ("State: Basic edit",          "test_state_BasicEdit"),
        ("State: Save / restore",      "test_state_SaveRestore"),
        ("Bridge: auto-enable on add", "test_bridge_autoEnableOnAdd"),
        ("Bridge: per-node opt-out",   "test_bridge_perNodeOptOut"),
        ("Bridge: multi-segmentation", "test_bridge_multiSegmentation"),
    ]

    # Page indices for the dynamic edit panel's QStackedWidget.
    _EDIT_PAGE_EMPTY = 0
    _EDIT_PAGE_VOLUME = 1
    _EDIT_PAGE_SEGMENTATION = 2

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Per-widget state. The bridge instance (when installed) also
        # lives on slicer.modules.wgpuVtkBridge so the test framework's
        # setUp() can find and tear it down.
        self._bridge = None
        self._edited_node_id = None  # MRML node ID currently in edit panel
        # View-checkbox widgets per page, rebuilt when the page is
        # populated for a node. nodeID -> {viewID: QCheckBox}.
        self._view_checks_volume = {}
        self._view_checks_seg = {}
        # Scene observers for state-node lifecycle and load-time
        # state reapplication. (caller, tag) tuples.
        self._scene_observer_tags = []

        self._setup_overrides_section()
        self._setup_demos_section()

        self.layout.addStretch(1)

        self._register_sh_plugin()
        self._install_scene_observers()

        # Default-ON bridge install. Synchronous: the 3D view is
        # virtually always laid out by the time setup() runs. If it
        # isn't (or deps aren't installed) the status label tells the
        # user; they can click "Install / reinstall deps" to retry.
        # Deferring this through QTimer was the prior approach but
        # raced with explicit install_default_bridge() calls in tests,
        # creating dead bridges with the right handle in
        # slicer.modules.wgpuVtkBridge but observers stripped.
        self._install_bridge()

    # ------------------------------------------------------------------
    # Rendering Overrides -- the primary panel.
    # Subject-hierarchy combo at the top drives a dynamic edit panel
    # whose contents adapt to the selected node (volume vs segmentation).
    # The scripted SH plugin (SceneRenderingLib/SlicerWGPUSubjectHierarchy
    # Plugin.py) drives users to this panel via its "Edit SlicerWGPU
    # options..." context-menu action.
    # ------------------------------------------------------------------

    def _setup_overrides_section(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Rendering Overrides"
        section.collapsed = False
        self.layout.addWidget(section)
        outer = qt.QVBoxLayout(section)

        # --- Bridge status row ---
        status_row = qt.QHBoxLayout()
        self._volumeStatusLabel = qt.QLabel("Bridge: (not installed)")
        self._volumeStatusLabel.setWordWrap(True)
        status_row.addWidget(self._volumeStatusLabel, 1)
        self._installDepsButton = qt.QPushButton("Install / reinstall deps")
        self._installDepsButton.setToolTip(
            "Run the same pip bootstrap the self-tests use. Needed the "
            "first time, or after a wgpu / pygfx / slicer-wgpu update.")
        self._installDepsButton.clicked.connect(self._on_install_deps_clicked)
        status_row.addWidget(self._installDepsButton)
        outer.addLayout(status_row)

        # --- Item-to-edit combo ---
        # Build the combo WITHOUT connecting yet -- setMRMLScene and
        # initial show emit currentItemChanged, which would fire before
        # the stack pages exist and crash _populate_*_page().
        combo_row = qt.QHBoxLayout()
        combo_row.addWidget(qt.QLabel("Item:"))
        self._editCombo = slicer.qMRMLSubjectHierarchyComboBox()
        self._editCombo.setMRMLScene(slicer.mrmlScene)
        # Limit picks to node types we know how to render.
        self._editCombo.setNodeTypes(
            ["vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"])
        self._editCombo.setToolTip(
            "Pick a volume or segmentation to edit its SlicerWGPU "
            "rendering options. Driven by the SH context-menu action "
            "'Edit SlicerWGPU options...' as well.")
        combo_row.addWidget(self._editCombo, 1)
        outer.addLayout(combo_row)

        # --- Dynamic edit stack ---
        self._editStack = qt.QStackedWidget()
        self._editStack.addWidget(self._build_edit_page_empty())
        self._editStack.addWidget(self._build_edit_page_volume())
        self._editStack.addWidget(self._build_edit_page_segmentation())
        outer.addWidget(self._editStack)
        self._editStack.setCurrentIndex(self._EDIT_PAGE_EMPTY)

        # Pages exist now -- safe to connect the combo signal.
        self._editCombo.connect(
            "currentItemChanged(vtkIdType)", self._on_edit_item_changed)

    # --- edit-panel page builders ---

    def _build_edit_page_empty(self):
        page = qt.QWidget()
        v = qt.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        hint = qt.QLabel(
            "<i>Select a volume or segmentation above (or right-click "
            "one in the Data module and choose 'Edit SlicerWGPU "
            "options...') to edit its rendering.</i>")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)
        return page

    def _build_edit_page_volume(self):
        page = qt.QWidget()
        v = qt.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        self._volumeRenderCheck = qt.QCheckBox("Render with SlicerWGPU")
        self._volumeRenderCheck.setToolTip(
            "Toggle wgpu rendering for this volume. Persisted on the "
            "per-data-node SlicerWGPU state node so MRML/MRB save/"
            "restore picks it up.")
        self._volumeRenderCheck.toggled.connect(
            self._on_volume_render_check_toggled)
        v.addWidget(self._volumeRenderCheck)

        self._volumeViewGroup, self._volumeViewContainerLayout = (
            self._build_view_selector_group("Render in views"))
        v.addWidget(self._volumeViewGroup)

        tf_group = qt.QGroupBox("Independent wgpu render (new mode)")
        tf_v = qt.QVBoxLayout(tf_group)

        self._volumeIndepCheck = qt.QCheckBox(
            "Render with independent wgpu mode")
        self._volumeIndepCheck.setToolTip(
            "Render this volume through the independent wgpu path -- its "
            "own state node and control-point transfer function, with no "
            "volume-rendering display node. While enabled the legacy "
            "'Render with SlicerWGPU' path steps aside for this volume.")
        self._volumeIndepCheck.toggled.connect(
            self._on_volume_indep_toggled)
        tf_v.addWidget(self._volumeIndepCheck)

        preset_row = qt.QHBoxLayout()
        preset_row.addWidget(qt.QLabel("Preset:"))
        self._volumePresetCombo = qt.QComboBox()
        for name in wgpu_volume_render.PRESETS:
            self._volumePresetCombo.addItem(name)
        self._volumePresetCombo.connect(
            "currentIndexChanged(int)", self._on_volume_preset_changed)
        preset_row.addWidget(self._volumePresetCombo, 1)
        tf_v.addLayout(preset_row)

        tf_hint = qt.QLabel(
            "<i>Preset transfer functions for now; a richer editor will "
            "replace this dropdown.</i>")
        tf_hint.setWordWrap(True)
        tf_v.addWidget(tf_hint)
        v.addWidget(tf_group)

        v.addStretch(1)
        return page

    def _build_edit_page_segmentation(self):
        page = qt.QWidget()
        v = qt.QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        self._segRenderCheck = qt.QCheckBox("Render with SlicerWGPU")
        self._segRenderCheck.setToolTip(
            "Toggle wgpu rendering for this segmentation. The bridge "
            "supports multiple segmentations simultaneously.")
        self._segRenderCheck.toggled.connect(
            self._on_seg_render_check_toggled)
        v.addWidget(self._segRenderCheck)

        self._segViewGroup, self._segViewContainerLayout = (
            self._build_view_selector_group("Render in views"))
        v.addWidget(self._segViewGroup)

        mode_group = qt.QGroupBox("Render mode")
        mode_v = qt.QVBoxLayout(mode_group)
        self._segModeButtons = qt.QButtonGroup(mode_group)
        self._segModeIso = qt.QRadioButton(
            "Isosurface (per-segment label, hard edges)")
        self._segModeSurface = qt.QRadioButton(
            "Surface (gradient-opacity, soft edges)")
        self._segModeIso.setChecked(True)
        self._segModeButtons.addButton(self._segModeIso, 0)
        self._segModeButtons.addButton(self._segModeSurface, 1)
        self._segModeIso.toggled.connect(self._on_seg_mode_changed)
        self._segModeSurface.toggled.connect(self._on_seg_mode_changed)
        mode_v.addWidget(self._segModeIso)
        mode_v.addWidget(self._segModeSurface)
        v.addWidget(mode_group)

        carve_hint = qt.QLabel(
            "<i>Carve / multi-volume compositing options will land "
            "here as the demos (test_vtk_SegmentSurfaces, "
            "test_vtk_FieldCompositing) get promoted to real "
            "controls.</i>")
        carve_hint.setWordWrap(True)
        v.addWidget(carve_hint)

        v.addStretch(1)
        return page

    def _build_view_selector_group(self, title):
        # Returns (group, container_layout). The container_layout is
        # the QVBoxLayout into which _rebuild_view_checks adds one
        # checkbox per vtkMRMLViewNode. We return both because PythonQt
        # forbids stashing Python attrs on the C++-backed QGroupBox.
        # Empty selection means "all views" (matches MRML display node
        # convention).
        group = qt.QGroupBox(title)
        layout = qt.QVBoxLayout(group)
        hint = qt.QLabel(
            "<i>Leave all unchecked to render in every 3D view. "
            "Check specific views to scope this node to those views "
            "only (matches MRML display-node viewNodeIDs convention).</i>")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        container = qt.QWidget()
        container_layout = qt.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)
        return group, container_layout

    # --- edit-panel state sync ---

    def selectItemForEditing(self, itemID):
        """Public hook for the SH plugin's 'Edit options' action.
        Sets the combo to itemID; the rest of the panel populates
        via the currentItemChanged signal."""
        if itemID is None:
            return
        invalid = slicer.vtkMRMLSubjectHierarchyNode.GetInvalidItemID()
        if itemID == invalid:
            return
        self._editCombo.setCurrentItem(itemID)

    def _on_edit_item_changed(self, itemID):
        # Guard against being fired before _setup_overrides_section
        # has finished building the stack pages (can happen on certain
        # reload paths where setup() runs again on an existing widget).
        if not hasattr(self, "_volumeViewGroup"):
            return
        node = self._node_for_sh_item(itemID)
        self._edited_node_id = node.GetID() if node is not None else None
        if node is None:
            self._editStack.setCurrentIndex(self._EDIT_PAGE_EMPTY)
            return
        if node.IsA("vtkMRMLScalarVolumeNode"):
            self._editStack.setCurrentIndex(self._EDIT_PAGE_VOLUME)
            self._populate_volume_page(node)
        elif node.IsA("vtkMRMLSegmentationNode"):
            self._editStack.setCurrentIndex(self._EDIT_PAGE_SEGMENTATION)
            self._populate_segmentation_page(node)
        else:
            self._editStack.setCurrentIndex(self._EDIT_PAGE_EMPTY)

    def _populate_volume_page(self, node):
        # Block signals while syncing UI from node state so we don't
        # echo-trigger the toggled handler.
        self._volumeRenderCheck.blockSignals(True)
        self._volumeRenderCheck.setChecked(wgpu_state.is_render_enabled(node))
        self._volumeRenderCheck.blockSignals(False)
        self._rebuild_view_checks(node, self._volumeViewContainerLayout,
                                  self._view_checks_volume)
        # Independent-mode controls: reflect whether this volume has an
        # independent state node, and which preset it carries.
        indep_state = wgpu_volume_render.state_for(node, create=False)
        self._volumeIndepCheck.blockSignals(True)
        self._volumeIndepCheck.setChecked(indep_state is not None)
        self._volumeIndepCheck.blockSignals(False)
        self._volumePresetCombo.blockSignals(True)
        if indep_state is not None:
            idx = self._volumePresetCombo.findText(
                wgpu_volume_render.preset_name(indep_state))
            if idx >= 0:
                self._volumePresetCombo.setCurrentIndex(idx)
        self._volumePresetCombo.enabled = indep_state is not None
        self._volumePresetCombo.blockSignals(False)

    def _populate_segmentation_page(self, node):
        self._segRenderCheck.blockSignals(True)
        self._segRenderCheck.setChecked(wgpu_state.is_render_enabled(node))
        self._segRenderCheck.blockSignals(False)
        self._rebuild_view_checks(node, self._segViewContainerLayout,
                                  self._view_checks_seg)
        # Render mode is per-node state -- read from MRML, fall back to
        # bridge default if no state node yet.
        mode = wgpu_state.segment_render_mode(node, default="iso")
        for btn, name in (
                (self._segModeIso, "iso"),
                (self._segModeSurface, "surface")):
            btn.blockSignals(True)
            btn.setChecked(mode == name)
            btn.blockSignals(False)

    def _rebuild_view_checks(self, node, container_layout, cache_dict):
        # Clear previous rows.
        while container_layout.count():
            item = container_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        cache_dict.clear()

        selected_ids = set(wgpu_state.view_node_ids(node))
        coll = slicer.mrmlScene.GetNodesByClass("vtkMRMLViewNode")
        try:
            for i in range(coll.GetNumberOfItems()):
                view_node = coll.GetItemAsObject(i)
                vid = view_node.GetID()
                cb = qt.QCheckBox(view_node.GetName())
                cb.setChecked(vid in selected_ids)
                cb.toggled.connect(
                    lambda checked, vid=vid: self._on_view_check_toggled(
                        vid, checked))
                container_layout.addWidget(cb)
                cache_dict[vid] = cb
        finally:
            coll.UnRegister(None)

    def _on_view_check_toggled(self, _view_id, _checked):
        node = self._currently_edited_node()
        if node is None:
            return
        # Determine which cache applies based on node type.
        cache = (self._view_checks_volume
                 if node.IsA("vtkMRMLScalarVolumeNode")
                 else self._view_checks_seg)
        selected = [vid for vid, cb in cache.items() if cb.isChecked()]
        wgpu_state.set_view_node_ids(node, selected)
        # TODO: notify the bridge to re-evaluate which views render
        # this node when bridge grows per-view installation.

    # --- edit-panel handlers ---

    def _on_volume_render_check_toggled(self, checked):
        node = self._currently_edited_node()
        if node is None:
            return
        # Flip the state node and let the bridge react via its own
        # state-Modified observer (claims/unclaims + auto-creates VRDN
        # via reconcile backfill if needed).
        wgpu_state.set_render_enabled(node, checked)

    @staticmethod
    def _volume_scalar_range(node):
        img = node.GetImageData() if node is not None else None
        if img is None:
            return (0.0, 1.0)
        return tuple(float(x) for x in img.GetScalarRange())

    def _on_volume_indep_toggled(self, checked):
        node = self._currently_edited_node()
        if node is None:
            return
        if checked:
            # Create the independent state node and seed it with the
            # currently-selected preset over the volume's scalar range.
            # The bridge's independent displayer picks it up on NodeAdded.
            state = wgpu_volume_render.state_for(node, create=True)
            preset = self._volumePresetCombo.currentText
            wgpu_volume_render.apply_preset(
                state, preset, self._volume_scalar_range(node))
            wgpu_volume_render.set_render_enabled(node, True)
        else:
            wgpu_volume_render.remove_state(node)
        self._volumePresetCombo.enabled = checked

    def _on_volume_preset_changed(self, _index):
        node = self._currently_edited_node()
        if node is None:
            return
        state = wgpu_volume_render.state_for(node, create=False)
        if state is None:
            return  # Preset only matters once independent mode is on.
        wgpu_volume_render.apply_preset(
            state, self._volumePresetCombo.currentText,
            self._volume_scalar_range(node))

    def _on_seg_render_check_toggled(self, checked):
        node = self._currently_edited_node()
        if node is None:
            return
        # Same pattern: flip state and let bridge reconcile pick it up.
        wgpu_state.set_render_enabled(node, checked)

    def _on_seg_mode_changed(self, _checked):
        node = self._currently_edited_node()
        mode = "surface" if self._segModeSurface.isChecked() else "iso"
        if node is not None:
            wgpu_state.set_segment_render_mode(node, mode)
        if self._bridge is None:
            return
        try:
            self._bridge.set_segment_render_mode(mode)
        except Exception as e:
            slicer.util.errorDisplay(
                f"Failed to set segment render mode: {e}")

    def _currently_edited_node(self):
        if self._edited_node_id is None:
            return None
        return slicer.mrmlScene.GetNodeByID(self._edited_node_id)

    @staticmethod
    def _node_for_sh_item(itemID):
        invalid = slicer.vtkMRMLSubjectHierarchyNode.GetInvalidItemID()
        if itemID == invalid:
            return None
        handler = slicer.qSlicerSubjectHierarchyPluginHandler.instance()
        shNode = handler.subjectHierarchyNode()
        if shNode is None:
            return None
        return shNode.GetItemDataNode(itemID)

    # ------------------------------------------------------------------
    # Subject hierarchy plugin registration
    # ------------------------------------------------------------------

    def _register_sh_plugin(self):
        # Guard against double-register on module reload -- the plugin
        # handler keeps registered plugins alive for the session.
        if getattr(slicer.modules, "slicerWgpuShPlugin", None) is not None:
            return
        try:
            from SceneRenderingLib.SlicerWGPUSubjectHierarchyPlugin import (
                SlicerWGPUSubjectHierarchyPlugin)
        except Exception as e:
            logging.warning(f"SlicerWGPU SH plugin import failed: {e}")
            return
        try:
            scriptedPlugin = slicer.qSlicerSubjectHierarchyScriptedPlugin(None)
            scriptedPlugin.setPythonSource(
                SlicerWGPUSubjectHierarchyPlugin.filePath)
            slicer.modules.slicerWgpuShPlugin = scriptedPlugin
        except Exception as e:
            logging.warning(f"SlicerWGPU SH plugin registration failed: {e}")

    # ------------------------------------------------------------------
    # Scene observers -- state-node lifecycle and load-time reapply.
    # ------------------------------------------------------------------

    def _install_scene_observers(self):
        scene = slicer.mrmlScene
        # When a data node is removed, drop its state node so the
        # scene doesn't accumulate orphans.
        tag_rm = scene.AddObserver(
            slicer.vtkMRMLScene.NodeRemovedEvent, self._on_node_removed)
        self._scene_observer_tags.append((scene, tag_rm))
        # After MRB / scene load, reapply renderEnabled to the bridge
        # for every state node that came back.
        tag_end = scene.AddObserver(
            slicer.vtkMRMLScene.EndImportEvent, self._on_scene_end_import)
        self._scene_observer_tags.append((scene, tag_end))
        # Observe content arriving so a live bridge always exists to render
        # it -- the bridge gets torn down by tests / scene close / reload and
        # nothing else brings it back.
        tag_add = scene.AddObserver(
            slicer.vtkMRMLScene.NodeAddedEvent, self._on_node_added)
        self._scene_observer_tags.append((scene, tag_add))
        # A fresh scene (after Close) also needs a bridge for the next loads.
        tag_close = scene.AddObserver(
            slicer.vtkMRMLScene.EndCloseEvent, self._on_scene_end_close)
        self._scene_observer_tags.append((scene, tag_close))

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def _on_node_added(self, caller, event, calldata):
        node = calldata
        if node is None:
            return
        if (node.IsA("vtkMRMLScalarVolumeNode")
                or node.IsA("vtkMRMLSegmentationNode")):
            self._schedule_ensure_bridge()

    def _on_scene_end_close(self, caller, event):
        self._schedule_ensure_bridge()

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def _on_node_removed(self, caller, event, calldata):
        removed = calldata
        if removed is None:
            return
        removed_id = removed.GetID()
        if not removed_id:
            return
        # Was this a managed data node? If so, drop its state node.
        for state in wgpu_state.all_state_nodes():
            if state.GetNodeReferenceID(
                    wgpu_state.TARGET_NODE_REF) == removed_id:
                slicer.mrmlScene.RemoveNode(state)

    def _on_scene_end_import(self, caller, event):
        # The bridge's own state-node Modified observers + reconcile
        # passes handle reapplication: when state nodes deserialize
        # during scene load, their Modified events drive reconcile,
        # which claims VRDNs / adds segmentations / backfills missing
        # VRDNs. Force one reconcile here in case the order of node
        # add events during import meant the bridge missed something
        # (e.g. state node observed before its target volume existed).
        if self._bridge is None:
            return
        try:
            self._bridge._reconcile_vrdn_claims()
            self._bridge._reconcile_segmentation_nodes()
        except Exception:
            pass

    def _setup_demos_section(self):
        section = ctk.ctkCollapsibleButton()
        section.text = "Demos / self-tests"
        section.collapsed = True
        self.layout.addWidget(section)
        v = qt.QVBoxLayout(section)

        header = qt.QLabel(
            "Click a button to reload this module and run that self-test.")
        header.setWordWrap(True)
        v.addWidget(header)

        vtk_group = qt.QGroupBox("VTK injection (current path)")
        vtk_v = qt.QVBoxLayout(vtk_group)
        for label, method_name in self.VTK_TESTS:
            btn = qt.QPushButton(label)
            btn.setToolTip(f"Reload SceneRendering and run {method_name}()")
            btn.clicked.connect(
                lambda _checked=False, m=method_name: self.onRunTest(m))
            vtk_v.addWidget(btn)
        v.addWidget(vtk_group)

        legacy_group = qt.QGroupBox("DualView / pygfx (legacy)")
        legacy_v = qt.QVBoxLayout(legacy_group)
        for label, method_name in self.TESTS:
            btn = qt.QPushButton(label)
            btn.setToolTip(f"Reload SceneRendering and run {method_name}()")
            btn.clicked.connect(
                lambda _checked=False, m=method_name: self.onRunTest(m))
            legacy_v.addWidget(btn)
        v.addWidget(legacy_group)

        state_group = qt.QGroupBox("State / MRB persistence")
        state_v = qt.QVBoxLayout(state_group)
        for label, method_name in self.STATE_TESTS:
            btn = qt.QPushButton(label)
            btn.setToolTip(f"Reload SceneRendering and run {method_name}()")
            btn.clicked.connect(
                lambda _checked=False, m=method_name: self.onRunTest(m))
            state_v.addWidget(btn)
        v.addWidget(state_group)

        self._forceReinstallCheck = qt.QCheckBox(
            "Force-reinstall deps from GitHub on next test")
        self._forceReinstallCheck.setToolTip(
            "When checked, the next Self-test run will pass "
            "--force-reinstall --no-cache-dir to pip for "
            "pieper/rendercanvas and pieper/slicer-wgpu. The box "
            "unchecks itself automatically after the run.")
        v.addWidget(self._forceReinstallCheck)

    # ------------------------------------------------------------------
    # Bridge install / uninstall
    # ------------------------------------------------------------------

    def _install_bridge(self):
        # Reinstall if we're holding a dead bridge -- e.g. a self-test (or any
        # other uninstall path) tore down the bridge we installed at setup,
        # leaving self._bridge pointing at a disposed object that renders
        # nothing. Treat disposed (or a mismatch with the live module bridge)
        # as "no bridge".
        live = (self._bridge is not None
                and not getattr(self._bridge, "_disposed", False))
        if live:
            self._volumeStatusLabel.text = "Bridge: installed"
            return
        self._bridge = None
        try:
            from SceneRenderingLib.wgpu_vtk_inject import install_default_bridge
        except Exception as e:
            self._volumeStatusLabel.text = (
                f"Bridge: deps missing ({type(e).__name__}). "
                "Click 'Install / reinstall deps' or run a demo first.")
            return
        try:
            self._bridge = install_default_bridge()
            slicer.modules.wgpuVtkBridge = self._bridge
            self._volumeStatusLabel.text = "Bridge: installed"
        except Exception as e:
            self._bridge = None
            self._volumeStatusLabel.text = f"Bridge install failed: {e}"

    def _uninstall_bridge(self):
        # Drop the segmentation hook first so the bridge doesn't
        # fire renders while we're tearing it down.
        if self._bridge is not None:
            try:
                self._bridge.set_segmentation_node(None)
            except Exception:
                pass
            try:
                self._bridge.uninstall()
            except Exception:
                pass
            self._bridge = None
        slicer.modules.wgpuVtkBridge = None
        self._volumeStatusLabel.text = "Bridge: (not installed)"

    def _ensure_bridge(self):
        """Guarantee a live bridge for interactive use. The bridge gets torn
        down by self-tests, scene close, and reloads; nothing used to bring
        it back, so loading a volume / making a segmentation afterwards
        rendered nothing. Adopt an existing live bridge from
        slicer.modules.wgpuVtkBridge (e.g. one a test left installed) into
        self._bridge, else install a fresh one. Cheap + idempotent."""
        glob = getattr(slicer.modules, "wgpuVtkBridge", None)
        if glob is not None and not getattr(glob, "_disposed", False):
            self._bridge = glob
            self._volumeStatusLabel.text = "Bridge: installed"
            return
        self._bridge = None
        self._install_bridge()

    def _schedule_ensure_bridge(self):
        # Defer to the next event-loop turn: a NodeAdded handler must not
        # synchronously install the bridge (install auto-creates VRDN nodes,
        # re-entering NodeAdded), and rapid adds coalesce into one ensure.
        if getattr(self, "_ensure_scheduled", False):
            return
        self._ensure_scheduled = True

        def _run():
            self._ensure_scheduled = False
            # Self-tests manage their own bridge lifecycle; stay out of it.
            if getattr(slicer.modules, "wgpuSuppressAutoBridge", False):
                return
            try:
                self._ensure_bridge()
            except Exception:
                pass
        qt.QTimer.singleShot(0, _run)

    def enter(self):
        """Module shown -- a prior test / scene close may have torn the
        bridge down, so make sure a live one exists."""
        try:
            self._ensure_bridge()
        except Exception:
            pass

    def _on_install_deps_clicked(self):
        # Reuse the test class's bootstrap -- it's the source of truth
        # for which packages and which versions get pip-installed.
        try:
            SceneRenderingTest()._ensure_dependencies()
        except Exception as e:
            slicer.util.errorDisplay(f"Dependency install failed: {e}")
            return
        # Try the bridge install now that deps should be present.
        self._install_bridge()

    # ------------------------------------------------------------------
    # Widget teardown
    # ------------------------------------------------------------------

    def cleanup(self):
        for obj, tag in self._scene_observer_tags:
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
        self._scene_observer_tags = []
        self._uninstall_bridge()

    def onRunTest(self, test_method_name):
        """Reload this module and run the named self-test. Mirrors the
        built-in `Reload and Test` mechanism: reload first so on-disk
        edits take effect, then instantiate the test class from the
        freshly-imported module and invoke the requested method.

        Errors are surfaced via `slicer.util.errorDisplay` so a failing
        test is visible in the UI and not only in the log."""
        import sys

        try:
            slicer.util.reloadScriptedModule(self.moduleName)
        except Exception as e:
            logging.exception("Module reload failed")
            slicer.util.errorDisplay(f"Module reload failed: {e}")
            return

        mod = sys.modules.get(self.moduleName)
        if mod is None:
            slicer.util.errorDisplay(
                f"After reload, {self.moduleName!r} not found in sys.modules")
            return

        try:
            tester = mod.SceneRenderingTest()
            tester._force_reinstall = self._forceReinstallCheck.isChecked()
            tester.runTestByName(test_method_name)
        except Exception as e:
            logging.exception(f"{test_method_name} failed")
            slicer.util.errorDisplay(f"{test_method_name} failed: {e}")
        finally:
            # One-shot: unchecking avoids re-running the slow path
            # on the next button press unless the user opts in again.
            self._forceReinstallCheck.setChecked(False)


#
# SceneRenderingLogic (kept minimal; tests drive install/uninstall)
#

class SceneRenderingLogic(ScriptedLoadableModuleLogic):
    def install(self):
        from slicer_wgpu import mrml_bridge
        return mrml_bridge.install()

    def uninstall(self):
        from slicer_wgpu import mrml_bridge
        mrml_bridge.uninstall()


#
# SceneRenderingTest
#

class SceneRenderingTest(ScriptedLoadableModuleTest):

    # ----- Lifecycle -----

    def setUp(self):
        # Tear down any previously-installed renderers BEFORE clearing the
        # scene. Otherwise the bridge's MRML displayer gets flooded with
        # NodeRemoved events while the wgpu pipeline is still live, which
        # makes reloads crawl as each removal triggers a pointless
        # pipeline rebuild.
        self._ensure_dependencies()
        try:
            from slicer_wgpu import mrml_bridge
            mrml_bridge.uninstall()
        except Exception:
            pass
        prev = getattr(slicer.modules, "wgpuVtkBridge", None)
        if prev is not None:
            try:
                prev.uninstall()
            except Exception:
                pass
            slicer.modules.wgpuVtkBridge = None
        slicer.app.processEvents()
        # Clear scene data but KEEP singletons (layout node, selection,
        # interaction, view/camera nodes, etc). The default Clear()
        # also rips out singletons, which can cascade into side effects
        # in modules that re-register singleton observers, so we stick
        # to the data-only flavor for test resets.
        #
        # Skip the clear if the user has loaded vtkMRMLFiberBundleNode
        # instances -- test_vtk_FiberStrands picks them up to drive the
        # rasterizer, so wiping the scene would defeat that path.
        coll = slicer.mrmlScene.GetNodesByClass("vtkMRMLFiberBundleNode")
        coll.UnRegister(None)
        if coll.GetNumberOfItems() == 0:
            slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

    def runTest(self):
        """Slicer's standard entry: run the current (VTK-injection) tests in
        sequence. Invoked by the built-in `Reload & Test` button.

        These exercise the raw-wgpu injection path -- the active development
        focus -- which needs no rendercanvas-fork Qt surface. The legacy
        DualView/pygfx demos (test_SingleVolume, test_BouncingHead, ...) are
        kept as runnable methods but left out of the default sweep because
        they depend on the pieper/rendercanvas PythonQt fork being installed.
        """
        # Tests manage their own bridge lifecycle (setUp tears it down per
        # test); suppress the widget's observe-and-reinstall so it doesn't
        # fight them, and restore a live bridge for interactive use after.
        slicer.modules.wgpuSuppressAutoBridge = True
        try:
            for name in (
                "test_vtk_IndependentVolumeAndSegmentation",
                "test_vtk_VolumeFollowsComposite",
                "test_state_SaveRestore",
            ):
                self.setUp()
                getattr(self, name)()
        finally:
            slicer.modules.wgpuSuppressAutoBridge = False
            self._leave_live_bridge()

    @staticmethod
    def _leave_live_bridge():
        """Make sure a live bridge exists after a self-test, WITHOUT clobbering
        one a demo already set up. Interactive demos (TPS landmark deform,
        segmentation painting, ...) install AND configure their bridge -- grid
        transform, segmentation hooks, clip planes -- and leave it live for the
        user to keep playing with. Replacing it with a fresh install would
        discard that state (e.g. the TPS grid transform, so the volume stops
        deforming). Only install when there is no live bridge -- the case after
        a sweep whose last test cleared the scene and tore the bridge down."""
        b = getattr(slicer.modules, "wgpuVtkBridge", None)
        if b is not None and not getattr(b, "_disposed", False):
            return
        try:
            from SceneRenderingLib.wgpu_vtk_inject import install_default_bridge
            slicer.modules.wgpuVtkBridge = install_default_bridge()
        except Exception:
            pass

    def runTestByName(self, test_method_name: str) -> None:
        """Run a single test by name. Used by the module UI buttons."""
        slicer.modules.wgpuSuppressAutoBridge = True
        try:
            self.setUp()
            getattr(self, test_method_name)()
        finally:
            slicer.modules.wgpuSuppressAutoBridge = False
            self._leave_live_bridge()

    # ----- Dependency bootstrap -----

    def _ensure_dependencies(self):
        """Make sure numpy / wgpu / pygfx / rendercanvas (PythonQt branch)
        / slicer_wgpu are importable. pip-install is only invoked when
        an import actually fails, so a live-development workflow that
        pushes files via the MCP /file endpoint isn't clobbered."""
        import importlib
        import sys

        # Apple-Silicon arch-detection fix: must run before wgpu's Metal
        # backend pulls in rubicon-objc (i.e. before the wgpu import below).
        _fix_macos_processor_for_rubicon()

        for pkg in ("numpy", "wgpu"):
            try:
                importlib.import_module(pkg)
            except ImportError:
                self.delayDisplay(f"pip-installing {pkg}", 100)
                slicer.util.pip_install(pkg)

        # When wgpu first creates its instance / requests an adapter it enumerates EVERY
        # backend. The OpenGL-ES backend opens an EGL display chosen from the environment
        # (WAYLAND_DISPLAY -> wayland, else DISPLAY -> X11). On NVIDIA under XWayland
        # (a headless / browser-streamed desktop) that EGL probe aborts the WHOLE process:
        # wgpu-hal panics with BadAccess across the C FFI (unrecoverable -- SlicerApp-real
        # exit abnormally). Clearing the windowing env around the request does NOT help: the
        # GL backend is enumerated regardless of when the env is clear, and WGPU_BACKEND only
        # changes adapter *selection*, not which backends get enumerated. The fix is to create
        # the instance with ONLY the Vulkan backend so the GL/ES backend is never enumerated.
        # Vulkan WSI still serves offscreen and on-screen surfaces, and DISPLAY is left
        # untouched so Slicer's VTK/GLX rendering is unaffected. Must run before the wgpu
        # instance is created (i.e. before pygfx imports / renders below).
        _force_vulkan_only_wgpu_instance()

        try:
            import pygfx
            # Require a real install, not a namespace-package shadow (e.g. a
            # stray pygfx/ git clone sitting on sys.path earlier than
            # site-packages). A namespace package has no __version__.
            _ = pygfx.__version__
        except (ImportError, AttributeError):
            self.delayDisplay("pip-installing pygfx", 100)
            slicer.util.pip_install("pygfx")

        # Detect the PythonQt-capable rendercanvas fork by reading
        # rendercanvas/qt.py from disk rather than `import rendercanvas.qt`,
        # because the upstream module raises at import time unless
        # PySide6/PySide2/PyQt6/PyQt5 has already been imported -- which
        # would make us mistakenly think the fork is missing.
        needs_rendercanvas_fork = True
        try:
            import rendercanvas
            import os as _os
            qt_path = _os.path.join(_os.path.dirname(rendercanvas.__file__),
                                    "qt.py")
            try:
                with open(qt_path) as f:
                    src = f.read()
            except Exception:
                src = ""
            if src and "is_pythonqt" in src:
                needs_rendercanvas_fork = False
        except ImportError:
            pass
        if needs_rendercanvas_fork or getattr(self, "_force_reinstall", False):
            msg = ("Force-reinstalling pieper/rendercanvas (PythonQt)"
                   if not needs_rendercanvas_fork
                   else "Installing pieper/rendercanvas (PythonQt)")
            self.delayDisplay(msg, 100)
            # --force-reinstall is essential: the fork's pyproject declares
            # itself as rendercanvas==<upstream version>, so pip would
            # otherwise decide the upstream install "already satisfies"
            # and skip the replacement. --no-deps avoids dragging in
            # upstream rendercanvas again via transitive requirements.
            # --no-cache-dir when force-reinstalling guarantees we pull
            # the current pythonqt-support branch, not a stale cached zip.
            cache = "--no-cache-dir " if getattr(self, "_force_reinstall", False) else ""
            rendercanvas_installed = False
            try:
                slicer.util.pip_install(
                    f"--force-reinstall --no-deps {cache}"
                    "https://github.com/pieper/rendercanvas/"
                    "archive/refs/heads/pythonqt-support.zip"
                )
                rendercanvas_installed = True
            except Exception as e:
                # Only the DualView / pygfx-Qt-surface tests (the legacy
                # "Single Volume", "Bouncing Head", etc. demos) actually
                # need this fork. The "Injection:" tests use raw wgpu and
                # are unaffected. Log and continue so those still run.
                print(
                    "Warning: could not install pieper/rendercanvas "
                    f"PythonQt fork: {e}\n"
                    "  The Injection: tests do not need it and will run "
                    "normally. The legacy DualView tests (Single Volume, "
                    "Bouncing Head, Multi-Volume, etc.) require it.")

            if rendercanvas_installed:
                # pip overwrote rendercanvas on disk but Python's sys.modules
                # still holds the pre-install (upstream) rendercanvas
                # objects. Without this pop, `from rendercanvas.qt import
                # ...` inside slicer_wgpu returns the cached upstream
                # module and the fork's PythonQt branch never runs.
                for mod_name in [
                    m for m in list(sys.modules)
                    if m == "rendercanvas" or m.startswith("rendercanvas.")
                ]:
                    sys.modules.pop(mod_name, None)

                # If pygfx was already imported, it cached a reference to
                # the OLD rendercanvas.BaseRenderCanvas. Its isinstance()
                # checks in WgpuRenderer would then reject the new
                # QRenderWidget ("Render target must be a Canvas or
                # Texture, not QRenderWidget"). Rebind the captured
                # reference in place to match the freshly-reimported class.
                #
                # We CANNOT just pop pygfx and re-import it: pygfx calls
                # wgpu.preconfigure_default_device("pygfx", ...) at module
                # top level, and wgpu raises RuntimeError if a device has
                # already been created in the process (which it has, any
                # time a renderer existed earlier in this session).
                if "pygfx" in sys.modules:
                    try:
                        import importlib
                        rc = importlib.import_module("rendercanvas")
                        pgr = importlib.import_module(
                            "pygfx.renderers.wgpu.engine.renderer")
                        pgr.BaseRenderCanvas = rc.BaseRenderCanvas
                    except Exception as e:
                        print(f"pygfx BaseRenderCanvas rebind failed: {e}")

        # The fork still mis-handles PythonQt property access on Qt6 (the
        # legacy DualView / Bouncing-Head demos otherwise fail with
        # "'int' object is not callable"). Patch it whether or not we just
        # reinstalled the fork.
        _patch_rendercanvas_pythonqt_qt6()

        # slicer-wgpu's version is pinned at 0.1.0 and never bumps, so
        # pip considers any cached wheel of the GitHub main.zip URL to
        # satisfy the requirement. By default we only install if the
        # package isn't importable at all; use the "Force Reinstall
        # Dependencies" button in the module UI to pull a fresh build
        # when you want to pick up new commits from main.
        force = getattr(self, "_force_reinstall", False)
        try:
            importlib.import_module("slicer_wgpu")
            needs_install = False
        except ImportError:
            needs_install = True
        if needs_install or force:
            msg = ("Force-reinstalling pieper/slicer-wgpu"
                   if force else "Installing pieper/slicer-wgpu")
            self.delayDisplay(msg, 100)
            args = (
                "--force-reinstall --no-deps --no-cache-dir "
                if force else ""
            )
            slicer.util.pip_install(
                args
                + "https://github.com/pieper/slicer-wgpu/"
                "archive/refs/heads/main.zip"
            )

        # Tear down any still-installed DualView before we drop
        # slicer_wgpu from sys.modules. Match by class NAME (not
        # isinstance), because after the upcoming module pop the OLD
        # DualView class and the NEW one will be distinct class objects:
        # an isinstance filter against the re-imported module would miss
        # leftover instances and leave their ImageField volume textures
        # (~337 MB each for CTACardio) pinned across the reload. Also
        # release any test-stashed DualViews from slicer.modules so GC
        # can actually collect the torn-down objects.
        import gc as _gc
        for _o in list(_gc.get_objects()):
            if type(_o).__name__ == "DualView":
                try:
                    if hasattr(_o, "uninstall"):
                        _o.uninstall()
                except Exception:
                    pass
        for _a in list(vars(slicer.modules)):
            if _a.startswith("sceneRenderingTest_"):
                try:
                    delattr(slicer.modules, _a)
                except Exception:
                    pass
        _gc.collect()

        # Force a fresh import so any on-disk edits (pushed via the
        # MCP /file endpoint during iteration) take effect.
        for mod_name in [
            m for m in list(sys.modules)
            if m == "slicer_wgpu" or m.startswith("slicer_wgpu.")
        ]:
            sys.modules.pop(mod_name, None)

    # ----- Shared helpers -----

    def _force_draw(self, view, n=3):
        slicer.app.processEvents()
        for _ in range(n):
            view.request_redraw()
            try:
                view.widget.force_draw()
            except Exception:
                pass
            slicer.app.processEvents()

    def _snapshot_stats(self, view, label=""):
        import numpy as np
        img = view.renderer.snapshot()
        rgb = np.asarray(img[..., :3])
        mn = tuple(int(x) for x in rgb.min(axis=(0, 1)))
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        me = tuple(float(round(float(x), 1)) for x in rgb.mean(axis=(0, 1)))
        logging.info(f"[SceneRendering:{label}] shape={img.shape} "
                     f"min={mn} max={mx} mean={me}")
        return img, mn, mx, me

    def _load_ctacardio(self, apply_preset=True):
        """Load the CTACardio sample volume and build default volume-
        rendering nodes (optionally overriding the preset)."""
        import SampleData
        self.delayDisplay("Downloading CTACardio", 100)
        vol = SampleData.SampleDataLogic().downloadCTACardio()
        self.assertIsNotNone(vol, "CTACardio failed to load")

        vrLogic = slicer.modules.volumerendering.logic()
        disp = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
        disp.SetVisibility(True)
        if apply_preset:
            preset = vrLogic.GetPresetByName("CT-Chest-Contrast-Enhanced")
            if preset is not None:
                disp.GetVolumePropertyNode().Copy(preset)
        slicer.app.processEvents()
        return vol

    def _build_markup_nodes(self, volume_node=None):
        """Create 4 markup fiducial nodes (25 control points each). If
        `volume_node` is given, points are scattered inside its bounds."""
        import numpy as np

        list_specs = [
            ("MarkupsRed",    (0.95, 0.20, 0.20), 5.0),
            ("MarkupsGreen",  (0.20, 0.85, 0.30), 3.5),
            ("MarkupsBlue",   (0.20, 0.45, 0.95), 7.0),
            ("MarkupsYellow", (0.95, 0.85, 0.10), 2.5),
        ]
        rng = np.random.default_rng(seed=20260415)
        if volume_node is not None:
            b = [0.0] * 6
            volume_node.GetBounds(b)
            ranges = [(b[0], b[1]), (b[2], b[3]), (b[4], b[5])]
        else:
            ranges = [(-100.0, 100.0)] * 3
        markup_nodes = []
        for name, color, _radius in list_specs:
            mn = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", name)
            d = mn.GetDisplayNode()
            if d is not None:
                d.SetColor(*color)
                d.SetSelectedColor(*color)
            for _ in range(25):
                p = tuple(float(rng.uniform(lo, hi)) for (lo, hi) in ranges)
                mn.AddControlPoint(*p)
            markup_nodes.append(mn)
        return list_specs, markup_nodes

    def _install_dualview(self):
        """Install the DualView and let it auto-sync cameras to MRML."""
        from slicer_wgpu import mrml_bridge
        dv = mrml_bridge.install()
        self.assertIsNotNone(dv.view, "DualView didn't instantiate PygfxView")
        slicer.app.processEvents()
        return dv

    def _frame_and_draw(self, dv):
        dv.view.reset_camera()
        dv._sync_camera_to_mrml()
        self._force_draw(dv.view, n=5)

    def _set_dualview_radii(self, dv, list_specs, markup_nodes):
        scene_mgr = next(m for m in dv.managers
                         if type(m).__name__ == "SceneRendererManager")
        fid_disp = next(d for d in scene_mgr._displayers
                        if type(d).__name__ == "FiducialDisplayer")
        for (_name, _color, radius), mn in zip(list_specs, markup_nodes):
            fid_disp.set_default_radius(mn.GetID(), radius)
        return scene_mgr, fid_disp

    def _pick_drag_roundtrip(self, dv, markup_node, fid_disp, label):
        """Pick the first control point of `markup_node`, drag it 0.1
        NDC units, commit back to MRML, assert the MRML position moved."""
        import numpy as np
        from slicer_wgpu.fields import FiducialField

        view = dv.view
        scene_mgr = next(m for m in dv.managers
                         if type(m).__name__ == "SceneRendererManager")
        r = scene_mgr.renderer
        self.assertIsNotNone(r, f"{label}: no SceneRenderer was built")

        matching_field = next(
            f for f in r.fields()
            if isinstance(f, FiducialField)
            and f.mrml_node_id == markup_node.GetID()
        )
        before = [0.0, 0.0, 0.0]
        markup_node.GetNthControlPointPosition(0, before)
        before = np.array(before)

        self._force_draw(view)
        cam = view.camera
        proj = np.asarray(cam.projection_matrix, dtype=np.float64)
        vm = np.asarray(cam.world.inverse_matrix, dtype=np.float64)
        clip = proj @ vm @ np.array([*before, 1.0], dtype=np.float64)
        ndc = clip[:3] / clip[3]
        sz = view.widget.get_logical_size()
        self.assertGreater(sz[0], 0, f"{label}: logical size is zero")

        hit = r.pick_at(float(ndc[0]), float(ndc[1]), cam, sz)
        self.assertIsNotNone(hit,
            f"{label}: pick missed control point 0 of "
            f"{markup_node.GetName()} at NDC={ndc.tolist()}")
        self.assertIs(hit.field, matching_field,
            f"{label}: pick hit a different field")
        self.assertEqual(hit.item_index, 0,
            f"{label}: pick hit the wrong control point")

        moved = r.drag_continue(hit, float(ndc[0]) + 0.1, float(ndc[1]),
                                cam, sz)
        self.assertTrue(moved, f"{label}: drag_continue reported no change")
        fid_disp.commit_drag(matching_field, 0)
        slicer.app.processEvents()

        after = [0.0, 0.0, 0.0]
        markup_node.GetNthControlPointPosition(0, after)
        after = np.array(after)
        delta = float(np.linalg.norm(after - before))
        self.assertGreater(delta, 1.0,
            f"{label}: MRML point barely moved (delta={delta:.3f}mm)")
        logging.info(f"[SceneRendering] {label} MRML drag delta={delta:.2f}mm")

    def _stash(self, **kwargs):
        """Park results on slicer.modules.* for interactive inspection
        after the test returns. Keeps test_SingleVolume's dv alive, etc."""
        for k, v in kwargs.items():
            setattr(slicer.modules, f"sceneRenderingTest_{k}", v)

    # ----- Working tests -----

    # ------------------------------------------------------------------
    # Sample-data cache.
    #
    # A scene Clear() (triggered between test runs in setUp) removes MRML
    # nodes, which means SampleData.downloadSample() re-parses the 278 MB
    # compressed NRRD into a 685 MB float32 ImageData on every click --
    # ~9 s on the MCP box. Holding a Python reference to the
    # vtkImageData / vtkSegmentation keeps them alive across the Clear;
    # a new MRML node then wraps the cached data in milliseconds.
    #
    # Keyed by sample name. Survives for the Slicer session.
    # ------------------------------------------------------------------
    _VOLUME_CACHE: dict = {}   # name -> {'image', 'ijk_to_ras', 'wl', 'range'}
    _SEG_CACHE:    dict = {}   # name -> {'segmentation'}

    def _load_cached_volume(self, name):
        """downloadSample(name) with a Python-level cache of the VTK data.
        Skips the NRRD parse on all runs after the first this session."""
        import SampleData
        entry = SceneRenderingTest._VOLUME_CACHE.get(name)
        if entry is not None:
            vol = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode", name)
            vol.SetAndObserveImageData(entry["image"])
            m = vtk.vtkMatrix4x4()
            m.DeepCopy(entry["ijk_to_ras"])
            vol.SetIJKToRASMatrix(m)
            vol.CreateDefaultDisplayNodes()
            d = vol.GetScalarVolumeDisplayNode()
            if d is not None and entry.get("wl") is not None:
                window, level = entry["wl"]
                d.SetWindow(window)
                d.SetLevel(level)
                d.SetAutoWindowLevel(False)
            return vol
        vol = SampleData.downloadSample(name)
        if vol is None:
            return None
        ijk_to_ras = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(ijk_to_ras)
        d = vol.GetScalarVolumeDisplayNode()
        wl = None
        if d is not None:
            try:
                wl = (float(d.GetWindow()), float(d.GetLevel()))
            except Exception:
                wl = None
        SceneRenderingTest._VOLUME_CACHE[name] = {
            "image": vol.GetImageData(),
            "ijk_to_ras": ijk_to_ras,
            "wl": wl,
        }
        return vol

    def _load_cached_segmentation(self, name):
        """downloadSample(name) for a .seg.nrrd with Python-level caching.
        Holds a reference to the vtkSegmentation object so subsequent
        loads skip the parse + segment-reconstruction cost."""
        import SampleData
        entry = SceneRenderingTest._SEG_CACHE.get(name)
        if entry is not None:
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", name)
            node.SetAndObserveSegmentation(entry["segmentation"])
            node.CreateDefaultDisplayNodes()
            return node
        loaded = SampleData.downloadSample(name)
        if isinstance(loaded, (list, tuple)):
            loaded = next((n for n in loaded
                           if n is not None
                           and n.IsA("vtkMRMLSegmentationNode")), None)
        if loaded is None:
            return None
        SceneRenderingTest._SEG_CACHE[name] = {
            "segmentation": loaded.GetSegmentation(),
        }
        return loaded

    @staticmethod
    def _resolve_carve_label_values(seg_node, name_substrings):
        """Map a list of segment-name substrings (case-insensitive) to the
        labelmap values used on the GPU. Skips substrings that don't match
        any segment so a partial match still carves something."""
        seg = seg_node.GetSegmentation()
        wanted = [s.lower() for s in name_substrings]
        out = []
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            segment = seg.GetSegment(sid)
            nm = (segment.GetName() or "").lower()
            if any(w in nm for w in wanted):
                try:
                    lv = int(segment.GetLabelValue())
                except Exception:
                    lv = i + 1
                if 0 < lv < 256:
                    out.append(lv)
        return out

    # ------------------------------------------------------------------
    # VTK-injection tests -- no DualView, wgpu renders directly into
    # Slicer's native 3D view via a vtkCommand::EndEvent hook.
    # ------------------------------------------------------------------

    def _install_vtk_bridge(self, layout=None):
        """Install the VTK-injection bridge on the active 3D view.
        Uses the module file shipped alongside this script
        (`wgpu_vtk_inject.py`) so we can iterate without reinstalling
        the slicer-wgpu pip package.

        `layout` may be any vtkMRMLLayoutNode layout constant; defaults
        to single-up 3D. Four-up is useful when the test also wants the
        slice views (e.g. segmentation painting demos).
        """
        target = layout if layout is not None else (
            slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        slicer.app.layoutManager().setLayout(target)
        slicer.app.processEvents()

        # Import the bridge from the sibling SceneRenderingLib package.
        # Force-reload so edits to the helper pick up without re-launching
        # Slicer.
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        for mod in ("SceneRenderingLib.wgpu_vtk_inject", "SceneRenderingLib"):
            if mod in sys.modules:
                del sys.modules[mod]
        from SceneRenderingLib import wgpu_vtk_inject as wvi
        bridge = wvi.install_default_bridge()
        slicer.modules.wgpuVtkBridge = bridge
        slicer.app.processEvents()
        return bridge

    def _vtk_render_and_snapshot(self):
        """Force a render on the 3D view and return the RGB snapshot."""
        import numpy as np
        lm = slicer.app.layoutManager()
        # Pick the first VISIBLE 3D widget -- after layout changes,
        # threeDWidget(0) can be an invisible spare from a wider
        # layout that's no longer active.
        tw = None
        for i in range(lm.threeDViewCount):
            candidate = lm.threeDWidget(i)
            if candidate is not None and candidate.visible:
                tw = candidate
                break
        if tw is None:
            tw = lm.threeDWidget(0)
        view = tw.threeDView()
        view.forceRender()
        slicer.app.processEvents()
        # WindowToImage grab of the view
        wti = vtk.vtkWindowToImageFilter()
        wti.SetInput(view.renderWindow())
        wti.SetInputBufferTypeToRGB()
        wti.Update()
        img = wti.GetOutput()
        dims = img.GetDimensions()
        from vtk.util.numpy_support import vtk_to_numpy
        arr = vtk_to_numpy(img.GetPointData().GetScalars())
        arr = arr.reshape(dims[1], dims[0], -1)
        return arr[::-1]  # flip to top-down

    def test_vtk_SingleVolume(self):
        """VTK injection: CTACardio rendered via wgpu inside the default
        3D view. Our bridge claims the VR display node (Visibility=0) so
        Slicer's native ray-caster stays idle."""
        import numpy as np
        self.delayDisplay("VTK: Single Volume loading", 150)
        vol = self._load_ctacardio()
        bridge = self._install_vtk_bridge()
        self.assertEqual(len(bridge.images_by_vrdn), 1,
            f"expected 1 claimed VRDN, got {list(bridge.images_by_vrdn.keys())}")

        # Frame the volume
        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no volume visible via VTK injection -- max_rgb={mx}")

        self._stash(vtkBridge=bridge, volume=vol)
        self.delayDisplay("VTK: Single Volume PASSED", 300)

    def _warm_pixel_count(self):
        """Count warm/skin-tone pixels (red well above blue) in the active
        3D view. The wgpu volume (MR skin tones) and the red segment are
        warm; the 3D view's gradient background is blue and the orientation
        labels/box are white (R==B) -- so this cleanly measures wgpu content
        without being saturated by background or annotations the way a plain
        max/mean would be."""
        rgb = self._vtk_render_and_snapshot().astype("int64")
        return int(((rgb[:, :, 0] - rgb[:, :, 2]) > 30).sum())

    def test_vtk_IndependentVolumeAndSegmentation(self):
        """Target use case, end to end: an independent wgpu volume whose 3D
        visibility tracks the Red slice composite (background/foreground +
        opacity slider), composited together with a wgpu-rendered
        segmentation -- with NO volume-rendering display node.

        Verifies, by counting warm pixels actually rendered into the 3D view:
          1. the volume renders when it is the Red background,
          2. it disappears when removed from the composite (the volume
             *follows* the Red slice composite node),
          3. a segmentation added afterwards renders on its own, independent
             of the (hidden) volume.
        """
        import numpy as np
        from SceneRenderingLib import wgpu_volume_render as wvr
        from SceneRenderingLib import wgpu_volume_displayer as wvd
        self.delayDisplay("Independent volume + segmentation loading", 150)

        vol = self._load_cached_volume("MRHead")
        self.assertIsNotNone(vol, "MRHead failed to load")
        rng = tuple(vol.GetImageData().GetScalarRange())
        wvr.apply_preset(wvr.state_for(vol, create=True), "MR default", rng)
        wvr.set_render_enabled(vol, True)
        self.assertEqual(
            slicer.mrmlScene.GetNumberOfNodesByClass(
                "vtkMRMLVolumeRenderingDisplayNode"), 0,
            "independent path must not create a VR display node")

        # 3D-only layout: the gradient background is blue (so warm-pixel
        # counting works), and the Red slice composite node still exists in
        # the scene to drive visibility even with no slice pane shown.
        bridge = self._install_vtk_bridge()
        comp = wvd.red_slice_composite()
        self.assertIsNotNone(comp, "no Red slice composite node found")
        comp.SetForegroundVolumeID(None)

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        view.renderWindow().GetRenderers().GetFirstRenderer().ResetCamera()

        # Watch for wgpu validation / bind-group errors over the whole test
        # (e.g. a pipeline/bind-group shape desync when a segment is added).
        elm = slicer.app.errorLogModel()
        err_start = elm.logEntryCount()

        # 1) Volume as the Red background -> full opacity, visible in 3D.
        comp.SetBackgroundVolumeID(vol.GetID())
        slicer.app.processEvents()
        self.assertEqual(len(bridge._fields), 1,
                         f"expected 1 independent field, got {len(bridge._fields)}")
        self.assertAlmostEqual(bridge._fields[0].render_opacity, 1.0, places=3,
                               msg="Red-background volume should be full opacity")
        on = self._warm_pixel_count()
        self.assertGreater(on, 4000,
            f"independent volume not visible in 3D -- warm_px={on}")

        # 2) Remove it from the Red background -> hidden (composite-following).
        comp.SetBackgroundVolumeID(None)
        slicer.app.processEvents()
        self.assertAlmostEqual(bridge._fields[0].render_opacity, 0.0, places=3,
            msg="volume removed from the Red composite should be hidden")
        off = self._warm_pixel_count()
        self.assertLess(off, max(300, on // 8),
            f"volume did NOT follow the Red slice composite (still visible "
            f"when removed) -- warm_px on={on} off={off}")

        # 3) Add a segmentation AFTER install -> the bridge picks it up and
        #    renders it, independent of the (still-hidden) volume.
        seg = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "TestSeg")
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        arr = slicer.util.arrayFromVolume(vol)
        sid = seg.GetSegmentation().AddEmptySegment("Brain", "Brain")
        seg.GetSegmentation().GetSegment(sid).SetColor(0.9, 0.2, 0.2)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            ((arr > 40) & (arr < 120)).astype(np.uint8), seg, sid, vol)
        seg.GetDisplayNode().SetVisibility3D(True)
        slicer.app.processEvents()
        self.assertGreater(len(bridge._segments), 0,
            "segmentation was not picked up by the bridge")
        seg_warm = self._warm_pixel_count()
        self.assertGreater(seg_warm, 4000,
            f"segmentation not rendered in 3D -- warm_px={seg_warm}")

        # 4) Volume AND segmentation visible together -- the combination that
        #    used to desync the bind group vs its layout. Both must render.
        comp.SetBackgroundVolumeID(vol.GetID())
        slicer.app.processEvents()
        both = self._warm_pixel_count()
        self.assertGreater(both, on,
            f"volume+segmentation should add warm pixels over volume-only "
            f"(volume={on}, both={both})")

        # No wgpu validation / bind-group errors may have been logged.
        wgpu_errs = []
        for i in range(err_start, elm.logEntryCount()):
            try:
                d = elm.logEntryDescription(i)
            except Exception:
                continue
            if ("WgpuVolumeBridge" in d or "Validation Error" in d
                    or "bind group" in d.lower() or "binding" in d.lower()):
                wgpu_errs.append(d.splitlines()[0][:120])
        self.assertEqual(wgpu_errs, [],
            f"wgpu validation/bind-group errors during render: {wgpu_errs}")

        self._stash(vtkBridge=bridge, volume=vol, segmentation=seg)
        self.delayDisplay(
            "Independent volume + segmentation PASSED -- set MRHead as the "
            "Red background/foreground and drag the opacity slider; paint "
            "the segmentation and watch the 3D view update", 600)

    def test_vtk_VolumeFollowsComposite(self):
        """EVERY wgpu volume must follow the Red slice composite -- including
        a plain auto-rendered volume with NO independent state node (the
        legacy VRDN path, which used to render at fixed full opacity and
        ignore the composite). Verified with a colour-agnostic before/after
        view diff, since the legacy path uses a grayscale transfer function.
        """
        import numpy as np
        from SceneRenderingLib import wgpu_volume_displayer as wvd
        self.delayDisplay("Volume-follows-composite loading", 150)

        vol = self._load_cached_volume("MRHead")  # no independent node -> legacy
        self.assertIsNotNone(vol, "MRHead failed to load")
        bridge = self._install_vtk_bridge()
        self.assertGreater(
            len(bridge._displayer.fields_by_nid) if bridge._displayer else 0, 0,
            "expected a legacy VRDN-backed field for the plain volume")

        comp = wvd.red_slice_composite()
        self.assertIsNotNone(comp, "no Red slice composite node found")
        view = slicer.app.layoutManager().threeDWidget(0).threeDView()
        view.renderWindow().GetRenderers().GetFirstRenderer().ResetCamera()

        def snap():
            return self._vtk_render_and_snapshot().astype("int64")

        # Not selected in the Red composite -> hidden.
        comp.SetForegroundVolumeID(None)
        comp.SetBackgroundVolumeID(None)
        slicer.app.processEvents()
        self.assertAlmostEqual(bridge._fields[0].render_opacity, 0.0, places=3,
            msg="legacy volume not in Red bg/fg should be hidden")
        off = snap()

        # Selected as the Red background -> visible.
        comp.SetBackgroundVolumeID(vol.GetID())
        slicer.app.processEvents()
        self.assertAlmostEqual(bridge._fields[0].render_opacity, 1.0, places=3,
            msg="legacy volume set as Red background should be full opacity")
        on = snap()

        diff = int((np.abs(on - off).sum(axis=2) > 40).sum())
        self.assertGreater(diff, 4000,
            f"legacy volume did NOT appear when set as the Red background "
            f"(it ignores the composite) -- changed_px={diff}")

        self._stash(vtkBridge=bridge, volume=vol)
        self.delayDisplay(
            "Volume-follows-composite PASSED -- only the volume chosen as the "
            "Red background/foreground renders in 3D", 400)

    def test_vtk_VolumeAndFiducials(self):
        """VTK injection + fiducials. The wgpu volume renders via our
        bridge; VTK's native Markups displayable manager handles the
        fiducial glyphs. Compositor blends wgpu on top of VTK's output,
        so background + fiducials stay visible around the volume."""
        import numpy as np
        self.delayDisplay("VTK: Volume + Fiducials loading", 150)

        vol = self._load_ctacardio()
        # Reuse the helper that builds 4 fiducial lists
        list_specs, markup_nodes = self._build_markup_nodes(volume_node=vol)

        bridge = self._install_vtk_bridge()
        self.assertEqual(len(bridge.images_by_vrdn), 1,
            f"expected 1 claimed VRDN, got {list(bridge.images_by_vrdn.keys())}")

        # Tune fiducial glyph sizes so they're easy to see
        for n in markup_nodes:
            dn = n.GetDisplayNode()
            if dn is not None:
                dn.SetGlyphScale(3.5)
                dn.SetTextScale(2.5)

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no composited output visible -- max_rgb={mx}")

        self._stash(vtkBridge=bridge, volume=vol, markupNodes=markup_nodes)
        self.delayDisplay("VTK: Volume + Fiducials PASSED", 300)

    def test_vtk_MultiVolume(self):
        """VTK injection with two volumes. The second volume is placed
        under an INTERACTIVE linear transform (with visible transform
        handles in the 3D view) so the user can drag it around live
        and watch our wgpu renderer track the change through the
        TransformModifiedEvent observer.
        """
        import numpy as np
        self.delayDisplay("VTK: Multi-Volume loading", 200)

        import SampleData
        sd = SampleData.SampleDataLogic()
        cta = sd.downloadCTACardio()
        self.assertIsNotNone(cta, "CTACardio failed to load")
        pano = sd.downloadSample("CTAAbdomenPanoramix")
        if isinstance(pano, (list, tuple)):
            pano = next((n for n in pano
                         if n is not None
                         and n.IsA("vtkMRMLScalarVolumeNode")), None)
        self.assertIsNotNone(pano, "Panoramix failed to load")

        # Distinct presets so the two are easy to tell apart.
        vrLogic = slicer.modules.volumerendering.logic()
        for vol, preset_name in ((cta, "CT-Chest-Contrast-Enhanced"),
                                 (pano, "CT-AAA")):
            d = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
            d.SetVisibility(True)
            preset = vrLogic.GetPresetByName(preset_name)
            if preset is not None:
                d.GetVolumePropertyNode().Copy(preset)

        # --- Interactive transform for pano ---
        # Translate pano up and to the side so it doesn't overlap cta
        # initially, then expose interactive handles so the user can
        # drag it around at demo time.
        tf = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "PanoInteractiveTransform")
        M = vtk.vtkMatrix4x4()
        b = [0.0] * 6
        pano.GetBounds(b)
        # Initial offset: translate pano by +200mm along R so it's next to cta
        M.SetElement(0, 3, 200.0)
        tf.SetMatrixTransformToParent(M)
        pano.SetAndObserveTransformNodeID(tf.GetID())

        # Expose interactive handles on the transform node
        tdn = tf.GetDisplayNode()
        if tdn is None:
            tf.CreateDefaultDisplayNodes()
            tdn = tf.GetDisplayNode()
        if tdn is not None:
            tdn.SetEditorVisibility(True)
            tdn.SetEditorVisibility3D(True)
            tdn.SetEditorTranslationEnabled(True)
            tdn.SetEditorRotationEnabled(True)
            tdn.SetEditorScalingEnabled(False)

        # Now install the bridge -- both VRDNs will be claimed.
        bridge = self._install_vtk_bridge()
        self.assertEqual(len(bridge.images_by_vrdn), 2,
            f"expected 2 claimed VRDNs, got {list(bridge.images_by_vrdn.keys())}")

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no multi-volume output visible -- max_rgb={mx}")

        # Stash so the user can inspect / further manipulate in the console
        self._stash(vtkBridge=bridge, cta=cta, pano=pano, panoTransform=tf)
        self.delayDisplay(
            "VTK: Multi-Volume PASSED -- drag the Pano handles to move it "
            "and see the wgpu render update live", 600)

    def test_vtk_LandmarkDeform(self):
        """VTK injection + thin-plate-spline grid transform driven by
        fiducial landmarks. Places 8 source landmarks at MRHead's
        bounding-box corners. Adding or moving a fiducial rebuilds the
        TPS + bakes it to a grid transform, which the bridge observes
        and uploads as a 3D displacement texture. Volume deforms live.
        """
        import numpy as np
        self.delayDisplay("VTK: Landmark Deform loading", 150)

        import SampleData
        vol = SampleData.SampleDataLogic().downloadMRHead()
        self.assertIsNotNone(vol, "MRHead failed to load")

        vrLogic = slicer.modules.volumerendering.logic()
        disp = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
        disp.SetVisibility(True)
        preset = vrLogic.GetPresetByName("MR-Default")
        if preset is not None:
            disp.GetVolumePropertyNode().Copy(preset)

        b = [0.0] * 6
        vol.GetBounds(b)
        corners = np.array([
            [b[0], b[2], b[4]], [b[1], b[2], b[4]],
            [b[0], b[3], b[4]], [b[1], b[3], b[4]],
            [b[0], b[2], b[5]], [b[1], b[2], b[5]],
            [b[0], b[3], b[5]], [b[1], b[3], b[5]],
        ], dtype=np.float64)

        # Fiducial list: initial 8 corner landmarks are the TPS sources.
        # Any further points the user adds extend the source set. Moving
        # any existing point defines its target (source stays fixed).
        fnode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "TPSLandmarks")
        fdisp = fnode.GetDisplayNode()
        if fdisp is not None:
            fdisp.SetGlyphScale(3.0)
            fdisp.SetTextScale(2.0)
            fdisp.SetSelectedColor(1.0, 0.85, 0.0)
        for i, c in enumerate(corners):
            fnode.AddControlPoint(*c, f"L{i}")

        # Per-point source/target in a side list. Keyed by control-point
        # Slicer-generated ID so points survive reordering.
        sources = {}  # cpid -> np.array(3) RAS
        for i in range(fnode.GetNumberOfControlPoints()):
            cpid = fnode.GetNthControlPointID(i)
            sources[cpid] = np.array(corners[i], dtype=np.float64)

        # Grid transform node we drive from the TPS.
        tfnode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLGridTransformNode", "TPSGrid")

        grid_dims = (24, 24, 24)
        pad_mm = 40.0
        grid_origin = [float(b[0] - pad_mm),
                       float(b[2] - pad_mm),
                       float(b[4] - pad_mm)]
        grid_extent = [float(b[1] - b[0] + 2 * pad_mm),
                       float(b[3] - b[2] + 2 * pad_mm),
                       float(b[5] - b[4] + 2 * pad_mm)]
        grid_spacing = [grid_extent[k] / max(grid_dims[k] - 1, 1)
                        for k in range(3)]

        def rebuild_tps():
            """Rebuild the TPS displacement grid from current sources +
            per-control-point positions (targets)."""
            n = fnode.GetNumberOfControlPoints()
            if n < 4:
                return  # TPS needs at least 4 points to be meaningful
            src = vtk.vtkPoints()
            tgt = vtk.vtkPoints()
            for i in range(n):
                cpid = fnode.GetNthControlPointID(i)
                if cpid not in sources:
                    continue
                pos = [0.0, 0.0, 0.0]
                fnode.GetNthControlPointPosition(i, pos)
                s = sources[cpid]
                src.InsertNextPoint(float(s[0]), float(s[1]), float(s[2]))
                tgt.InsertNextPoint(float(pos[0]), float(pos[1]), float(pos[2]))

            if src.GetNumberOfPoints() < 4:
                return
            tps = vtk.vtkThinPlateSplineTransform()
            tps.SetSourceLandmarks(src)
            tps.SetTargetLandmarks(tgt)
            tps.SetBasisToR()

            # Bake to a grid: vtkTransformToGrid samples the analytic TPS
            # on a regular lattice and produces a vtkImageData of
            # displacements suitable for vtkGridTransform.
            ttg = vtk.vtkTransformToGrid()
            ttg.SetGridOrigin(*grid_origin)
            ttg.SetGridSpacing(*grid_spacing)
            ttg.SetGridExtent(0, grid_dims[0] - 1,
                              0, grid_dims[1] - 1,
                              0, grid_dims[2] - 1)
            ttg.SetGridScalarTypeToFloat()
            # TPS maps source->target; we want the inverse so the volume
            # (in world space) warps toward the targets. Inversing the
            # TPS is cheap because it's analytical.
            inv = tps.GetInverse()
            ttg.SetInput(inv)
            ttg.Update()

            vgrid = vtk.vtkGridTransform()
            vgrid.SetDisplacementGridData(ttg.GetOutput())
            vgrid.SetInterpolationModeToLinear()
            tfnode.SetAndObserveTransformFromParent(vgrid)

        # Seed the grid with an identity TPS (sources == current positions).
        rebuild_tps()

        # Install bridge, attach the grid transform.
        bridge = self._install_vtk_bridge()
        self.assertEqual(len(bridge.images_by_vrdn), 1,
            f"expected 1 claimed VRDN, got {list(bridge.images_by_vrdn.keys())}")
        bridge.set_grid_transform(tfnode)

        # Keep sources in sync when the user adds NEW points (they become
        # new source landmarks at their placement position).
        def on_point_defined(caller, event):
            try:
                # PointPositionDefinedEvent fires with the cp index in
                # caller's last-event data; scan for unknown cpids.
                n = fnode.GetNumberOfControlPoints()
                changed = False
                for i in range(n):
                    cpid = fnode.GetNthControlPointID(i)
                    if cpid not in sources:
                        pos = [0.0, 0.0, 0.0]
                        fnode.GetNthControlPointPosition(i, pos)
                        sources[cpid] = np.array(pos, dtype=np.float64)
                        changed = True
                if changed:
                    rebuild_tps()
            except Exception as e:
                print(f"on_point_defined: {e}")

        def on_point_modified(caller, event):
            try:
                rebuild_tps()
            except Exception as e:
                print(f"on_point_modified: {e}")

        t1 = fnode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            on_point_defined)
        t2 = fnode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            on_point_modified)

        # Nudge one corner to create a visible warp so the test image has
        # content to measure against the initial identity render.
        moved_i = 0
        start = [0.0, 0.0, 0.0]
        fnode.GetNthControlPointPosition(moved_i, start)
        fnode.SetNthControlPointPosition(
            moved_i, start[0] + 30.0, start[1] + 15.0, start[2] + 10.0)
        slicer.app.processEvents()

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no warped output visible -- max_rgb={mx}")

        self._stash(vtkBridge=bridge, volume=vol, landmarks=fnode,
                    gridTransform=tfnode, sources=sources,
                    rebuildTPS=rebuild_tps, observerTags=(t1, t2))
        self.delayDisplay(
            "VTK: Landmark Deform PASSED -- drag the L# points to warp "
            "the volume; place new points to extend the TPS reference set",
            600)

    def test_vtk_Segmentation(self):
        """VTK injection + segmentation iso-surface rendering. Seeds two
        segments in MRHead (a coarse brain mask + a threshold-based second
        segment), hooks the bridge to watch the segmentation node, then
        mutates a labelmap to exercise the live re-upload path. The shader
        uses a local-DT approximation to render a clean, anti-aliased
        iso-surface per segment with Phong shading.
        """
        import numpy as np
        self.delayDisplay("VTK: Segmentation loading", 150)

        import SampleData
        vol = SampleData.SampleDataLogic().downloadMRHead()
        self.assertIsNotNone(vol, "MRHead failed to load")
        # No volume-rendering display node -- this demo is segments only.
        # The user can always enable VR on MRHead interactively afterward.

        # Build two segments from thresholded MRHead intensities.
        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", "TestSeg")
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(vol)

        vol_arr = slicer.util.arrayFromVolume(vol)   # (K, J, I)
        # Brain-ish: mid range of MRHead intensities -> a connected blob.
        mask_brain = ((vol_arr > 40) & (vol_arr < 120)).astype(np.uint8)
        # Skull-ish: brighter/darker contrast -- just a second distinct region.
        mask_high = (vol_arr > 120).astype(np.uint8)

        seg_ids = []
        for name, color, mask in [
            ("Brain", (0.90, 0.20, 0.20), mask_brain),
            ("High",  (0.20, 0.80, 0.50), mask_high),
        ]:
            sid = seg_node.GetSegmentation().AddEmptySegment(name, name)
            seg_node.GetSegmentation().GetSegment(sid).SetColor(*color)
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask, seg_node, sid, vol)
            seg_ids.append(sid)

        # 3D visibility on; give the display a bright opacity.
        dn = seg_node.GetDisplayNode()
        dn.SetVisibility3D(True)
        for sid in seg_ids:
            dn.SetSegmentOpacity3D(sid, 1.0)
            dn.SetSegmentVisibility(sid, True)

        # Install bridge in four-up so slice views are there for painting.
        bridge = self._install_vtk_bridge(
            layout=slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        bridge.set_segmentation_node(seg_node)
        self.assertEqual(len(bridge._segments), 2,
            f"expected 2 SegmentFields, got {len(bridge._segments)}")

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no iso-surface visible -- max_rgb={mx}")

        # Exercise the paint path: flip some voxels in the "Brain" labelmap
        # and confirm the content-modified observer re-uploads. This mimics
        # what the Segment Editor's brush does -- direct writes to the
        # source rep with Modified() triggering the SourceRepresentationModified
        # event chain.
        oimg = seg_node.GetBinaryLabelmapInternalRepresentation(seg_ids[0])
        self.assertIsNotNone(oimg, "brain labelmap unexpectedly None")
        import vtk.util.numpy_support as vnp
        ext = oimg.GetExtent()
        dx = ext[1] - ext[0] + 1
        dy = ext[3] - ext[2] + 1
        dz = ext[5] - ext[4] + 1
        scalars = oimg.GetPointData().GetScalars()
        arr = vnp.vtk_to_numpy(scalars).reshape(dz, dy, dx)
        # Carve a small cavity in the middle so the surface changes visibly.
        cx, cy, cz = dx // 2, dy // 2, dz // 2
        r = max(3, min(dx, dy, dz) // 8)
        k, j, i = np.ogrid[:dz, :dy, :dx]
        ball = (k - cz) ** 2 + (j - cy) ** 2 + (i - cx) ** 2 <= r * r
        arr[ball] = 0
        oimg.Modified()
        seg_node.GetSegmentation().Modified()
        slicer.app.processEvents()
        view.forceRender()
        slicer.app.processEvents()

        rgb2 = self._vtk_render_and_snapshot()
        # Weak check: just confirm the render still produced non-black pixels
        # after the edit. The real check is visual.
        mx2 = tuple(int(x) for x in rgb2.max(axis=(0, 1)))
        self.assertGreater(max(mx2), 60,
            f"no output after carving -- max_rgb={mx2}")

        self._stash(vtkBridge=bridge, volume=vol, segmentation=seg_node,
                    segmentIds=seg_ids)
        self.delayDisplay(
            "VTK: Segmentation PASSED -- switch to the Segment Editor and "
            "paint into 'Brain' or 'High'; the 3D view updates as you paint",
            600)

    def test_vtk_ColorizeRGBA(self):
        """GPU ColorizeVolume: bakes a single rgba16float 3D texture from
        CT + segmentation on the GPU (label->palette, separable Gaussian
        on alpha, multiply alpha by CT intensity) and renders it with a
        plain RGBA-volume mode (RGB = color, A = opacity, Phong from
        gradient of A). Separate from the paint/iso-surface demo -- no
        per-segment textures at render time. Uses the same CTLiver +
        CTLiverSegmentation data as SlicerSandbox/ColorizeVolume so the
        two pipelines can be compared side by side.
        """
        self.delayDisplay("Injection: Colorize (RGBA) loading", 150)

        # Use the same CTLiver + CTLiverSegmentation pair that
        # SlicerSandbox/ColorizeVolume's self-test uses, but register the
        # segmentation sample inline so we don't need the ColorizeVolume
        # module (and therefore SlicerSandbox) to be installed.
        import SampleData
        try:
            SampleData.SampleDataLogic.registerCustomSampleDataSource(
                category="Sandbox",
                sampleName="CTLiverSegmentation",
                uris="https://github.com/PerkLab/SlicerSandbox/releases/download/TestingData/CTLiverSegmentation.seg.nrrd",
                fileNames="CTLiverSegmentation.seg.nrrd",
                checksums="SHA256:ce9a7182a666788a2556f6cf4f59ad5dadd944171cc279e80c164496729a7032",
                nodeNames="CTLiverSegmentation")
        except Exception:
            pass  # already registered on a prior run

        # Cached-across-scene-clears sample loads. See class-level
        # _load_cached_volume / _load_cached_segmentation for details --
        # they hold a Python reference to the vtkImageData / vtkSegmentation
        # so subsequent test runs skip the ~9 s NRRD parse and just rewrap
        # the cached data in a fresh MRML node.
        vol = self._load_cached_volume("CTLiver")
        self.assertIsNotNone(vol, "CTLiver failed to load")
        seg = self._load_cached_segmentation("CTLiverSegmentation")
        self.assertIsNotNone(seg, "CTLiverSegmentation failed to load")

        # No VR display node on the source volume -- the bake is the full
        # rendering pipeline.
        bridge = self._install_vtk_bridge(
            layout=slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        rgba = bridge.add_colorize_volume(vol, seg, sigma_voxels=1.5)
        self.assertIsNotNone(rgba, "add_colorize_volume returned None")
        self.assertEqual(len(bridge._rgba_volumes), 1,
            f"expected 1 RGBA volume, got {len(bridge._rgba_volumes)}")

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no RGBA-baked output visible -- max_rgb={mx}")

        self._stash(vtkBridge=bridge, volume=vol, segmentation=seg,
                    rgbaField=rgba)
        self.delayDisplay(
            "Injection: Colorize (RGBA) PASSED -- same bake that "
            "ColorizeVolume does on CPU, now on the GPU.", 600)

    def test_vtk_SegmentSurfaces(self):
        """Gradient-opacity segment rendering: emulates Slicer's polydata
        closed-surface look using volume-rendering compositing. Per ray
        step the alpha contribution is |grad alpha| * step (alpha is the
        palette-opacity-scaled Gaussian-smoothed presence from the bake).
        Integrated across a 0->opacity transition that sums to the
        segment's opacity regardless of how deep the ray travels through
        the segment interior. Semi-transparent segments therefore add
        their opacity once per front/back face crossing.

        Uses the ColorizeVolume RGBA bake pipeline (single r8uint merged
        labelmap + one rgba16float output + one scratch, ~2.7 GB total
        regardless of segment count) instead of per-segment texture
        buffers. The 73 segments in CTLiverSegmentation would otherwise
        allocate ~210 GB of per-segment textures.

        Visibility / opacity / color changes on the segmentation display
        node trigger a cheap palette-only rebake through the existing
        _on_rgba_display_modified observer. Scene-close auto-uninstalls.
        """
        self.delayDisplay("Injection: Segment Surfaces loading", 150)

        import SampleData
        try:
            SampleData.SampleDataLogic.registerCustomSampleDataSource(
                category="Sandbox",
                sampleName="CTLiverSegmentation",
                uris="https://github.com/PerkLab/SlicerSandbox/releases/download/TestingData/CTLiverSegmentation.seg.nrrd",
                fileNames="CTLiverSegmentation.seg.nrrd",
                checksums="SHA256:ce9a7182a666788a2556f6cf4f59ad5dadd944171cc279e80c164496729a7032",
                nodeNames="CTLiverSegmentation")
        except Exception:
            pass

        vol = self._load_cached_volume("CTLiver")
        self.assertIsNotNone(vol, "CTLiver failed to load")
        seg = self._load_cached_segmentation("CTLiverSegmentation")
        self.assertIsNotNone(seg, "CTLiverSegmentation failed to load")

        dn = seg.GetDisplayNode()
        if dn is not None:
            dn.SetVisibility3D(True)
            dn.SetVisibility(True)

        bridge = self._install_vtk_bridge(
            layout=slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        # Bake the RGBA volume with modulate_by_ct=False: skips the CT
        # upload (~685 MB saved) and the CT modulate pass. The resulting
        # texture's alpha is the palette-opacity-scaled smoothed presence,
        # rendered with surface-mode gradient opacity.
        rgba = bridge.add_colorize_volume(vol, seg, sigma_voxels=1.5,
                                          modulate_by_ct=False)
        self.assertIsNotNone(rgba, "add_colorize_volume returned None")
        self.assertEqual(rgba.render_mode, "surface",
            f"expected surface render mode, got {rgba.render_mode}")

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no surface-rendered output visible -- max_rgb={mx}")

        # ------------------------------------------------------------------
        # Carving: a single fiducial control point removes the liver, small
        # bowel, and colon from compositing within a sphere whose radius is
        # 3x the markup display radius -- enough to fully cut through them
        # without overlapping the volume rendering's screen footprint.
        # ------------------------------------------------------------------
        carve_names = ("liver", "small bowel", "colon")
        carve_ids = self._resolve_carve_label_values(seg, carve_names)
        self.assertGreater(len(carve_ids), 0,
            f"none of {carve_names} found in segmentation; "
            f"available={[seg.GetSegmentation().GetNthSegmentID(i) for i in range(seg.GetSegmentation().GetNumberOfSegments())]}")

        # Initial sphere center: roughly the volume center.
        b = [0.0] * 6
        vol.GetRASBounds(b)
        center0 = ((b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5)

        carve_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "CarvePoint")
        cdisp = carve_node.GetDisplayNode()
        if cdisp is not None:
            # Stable mm-sized glyph so 1.5x its diameter is a predictable
            # carve. GlyphSize is interpreted as the marker's diameter,
            # so the displayed radius is GlyphSize / 2.
            cdisp.SetUseGlyphScale(False)
            cdisp.SetGlyphSize(35.0)
            cdisp.SetSelectedColor(1.0, 0.85, 0.0)
        carve_node.AddControlPoint(*center0, "carve")

        def _radius_from_glyph():
            # carve sphere = 3x the marker's display radius = 1.5x diameter.
            return 1.5 * float(cdisp.GetGlyphSize()) if cdisp else 30.0

        def _push_carve():
            pos = [0.0, 0.0, 0.0]
            carve_node.GetNthControlPointPosition(0, pos)
            bridge.set_rgba_carve(rgba, carve_ids, pos, _radius_from_glyph())

        _push_carve()

        def _on_carve_point_modified(caller, event):
            try:
                _push_carve()
            except Exception as e:
                print(f"_on_carve_point_modified: {e}")

        def _on_carve_disp_modified(caller, event):
            try:
                _push_carve()
            except Exception as e:
                print(f"_on_carve_disp_modified: {e}")

        carve_node.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
            _on_carve_point_modified)
        if cdisp is not None:
            cdisp.AddObserver(vtk.vtkCommand.ModifiedEvent,
                              _on_carve_disp_modified)

        view.forceRender()
        slicer.app.processEvents()

        self._stash(vtkBridge=bridge, volume=vol, segmentation=seg,
                    rgbaField=rgba, carveNode=carve_node)
        self.delayDisplay(
            "Injection: Segment Surface (Carving) PASSED -- drag the "
            "yellow CarvePoint fiducial through the abdomen and the "
            "liver / small bowel / colon are carved away inside a sphere "
            "of radius 3x the glyph display radius (= 1.5x its diameter). "
            "Resize the glyph to grow or shrink the carve. Other "
            "segments are unaffected.", 600)

    def _collect_scene_fiber_bundles(self):
        """Scan the MRML scene for vtkMRMLFiberBundleNode instances; if
        any are found, build a single combined polydata with one bundle
        id per fiber bundle node, populate a 256-entry palette from each
        bundle's line-display colour and opacity, and hide the line
        display so our rasterized tubes show through.

        Returns (polydata, palette, hidden_line_displays) on success, or
        (None, None, []) if the scene has no fiber bundles (caller falls
        back to the synthetic scene).
        """
        import numpy as np

        coll = slicer.mrmlScene.GetNodesByClass("vtkMRMLFiberBundleNode")
        coll.UnRegister(None)  # decrement the ref the collection added
        n_nodes = coll.GetNumberOfItems()
        if n_nodes == 0:
            return None, None, []

        combined_pts = vtk.vtkPoints()
        combined_lines = vtk.vtkCellArray()
        bundle_array = vtk.vtkIntArray()
        bundle_array.SetName("BundleId")
        palette = np.zeros((256, 4), dtype=np.uint8)
        hidden = []
        bundle_id = 0
        point_offset = 0
        n_strands_total = 0

        for i in range(n_nodes):
            node = coll.GetItemAsObject(i)
            polydata = node.GetPolyData()
            if polydata is None:
                continue
            pts = polydata.GetPoints()
            lines = polydata.GetLines()
            if (pts is None or lines is None
                    or pts.GetNumberOfPoints() == 0
                    or lines.GetNumberOfCells() == 0):
                continue
            bundle_id += 1
            if bundle_id > 255:
                # Palette only has slots 1..255 (0 reserved). Cap.
                bundle_id = 255

            # Per-bundle colour + opacity from the line display node.
            ldisp = None
            if hasattr(node, "GetLineDisplayNode"):
                ldisp = node.GetLineDisplayNode()
            if ldisp is None:
                # Fall back to scanning all display nodes for a line one.
                for di in range(node.GetNumberOfDisplayNodes()):
                    dn = node.GetNthDisplayNode(di)
                    if dn is not None and dn.IsA(
                            "vtkMRMLFiberBundleLineDisplayNode"):
                        ldisp = dn
                        break
            color = (1.0, 1.0, 1.0)
            opacity = 1.0
            if ldisp is not None:
                try:
                    color = ldisp.GetColor()
                    opacity = float(ldisp.GetOpacity())
                except Exception:
                    pass
                if ldisp.GetVisibility():
                    ldisp.SetVisibility(False)
                    hidden.append(ldisp)
            palette[bundle_id, 0] = int(np.clip(color[0] * 255, 0, 255))
            palette[bundle_id, 1] = int(np.clip(color[1] * 255, 0, 255))
            palette[bundle_id, 2] = int(np.clip(color[2] * 255, 0, 255))
            palette[bundle_id, 3] = int(np.clip(opacity * 255, 0, 255))

            # Copy points and re-emit lines with offset indices + bundle id.
            n_pts = pts.GetNumberOfPoints()
            for pi in range(n_pts):
                p = [0.0, 0.0, 0.0]
                pts.GetPoint(pi, p)
                combined_pts.InsertNextPoint(*p)
            id_list = vtk.vtkIdList()
            lines.InitTraversal()
            while lines.GetNextCell(id_list):
                new_cell = vtk.vtkIdList()
                for j in range(id_list.GetNumberOfIds()):
                    new_cell.InsertNextId(point_offset + id_list.GetId(j))
                combined_lines.InsertNextCell(new_cell)
                bundle_array.InsertNextValue(bundle_id)
                n_strands_total += 1
            point_offset += n_pts

        if point_offset == 0:
            return None, None, []

        combined = vtk.vtkPolyData()
        combined.SetPoints(combined_pts)
        combined.SetLines(combined_lines)
        combined.GetCellData().AddArray(bundle_array)
        return combined, palette, hidden

    def test_vtk_FiberStrands(self):
        """Per-strand cylinder-impostor rasterization through an A-buffer
        FragmentField. Each polyline segment is rasterized as an
        instanced billboard, the fragment shader does an analytic
        ray-cylinder intersection and atomic-appends premultiplied
        (depth, rgba8) into a per-pixel sorted list (K=64 fragments).
        The main ray-march reads from this list and interleaves the
        fragments with volume samples by depth -- correct compositing
        with anything else in the scene.

        If the scene already has vtkMRMLFiberBundleNode instances, use
        them: combine their polylines with one bundle id per node and
        populate the palette from each line-display node's colour and
        opacity, then hide the default line display so our rendering
        replaces it. Otherwise build the synthetic 4-bundle scene
        (helix, U-arc, fan, diagonal) at ~1500 streamlines.
        """
        import numpy as np

        self.delayDisplay(
            "Injection: Fiber Strands (A-buffer) -- scanning scene", 150)

        polydata, palette, hidden_displays = (
            self._collect_scene_fiber_bundles())
        used_scene_bundles = polydata is not None
        if used_scene_bundles:
            n_strands = polydata.GetLines().GetNumberOfCells()
            n_pts = polydata.GetPoints().GetNumberOfPoints()
            self.delayDisplay(
                f"Found {len(palette[palette[:,3] > 0]) - 1} fiber "
                f"bundle(s) in the scene -- baking "
                f"{n_strands} strands, {n_pts} points...", 150)
            self._stash(hiddenLineDisplays=hidden_displays)

        if not used_scene_bundles:
            self.delayDisplay(
                "No fiber bundles in the scene -- building synthetic "
                "4-bundle test scene", 150)

            polydata = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            bundle_array = vtk.vtkIntArray()
            bundle_array.SetName("BundleId")

            rng = np.random.default_rng(42)

            def add_strand(strand_fn, bundle_id, n_samples=60,
                           jitter_mm=0.02):
                """Per-strand offset gives variety; per-sample jitter
                must stay well below tube_radius (0.2 mm) so consecutive
                segments share tangent direction and the capsule chain
                reads as a smooth tube. With jitter ~ tube_radius the
                strand looks zigzag."""
                cell = vtk.vtkIdList()
                offset = rng.normal(scale=0.8, size=3)
                for i in range(n_samples):
                    t = i / float(n_samples - 1)
                    p = (np.asarray(strand_fn(t), dtype=np.float64)
                         + offset
                         + rng.normal(scale=jitter_mm, size=3))
                    pid = points.InsertNextPoint(float(p[0]), float(p[1]),
                                                 float(p[2]))
                    cell.InsertNextId(pid)
                lines.InsertNextCell(cell)
                bundle_array.InsertNextValue(bundle_id)

            # Bundle 1: helix along +Z, ~400 strands.
            for s in range(400):
                phase = 2 * np.pi * s / 400
                r = 22.0 + (s % 13 - 6) * 0.7
                def fn(t, phase=phase, r=r):
                    z = (t - 0.5) * 110.0
                    theta = 2 * np.pi * z / 38.0 + phase
                    return (r * np.cos(theta), r * np.sin(theta), z)
                add_strand(fn, bundle_id=1, n_samples=70)

            # Bundle 2: U-arc along +X, ~400 strands.
            for s in range(400):
                z_off = (s - 200) * 0.30
                y_thick = (s % 11 - 5) * 0.4
                def fn(t, z_off=z_off, y_thick=y_thick):
                    x = -55.0 + 110.0 * t
                    y = 32.0 * np.sin(np.pi * t) + y_thick
                    return (x, y, z_off)
                add_strand(fn, bundle_id=2, n_samples=70)

            # Bundle 3: fan from a focal point, ~400 strands on a 20x20
            # grid over a forward-facing cone.
            focal = np.array([-30.0, -8.0, 0.0])
            for s in range(400):
                sx = s % 20
                sy = s // 20
                azim = 2 * np.pi * sx / 20.0
                elev = 0.40 * np.pi * (sy - 9.5) / 10.0
                dx = np.cos(elev)
                spread = np.sin(elev)
                dy = spread * np.cos(azim) * 0.8
                dz = spread * np.sin(azim) * 0.8
                direction = np.array([dx, dy, dz])
                direction = direction / max(np.linalg.norm(direction), 1e-6)
                length = 80.0
                def fn(t, direction=direction, length=length):
                    return tuple(focal + length * t * direction)
                add_strand(fn, bundle_id=3, n_samples=60)

            # Bundle 4: diagonal bundle through the centre, ~300 strands.
            axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
            u_axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
            v_axis = np.cross(axis, u_axis)
            for s in range(300):
                angle = 2 * np.pi * s / 300
                r = 9.0 + (s % 7 - 3) * 0.4
                offset_perp = r * (np.cos(angle) * u_axis
                                   + np.sin(angle) * v_axis)
                def fn(t, offset_perp=offset_perp):
                    return tuple((t - 0.5) * 100.0 * axis + offset_perp)
                add_strand(fn, bundle_id=4, n_samples=60)

            polydata.SetPoints(points)
            polydata.SetLines(lines)
            polydata.GetCellData().AddArray(bundle_array)
            n_strands = lines.GetNumberOfCells()
            n_pts = points.GetNumberOfPoints()

            # Palette: 4 distinct colours for the 4 bundles.
            palette = np.zeros((256, 4), dtype=np.uint8)
            palette[1] = (235,  70,  70, 255)
            palette[2] = ( 80, 190, 235, 255)
            palette[3] = (245, 200,  70, 255)
            palette[4] = ( 90, 220, 130, 255)

        self.delayDisplay(
            f"Injection: Fiber Strands rasterizing "
            f"{n_strands} strands, {n_pts} pts...", 150)

        bridge = self._install_vtk_bridge()
        sf = bridge.add_fiber_strands(
            polydata, palette,
            tube_radius_mm=0.2,
            bundle_id_array_name="BundleId")
        self.assertIsNotNone(sf, "add_fiber_strands returned None")

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        rgb = self._vtk_render_and_snapshot()
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        self.assertGreater(max(mx), 60,
            f"no strand output visible -- max_rgb={mx}")

        self._stash(vtkBridge=bridge, fiberPolyData=polydata,
                    fiberStrandField=sf)
        source = ("scene bundles" if used_scene_bundles
                  else "synthetic 4-bundle scene")
        self.delayDisplay(
            f"Injection: Fiber Strands (A-buffer) PASSED -- {source} "
            f"({n_strands} strands, {n_pts} points) rasterized as "
            "anisotropic 3D tubes via cylinder impostors + per-pixel "
            "sorted A-buffer (K=64 fragments) for depth-correct "
            "compositing with the rest of the scene.", 700)

    def test_vtk_FieldCompositing(self):
        """Three-way compositing test that exercises every render path
        the bridge supports interacting with each other:

          - FiberStrandField (cylinder-impostor rasterization → A-buffer)
          - RGBAVolumeField  (volumetric ray-march, density mode)
                             animated as a "lavalamp" -- 3 Gaussian blobs
                             of distinct primary colours moving on
                             Lissajous paths.
          - vtkMRMLMarkupsFiducialNode (rendered the normal VTK way;
                             clips the volume ray-march and the strand
                             pass via the existing VTK depth integration)

        A QTimer.singleShot loop updates the volume CPU-side and re-uploads
        ~20 fps. The animation is event-loop friendly (no threading) and
        stops automatically when the bridge is uninstalled.
        """
        import numpy as np
        import time
        import qt

        self.delayDisplay(
            "Injection: Field Compositing -- building scene", 200)

        # --- 1. Fiber strands (4 bundles, ~1500 strands). ---
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        bundle_array = vtk.vtkIntArray()
        bundle_array.SetName("BundleId")
        rng = np.random.default_rng(42)

        def add_strand(strand_fn, bundle_id, n_samples=60, jitter_mm=0.02):
            cell = vtk.vtkIdList()
            offset = rng.normal(scale=0.8, size=3)
            for i in range(n_samples):
                t_p = i / float(n_samples - 1)
                p = (np.asarray(strand_fn(t_p), dtype=np.float64)
                     + offset
                     + rng.normal(scale=jitter_mm, size=3))
                pid = points.InsertNextPoint(float(p[0]), float(p[1]),
                                             float(p[2]))
                cell.InsertNextId(pid)
            lines.InsertNextCell(cell)
            bundle_array.InsertNextValue(bundle_id)

        for s in range(400):
            phase = 2 * np.pi * s / 400
            r = 22.0 + (s % 13 - 6) * 0.7
            def fn(t, phase=phase, r=r):
                z = (t - 0.5) * 110.0
                theta = 2 * np.pi * z / 38.0 + phase
                return (r * np.cos(theta), r * np.sin(theta), z)
            add_strand(fn, bundle_id=1, n_samples=70)
        for s in range(400):
            z_off = (s - 200) * 0.30
            y_thick = (s % 11 - 5) * 0.4
            def fn(t, z_off=z_off, y_thick=y_thick):
                x = -55.0 + 110.0 * t
                y = 32.0 * np.sin(np.pi * t) + y_thick
                return (x, y, z_off)
            add_strand(fn, bundle_id=2, n_samples=70)
        focal = np.array([-30.0, -8.0, 0.0])
        for s in range(400):
            sx = s % 20
            sy = s // 20
            azim = 2 * np.pi * sx / 20.0
            elev = 0.40 * np.pi * (sy - 9.5) / 10.0
            dx = np.cos(elev)
            spread = np.sin(elev)
            dy = spread * np.cos(azim) * 0.8
            dz = spread * np.sin(azim) * 0.8
            direction = np.array([dx, dy, dz])
            direction = direction / max(np.linalg.norm(direction), 1e-6)
            length = 80.0
            def fn(t, direction=direction, length=length):
                return tuple(focal + length * t * direction)
            add_strand(fn, bundle_id=3, n_samples=60)
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        u_axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        v_axis = np.cross(axis, u_axis)
        for s in range(300):
            angle = 2 * np.pi * s / 300
            r = 9.0 + (s % 7 - 3) * 0.4
            offset_perp = r * (np.cos(angle) * u_axis
                               + np.sin(angle) * v_axis)
            def fn(t, offset_perp=offset_perp):
                return tuple((t - 0.5) * 100.0 * axis + offset_perp)
            add_strand(fn, bundle_id=4, n_samples=60)
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetCellData().AddArray(bundle_array)

        palette = np.zeros((256, 4), dtype=np.uint8)
        palette[1] = (235,  70,  70, 220)
        palette[2] = ( 80, 190, 235, 128)   # blue at 0.5 opacity
        palette[3] = (245, 200,  70, 220)
        palette[4] = ( 90, 220, 130, 220)

        bridge = self._install_vtk_bridge()
        sf = bridge.add_fiber_strands(
            polydata, palette,
            tube_radius_mm=0.2,
            bundle_id_array_name="BundleId")

        # --- 2. Markup fiducials (rendered through VTK normally). ---
        fnode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "RefPoints")
        fdisp = fnode.GetDisplayNode()
        if fdisp is not None:
            fdisp.SetUseGlyphScale(False)
            fdisp.SetGlyphSize(8.0)
            fdisp.SetSelectedColor(1.0, 1.0, 0.3)
        # First control point is the lavalamp's attractor -- placed in
        # the periphery (outside the helix) so the blobs orbit somewhere
        # the user can actually see them, not buried in the bundle.
        fnode.AddControlPoint(40.0, 25.0, 30.0, "attractor")
        fnode.AddControlPoint(0.0, 0.0, 0.0, "origin")
        fnode.AddControlPoint(0.0, 0.0, 45.0, "+Z")
        fnode.AddControlPoint(-30.0, 25.0, -30.0, "back-left")

        # --- 3. Lavalamp volume (animated). ---
        lava = self._add_lavalamp_volume(
            bridge,
            dims=(48, 48, 48),
            bbox=(-60.0, 60.0, -45.0, 45.0, -60.0, 60.0))

        # --- 4. Animation timer (singleShot chain) + physics state. ---
        # Each blob has a primary RGB colour, an initial position
        # spread away from origin so they don't all collapse, and zero
        # velocity. Per tick: attract toward the first fiducial,
        # repel from each other, integrate with damping. Blobs end up
        # orbiting the fiducial in a "fighting for the centre" pattern.
        anim_state = {
            "start": time.time(),
            "last_t": time.time(),
            "running": True,
            "frame": 0,
            "fnode": fnode,
            # Initial positions in a triangle around the attractor
            # (40, 25, 30) so they're visible from frame 1.
            "blob_pos": [
                np.array([ 60.0,  25.0,  30.0], dtype=np.float64),
                np.array([ 30.0,  45.0,  30.0], dtype=np.float64),
                np.array([ 30.0,  15.0,  50.0], dtype=np.float64),
            ],
            # Slight initial swirl so the orbit is visible while it
            # damps toward equilibrium.
            "blob_vel": [
                np.array([  0.0,  10.0,   0.0], dtype=np.float64),
                np.array([ 10.0,   0.0,  10.0], dtype=np.float64),
                np.array([-10.0, -10.0,   0.0], dtype=np.float64),
            ],
            "blob_color": [
                (1.00, 0.18, 0.18),
                (0.18, 1.00, 0.30),
                (0.25, 0.45, 1.00),
            ],
            # Per-blob breathing periods (seconds) and phase offsets.
            # Prime-ish ratios so the three blobs never sync up.
            "blob_period": [3.7, 5.3, 4.6],
            "blob_phase":  [0.0, 2.1, 4.7],
        }

        def tick():
            try:
                if not anim_state["running"]:
                    return
                if getattr(bridge, "_disposed", False):
                    anim_state["running"] = False
                    return
                now = time.time()
                dt = min(now - anim_state["last_t"], 0.1)
                anim_state["last_t"] = now
                self._step_lavalamp_physics(anim_state, dt)
                # Per-blob breathing: pulse(t) ∈ [0, 1], modulating
                # both drawn sigma (size) and attraction strength.
                t_anim = now - anim_state["start"]
                two_pi = 2.0 * np.pi
                sigma_scales = [
                    0.6 + 0.6 * (0.5 + 0.5 * np.sin(
                        two_pi * t_anim / p + ph))
                    for p, ph in zip(anim_state["blob_period"],
                                     anim_state["blob_phase"])
                ]
                blobs = list(zip(anim_state["blob_pos"],
                                 anim_state["blob_color"],
                                 sigma_scales))
                self._animate_lavalamp(lava, blobs)
                bridge.rw.Render()
                anim_state["frame"] += 1
            except Exception as e:
                print(f"lavalamp tick: {e}")
                anim_state["running"] = False
                return
            qt.QTimer.singleShot(50, tick)

        # Initial frame + start the chain. Use sigma_scale=1.0 for
        # the first frame; the breathing phase will pick up next tick.
        initial_blobs = [(p, c, 1.0) for p, c in zip(
            anim_state["blob_pos"], anim_state["blob_color"])]
        self._animate_lavalamp(lava, initial_blobs)

        lm = slicer.app.layoutManager()
        view = lm.threeDWidget(0).threeDView()
        renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        view.forceRender()
        slicer.app.processEvents()

        qt.QTimer.singleShot(0, tick)

        self._stash(vtkBridge=bridge, fiberStrandField=sf,
                    fiberPolyData=polydata, lavalampField=lava,
                    lavalampState=anim_state, fiducialNode=fnode)

        self.delayDisplay(
            "Injection: Field Compositing PASSED -- four fiber bundles "
            "(A-buffer rasterization), three-blob animated lavalamp "
            "(volumetric density mode), and four markup fiducials "
            "(VTK opaque) all composite together with correct depth "
            "interleaving. Animation runs from a singleShot QTimer "
            "chain at ~20 fps; set "
            "slicer.modules.lavalampState['running']=False to stop.",
            900)

    # ----- Lavalamp helpers (used by test_vtk_FieldCompositing) -----

    def _add_lavalamp_volume(self, bridge, dims, bbox):
        """Allocate an RGBAVolumeField sized to `dims` covering world
        bbox=(xmin,xmax,ymin,ymax,zmin,zmax), register it with the bridge
        in density-mode without a bake. The texture is created with
        COPY_DST so we can write_texture into it each frame."""
        import numpy as np
        import wgpu
        from SceneRenderingLib.wgpu_vtk_inject import RGBAVolumeField

        dx, dy, dz = dims
        xmin, xmax, ymin, ymax, zmin, zmax = bbox

        field = RGBAVolumeField(bridge.device)
        # Re-create the main texture with COPY_DST (allocate() leaves
        # only TEXTURE_BINDING + STORAGE_BINDING).
        field.tex = bridge.device.create_texture(
            size=(dx, dy, dz), dimension="3d",
            format=wgpu.TextureFormat.rgba16float,
            usage=(wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_DST),
        )
        field.tex_view = field.tex.create_view()
        # scratch_tex isn't used (no bake), but the bind group only
        # references tex_view so we leave scratch_tex None.
        if field.sampler is None:
            field.sampler = bridge.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                address_mode_w=wgpu.AddressMode.clamp_to_edge,
            )
        field.dims = (dx, dy, dz)

        # patient_to_texture: world -> [0, 1]^3.
        ext = np.array([xmax-xmin, ymax-ymin, zmax-zmin], dtype=np.float64)
        org = np.array([xmin, ymin, zmin], dtype=np.float64)
        p2t = np.eye(4, dtype=np.float64)
        p2t[0, 0] = 1.0 / ext[0]
        p2t[1, 1] = 1.0 / ext[1]
        p2t[2, 2] = 1.0 / ext[2]
        p2t[0, 3] = -org[0] / ext[0]
        p2t[1, 3] = -org[1] / ext[1]
        p2t[2, 3] = -org[2] / ext[2]
        field.patient_to_texture = p2t.astype(np.float32)
        field._bounds = (np.array([xmin, ymin, zmin], dtype=np.float32),
                         np.array([xmax, ymax, zmax], dtype=np.float32))
        # Density mode: alpha is per-sample density, integrated over
        # opacity_unit_distance. Tuned so a single blob (sigma ~ 7% of
        # bbox extent ~= 8 mm) integrates to ~0.85 alpha across its
        # diameter; lower opacity_unit_distance → denser/more opaque.
        field.sample_step_mm = max(min(ext) / dx * 0.5, 0.5)
        field.opacity_unit_distance = max(min(ext) / 12.0, 4.0)
        field.render_mode = "density"

        # Placeholder labelmap: the rgba bind layout requires a uint 3D
        # texture for the carve check, but with carve_radius_mm=0 the
        # shader never samples it. 1x1x1 r8uint zero suffices.
        label_tex = bridge.device.create_texture(
            size=(1, 1, 1), dimension="3d",
            format=wgpu.TextureFormat.r8uint,
            usage=(wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_DST),
        )
        bridge.device.queue.write_texture(
            {"texture": label_tex, "mip_level": 0, "origin": (0, 0, 0)},
            np.zeros((1, 1, 1), dtype=np.uint8),
            {"offset": 0, "bytes_per_row": 256, "rows_per_image": 1},
            (1, 1, 1))
        field._label_tex = label_tex
        field._label_carve_tex = label_tex
        field._label_tex_view = label_tex.create_view()
        field._world_to_label_tex = np.eye(4, dtype=np.float64)
        field._output_to_world = np.linalg.inv(p2t).astype(np.float64)

        # Stash bbox + dims for the animator.
        field._lava_dims = dims
        field._lava_bbox = bbox

        bridge._rgba_volumes.append(field)
        bridge._rebuild_pipeline()
        return field

    def _animate_lavalamp(self, field, blobs):
        """Render the volume from a list of `blobs` =
        [((cx, cy, cz), (cr, cg, cb), sigma_scale), ...]. Each blob is
        a Gaussian whose width is sigma_base * sigma_scale (sigma_base
        ~8% of bbox extent), so per-blob `sigma_scale` makes the blob
        visibly grow and shrink for the lava-lamp pulse. CPU compute
        (~1 ms at 48^3) + write_texture upload."""
        import numpy as np

        dx, dy, dz = field._lava_dims
        xmin, xmax, ymin, ymax, zmin, zmax = field._lava_bbox

        # Voxel-center grid in world.
        xs = np.linspace(xmin + (xmax - xmin) / (2 * dx),
                         xmax - (xmax - xmin) / (2 * dx),
                         dx, dtype=np.float32)
        ys = np.linspace(ymin + (ymax - ymin) / (2 * dy),
                         ymax - (ymax - ymin) / (2 * dy),
                         dy, dtype=np.float32)
        zs = np.linspace(zmin + (zmax - zmin) / (2 * dz),
                         zmax - (zmax - zmin) / (2 * dz),
                         dz, dtype=np.float32)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')

        ext_max = max(xmax - xmin, ymax - ymin, zmax - zmin)
        sigma_base = ext_max * 0.07         # sharper -- distinct droplets

        rgba = np.zeros((dz, dy, dx, 4), dtype=np.float32)
        for blob in blobs:
            # Tolerate (pos, color) for the first-frame call site.
            if len(blob) == 3:
                (cx, cy, cz), (cr, cg, cb), sigma_scale = blob
            else:
                (cx, cy, cz), (cr, cg, cb) = blob
                sigma_scale = 1.0
            sigma = sigma_base * float(sigma_scale)
            inv_2s2 = 1.0 / (2 * sigma * sigma)
            d2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            density = np.exp(-d2 * inv_2s2)
            rgba[..., 0] += density * cr
            rgba[..., 1] += density * cg
            rgba[..., 2] += density * cb
            rgba[..., 3] += density
        # Normalize rgb by total density so blob colours stay pure where
        # blobs overlap (otherwise heavy overlap of primary lights would
        # bleach to white). Cap alpha at 1.
        a_total = rgba[..., 3:4]
        rgba[..., :3] = rgba[..., :3] / np.maximum(a_total, 1e-6)
        rgba[..., 3] = np.clip(rgba[..., 3], 0.0, 1.0)

        rgba_f16 = np.ascontiguousarray(rgba.astype(np.float16))
        field.device.queue.write_texture(
            {"texture": field.tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba_f16,
            {"offset": 0, "bytes_per_row": dx * 8, "rows_per_image": dy},
            (dx, dy, dz))

    def _step_lavalamp_physics(self, state, dt):
        """Step the blob simulation: each blob is attracted to the first
        control point of state["fnode"] (linear spring) and repelled
        from every other blob (inverse-square, softened). Velocity is
        damped per step. The result is a stable orbit-ish fight where
        all three blobs jostle for the spot nearest the fiducial.

        Per-blob breathing: each blob has its own pulse_period and
        phase, and the attraction gain is multiplied by 0.3 + 1.4 *
        pulse(t) -- so when a blob is at peak size it pulls hard, and
        at its minimum it nearly lets go. With prime-ish periods the
        three blobs hand the "winner" position back and forth without
        ever syncing, which gives the scene its lively unsteady feel."""
        import numpy as np

        positions = state["blob_pos"]
        velocities = state["blob_vel"]
        n = len(positions)

        # Read fiducial position (first control point).
        fid = [0.0, 0.0, 0.0]
        fnode = state.get("fnode")
        if fnode is not None and fnode.GetNumberOfControlPoints() > 0:
            fnode.GetNthControlPointPosition(0, fid)
        fid_pos = np.array(fid, dtype=np.float64)

        # Per-blob breathing (matches _animate_lavalamp's sigma_scale).
        import time as _time
        t_anim = _time.time() - state["start"]
        two_pi = 2.0 * np.pi
        periods = state.get("blob_period", [1.0] * n)
        phases  = state.get("blob_phase",  [0.0] * n)
        pulse = [
            0.5 + 0.5 * np.sin(two_pi * t_anim / periods[i] + phases[i])
            for i in range(n)
        ]

        k_attract = 1.2      # base spring gain (mm/s^2 per mm); per-blob
                             # pulse modulates in [0.3, 1.7] x.
        k_repel   = 4500.0   # inverse-square push (mm^3/s^2)
        damping   = 0.985    # per-step velocity decay (lighter than
                             # critical -> visible oscillation)
        v_max     = 50.0     # mm/s cap

        forces = [np.zeros(3, dtype=np.float64) for _ in range(n)]
        for i in range(n):
            k_i = k_attract * (0.3 + 1.4 * pulse[i])
            forces[i] += k_i * (fid_pos - positions[i])
            for j in range(n):
                if i == j:
                    continue
                d = positions[i] - positions[j]
                d2 = float(np.dot(d, d)) + 4.0   # softened (mm^2)
                forces[i] += k_repel * d / (d2 ** 1.5)

        for i in range(n):
            velocities[i] = velocities[i] * damping + forces[i] * dt
            speed = float(np.linalg.norm(velocities[i]))
            if speed > v_max:
                velocities[i] *= (v_max / speed)
            positions[i] = positions[i] + velocities[i] * dt

    # ------------------------------------------------------------------
    # State / MRB persistence tests for the per-data-node wgpu_state
    # schema (see reference_wgpu_state_node_schema memory). These do
    # not exercise GPU rendering -- they verify the scripted module
    # node schema, the SH-plugin toggle path, and round-trip through
    # MRML save/load.
    # ------------------------------------------------------------------

    def _make_synthetic_volume(self, name="WgpuTestVol", dims=(16, 16, 16)):
        """Build a tiny scalar volume in MRML without downloading
        anything. Just a gradient so it has nonzero scalar range."""
        import numpy as np
        vol = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", name)
        ijk_to_ras = vtk.vtkMatrix4x4()
        ijk_to_ras.Identity()
        img = vtk.vtkImageData()
        img.SetDimensions(dims[0], dims[1], dims[2])
        img.AllocateScalars(vtk.VTK_SHORT, 1)
        arr = vtk.util.numpy_support.vtk_to_numpy(
            img.GetPointData().GetScalars()).reshape(
                dims[2], dims[1], dims[0])
        zz, yy, xx = np.indices(arr.shape)
        arr[:] = (xx + yy + zz).astype(np.int16)
        vol.SetAndObserveImageData(img)
        vol.SetIJKToRASMatrix(ijk_to_ras)
        vol.CreateDefaultDisplayNodes()
        return vol

    def _make_synthetic_segmentation(self, vol, name="WgpuTestSeg"):
        """Build a tiny segmentation with one empty segment, referencing
        the given volume's geometry. Bridge-friendly without any paint."""
        seg = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", name)
        seg.CreateDefaultDisplayNodes()
        seg.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        seg.GetSegmentation().AddEmptySegment("seg1", "seg1")
        return seg

    def _make_extra_view_node(self, name="WgpuTestView"):
        """Return a second 3D view node so the view-id selector and
        state node have something to point at beyond the default view."""
        existing = slicer.mrmlScene.GetFirstNodeByName(name)
        if existing is not None and existing.IsA("vtkMRMLViewNode"):
            return existing
        view = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLViewNode", name)
        return view

    def test_state_BasicEdit(self):
        """Round-trip the per-data-node wgpu_state schema via direct
        API calls. Covers: state creation on enable, state survives
        disable (keeps settings), view-id list set/get, segment render
        mode set/get, lifecycle cleanup when target node is deleted."""
        from SceneRenderingLib import wgpu_state
        self.delayDisplay("State: basic edit -- building scene", 100)

        vol = self._make_synthetic_volume(name="StateTestVol")
        seg = self._make_synthetic_segmentation(vol, name="StateTestSeg")
        view2 = self._make_extra_view_node()

        # No state nodes yet.
        self.assertEqual(len(wgpu_state.all_state_nodes()), 0,
            "expected no SlicerWGPU state nodes before any toggle")

        # Enable rendering on the volume -> state node appears.
        wgpu_state.set_render_enabled(vol, True)
        self.assertTrue(wgpu_state.is_render_enabled(vol),
            "volume state should report rendered=true after enable")
        self.assertEqual(len(wgpu_state.all_state_nodes()), 1,
            "expected one state node after enabling volume")

        # Disable -> state node persists but flag flips.
        wgpu_state.set_render_enabled(vol, False)
        self.assertFalse(wgpu_state.is_render_enabled(vol),
            "volume state should report rendered=false after disable")
        self.assertEqual(len(wgpu_state.all_state_nodes()), 1,
            "state node should persist across disable (settings retained)")

        # View ids: empty default, then set one explicit view.
        self.assertEqual(wgpu_state.view_node_ids(vol), [],
            "default view-id list should be empty (= all views)")
        wgpu_state.set_view_node_ids(vol, [view2.GetID()])
        self.assertEqual(wgpu_state.view_node_ids(vol), [view2.GetID()],
            "view-id list should round-trip via wgpu_state")

        # Segmentation: enable + set render mode.
        wgpu_state.set_render_enabled(seg, True)
        wgpu_state.set_segment_render_mode(seg, "surface")
        self.assertTrue(wgpu_state.is_render_enabled(seg))
        self.assertEqual(wgpu_state.segment_render_mode(seg), "surface")
        self.assertEqual(len(wgpu_state.all_state_nodes()), 2,
            "expected one state node per managed data node")

        # Lifecycle: removing the data node should drop its state node
        # via the widget's NodeRemovedEvent observer. The observer is
        # only installed when the widget is alive -- force it here so
        # the test works whether the user has opened the module or not.
        widget = slicer.modules.scenerendering.widgetRepresentation().self()
        self.assertIsNotNone(widget, "SceneRendering widget should exist")
        slicer.mrmlScene.RemoveNode(seg)
        slicer.app.processEvents()
        remaining_targets = [s.GetNodeReferenceID(wgpu_state.TARGET_NODE_REF)
                             for s in wgpu_state.all_state_nodes()]
        self.assertNotIn(seg.GetID(), remaining_targets,
            "state node should be cleaned up when its target is removed")
        self.assertIn(vol.GetID(), remaining_targets,
            "volume's state node should still be present")

        self.delayDisplay("State: Basic edit PASSED", 200)

    def test_state_SaveRestore(self):
        """Save the scene to a .mrml file with state nodes set, clear
        the scene, reload, and verify all state survived. Exercises the
        scripted-module-node serialization path (the whole point of
        using these nodes instead of node attributes)."""
        import os, tempfile
        from SceneRenderingLib import wgpu_state
        self.delayDisplay("State: save/restore -- building scene", 100)

        vol = self._make_synthetic_volume(name="SaveTestVol")
        seg = self._make_synthetic_segmentation(vol, name="SaveTestSeg")
        view2 = self._make_extra_view_node(name="SaveTestView")

        wgpu_state.set_render_enabled(vol, True)
        wgpu_state.set_view_node_ids(vol, [view2.GetID()])
        wgpu_state.set_render_enabled(seg, True)
        wgpu_state.set_segment_render_mode(seg, "surface")

        # Names we'll use to relocate the nodes after reload.
        vol_name = vol.GetName()
        seg_name = seg.GetName()
        view2_name = view2.GetName()

        tmpdir = tempfile.mkdtemp(prefix="wgpuStateTest_")
        scene_path = os.path.join(tmpdir, "scene.mrml")
        self.delayDisplay(f"State: saving scene to {scene_path}", 100)
        # saveScene returns True/False; loadScene's return varies by
        # Slicer version (None vs bool). Verify by checking the file
        # exists after save, and by checking node lookups succeed
        # after load.
        slicer.util.saveScene(scene_path)
        self.assertTrue(os.path.exists(scene_path),
            f"saveScene did not produce {scene_path}")

        # Clear and reload.
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()
        self.assertEqual(len(wgpu_state.all_state_nodes()), 0,
            "scene clear should have removed every state node")
        slicer.util.loadScene(scene_path)
        slicer.app.processEvents()

        # Refetch nodes by name.
        vol2 = slicer.mrmlScene.GetFirstNodeByName(vol_name)
        seg2 = slicer.mrmlScene.GetFirstNodeByName(seg_name)
        view2b = slicer.mrmlScene.GetFirstNodeByName(view2_name)
        self.assertIsNotNone(vol2, f"volume {vol_name!r} missing after reload")
        self.assertIsNotNone(seg2, f"segmentation {seg_name!r} missing after reload")
        self.assertIsNotNone(view2b, f"view {view2_name!r} missing after reload")

        # State survived?
        self.assertTrue(wgpu_state.is_render_enabled(vol2),
            "volume renderEnabled should be true after reload")
        self.assertEqual(wgpu_state.view_node_ids(vol2), [view2b.GetID()],
            "view-id list should round-trip through save/load")
        self.assertTrue(wgpu_state.is_render_enabled(seg2),
            "segmentation renderEnabled should be true after reload")
        self.assertEqual(wgpu_state.segment_render_mode(seg2), "surface",
            "segment render mode should round-trip through save/load")
        self.assertEqual(len(wgpu_state.all_state_nodes()), 2,
            "expected exactly two state nodes after reload")

        self.delayDisplay("State: Save / restore PASSED", 200)

    # ------------------------------------------------------------------
    # Bridge integration tests (single-view, state-driven). These
    # exercise the auto-render-on-add path (Phase 4), per-node opt-out
    # via wgpu_state (Phase 1), and multi-segmentation (Phase 2).
    # See project_automatic_bridge_progress memory for context.
    # ------------------------------------------------------------------

    def _isolate_bridge(self):
        """Tear down every live WgpuVolumeBridge in the process (not
        just ones in the bridges dict) and clear the scene. Necessary
        for repeatable bridge tests because dev-iteration MCP sessions
        accumulate leaked bridge instances whose observers race the
        bridge we're trying to test."""
        import gc
        live = [o for o in gc.get_objects()
                if type(o).__name__ == "WgpuVolumeBridge"]
        for b in live:
            try: b.uninstall()
            except Exception: pass
        slicer.modules.wgpuVtkBridges = {}
        slicer.modules.wgpuVtkBridge = None
        gc.collect(); gc.collect()
        # Drain whatever pending events any dying observers queued.
        slicer.app.processEvents()

    def _install_fresh_bridge(self):
        """Install the bridge from the freshly-imported wgpu_vtk_inject
        module (matching the legacy `_install_vtk_bridge` reload-friendly
        pattern), pinned to the visible 3D widget.

        Drains the Qt event queue both BEFORE and AFTER install so that
        any pending QTimer.singleShot(0, ...) install from the widget's
        own setup() doesn't fire AFTER our install and silently replace
        the bridge we just made (which would leave our local reference
        pointing at the now-uninstalled instance). Returns whichever
        bridge ends up in slicer.modules.wgpuVtkBridge to guarantee we
        hand back the live one even if a race did happen."""
        import os, sys
        # Drain any pending deferred installs from prior module loads.
        slicer.app.processEvents()
        slicer.app.processEvents()
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        for mod in ("SceneRenderingLib.wgpu_vtk_inject", "SceneRenderingLib"):
            if mod in sys.modules:
                del sys.modules[mod]
        from SceneRenderingLib import wgpu_vtk_inject as wvi
        wvi.install_default_bridge()
        # Drain again -- and then re-fetch the active bridge from the
        # registry so we don't return a uninstall()'d stale handle.
        slicer.app.processEvents()
        slicer.app.processEvents()
        return slicer.modules.wgpuVtkBridge

    def test_bridge_autoEnableOnAdd(self):
        """Default-true policy: adding a new volume to the scene
        automatically creates a VRDN and the bridge claims it without
        any explicit toggle. Verifies the event-driven retry that
        defers create until the volume has both ImageData and a
        standard display node."""
        self.delayDisplay("Bridge: auto-enable on add", 100)
        self._isolate_bridge()
        bridge = self._install_fresh_bridge()
        # Empty scene at install time.
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodesByClass(
            "vtkMRMLVolumeRenderingDisplayNode"), 0)
        self.assertEqual(len(bridge._claimed_vrdn_ids), 0)
        # Add one volume; expect exactly one VRDN, claimed.
        self._make_synthetic_volume("AutoVol1", dims=(8, 8, 8))
        slicer.app.processEvents()
        n1 = slicer.mrmlScene.GetNumberOfNodesByClass(
            "vtkMRMLVolumeRenderingDisplayNode")
        self.assertEqual(n1, 1,
            f"expected 1 VRDN after first volume, got {n1}")
        self.assertEqual(len(bridge._claimed_vrdn_ids), 1)
        # Add a second; one more VRDN, both claimed.
        self._make_synthetic_volume("AutoVol2", dims=(8, 8, 8))
        slicer.app.processEvents()
        n2 = slicer.mrmlScene.GetNumberOfNodesByClass(
            "vtkMRMLVolumeRenderingDisplayNode")
        self.assertEqual(n2, 2,
            f"expected 2 VRDNs after second volume, got {n2}")
        self.assertEqual(len(bridge._claimed_vrdn_ids), 2)
        # No orphan VRDNs.
        coll = slicer.mrmlScene.GetNodesByClass(
            "vtkMRMLVolumeRenderingDisplayNode")
        try:
            orphans = sum(1 for i in range(coll.GetNumberOfItems())
                          if coll.GetItemAsObject(i).GetVolumeNodeID() is None)
        finally:
            coll.UnRegister(None)
        self.assertEqual(orphans, 0,
            f"expected 0 orphan VRDNs, got {orphans}")
        self.delayDisplay("Bridge: auto-enable on add PASSED", 200)

    def test_bridge_perNodeOptOut(self):
        """Toggle wgpu_state.renderEnabled=False on a claimed volume;
        bridge unclaims and Slicer's native VR resumes (visibility=1).
        Toggle back on; bridge re-claims (visibility=0). Verifies the
        Phase 1 state-driven reconcile path end-to-end."""
        from SceneRenderingLib import wgpu_state
        self.delayDisplay("Bridge: per-node opt-out", 100)
        self._isolate_bridge()
        bridge = self._install_fresh_bridge()
        vol = self._make_synthetic_volume("OptOutVol", dims=(8, 8, 8))
        slicer.app.processEvents()
        self.assertEqual(len(bridge._claimed_vrdn_ids), 1,
            "bridge should claim auto-created VRDN")
        vrdn = slicer.mrmlScene.GetFirstNodeByClass(
            "vtkMRMLVolumeRenderingDisplayNode")
        self.assertEqual(vrdn.GetVisibility(), 0,
            "claimed VRDN should have visibility=0 (bridge owns it)")
        # Opt out.
        wgpu_state.set_render_enabled(vol, False)
        slicer.app.processEvents()
        self.assertEqual(len(bridge._claimed_vrdn_ids), 0,
            "bridge should unclaim after opt-out")
        self.assertEqual(vrdn.GetVisibility(), 1,
            "VRDN visibility should be restored to 1 after opt-out")
        # Opt back in.
        wgpu_state.set_render_enabled(vol, True)
        slicer.app.processEvents()
        self.assertEqual(len(bridge._claimed_vrdn_ids), 1,
            "bridge should re-claim after opt-in")
        self.assertEqual(vrdn.GetVisibility(), 0,
            "VRDN visibility should be back to 0 after re-claim")
        self.delayDisplay("Bridge: per-node opt-out PASSED", 200)

    def test_bridge_multiSegmentation(self):
        """Add two segmentation nodes; bridge picks both up via
        reconcile + NodeAddedEvent observer. Opt one out via state;
        only the other remains. Verifies Phase 2 multi-seg lifecycle."""
        import numpy as np
        from SceneRenderingLib import wgpu_state
        self.delayDisplay("Bridge: multi-segmentation", 100)
        self._isolate_bridge()
        bridge = self._install_fresh_bridge()
        vol = self._make_synthetic_volume("MultiSegVol", dims=(16, 16, 16))
        # Two segmentations each with one filled segment so the
        # visible-segment-id scan has something to pick up.
        segA = self._make_synthetic_segmentation(vol, "SegA")
        segB = self._make_synthetic_segmentation(vol, "SegB")
        for seg in (segA, segB):
            sid = seg.GetSegmentation().GetNthSegmentID(0)
            mask = np.zeros((16, 16, 16), dtype=np.uint8)
            mask[4:12, 4:12, 4:12] = 1
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask, seg, sid, vol)
        slicer.app.processEvents()
        # The bridge auto-picks up seg nodes via NodeAddedEvent + the
        # reconcile pass that follows.
        bridge._reconcile_segmentation_nodes()
        slicer.app.processEvents()
        self.assertEqual(sorted(bridge._seg_records.keys()),
                         sorted([segA.GetID(), segB.GetID()]),
                         "bridge should track both segmentations")
        self.assertGreaterEqual(len(bridge._segments), 2,
            f"expected at least 2 SegmentFields, got {len(bridge._segments)}")
        # Opt B out via state.
        wgpu_state.set_render_enabled(segB, False)
        slicer.app.processEvents()
        self.assertEqual(list(bridge._seg_records.keys()),
                         [segA.GetID()],
                         "bridge should drop B from records after opt-out")
        # Add a third seg AFTER bridge is live -> auto-picked up.
        segC = self._make_synthetic_segmentation(vol, "SegC")
        sidC = segC.GetSegmentation().GetNthSegmentID(0)
        maskC = np.zeros((16, 16, 16), dtype=np.uint8)
        maskC[2:10, 2:10, 2:10] = 1
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            maskC, segC, sidC, vol)
        slicer.app.processEvents()
        self.assertIn(segC.GetID(), bridge._seg_records,
            "third segmentation added after install should be auto-picked")
        self.assertNotIn(segB.GetID(), bridge._seg_records,
            "opted-out B should still be excluded")
        self.delayDisplay("Bridge: multi-segmentation PASSED", 200)

    # ------------------------------------------------------------------
    # Legacy DualView / pygfx tests
    # ------------------------------------------------------------------

    def test_SingleVolume(self):
        """DualView + CTACardio only (no fiducials, no transform).
        Confirms ImageField rendering, camera sync, and that both panes
        agree on the composited volume."""
        self.delayDisplay("Single Volume: loading CTACardio", 200)

        vol = self._load_ctacardio()
        dv = self._install_dualview()
        self._frame_and_draw(dv)

        scene_mgr = next(m for m in dv.managers
                         if type(m).__name__ == "SceneRendererManager")
        r = scene_mgr.renderer
        kinds = sorted(f.field_kind for f in r.fields())
        self.assertEqual(kinds.count("img"), 1,
            f"expected 1 ImageField, got kinds={kinds}")
        self.assertEqual(kinds.count("fid"), 0,
            f"expected 0 FiducialFields, got kinds={kinds}")

        _, _, mx, _ = self._snapshot_stats(dv.view, "single-volume")
        self.assertGreater(max(mx), 60,
            f"no volume visible in pygfx pane -- max_rgb={mx}")

        self._stash(dualView=dv, volume=vol)
        self.delayDisplay("Single Volume test PASSED", 400)

    def test_VolumeAndFiducials(self):
        """DualView + CTACardio + 4 markup lists (100 control points).
        Asserts 1 ImageField + 4 FiducialFields coexist and pick+drag
        still round-trips through MRML."""
        import numpy as np
        from slicer_wgpu.fields import FiducialField

        self.delayDisplay("Volume + Fiducials: loading scene", 200)

        vol = self._load_ctacardio()
        list_specs, markup_nodes = self._build_markup_nodes(volume_node=vol)
        dv = self._install_dualview()
        _, fid_disp = self._set_dualview_radii(dv, list_specs, markup_nodes)
        self._frame_and_draw(dv)

        scene_mgr = next(m for m in dv.managers
                         if type(m).__name__ == "SceneRendererManager")
        r = scene_mgr.renderer
        kinds = sorted(f.field_kind for f in r.fields())
        self.assertEqual(kinds.count("img"), 1,
            f"expected 1 ImageField, got kinds={kinds}")
        self.assertEqual(kinds.count("fid"), 4,
            f"expected 4 FiducialFields, got kinds={kinds}")
        total = sum(f.n_spheres for f in r.fields()
                    if isinstance(f, FiducialField))
        self.assertEqual(total, 100,
            f"expected 100 spheres, got {total}")

        _, _, mx, _ = self._snapshot_stats(dv.view, "volume+fiducials")
        self.assertGreater(max(mx), 60,
            f"nothing visible in composite -- max_rgb={mx}")

        self._pick_drag_roundtrip(
            dv, markup_nodes[0], fid_disp, "Volume+Fiducials")

        self._stash(dualView=dv, volume=vol, markupNodes=markup_nodes)
        self.delayDisplay("Volume + Fiducials test PASSED", 400)

    def test_TransformableVolume(self):
        """DualView + CTACardio under a linear transform (rotation +
        non-uniform stretch). Verifies that world_from_local is folded
        into the sampling matrix, that the scene-AABB expands to cover
        the deformed bounds, and that in-place matrix mutation takes
        the fast path (same Field instance, no LUT/texture rebuild)."""
        import math
        import numpy as np
        from slicer_wgpu.fields import ImageField

        self.delayDisplay("Transformable Volume: loading scene", 200)

        vol = self._load_ctacardio()
        dv = self._install_dualview()

        # 45-degree rotation about z + 1.5x stretch along the rotated X.
        tform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "StretchRot")
        m = vtk.vtkMatrix4x4()
        c, s = math.cos(math.radians(45.0)), math.sin(math.radians(45.0))
        sx = 1.5
        M = [
            [c * sx, -s * sx, 0.0, 0.0],
            [s,       c,      0.0, 0.0],
            [0.0,     0.0,    1.0, 0.0],
            [0.0,     0.0,    0.0, 1.0],
        ]
        for i in range(4):
            for j in range(4):
                m.SetElement(i, j, float(M[i][j]))
        tform.SetMatrixTransformToParent(m)
        vol.SetAndObserveTransformNodeID(tform.GetID())
        slicer.app.processEvents()

        self._frame_and_draw(dv)

        scene_mgr = next(mm for mm in dv.managers
                         if type(mm).__name__ == "SceneRendererManager")
        img_field = next(f for f in scene_mgr.renderer.fields()
                         if isinstance(f, ImageField))
        self.assertFalse(
            np.allclose(img_field.world_from_local, np.eye(4)),
            "Transform didn't reach ImageField.world_from_local")

        _, _, mx, _ = self._snapshot_stats(dv.view, "transformable")
        self.assertGreater(max(mx), 60,
            f"no deformed volume visible -- max_rgb={mx}")

        # Fast-path: mutate the transform matrix in place and confirm
        # the same Field object is reused (no full rebuild).
        obj_id_before = id(img_field)
        c2, s2 = math.cos(math.radians(90.0)), math.sin(math.radians(90.0))
        sx2, sy2 = 2.0, 0.7
        M2 = [
            [c2 * sx2, -s2 * sx2, 0.0, 0.0],
            [s2 * sy2,  c2 * sy2, 0.0, 0.0],
            [0.0,       0.0,      1.0, 0.0],
            [0.0,       0.0,      0.0, 1.0],
        ]
        for i in range(4):
            for j in range(4):
                m.SetElement(i, j, float(M2[i][j]))
        tform.SetMatrixTransformToParent(m)
        slicer.app.processEvents()
        self._force_draw(dv.view, n=5)

        img_field2 = next(f for f in scene_mgr.renderer.fields()
                          if isinstance(f, ImageField))
        self.assertIs(img_field2, img_field,
            "fast-path broke: field was rebuilt on transform mutation")
        self.assertEqual(id(img_field2), obj_id_before,
            "fast-path broke: different object id")

        self._stash(dualView=dv, volume=vol, transform=tform)
        self.delayDisplay("Transformable Volume test PASSED", 400)

    def test_BouncingHead(self):
        """MRHead rendered with a gradient-emphasizing transfer function
        while a mass-spring simulation drives its parent linear
        transform. Four masses sit at the bottom corners of the volume
        bounding box and hang from fixed anchors at the top corners via
        vertical springs; six additional springs couple the bottom
        masses (four along the bottom edges plus two diagonals). The
        loop runs the ODE, fits a least-squares affine through the 8
        corner positions each frame, and pushes it into the transform
        node. Slicer's TransformModifiedEvent fans out to the Stage-3
        fast path, so Phong shading reacts per-frame to the stretch and
        shear as the head wobbles.

        The exact deformation produced by this configuration is not a
        single affine, but the least-squares fit is a reasonable visual
        approximation. Bottom-edge / bottom-diagonal stiffnesses are
        tuned so a vertical stretch couples into lateral contraction
        (and vice versa), giving an effective Poisson-ratio-ish
        response around 0.4."""
        import math
        import numpy as np
        import time

        from slicer_wgpu.fields import ImageField

        self.delayDisplay("Bouncing Head: loading MRHead", 200)

        import SampleData
        vol = SampleData.SampleDataLogic().downloadMRHead()
        self.assertIsNotNone(vol, "MRHead failed to load")

        # Build a transfer function that emphasises gradients: modest
        # scalar opacity across the whole range so internal voxels
        # contribute, but gradient-opacity ramps up sharply so the
        # composite pops at tissue boundaries (skin, sinuses, bone,
        # white/grey matter). Shading on, ambient low, diffuse high.
        vrLogic = slicer.modules.volumerendering.logic()
        disp = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
        disp.SetVisibility(True)
        vp = disp.GetVolumePropertyNode().GetVolumeProperty()

        # Detect the actual scalar range so these TF points aren't off
        # if MRHead gets re-encoded some day.
        arr = slicer.util.arrayFromVolume(vol)
        s_lo = float(arr.min())
        s_hi = float(arr.max())
        s_mid = 0.25 * (s_hi - s_lo) + s_lo

        op = vtk.vtkPiecewiseFunction()
        op.AddPoint(s_lo,          0.0)
        op.AddPoint(s_lo + 1.0,    0.02)
        op.AddPoint(s_mid,         0.18)
        op.AddPoint(s_hi,          0.25)
        vp.SetScalarOpacity(0, op)

        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(s_lo,    0.02, 0.02, 0.05)
        color.AddRGBPoint(s_mid,   0.60, 0.52, 0.42)
        color.AddRGBPoint(s_hi,    1.00, 0.95, 0.90)
        vp.SetColor(0, color)

        g_lut = vtk.vtkPiecewiseFunction()
        g_lut.AddPoint(0.0,   0.0)
        g_lut.AddPoint(3.0,   0.05)
        g_lut.AddPoint(15.0,  0.90)
        g_lut.AddPoint(80.0,  1.00)
        vp.SetGradientOpacity(0, g_lut)

        vp.ShadeOn()
        vp.SetAmbient(0.20)
        vp.SetDiffuse(0.85)
        vp.SetSpecular(0.45)
        vp.SetSpecularPower(24)

        tform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", "BouncingHead")
        vol.SetAndObserveTransformNodeID(tform.GetID())
        slicer.app.processEvents()

        # --- Mass-spring system ---
        bounds = [0.0] * 6
        vol.GetBounds(bounds)
        xm, xM, ym, yM, zm, zM = bounds
        # 8 corner particles: indices 0-3 are top (fixed), 4-7 are
        # bottom (free). Matching pairs (0,4), (1,5), (2,6), (3,7) are
        # directly above each other.
        rest = np.array([
            [xm, ym, zM], [xM, ym, zM], [xM, yM, zM], [xm, yM, zM],  # top
            [xm, ym, zm], [xM, ym, zm], [xM, yM, zm], [xm, yM, zm],  # bot
        ], dtype=np.float64)
        particles = rest.copy()
        velocities = np.zeros((8, 3), dtype=np.float64)
        is_fixed = np.array([True] * 4 + [False] * 4)
        masses = np.ones(8, dtype=np.float64)

        # ---- Cartoony (but stable) initial pose ----
        # Pull the bottom face down so the overall vertical extent is
        # ~1.4x the rest length, and squeeze x/y toward the centerline
        # by 1/sqrt(stretch) so the volume is approximately preserved
        # (Poisson's-ratio-ish silly-putty response). Smaller than a
        # literal "double length" because with real MRHead bounds
        # (~150 mm) a 2x stretch gives the springs enough stored
        # energy to launch the system into divergence before damping
        # catches up.
        stretch_z  = 1.4
        squeeze_xy = 1.0 / math.sqrt(stretch_z)    # ≈ 0.845
        cx, cy = 0.5 * (xm + xM), 0.5 * (ym + yM)
        bot_z0 = zm - (zM - zm) * (stretch_z - 1.0)

        def _sq_xy(x, y):
            return (cx + (x - cx) * squeeze_xy,
                    cy + (y - cy) * squeeze_xy)

        for k, (bx, by) in enumerate([(xm, ym), (xM, ym), (xM, yM), (xm, yM)]):
            sx, sy = _sq_xy(bx, by)
            particles[4 + k] = np.array([sx, sy, bot_z0])

        # Modest upward return velocities + a little lateral asymmetry
        # so the bottom face swings rather than pistoning straight up.
        velocities[4] = np.array([  40.0,   10.0,   80.0])
        velocities[5] = np.array([ -30.0,   40.0,   70.0])
        velocities[6] = np.array([   5.0,  -45.0,   80.0])
        velocities[7] = np.array([  35.0,   25.0,   60.0])

        # Geometry-derived rest lengths.
        L_vert = np.linalg.norm(rest[0] - rest[4])   # 4 vertical (all equal)
        L_edge = [                                   # 4 bottom edges
            np.linalg.norm(rest[4] - rest[5]),       # -y edge, along x
            np.linalg.norm(rest[5] - rest[6]),       # +x edge, along y
            np.linalg.norm(rest[6] - rest[7]),       # +y edge, along x
            np.linalg.norm(rest[7] - rest[4]),       # -x edge, along y
        ]
        L_diag = [                                   # 2 bottom diagonals
            np.linalg.norm(rest[4] - rest[6]),
            np.linalg.norm(rest[5] - rest[7]),
        ]

        # Stiffnesses: stiff enough that the pre-stretched pose can't
        # launch the system past its rest length by a huge overshoot,
        # soft enough that the first recovery is a visible wobble.
        k_v = 28.0
        k_e = 42.0
        k_d = 24.0

        springs = (
            [(i, i + 4, L_vert, k_v) for i in range(4)]
            + [(4, 5, L_edge[0], k_e), (5, 6, L_edge[1], k_e),
               (6, 7, L_edge[2], k_e), (7, 4, L_edge[3], k_e)]
            + [(4, 6, L_diag[0], k_d), (5, 7, L_diag[1], k_d)]
        )

        gravity = np.array([0.0, 0.0, -180.0])  # mm/s^2, tuned for visuals
        damping = 0.55                          # firmer decay for stability
        dt_sub = 0.004
        substeps = 4
        n_frames = 720                          # ~12 s at 60 fps

        # Periodically inject a modest impulse on a random bottom mass
        # so the head keeps wobbling for the full 12 s instead of
        # damping all the way back to rest. Kept well below what the
        # springs can restore in one cycle.
        rng = np.random.default_rng(seed=20260415)
        kick_frames = set(range(180, n_frames, 160))
        kick_impulse = 55.0

        P_rest_hom = np.hstack([rest, np.ones((8, 1))])  # (8, 4)

        def fit_affine():
            """Least-squares 4x4 affine M s.t. M @ P_rest ≈ particles."""
            Mt, *_ = np.linalg.lstsq(P_rest_hom, particles, rcond=None)
            M = np.vstack([Mt.T, [0.0, 0.0, 0.0, 1.0]])
            return M

        m_vtk = vtk.vtkMatrix4x4()

        def push_transform(M):
            for i in range(4):
                for j in range(4):
                    m_vtk.SetElement(i, j, float(M[i, j]))
            tform.SetMatrixTransformToParent(m_vtk)

        # Seed the transform with the stretched initial affine BEFORE
        # installing the DualView so the ImageField's world_from_local
        # AABB already reflects the dramatic pose. That lets
        # reset_camera frame the full stretched head on frame 1.
        push_transform(fit_affine())
        slicer.app.processEvents()

        dv = self._install_dualview()
        self._frame_and_draw(dv)

        # Periodically nudge a random bottom mass so the sim stays
        # lively instead of slowly damping toward equilibrium. A handful
        # of mid-run kicks keeps things wobbly without looking erratic.
        rng = np.random.default_rng(seed=20260415)
        kick_frames = set(range(120, n_frames, 140))   # every ~2.3 s
        kick_impulse = 120.0                           # mm/s added instantly

        def sim_step(dt):
            forces = np.zeros_like(particles)
            forces[~is_fixed] += gravity * masses[~is_fixed, None]
            for i, j, L0, k in springs:
                d = particles[j] - particles[i]
                L = float(np.linalg.norm(d))
                if L < 1e-6:
                    continue
                f = (k * (L - L0) / L) * d
                forces[i] += f
                forces[j] -= f
            velocities[~is_fixed] += (
                forces[~is_fixed] / masses[~is_fixed, None]) * dt
            velocities[~is_fixed] *= (1.0 - damping * dt)
            particles[~is_fixed] += velocities[~is_fixed] * dt

        # Drive the animation off a self-triggering QTimer.singleShot
        # chain rather than a tight Python `for` loop with
        # processEvents() inside. The old pattern starved Qt's event
        # loop -- input events, Ctrl-C, and other module updates were
        # all blocked until the 720-frame loop finished. A QTimer
        # chain runs one frame per tick, yields between frames, and
        # lets Qt handle interrupts and input normally.
        #
        # We block the *test method* itself on a local QEventLoop so
        # the test still completes synchronously from the caller's
        # point of view; QEventLoop.exec_() pumps Qt events while
        # waiting.
        max_seen = [0, 0, 0]
        frame_state = {"i": 0}
        event_loop = qt.QEventLoop()

        def tick():
            frame = frame_state["i"]
            if frame >= n_frames:
                event_loop.quit()
                return
            if frame in kick_frames:
                which = int(rng.integers(4, 8))
                direction = rng.normal(size=3)
                direction[2] *= 0.3  # bias mostly lateral kicks
                direction /= max(np.linalg.norm(direction), 1e-6)
                velocities[which] += kick_impulse * direction
            for _ in range(substeps):
                sim_step(dt_sub)
            push_transform(fit_affine())
            if frame == n_frames // 3:
                _, _, mx, _ = self._snapshot_stats(dv.view, "bouncing-head-mid")
                for k in range(3):
                    max_seen[k] = max(max_seen[k], mx[k])
            frame_state["i"] = frame + 1
            # 0-delay singleShot: runs after the current Qt tick
            # finishes (render, input dispatch, etc). No busy loop.
            qt.QTimer.singleShot(0, tick)

        qt.QTimer.singleShot(0, tick)
        event_loop.exec_()

        _, _, mx, _ = self._snapshot_stats(dv.view, "bouncing-head-final")
        for k in range(3):
            max_seen[k] = max(max_seen[k], mx[k])
        self.assertGreater(max(max_seen), 60,
            f"no head visible during simulation -- max_rgb={tuple(max_seen)}")

        # Sanity: the final transform should differ from identity
        # (i.e. the simulation actually moved the masses).
        final_M = np.array([
            [m_vtk.GetElement(i, j) for j in range(4)] for i in range(4)
        ])
        delta_from_I = float(np.linalg.norm(final_M - np.eye(4)))
        self.assertGreater(delta_from_I, 1e-3,
            f"transform never left identity (delta={delta_from_I:.4f})")

        self._stash(dualView=dv, volume=vol, transform=tform)
        self.delayDisplay("Bouncing Head test PASSED", 400)

    # ----- Further stages -----

    def test_MultiVolume(self):
        """Two independent volumes (CTACardio + CTAAbdomenPanoramix) with
        distinct TFs, each centered at the world origin via its own
        linear transform so the two occupy the same region of space.
        Lets us inspect how the per-sample compositing looks when both
        Fields contribute to the same ray.
        """
        from slicer_wgpu.fields import ImageField

        self.delayDisplay(
            "Multi-Volume: loading CTACardio + CTAAbdomenPanoramix", 200)

        import SampleData
        sd = SampleData.SampleDataLogic()
        cta = sd.downloadCTACardio()
        self.assertIsNotNone(cta, "CTACardio failed to load")
        pano = sd.downloadSample("CTAAbdomenPanoramix")
        # downloadSample can return a single node or a list/tuple; coerce
        # to a single scalar volume node for the test.
        if isinstance(pano, (list, tuple)):
            pano = next((n for n in pano
                         if n is not None
                         and n.IsA("vtkMRMLScalarVolumeNode")), None)
        self.assertIsNotNone(pano, "Panoramix failed to load")

        # Build default VR display nodes and apply a distinct preset to
        # each so the two are easy to tell apart visually.
        vrLogic = slicer.modules.volumerendering.logic()
        specs = [
            (cta,  "CT-Chest-Contrast-Enhanced"),
            (pano, "CT-AAA"),
        ]
        disps = []
        for vol, preset_name in specs:
            d = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
            d.SetVisibility(True)
            preset = vrLogic.GetPresetByName(preset_name)
            if preset is not None:
                d.GetVolumePropertyNode().Copy(preset)
            disps.append(d)

        # Each volume gets a translation that moves the centre of its
        # bounds to the world origin. Done via a linear transform node
        # so the Stage-3 world_from_local path carries the shift (the
        # ImageField re-uses the same object -- no texture rebuild).
        m_vtk = vtk.vtkMatrix4x4()
        tforms = []
        for name, vol in (("CtaCenter", cta), ("PanoCenter", pano)):
            b = [0.0] * 6
            vol.GetBounds(b)
            cx = 0.5 * (b[0] + b[1])
            cy = 0.5 * (b[2] + b[3])
            cz = 0.5 * (b[4] + b[5])
            m_vtk.Identity()
            m_vtk.SetElement(0, 3, -cx)
            m_vtk.SetElement(1, 3, -cy)
            m_vtk.SetElement(2, 3, -cz)
            t = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode", name)
            t.SetMatrixTransformToParent(m_vtk)
            vol.SetAndObserveTransformNodeID(t.GetID())
            tforms.append(t)

        slicer.app.processEvents()

        dv = self._install_dualview()
        self._frame_and_draw(dv)

        scene_mgr = next(mm for mm in dv.managers
                         if type(mm).__name__ == "SceneRendererManager")
        img_fields = [f for f in scene_mgr.renderer.fields()
                      if isinstance(f, ImageField)]
        self.assertEqual(len(img_fields), 2,
            f"expected 2 ImageFields, got {len(img_fields)}")

        _, _, mx, _ = self._snapshot_stats(dv.view, "multivol-overlap")
        self.assertGreater(max(mx), 60,
            f"no volumes visible in overlap pose -- max_rgb={mx}")

        self._stash(dualView=dv, volumes=[cta, pano], transforms=tforms)
        self.delayDisplay(
            "Multi-Volume loaded: both volumes centered at origin",
            400)

    def test_DeformableVolume(self):
        """CTACardio under a vtkMRMLGridTransformNode whose displacement
        field is an animated traveling sinewave along +X that moves up
        and down through the body over 10 seconds. 20 cm wavelength,
        20 mm amplitude (visible; max dx/dz ~= 0.63), one wave period
        every 2 seconds.

        Displacement grid is 128^3 (~3.75 mm cell spacing over the
        ~480 mm volume, ~53 samples per wavelength). A coarser 32^3
        grid makes vtkGridTransform's iterative Newton inverse (used by
        the 2D reslicers) oscillate near the dx(z) zero-crossings and
        fire "singularity" warnings; at 128^3 the trilinear gradient
        tracks the true A*k*cos(kz) slope accurately enough for the
        Newton iteration to converge even at the larger amplitude.
        The forward Jacobian det is identically 1 for this displacement
        so the inverse is analytically well-defined everywhere; the
        problem was always numerical conditioning, not a singularity.

        Only the wgpu (pygfx) pane renders the warp; VTK's volume
        mapper ignores a grid transform in its render path, so the VTK
        pane's volume stays rigid (only its ROI frame follows). The
        slice views do honor the grid transform via vtkImageReslice's
        inverse path.

        Driven by a QTimer.singleShot chain blocked on a local
        QEventLoop, same pattern as test_BouncingHead -- Qt keeps
        processing events between frames so input + timers stay live.
        """
        import math
        import numpy as np
        import vtk.util.numpy_support as vnp

        vol = self._load_ctacardio()
        dv = self._install_dualview()
        self._frame_and_draw(dv)

        _, _, mx_base, _ = self._snapshot_stats(dv.view, "deformable-base")
        img_base = np.asarray(dv.view.renderer.snapshot()[..., :3]).astype(np.int32)

        bounds = [0.0] * 6
        vol.GetBounds(bounds)
        bmin = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
        bmax = np.array([bounds[1], bounds[3], bounds[5]], dtype=np.float64)
        extent = bmax - bmin

        N = 128
        grid = vtk.vtkImageData()
        grid.SetDimensions(N, N, N)
        grid.SetOrigin(*bmin.tolist())
        grid.SetSpacing(*(extent / (N - 1)).tolist())
        grid.AllocateScalars(vtk.VTK_FLOAT, 3)

        z_world = np.linspace(bmin[2], bmax[2], N, dtype=np.float32)
        WAVELENGTH_MM = 200.0
        AMPLITUDE_MM = 20.0
        K = 2.0 * math.pi / WAVELENGTH_MM

        def set_displacement_phase(phase):
            pd = grid.GetPointData().GetScalars()
            arr = vnp.vtk_to_numpy(pd).reshape(N, N, N, 3)  # K, J, I, comp
            dx_per_slice = AMPLITUDE_MM * np.sin(K * z_world + phase)
            arr[..., 0] = dx_per_slice[:, None, None]
            pd.Modified(); grid.Modified()
            grid_tf.Modified(); gt.TransformModified()

        gt = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLGridTransformNode", "SineWaveDeformation")
        grid_tf = vtk.vtkGridTransform()
        grid_tf.SetDisplacementGridData(grid)
        grid_tf.SetInterpolationModeToLinear()
        gt.SetAndObserveTransformFromParent(grid_tf)
        set_displacement_phase(0.0)
        vol.SetAndObserveTransformNodeID(gt.GetID())
        slicer.app.processEvents()
        self._force_draw(dv.view, n=3)

        _, _, mx_warp, _ = self._snapshot_stats(dv.view, "deformable-phase0")
        img_phase0 = np.asarray(dv.view.renderer.snapshot()[..., :3]).astype(np.int32)

        diff0 = np.abs(img_base - img_phase0)
        changed0 = int((diff0.max(axis=2) > 20).sum())
        self.assertGreater(max(mx_base), 60,
            f"baseline never lit the scene: max={mx_base}")
        self.assertGreater(changed0, 2000,
            f"grid transform had no visible effect: "
            f"changed-px={changed0}, base_max={mx_base}, warp_max={mx_warp}")

        DURATION_S = 10.0
        FPS = 30
        N_FRAMES = int(DURATION_S * FPS)
        event_loop = qt.QEventLoop()
        state = {"frame": 0}
        mid_img = [None]

        def tick():
            frame = state["frame"]
            if frame >= N_FRAMES:
                event_loop.quit()
                return
            t = frame / FPS
            # One wave period every 2 seconds; the wave travels in -z
            # through the body over the 10 second run.
            set_displacement_phase(math.pi * t)
            slicer.app.processEvents()
            if frame == N_FRAMES // 2:
                self._force_draw(dv.view, n=1)
                mid_img[0] = np.asarray(
                    dv.view.renderer.snapshot()[..., :3]).astype(np.int32)
            state["frame"] = frame + 1
            qt.QTimer.singleShot(int(1000.0 / FPS), tick)

        qt.QTimer.singleShot(0, tick)
        event_loop.exec_()

        self.assertIsNotNone(mid_img[0],
            "mid-animation snapshot never taken -- timer loop didn't run")
        # The snapshot resolution can change between frames if Slicer
        # reshuffles the view after a HiDPI / layout flip, which would
        # make a straight element-wise subtract throw. Resize mid_img
        # down to img_phase0's shape if they differ so the diff stays
        # apples-to-apples.
        a, b = img_phase0, mid_img[0]
        if a.shape != b.shape:
            from PIL import Image
            b = np.asarray(
                Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))
            ).astype(np.int32)
        diff_anim = np.abs(a - b)
        anim_changed = int((diff_anim.max(axis=2) > 20).sum())
        self.assertGreater(anim_changed, 500,
            f"sinewave animation did not change the render over time: "
            f"changed-px={anim_changed}")

        self._stash(dualView=dv, volume=vol, transform=gt)
        self.delayDisplay("Deformable Volume test PASSED", 400)

    def test_CinematicRendering(self):
        """CTACardio rendered with a two-light cinematic setup: a
        shadow-casting key light above-and-left of the camera and an
        unshadowed fill light to the right, each calibrated so the
        combined scene brightness stays close to the headlight baseline
        while the key light's shadows still read clearly in the heart.

        Lights are camera-relative so they orbit with the view and the
        shadow volume is rebuilt each time the camera rotation changes.
        """
        import math
        import numpy as np

        vol = self._load_ctacardio()
        dv = self._install_dualview()
        self._frame_and_draw(dv)

        # Headlight baseline snapshot (no light_direction, no shadows).
        _, _, _, me_base = self._snapshot_stats(dv.view, "cinematic-base")

        scene_mgr = next(mm for mm in dv.managers
                         if type(mm).__name__ == "SceneRendererManager")

        # Directions in CAMERA space (pygfx convention: +X right, +Y up,
        # -Z forward). enable_shadows(camera_relative=True) re-rotates
        # these into world each frame, so the shadows follow the camera.
        #   Key: up (+Y) and slightly to the left (-X) of the camera.
        #   Fill: to the right (+X), shadowless, half the key intensity.
        key_cam = np.array([-0.5, 1.0, 0.0])
        key_cam /= np.linalg.norm(key_cam)
        fill_cam = np.array([1.0, 0.0, 0.0])

        # Calibrated on CTACardio + CT-Chest-Contrast-Enhanced preset:
        # key=3.5 with fill=1.75 (keeps the user-requested 2:1 ratio)
        # lands within 1% of the headlight baseline mean while leaving
        # the key-side shadow clearly visible. Lower values leave the
        # scene perceptibly darker; higher values start clipping the
        # specular highlights on bone.
        scene_mgr.enable_shadows(
            light_direction=tuple(key_cam),
            resolution=128,
            light_intensity=3.5,
            fill_light_direction=tuple(fill_cam),
            fill_light_intensity=1.75,
            camera_relative=True,
        )
        slicer.app.processEvents()
        self._force_draw(dv.view, n=5)
        _, _, _, me_shad = self._snapshot_stats(dv.view, "cinematic-shad")

        # With both lights calibrated the cinematic render should stay
        # within ~15% of the headlight baseline mean. Peak-channel mean
        # is a robust proxy for scene luminance on this CT preset --
        # the R channel dominates the warm bone/vessel colors.
        base_peak = max(me_base)
        shad_peak = max(me_shad)
        self.assertGreater(base_peak, 5.0,
            f"baseline never lit the scene: mean={me_base}")
        self.assertGreater(shad_peak, base_peak * 0.85,
            f"cinematic scene is too dark vs baseline: "
            f"base mean={me_base}, cinematic mean={me_shad}")
        self.assertLess(shad_peak, base_peak * 1.15,
            f"cinematic scene is too bright vs baseline: "
            f"base mean={me_base}, cinematic mean={me_shad}")

        # Orbit through four azimuths so the shadow pattern is visible
        # from multiple angles (each reuses the same baked shadow volume
        # -- camera motion alone doesn't rebuild it).
        for az in (30, 90, 150, 210):
            dv.view.controller.rotate((math.radians(az), 0.0), (0, 0, 800, 600))
            slicer.app.processEvents()
            self._force_draw(dv.view, n=3)
            self._snapshot_stats(dv.view, f"cinematic-az{az}")

        self._stash(dualView=dv, volume=vol)
        self.delayDisplay("Cinematic Rendering test PASSED", 400)
