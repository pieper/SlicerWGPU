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

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


MODULE_NAME = "SceneRendering"


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
        ("Injection: Colorize (RGBA)",         "test_vtk_ColorizeRGBA"),
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

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        container = qt.QGroupBox("Self-tests")
        vbox = qt.QVBoxLayout(container)

        header = qt.QLabel(
            "Click a button to reload this module and run that self-test.")
        header.setWordWrap(True)
        vbox.addWidget(header)

        # --- VTK-injection self-tests (new path, runs in default 3D view) ---
        vtk_label = qt.QLabel("<b>VTK injection (new)</b>")
        vbox.addWidget(vtk_label)
        for label, method_name in self.VTK_TESTS:
            btn = qt.QPushButton(label)
            btn.setToolTip(f"Reload SceneRendering and run {method_name}()")
            btn.clicked.connect(
                lambda _checked=False, m=method_name: self.onRunTest(m))
            vbox.addWidget(btn)

        # --- Legacy DualView/pygfx self-tests ---
        legacy_label = qt.QLabel("<b>DualView / pygfx (legacy)</b>")
        vbox.addWidget(legacy_label)
        for label, method_name in self.TESTS:
            btn = qt.QPushButton(label)
            btn.setToolTip(f"Reload SceneRendering and run {method_name}()")
            # Capture method_name by default-arg to avoid late-binding on
            # the loop variable.
            btn.clicked.connect(
                lambda _checked=False, m=method_name: self.onRunTest(m))
            vbox.addWidget(btn)

        # Opt-in force-reinstall: by default _ensure_dependencies only
        # pip-installs when a dep isn't importable, because the github
        # zip URLs cache and re-downloading every run is slow. Toggle
        # this when you want the next test button press to pull a
        # fresh build of pieper/rendercanvas or pieper/slicer-wgpu.
        self._forceReinstallCheck = qt.QCheckBox(
            "Force-reinstall deps from GitHub on next test")
        self._forceReinstallCheck.setToolTip(
            "When checked, the next Self-test run will pass "
            "--force-reinstall --no-cache-dir to pip for "
            "pieper/rendercanvas and pieper/slicer-wgpu. The box "
            "unchecks itself automatically after the run.")
        vbox.addWidget(self._forceReinstallCheck)

        self.layout.addWidget(container)
        self.layout.addStretch(1)

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
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

    def runTest(self):
        """Slicer's standard entry: run every implemented test in
        sequence. Invoked by the built-in `Reload & Test` button."""
        for name in (
            "test_SingleVolume",
            "test_VolumeAndFiducials",
            "test_TransformableVolume",
        ):
            self.setUp()
            getattr(self, name)()

    def runTestByName(self, test_method_name: str) -> None:
        """Run a single test by name. Used by the module UI buttons."""
        self.setUp()
        getattr(self, test_method_name)()

    # ----- Dependency bootstrap -----

    def _ensure_dependencies(self):
        """Make sure numpy / wgpu / pygfx / rendercanvas (PythonQt branch)
        / slicer_wgpu are importable. pip-install is only invoked when
        an import actually fails, so a live-development workflow that
        pushes files via the MCP /file endpoint isn't clobbered."""
        import importlib
        import sys

        for pkg in ("numpy", "wgpu"):
            try:
                importlib.import_module(pkg)
            except ImportError:
                self.delayDisplay(f"pip-installing {pkg}", 100)
                slicer.util.pip_install(pkg)

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
        view = lm.threeDWidget(0).threeDView()
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
