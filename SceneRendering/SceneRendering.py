"""SceneRendering -- a Slicer module that exposes the slicer_wgpu
SceneRenderer (Field-compositing ray tracer) and runs end-to-end
self-tests.

Self-tests progress from the simplest verifiable pipeline upward, so
that a failure at any step pin-points which layer of the stack is
broken:

    Step A  bare PygfxView + pygfx.Mesh cube
            -- confirms WgpuRenderer + QRenderWidget plumbing works.
    Step B  SceneRenderer with a single giant FiducialField sphere
            -- confirms dynamic WGSL codegen + material-class minting +
               geometryless fullscreen-triangle pipeline.
    Step C  SceneRenderer with 4 coloured spheres at distinct positions
            -- confirms per-sphere uniform indexing and per-sphere
               colouring in the shader.
    Step D  pick_at() + drag_continue() directly on the SceneRenderer
            -- confirms the picking/drag math round-trips without any
               MRML involvement.
    Step E  mrml_bridge.install() DualView with 4 Markups nodes only
            (no volume) -- confirms the side-by-side pygfx/VTK layout,
            the Displayer -> Field machinery, and pick+drag round-trip
            from pygfx back into MRML.
    Step F  DualView + CTACardio volume rendering + the same 4 Markups
            nodes -- confirms ImageField + FiducialField co-exist in
            the same SceneRenderer, compositing works, and pick+drag
            still lands on a sphere when the volume is in front.

Running test_SceneRenderingFiducials executes all six in order.
"""

import logging
import os

import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


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
Field-compositing Scene Renderer (slicer_wgpu): one pygfx ray tracer
draws every contributing MRML node (volumes, markups, ...) by sampling
each Field at every ray step and compositing per-sample. The self-test
exercises the fiducial path in isolation.
"""
        self.parent.acknowledgementText = """
Built on slicer-wgpu (https://github.com/pieper/slicer-wgpu) and pygfx.
"""


#
# SceneRenderingWidget
#

class SceneRenderingWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SceneRendering.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = SceneRenderingLogic()

        self.ui.installButton.connect('clicked(bool)', self.onInstall)
        self.ui.uninstallButton.connect('clicked(bool)', self.onUninstall)

    def onInstall(self):
        self.logic.install()

    def onUninstall(self):
        self.logic.uninstall()


#
# SceneRenderingLogic
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

    # Subwindow held alive across steps so the user can interactively
    # inspect the state left by the test.
    _view = None

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SceneRenderingFiducials()

    # ----- Dependency bootstrap -----

    def _ensure_dependencies(self):
        """Install/refresh wgpu, pygfx, rendercanvas (PythonQt patch),
        and slicer_wgpu from source archives."""
        import importlib, sys

        for pkg in ("numpy", "wgpu", "pygfx"):
            try:
                importlib.import_module(pkg)
            except ImportError:
                self.delayDisplay(f"pip-installing {pkg}", 100)
                slicer.util.pip_install(pkg)

        needs_rendercanvas_fork = True
        try:
            import rendercanvas.qt as _rcqt
            try:
                src = open(_rcqt.__file__).read()
            except Exception:
                src = ""
            if src and "is_pythonqt" in src:
                needs_rendercanvas_fork = False
        except ImportError:
            pass
        if needs_rendercanvas_fork:
            self.delayDisplay("Installing pieper/rendercanvas (PythonQt patch)", 100)
            slicer.util.pip_install(
                "https://github.com/pieper/rendercanvas/archive/refs/heads/pythonqt-support.zip"
            )

        self.delayDisplay("Installing pieper/slicer-wgpu", 100)
        slicer.util.pip_install(
            "https://github.com/pieper/slicer-wgpu/archive/refs/heads/main.zip"
        )

        # Reload slicer_wgpu only. Popping rendercanvas.* here would leave
        # pygfx holding a stale reference to the old rendercanvas.base
        # BaseRenderCanvas (imported at module-load time), which then fails
        # isinstance() against the freshly-imported widget class.
        for mod_name in [m for m in list(sys.modules)
                         if m == "slicer_wgpu" or m.startswith("slicer_wgpu.")]:
            sys.modules.pop(mod_name, None)

    # ----- Helpers -----

    def _enable_debug_logging(self):
        """Turn on pygfx + wgpu logs at INFO so compilation errors and
        validation warnings are visible in the Slicer console. Returns the
        list of (logger, prev_level) pairs so the caller can restore."""
        import logging
        prev = []
        for name in ("pygfx", "wgpu"):
            lg = logging.getLogger(name)
            prev.append((lg, lg.level))
            lg.setLevel(logging.INFO)
            if not lg.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("[%(name)s:%(levelname)s] %(message)s"))
                lg.addHandler(h)
        return prev

    def _force_draw(self, view, n=3):
        """Pump the event loop and request multiple draws so pygfx/wgpu
        has definitely rendered before we snapshot."""
        slicer.app.processEvents()
        for _ in range(n):
            view.request_redraw()
            try:
                view.widget.force_draw()
            except Exception:
                pass
            slicer.app.processEvents()

    def _snapshot_stats(self, view, label=""):
        """Snapshot the current frame and return (rgba_array, min_rgb,
        max_rgb, mean_rgb). Also log them."""
        import numpy as np
        img = view.renderer.snapshot()
        rgb = np.asarray(img[..., :3])
        mn = tuple(int(x) for x in rgb.min(axis=(0, 1)))
        mx = tuple(int(x) for x in rgb.max(axis=(0, 1)))
        me = tuple(float(x) for x in rgb.mean(axis=(0, 1)))
        logging.info(f"[SceneRendering:{label}] "
                     f"shape={img.shape} min={mn} max={mx} mean={me}")
        return img, mn, mx, me

    def _new_view(self, width=480, height=360):
        """Build a bare PygfxView (no DualView, no managers) sized for
        snapshots. Left as self._view for interactive inspection."""
        from slicer_wgpu.mrml_bridge import PygfxView
        if self._view is not None:
            try:
                self._view.close()
            except Exception:
                pass
            try:
                self._view.widget.deleteLater()
            except Exception:
                pass
            self._view = None
        view = PygfxView()
        view.widget.resize(width, height)
        view.widget.show()
        slicer.app.processEvents()
        self._view = view
        return view

    def _log_adapter_info(self):
        try:
            import wgpu
            try:
                adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            except Exception:
                adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
            info = adapter.info if hasattr(adapter, "info") else {}
            logging.info(f"[SceneRendering] wgpu adapter info: {info}")
        except Exception as e:
            logging.info(f"[SceneRendering] adapter-info lookup failed: {e}")

    # ----- Steps -----

    def _step_a_mesh_baseline(self):
        """Confirm that a plain pygfx.Mesh renders into the snapshot
        texture at all. If this fails, the renderer itself is broken
        and subsequent steps can't diagnose SceneRenderer issues."""
        import numpy as np
        import pygfx

        self.delayDisplay("Step A: pygfx.Mesh baseline", 200)
        view = self._new_view()

        mesh = pygfx.Mesh(
            pygfx.box_geometry(80, 80, 80),
            pygfx.MeshPhongMaterial(color="#ff8000"),
        )
        view.scene.add(mesh)
        view.camera.local.position = (0, 200, 0)
        view.camera.look_at((0, 0, 0))
        self._force_draw(view)

        img, mn, mx, me = self._snapshot_stats(view, "step-A mesh")
        # The orange cube should push at least one channel above the
        # background (~13 on the default 0.05 clear colour).
        self.assertGreater(max(mx), 60,
            f"Step A: pygfx.Mesh didn't render -- max_rgb={mx}")

        view.scene.remove(mesh)
        self._force_draw(view)
        return view

    def _step_b_single_sphere(self, view):
        """Build a SceneRenderer with one huge FiducialField sphere and
        confirm it paints pixels. This is the narrowest test of the
        Field-compositing pipeline."""
        import numpy as np
        import pygfx

        from slicer_wgpu.fields import FiducialField
        from slicer_wgpu.scene_renderer import SceneRenderer

        self.delayDisplay("Step B: SceneRenderer + 1 giant sphere", 200)

        fid = FiducialField(
            centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            radii=np.array([40.0], dtype=np.float32),
            colors=np.array([[1.0, 0.2, 0.2, 1.0]], dtype=np.float32),
        )
        renderer = SceneRenderer.build_for_fields([fid])
        # Useful diagnostics:
        logging.info(f"[SceneRendering] SceneRenderer class={type(renderer).__name__} "
                     f"material={type(renderer.material).__name__}")
        logging.info(f"[SceneRendering] bounds_min={renderer.material.scene_bounds_min} "
                     f"bounds_max={renderer.material.scene_bounds_max} "
                     f"sample_step={renderer.material.sample_step}")
        wgsl = renderer._shader_wgsl
        logging.info(f"[SceneRendering] generated WGSL length={len(wgsl)} "
                     f"contains sample_field_fid0={'sample_field_fid0' in wgsl}")

        view.scene.add(renderer)
        view.camera.local.position = (0, 180, 0)
        view.camera.look_at((0, 0, 0))
        self._force_draw(view)

        img, mn, mx, me = self._snapshot_stats(view, "step-B single-sphere")
        self.assertGreater(max(mx), 60,
            f"Step B: FiducialField sphere didn't render -- max_rgb={mx}\n"
            f"First 400 chars of generated WGSL:\n{wgsl[:400]}")
        # Red sphere: R channel should dominate.
        self.assertGreater(mx[0], mx[1],
            f"Step B: red channel didn't dominate -- max_rgb={mx}")
        self.assertGreater(mx[0], mx[2],
            f"Step B: red channel didn't dominate blue -- max_rgb={mx}")

        view.scene.remove(renderer)
        self._force_draw(view)
        return renderer

    def _step_c_multi_sphere(self, view):
        """Multiple spheres with distinct colours -- verifies per-sphere
        uniform indexing."""
        import numpy as np

        from slicer_wgpu.fields import FiducialField
        from slicer_wgpu.scene_renderer import SceneRenderer

        self.delayDisplay("Step C: SceneRenderer + 4 coloured spheres", 200)

        centers = np.array([
            [-60.0,   0.0,   0.0],  # red
            [ 60.0,   0.0,   0.0],  # green
            [  0.0,   0.0, -60.0],  # blue
            [  0.0,   0.0,  60.0],  # yellow
        ], dtype=np.float32)
        radii = np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float32)
        colors = np.array([
            [1.0, 0.15, 0.15, 1.0],
            [0.15, 1.0, 0.15, 1.0],
            [0.15, 0.25, 1.0, 1.0],
            [1.0, 0.85, 0.15, 1.0],
        ], dtype=np.float32)

        fid = FiducialField(centers=centers, radii=radii, colors=colors)
        renderer = SceneRenderer.build_for_fields([fid])
        view.scene.add(renderer)

        # Frame the scene from above.
        view.camera.local.position = (0, 280, 30)
        view.camera.look_at((0, 0, 0))
        self._force_draw(view)

        img, mn, mx, me = self._snapshot_stats(view, "step-C multi-sphere")
        # All 3 channels should have been pushed above background by at
        # least one of the spheres.
        for ch, name in enumerate("RGB"):
            self.assertGreater(mx[ch], 80,
                f"Step C: channel {name} never exceeded 80 -- max_rgb={mx}")

        # Keep state for step D.
        return renderer, fid

    def _step_d_pick_and_drag(self, view, renderer, fid):
        """Pick the first sphere at its known NDC location, then drag
        it 0.2 NDC units in X and confirm the FiducialField state
        advances."""
        import numpy as np

        self.delayDisplay("Step D: pick + drag", 200)

        cam = view.camera
        sz = view.widget.get_logical_size()
        self.assertGreater(sz[0], 0)

        # Project sphere 0 (at x=-60) to NDC.
        proj = np.asarray(cam.projection_matrix, dtype=np.float64)
        vm = np.asarray(cam.world.inverse_matrix, dtype=np.float64)
        p_world = np.array([-60.0, 0.0, 0.0, 1.0], dtype=np.float64)
        clip = proj @ vm @ p_world
        ndc = clip[:3] / clip[3]

        hit = renderer.pick_at(float(ndc[0]), float(ndc[1]), cam, sz)
        self.assertIsNotNone(hit, f"Step D: pick_at missed sphere 0 at NDC={ndc}")
        self.assertIs(hit.field, fid, "Step D: pick_at hit the wrong field")
        self.assertEqual(hit.item_index, 0, "Step D: pick_at hit the wrong sphere")

        before = fid.get_center(0).copy()
        moved = renderer.drag_continue(hit, float(ndc[0]) + 0.2, float(ndc[1]),
                                       cam, sz)
        self.assertTrue(moved, "Step D: drag_continue reported no change")
        after = fid.get_center(0)
        delta = float(np.linalg.norm(after - before))
        self.assertGreater(delta, 1.0,
            f"Step D: sphere barely moved (delta={delta:.3f}mm)")
        logging.info(f"[SceneRendering] Step D drag delta={delta:.2f}mm "
                     f"({before.tolist()} -> {after.tolist()})")

    # ----- MRML / DualView helpers -----

    def _build_markup_nodes(self, volume_node=None):
        """Create 4 markup fiducial nodes with 25 random control points
        each (100 total). If `volume_node` is given, points are sampled
        from its bounding box so they sit in/around the volume; otherwise
        they span a fixed world cube. Returns (list_specs, markup_nodes).
        """
        import numpy as np

        list_specs = [
            ("MarkupsRed",    (0.95, 0.20, 0.20),  5.0),
            ("MarkupsGreen",  (0.20, 0.85, 0.30),  3.5),
            ("MarkupsBlue",   (0.20, 0.45, 0.95),  7.0),
            ("MarkupsYellow", (0.95, 0.85, 0.10),  2.5),
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

    def _dualview_pygfx_view(self, dv):
        """Return the underlying PygfxView from an installed DualView."""
        return dv.view

    def _set_dualview_radii(self, dv, list_specs, markup_nodes):
        from slicer_wgpu.displayers.fiducial import FiducialDisplayer
        scene_mgr = next(m for m in dv.managers
                         if type(m).__name__ == "SceneRendererManager")
        fid_disp = next(d for d in scene_mgr._displayers
                        if type(d).__name__ == "FiducialDisplayer")
        for (_name, _color, radius), mn in zip(list_specs, markup_nodes):
            fid_disp.set_default_radius(mn.GetID(), radius)
        return scene_mgr, fid_disp

    def _pick_drag_roundtrip(self, dv, markup_node, fid_disp, label):
        """Pick the first control point of `markup_node`, drag it 0.1 NDC
        units, commit back to MRML, and assert the MRML position moved.
        Reuses DualView's camera/view -- this is the integration test."""
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

        # Rely on DualView's install-time framing (which now also pushes
        # the pygfx camera back to MRML so both panes agree). Just make
        # sure a frame has rendered with the current uniform state.
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
            f"{label}: pick missed control point 0 of {markup_node.GetName()} "
            f"at NDC={ndc.tolist()}")
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

    # ----- DualView steps -----

    def _step_e_dualview_fiducials(self):
        """Install the DualView (pygfx + VTK side-by-side), add 4 markup
        nodes with 100 total points, assert the SceneRenderer ends up
        with 4 FiducialFields, and exercise pick+drag through the
        manager. No volume rendering yet."""
        import numpy as np

        from slicer_wgpu import mrml_bridge
        from slicer_wgpu.fields import FiducialField

        self.delayDisplay("Step E: DualView with fiducials (no volume)", 200)

        # Tear down the standalone view from earlier steps so the
        # DualView install gets a clean Qt layout.
        if self._view is not None:
            try: self._view.close()
            except Exception: pass
            try: self._view.widget.deleteLater()
            except Exception: pass
            self._view = None

        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

        list_specs, markup_nodes = self._build_markup_nodes()

        dv = mrml_bridge.install()
        self.assertIsNotNone(dv.view, "Step E: DualView didn't instantiate PygfxView")
        slicer.app.processEvents()

        scene_mgr, fid_disp = self._set_dualview_radii(
            dv, list_specs, markup_nodes)

        # Radii changed after install, so AABB did too -- reframe and
        # resync both cameras, then let the scene redraw.
        dv.view.reset_camera()
        dv._sync_camera_to_mrml()
        self._force_draw(dv.view)

        r = scene_mgr.renderer
        kinds = sorted(f.field_kind for f in r.fields())
        n_img = kinds.count("img")
        n_fid = kinds.count("fid")
        self.assertEqual(n_img, 0,
            f"Step E: expected 0 ImageFields (no volume loaded), got {n_img}")
        self.assertEqual(n_fid, 4,
            f"Step E: expected 4 FiducialFields, got {n_fid}")
        total = sum(f.n_spheres for f in r.fields()
                    if isinstance(f, FiducialField))
        self.assertEqual(total, 100,
            f"Step E: expected 100 spheres, got {total}")

        img, mn, mx, me = self._snapshot_stats(dv.view, "step-E dualview-fid")
        self.assertGreater(max(mx), 60,
            f"Step E: no fiducials visible -- max_rgb={mx}")

        # Pick+drag round-trip on the red list.
        self._pick_drag_roundtrip(dv, markup_nodes[0], fid_disp, "Step E")

        slicer.modules.sceneRenderingTestDualView = dv

    def _step_f_dualview_with_volume(self):
        """Reinstall the DualView after loading CTACardio + volume
        rendering, then drop in the same 4 Markups nodes. Assert the
        SceneRenderer ends up with 1 ImageField + 4 FiducialFields,
        and pick+drag still resolves a sphere with the volume composed
        underneath."""
        import numpy as np
        import SampleData

        from slicer_wgpu import mrml_bridge
        from slicer_wgpu.fields import FiducialField

        self.delayDisplay("Step F: DualView + volume + fiducials", 200)

        # Start from a clean DualView and a clean scene.
        try:
            mrml_bridge.uninstall()
        except Exception:
            pass
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

        self.delayDisplay("Step F: downloading CTACardio", 100)
        vol = SampleData.SampleDataLogic().downloadCTACardio()
        self.assertIsNotNone(vol, "Step F: CTACardio sample data failed to load")

        vrLogic = slicer.modules.volumerendering.logic()
        disp = vrLogic.CreateDefaultVolumeRenderingNodes(vol)
        disp.SetVisibility(True)
        preset = vrLogic.GetPresetByName("CT-Chest-Contrast-Enhanced")
        if preset is not None:
            disp.GetVolumePropertyNode().Copy(preset)
        slicer.app.processEvents()

        list_specs, markup_nodes = self._build_markup_nodes(volume_node=vol)

        dv = mrml_bridge.install()
        self.assertIsNotNone(dv.view, "Step F: DualView didn't instantiate PygfxView")
        slicer.app.processEvents()

        scene_mgr, fid_disp = self._set_dualview_radii(
            dv, list_specs, markup_nodes)

        # Reframe after radii change so both panes see the scene, and
        # push back to MRML so the side-by-side VTK view matches.
        dv.view.reset_camera()
        dv._sync_camera_to_mrml()
        self._force_draw(dv.view, n=5)

        r = scene_mgr.renderer
        kinds = sorted(f.field_kind for f in r.fields())
        n_img = kinds.count("img")
        n_fid = kinds.count("fid")
        self.assertEqual(n_img, 1,
            f"Step F: expected 1 ImageField, got {n_img}")
        self.assertEqual(n_fid, 4,
            f"Step F: expected 4 FiducialFields, got {n_fid}")
        total = sum(f.n_spheres for f in r.fields()
                    if isinstance(f, FiducialField))
        self.assertEqual(total, 100,
            f"Step F: expected 100 spheres, got {total}")

        img, mn, mx, me = self._snapshot_stats(dv.view, "step-F dualview-vol-fid")
        self.assertGreater(max(mx), 60,
            f"Step F: no content visible in composite snapshot -- max_rgb={mx}")

        # Pick+drag round-trip. Even with the volume in the scene, the
        # FiducialField's SDF pass should still register a sphere hit in
        # front of any volume contribution on the same ray.
        self._pick_drag_roundtrip(dv, markup_nodes[0], fid_disp, "Step F")

        slicer.modules.sceneRenderingTestDualView = dv

    # ----- Entry points -----

    def test_SceneRenderingFiducials(self):
        """Run all six diagnostic steps."""
        self.delayDisplay("Starting SceneRendering fiducial diagnostic", 100)
        self._ensure_dependencies()
        self._enable_debug_logging()
        self._log_adapter_info()

        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()

        view = self._step_a_mesh_baseline()
        renderer = self._step_b_single_sphere(view)
        renderer, fid = self._step_c_multi_sphere(view)
        self._step_d_pick_and_drag(view, renderer, fid)
        self._step_e_dualview_fiducials()
        self._step_f_dualview_with_volume()

        self.delayDisplay("SceneRendering fiducial test PASSED", 500)
