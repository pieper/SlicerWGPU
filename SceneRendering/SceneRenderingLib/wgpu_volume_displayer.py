"""WgpuVolumeStateDisplayer: mirrors independent wgpu volume-render state
nodes (see wgpu_volume_render.py) onto ImageFields, with NO dependency on
Slicer's volume-rendering display node or volume-property node.

It reuses ImageField.from_volume_node ONLY for the geometry/volume-upload
plumbing (bounds, patient-to-texture matrix, data range), then overwrites
the colour / opacity / gradient LUTs and shading from our own control-point
transfer function. The expensive 3D volume texture is built once and kept
across TF edits -- preset swaps cost O(LUT size), not O(voxels).

One displayer instance lives per bridge (per 3D view); it stamps each field's
`visible` flag from renderEnabled AND this view's id against the node's view
scoping, so the same state node renders only in the views it targets.
"""

import numpy as np
import vtk

from slicer_wgpu.displayers.base import Displayer
from slicer_wgpu.fields.image import ImageField

from . import wgpu_volume_render as wvr


def _vtk_to_numpy(m):
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)],
                    dtype=np.float64)


def _world_from_local(volume_node):
    t_node = volume_node.GetParentTransformNode()
    if t_node is None:
        return np.eye(4, dtype=np.float64)
    m = vtk.vtkMatrix4x4()
    t_node.GetMatrixTransformToWorld(m)
    return _vtk_to_numpy(m)


SLICE_COMPOSITE_LAYOUT = "Red"


def red_slice_composite():
    """The vtkMRMLSliceCompositeNode for the Red slice, or None. Resolved by
    LayoutName so it works even when the Red view isn't in the current
    layout -- the composite node persists in the scene regardless."""
    import slicer
    for comp in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
        if comp.GetLayoutName() == SLICE_COMPOSITE_LAYOUT:
            return comp
    return None


def composite_opacity(volume_id, comp):
    """Render-opacity for a volume from the Red slice composite node:
    background -> 1.0, foreground -> the ForegroundOpacity slider value,
    anything else -> 0.0 (hidden, so the 3D view shows only what's being
    compared in the slice viewer). Falls back to 1.0 when there is no Red
    composite, so a slice-less pure-3D setup still renders the field."""
    if comp is None:
        return 1.0
    if volume_id and volume_id == comp.GetBackgroundVolumeID():
        return 1.0
    if volume_id and volume_id == comp.GetForegroundVolumeID():
        return float(comp.GetForegroundOpacity())
    return 0.0


class WgpuVolumeStateDisplayer(Displayer):
    # Same MRML class as the legacy wgpu_state control nodes, so _make_field
    # filters by ModuleName (wvr.is_managed). Non-managed scripted module
    # nodes return None and never get a field.
    node_class = "vtkMRMLScriptedModuleNode"

    def __init__(self, *args, view_node_id=None, **kwargs):
        # Set BEFORE super().__init__ -- the base scans the scene and may
        # build fields during construction, and _apply_state reads it.
        self.view_node_id = view_node_id
        super().__init__(*args, **kwargs)

    # -------- view scoping --------

    def _targets_this_view(self, state):
        view_ids = wvr.view_node_ids(state)
        if not view_ids:
            return True
        if self.view_node_id is None:
            return False
        return self.view_node_id in view_ids

    # -------- Displayer hooks --------

    def _make_field(self, node):
        if not wvr.is_managed(node):
            return None
        vol = wvr.target_node(node)
        if vol is None or not vol.IsA("vtkMRMLScalarVolumeNode"):
            return None
        if vol.GetImageData() is None:
            return None
        # Geometry + volume upload + default ramp come from the library;
        # we immediately swap in our own transfer function below.
        field = ImageField.from_volume_node(vol, None)
        self._apply_state(field, node, vol)
        return field

    def _update_field(self, node, field):
        # Re-apply TF / shading / transform on the SAME field instance and
        # report non-structural (False) so the renderer just refills
        # uniforms and redraws without re-uploading the volume texture.
        if not wvr.is_managed(node):
            return False
        vol = wvr.target_node(node)
        if vol is None or vol.GetImageData() is None:
            return False
        self._apply_state(field, node, vol)
        return False

    def _extra_watch(self, node, tags):
        if not wvr.is_managed(node):
            return
        vol = wvr.target_node(node)
        if vol is not None:
            # Follow the volume under transform edits / re-parenting.
            tags.append((vol, vol.AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._handle_node_modified)))
            tags.append((vol, vol.AddObserver(
                getattr(vtk.vtkCommand, "TransformModifiedEvent",
                        vtk.vtkCommand.ModifiedEvent),
                self._handle_node_modified)))

    # -------- TF application --------

    def _apply_state(self, field, state, volume_node):
        data_range = getattr(field, "_data_range", (0.0, 1.0))
        color_pts = wvr.color_points(state)
        opacity_pts = wvr.opacity_points(state)
        grad_pts = wvr.gradient_points(state)
        # A node with no TF yet (just-created, or a sparse MRB) renders
        # transparent black otherwise -- fall back to a grayscale ramp so
        # the volume is at least visible.
        if not color_pts or not opacity_pts:
            color_pts, opacity_pts = wvr.default_points_for_range(data_range)

        clim = wvr.opacity_scalar_range(opacity_pts, data_range)
        lut = wvr.build_color_opacity_lut(color_pts, opacity_pts, clim)
        grad_lut, grad_range = wvr.build_gradient_lut(grad_pts)

        if field._lut_tex is not None:
            field._lut_tex.set_data(lut.astype(np.float32, copy=False))
        if field._grad_lut_tex is not None:
            field._grad_lut_tex.set_data(grad_lut.astype(np.float32, copy=False))

        field.clim = clim
        field.gradient_range = grad_range
        field.gradient_opacity_enabled = bool(grad_pts)

        ka, kd, ks, sh = wvr.shading(state)
        field.k_ambient, field.k_diffuse = ka, kd
        field.k_specular, field.shininess = ks, sh

        oud = wvr.opacity_unit_distance(state)
        if oud > 0.0:
            field.opacity_unit_distance = oud
        ssm = wvr.sample_step_mm(state)
        if ssm > 0.0:
            field.sample_step_mm = ssm

        self._apply_visibility(field, state, volume_node, red_slice_composite())
        field.set_world_from_local(_world_from_local(volume_node))
        field.touch()

    def _apply_visibility(self, field, state, volume_node, comp):
        """Set field.render_opacity (continuous [0,1]) and field.visible from
        renderEnabled + this view's scoping + the Red slice bg/fg + opacity
        slider. render_opacity is packed into the shader; visible gates the
        field out of the scene-bounds / AABB when fully hidden."""
        field._wgpu_volume_id = volume_node.GetID()
        if (wvr.is_render_enabled(state) and self._targets_this_view(state)):
            opacity = composite_opacity(volume_node.GetID(), comp)
        else:
            opacity = 0.0
        field.render_opacity = float(opacity)
        field.visible = opacity > 0.0

    def refresh_visibility(self):
        """Recompute render-opacity for every field from the current Red
        slice composite. Called by the bridge on slice-composite changes.
        Returns True if any field's opacity changed (so the caller can skip
        a redundant render)."""
        comp = red_slice_composite()
        changed = False
        for nid, field in list(self.fields_by_nid.items()):
            state = self.mrml_scene.GetNodeByID(nid)
            if state is None:
                continue
            vol = wvr.target_node(state)
            if vol is None:
                continue
            prev = getattr(field, "render_opacity", None)
            self._apply_visibility(field, state, vol, comp)
            if field.render_opacity != prev:
                changed = True
        return changed
