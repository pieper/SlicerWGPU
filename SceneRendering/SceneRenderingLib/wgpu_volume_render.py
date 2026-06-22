"""Independent wgpu volume-render state.

This is the "new mode": per-volume wgpu rendering state stored in its OWN
vtkMRMLScriptedModuleNode, with NO dependency on Slicer's volume-rendering
display node or vtkMRMLVolumePropertyNode. The transfer function is a plain
list of control points serialized as JSON parameters -- deliberately simple
so the TF interface can be rethought without touching VTK's piecewise/colour
transfer-function classes or widgets.

Coexists with the legacy VRDN-based path (see wgpu_vtk_inject.py). To keep
the two scans fully disjoint these nodes use ModuleName "SlicerWGPUVolume"
(NOT "SlicerWGPU"), so the legacy path's all_state_nodes() / VRDN-backfill
never see them and a volume is never double-rendered.

State stored per node:
  - targetNode reference (role 'targetNode')  -> the scalar volume
  - viewNode references (role 'viewNode', 0+) -> view scoping (empty = all)
  - renderEnabled            parameter (absence -> enabled)
  - preset                   parameter (name of the active TF preset)
  - colorPoints              parameter (JSON: [[scalar, r, g, b], ...])
  - opacityPoints            parameter (JSON: [[scalar, a], ...])
  - gradientPoints           parameter (JSON: [[gradmag, a], ...] or [])
  - ambient/diffuse/specular/shininess  parameters (Phong constants)
  - opacityUnitDistance      parameter (<=0 -> derive from spacing)
  - sampleStepMm             parameter (<=0 -> derive from spacing)
"""

import json

import numpy as np
import slicer


MODULE_NAME = "SlicerWGPUVolume"

TARGET_NODE_REF = "targetNode"
VIEW_NODE_REF = "viewNode"

PARAM_RENDER_ENABLED = "renderEnabled"
PARAM_PRESET = "preset"
PARAM_COLOR_POINTS = "colorPoints"
PARAM_OPACITY_POINTS = "opacityPoints"
PARAM_GRADIENT_POINTS = "gradientPoints"
PARAM_AMBIENT = "ambient"
PARAM_DIFFUSE = "diffuse"
PARAM_SPECULAR = "specular"
PARAM_SHININESS = "shininess"
PARAM_OPACITY_UNIT_DISTANCE = "opacityUnitDistance"
PARAM_SAMPLE_STEP_MM = "sampleStepMm"


# ---- transfer-function presets ----
#
# Each preset is a dict of control points + shading. Scalar positions are in
# the volume's native intensity units (HU for CT). Colours are 0..1 RGB.
# These are intentionally coarse starting points, not a curated library --
# the whole TF interface is slated for a rethink.

DEFAULT_PRESET = "Grayscale ramp"

PRESETS = {
    "Grayscale ramp": {
        # Filled in per-volume from the data range when applied (see
        # default_points_for_range); the static entry is a placeholder.
        "color": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        "opacity": [[0.0, 0.0], [1.0, 0.8]],
        "gradient": [],
        "ambient": 0.3, "diffuse": 0.7, "specular": 0.2, "shininess": 10.0,
    },
    "CT bone": {
        "color": [
            [-1000.0, 0.0, 0.0, 0.0],
            [150.0, 0.6, 0.35, 0.2],
            [400.0, 0.9, 0.8, 0.6],
            [1500.0, 1.0, 1.0, 0.95],
        ],
        "opacity": [
            [-1000.0, 0.0], [150.0, 0.0], [300.0, 0.6], [1500.0, 0.95],
        ],
        "gradient": [],
        "ambient": 0.25, "diffuse": 0.75, "specular": 0.35, "shininess": 20.0,
    },
    "CT soft tissue": {
        "color": [
            [-1000.0, 0.0, 0.0, 0.0],
            [-50.0, 0.55, 0.35, 0.3],
            [100.0, 0.9, 0.7, 0.6],
            [300.0, 1.0, 0.9, 0.85],
        ],
        "opacity": [
            [-1000.0, 0.0], [-100.0, 0.0], [40.0, 0.25], [300.0, 0.55],
        ],
        "gradient": [],
        "ambient": 0.35, "diffuse": 0.65, "specular": 0.15, "shininess": 8.0,
    },
    "MR default": {
        "color": [
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.4, 0.2, 0.15],
            [0.6, 0.85, 0.75, 0.65],
            [1.0, 1.0, 1.0, 1.0],
        ],
        "opacity": [[0.0, 0.0], [0.2, 0.0], [0.5, 0.4], [1.0, 0.85]],
        "gradient": [],
        "ambient": 0.3, "diffuse": 0.7, "specular": 0.2, "shininess": 12.0,
        # MR points are normalized 0..1; scaled to the data range on apply.
        "normalized": True,
    },
}


def default_points_for_range(scalar_range):
    """Color/opacity control points for the grayscale ramp over a data range."""
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    if hi <= lo:
        hi = lo + 1.0
    color = [[lo, 0.0, 0.0, 0.0], [hi, 1.0, 1.0, 1.0]]
    opacity = [[lo, 0.0], [lo + 0.5 * (hi - lo), 0.4], [hi, 0.85]]
    return color, opacity


# ---- lookup / creation ----

def state_for(volume_node, create=False):
    if volume_node is None:
        return None
    found = _find_state(volume_node)
    if found is not None:
        return found
    if not create:
        return None
    return _create_state(volume_node)


def all_state_nodes():
    coll = slicer.mrmlScene.GetNodesByClass("vtkMRMLScriptedModuleNode")
    try:
        out = []
        for i in range(coll.GetNumberOfItems()):
            node = coll.GetItemAsObject(i)
            if node.GetAttribute("ModuleName") == MODULE_NAME:
                out.append(node)
        return out
    finally:
        coll.UnRegister(None)


def is_managed(node):
    """True if node is one of our independent volume-render state nodes."""
    return (node is not None
            and node.IsA("vtkMRMLScriptedModuleNode")
            and node.GetAttribute("ModuleName") == MODULE_NAME)


def has_independent_node(volume_node):
    """True if this volume is rendered by the independent path -- the
    legacy VRDN path consults this to step aside and avoid double-render."""
    return _find_state(volume_node) is not None


def target_node(state):
    return state.GetNodeReference(TARGET_NODE_REF)


def remove_state(volume_node):
    state = _find_state(volume_node)
    if state is not None:
        slicer.mrmlScene.RemoveNode(state)


# ---- render-enabled flag ----

def is_render_enabled(state):
    # Absence -> enabled; only an explicit "false" opts out.
    return state.GetParameter(PARAM_RENDER_ENABLED) != "false"


def set_render_enabled(volume_node, enabled):
    state = state_for(volume_node, create=True)
    state.SetParameter(PARAM_RENDER_ENABLED, "true" if enabled else "false")


# ---- view scoping (empty list = all views) ----

def view_node_ids(state):
    n = state.GetNumberOfNodeReferences(VIEW_NODE_REF)
    return [state.GetNthNodeReferenceID(VIEW_NODE_REF, i) for i in range(n)]


def set_view_node_ids(volume_node, view_ids):
    state = state_for(volume_node, create=True)
    state.RemoveNodeReferenceIDs(VIEW_NODE_REF)
    for vid in view_ids:
        state.AddNodeReferenceID(VIEW_NODE_REF, vid)


# ---- transfer function ----

def color_points(state):
    return _load_points(state, PARAM_COLOR_POINTS)


def opacity_points(state):
    return _load_points(state, PARAM_OPACITY_POINTS)


def gradient_points(state):
    return _load_points(state, PARAM_GRADIENT_POINTS)


def shading(state):
    """Return (ambient, diffuse, specular, shininess)."""
    return (
        _get_float(state, PARAM_AMBIENT, 0.3),
        _get_float(state, PARAM_DIFFUSE, 0.7),
        _get_float(state, PARAM_SPECULAR, 0.2),
        _get_float(state, PARAM_SHININESS, 10.0),
    )


def opacity_unit_distance(state):
    return _get_float(state, PARAM_OPACITY_UNIT_DISTANCE, 0.0)


def sample_step_mm(state):
    return _get_float(state, PARAM_SAMPLE_STEP_MM, 0.0)


def preset_name(state):
    return state.GetParameter(PARAM_PRESET) or DEFAULT_PRESET


def apply_preset(state, preset_name_, scalar_range):
    """Write a preset's control points + shading onto the state node.
    scalar_range scales normalized presets and fills the grayscale ramp."""
    preset = PRESETS.get(preset_name_, PRESETS[DEFAULT_PRESET])
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    if hi <= lo:
        hi = lo + 1.0

    if preset_name_ == DEFAULT_PRESET:
        color, opacity = default_points_for_range((lo, hi))
        gradient = []
    elif preset.get("normalized"):
        color = [[lo + p[0] * (hi - lo), p[1], p[2], p[3]]
                 for p in preset["color"]]
        opacity = [[lo + p[0] * (hi - lo), p[1]] for p in preset["opacity"]]
        gradient = list(preset.get("gradient", []))
    else:
        color = [list(p) for p in preset["color"]]
        opacity = [list(p) for p in preset["opacity"]]
        gradient = list(preset.get("gradient", []))

    # Batch the parameter writes into a single Modified so observers
    # (the bridge's displayer) rebuild the LUT once, not once per write.
    was_modifying = state.StartModify()
    try:
        state.SetParameter(PARAM_PRESET, preset_name_)
        _store_points(state, PARAM_COLOR_POINTS, color)
        _store_points(state, PARAM_OPACITY_POINTS, opacity)
        _store_points(state, PARAM_GRADIENT_POINTS, gradient)
        state.SetParameter(PARAM_AMBIENT, str(preset.get("ambient", 0.3)))
        state.SetParameter(PARAM_DIFFUSE, str(preset.get("diffuse", 0.7)))
        state.SetParameter(PARAM_SPECULAR, str(preset.get("specular", 0.2)))
        state.SetParameter(PARAM_SHININESS, str(preset.get("shininess", 10.0)))
    finally:
        state.EndModify(was_modifying)


# ---- LUT building (numpy, no VTK transfer-function classes) ----

def build_color_opacity_lut(color_pts, opacity_pts, scalar_range, n=256):
    """(n,4) f32 RGBA LUT sampled uniformly across scalar_range by linear
    interpolation of the control points. RGB from color points, A from
    opacity points -- the two lists are sampled independently so they can
    have different control-point positions."""
    lo, hi = float(scalar_range[0]), float(scalar_range[1])
    if hi <= lo:
        hi = lo + 1.0
    xs = np.linspace(lo, hi, n, dtype=np.float64)
    lut = np.zeros((n, 4), dtype=np.float32)
    if color_pts:
        cp = np.asarray(color_pts, dtype=np.float64)
        order = np.argsort(cp[:, 0])
        cp = cp[order]
        for ch in range(3):
            lut[:, ch] = np.interp(xs, cp[:, 0], cp[:, ch + 1]).astype(np.float32)
    if opacity_pts:
        op = np.asarray(opacity_pts, dtype=np.float64)
        order = np.argsort(op[:, 0])
        op = op[order]
        lut[:, 3] = np.interp(xs, op[:, 0], op[:, 1]).astype(np.float32)
    return lut


def build_gradient_lut(gradient_pts, n=256):
    """((n,1) f32, (gmin, gmax)). Flat 1.0 when no gradient points."""
    if not gradient_pts:
        return np.ones((n, 1), dtype=np.float32), (0.0, 1.0)
    gp = np.asarray(gradient_pts, dtype=np.float64)
    order = np.argsort(gp[:, 0])
    gp = gp[order]
    gmin, gmax = float(gp[0, 0]), float(gp[-1, 0])
    if gmax <= gmin:
        gmax = gmin + 1.0
    xs = np.linspace(gmin, gmax, n, dtype=np.float64)
    lut = np.interp(xs, gp[:, 0], gp[:, 1]).astype(np.float32)
    return lut.reshape(n, 1), (gmin, gmax)


def opacity_scalar_range(opacity_pts, data_range):
    """clim for the LUT: span of the opacity control points clamped to the
    data range (mirrors the legacy path's GetScalarOpacity().GetRange())."""
    dmin, dmax = float(data_range[0]), float(data_range[1])
    if not opacity_pts:
        return dmin, dmax
    xs = [float(p[0]) for p in opacity_pts]
    lo = max(min(xs), dmin)
    hi = min(max(xs), dmax)
    if hi <= lo:
        lo, hi = dmin, dmax
    return lo, hi


# ---- internals ----

def _find_state(volume_node):
    target_id = volume_node.GetID()
    for state in all_state_nodes():
        if state.GetNodeReferenceID(TARGET_NODE_REF) == target_id:
            return state
    return None


def _create_state(volume_node):
    state = slicer.vtkMRMLScriptedModuleNode()
    state.SetName(f"SlicerWGPUVolume: {volume_node.GetName()}")
    state.SetAttribute("ModuleName", MODULE_NAME)
    state.HideFromEditorsOn()
    slicer.mrmlScene.AddNode(state)
    state.SetAndObserveNodeReferenceID(TARGET_NODE_REF, volume_node.GetID())
    return state


def _store_points(state, param, points):
    state.SetParameter(param, json.dumps(points))


def _load_points(state, param):
    raw = state.GetParameter(param)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return []


def _get_float(state, param, default):
    raw = state.GetParameter(param)
    if not raw:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default
