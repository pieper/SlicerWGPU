"""SlicerWGPU per-data-node rendering state stored as
vtkMRMLScriptedModuleNode in the MRML scene.

Each managed data node (volume or segmentation) gets one state node
holding:

  - target node reference (role 'targetNode')
  - view node references (role 'viewNode', multiple, empty = all views)
  - renderEnabled parameter
  - segmentRenderMode parameter (segmentations only)

Save / restore through MRML / MRB happens automatically because
vtkMRMLScriptedModuleNode serializes its parameters and node
references. Reapplying renderEnabled to the running bridge after
scene import is the caller's responsibility -- the bridge currently
has no concept of these state nodes (see TODOs in wgpu_vtk_inject.py).

The bridge ALSO needs to grow per-view installation and consult
viewNodeIDs to support side-by-side VTK / wgpu comparison; until
that lands the view-id list round-trips in MRML but is not honored
at render time.
"""

import slicer


MODULE_NAME = "SlicerWGPU"

# Node-reference roles on the state node.
TARGET_NODE_REF = "targetNode"
VIEW_NODE_REF = "viewNode"

# String parameters on the state node.
PARAM_RENDER_ENABLED = "renderEnabled"
PARAM_SEGMENT_RENDER_MODE = "segmentRenderMode"


# ---- lookup / creation ----

def state_for(data_node, create=False):
    """Return the state node for data_node. Create one when create=True
    and none exists. Returns None when data_node is None or when no
    state exists and create=False."""
    if data_node is None:
        return None
    found = _find_state(data_node)
    if found is not None:
        return found
    if not create:
        return None
    return _create_state(data_node)


def all_state_nodes():
    """List every SlicerWGPU state node currently in the scene."""
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


def remove_state(data_node):
    """Remove the state node for data_node, if any."""
    state = _find_state(data_node)
    if state is not None:
        slicer.mrmlScene.RemoveNode(state)


def target_node(state):
    """Return the data node referenced by state, or None."""
    return state.GetNodeReference(TARGET_NODE_REF)


# ---- render-enabled flag ----

def is_render_enabled(data_node):
    # Semantics: absence of the renderEnabled parameter -> enabled.
    # Only an explicit "false" opts out. This way side-effect helpers
    # like set_view_node_ids that create the state node without
    # touching renderEnabled don't accidentally disable the node.
    state = _find_state(data_node)
    if state is None:
        return False
    val = state.GetParameter(PARAM_RENDER_ENABLED)
    return val != "false"


def set_render_enabled(data_node, enabled):
    # Always create the state node so an explicit opt-OUT (enabled=
    # False) is recorded and survives MRB save/restore. The bridge
    # treats "no state node" as "implicit opt-in" -- without this we
    # couldn't durably override that default.
    state = state_for(data_node, create=True)
    state.SetParameter(PARAM_RENDER_ENABLED, "true" if enabled else "false")


# ---- view scoping (empty list = all views) ----

def view_node_ids(data_node):
    state = _find_state(data_node)
    if state is None:
        return []
    n = state.GetNumberOfNodeReferences(VIEW_NODE_REF)
    return [state.GetNthNodeReferenceID(VIEW_NODE_REF, i) for i in range(n)]


def set_view_node_ids(data_node, view_ids):
    state = state_for(data_node, create=True)
    state.RemoveNodeReferenceIDs(VIEW_NODE_REF)
    for vid in view_ids:
        state.AddNodeReferenceID(VIEW_NODE_REF, vid)


# ---- segmentation render mode ----

def segment_render_mode(data_node, default="iso"):
    state = _find_state(data_node)
    if state is None:
        return default
    mode = state.GetParameter(PARAM_SEGMENT_RENDER_MODE)
    return mode if mode else default


def set_segment_render_mode(data_node, mode):
    state = state_for(data_node, create=True)
    state.SetParameter(PARAM_SEGMENT_RENDER_MODE, mode)


# ---- internals ----

def _find_state(data_node):
    target_id = data_node.GetID()
    for state in all_state_nodes():
        if state.GetNodeReferenceID(TARGET_NODE_REF) == target_id:
            return state
    return None


def _create_state(data_node):
    state = slicer.vtkMRMLScriptedModuleNode()
    state.SetName(f"SlicerWGPU: {data_node.GetName()}")
    # ModuleName attribute is the convention every scripted module
    # uses to claim its own ScriptedModuleNodes -- see HeartValves SH
    # plugin for the same idiom.
    state.SetAttribute("ModuleName", MODULE_NAME)
    # Keep these out of node selectors so they don't clutter UI;
    # subject hierarchy will still show them under 'Other'.
    state.HideFromEditorsOn()
    slicer.mrmlScene.AddNode(state)
    state.SetAndObserveNodeReferenceID(TARGET_NODE_REF, data_node.GetID())
    return state
