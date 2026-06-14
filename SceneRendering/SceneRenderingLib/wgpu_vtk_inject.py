"""wgpu_vtk_inject.py

Thin VTK-injection bridge: wgpu ray-march rendered into Slicer's native 3D
view via a vtkCommand::EndEvent hook. Reuses slicer_wgpu for everything
except the VTK plumbing and the raw wgpu pipeline.

Responsibilities of this module:
  - Install a VolumeRenderingDisplayer to observe MRML VR nodes; it
    builds/updates ImageField instances for us on TF / transform changes.
  - Claim each VR display node (set Visibility=0) so Slicer's native VR
    mapper doesn't double-render.
  - Own a raw wgpu.RenderPipeline that ray-marches N ImageFields to an
    offscreen RGBA texture.
  - On EndEvent: read VTK's current color buffer, alpha-composite the
    wgpu output on top, glTexSubImage2D back into VTK's RenderFramebuffer.

Uses from slicer_wgpu:
  - fields.ImageField (per-volume GPU resources)
  - displayers.VolumeRenderingDisplayer (MRML observation + Field updates)
"""

from __future__ import annotations

import ctypes
import sys
import weakref
from ctypes import cdll, c_ubyte

import numpy as np
import slicer
import vtk
import wgpu


def _load_gl_library():
    """Return an object that looks like a ctypes DLL of libGL.

    macOS : OpenGL.framework (Core profile 3.2+ gives direct access to
            glBindFramebuffer / glReadPixels / glTexSubImage2D).
    Linux : libGL.so.1.
    Windows: opengl32.dll wrapped by a thin shim that falls back to
             wglGetProcAddress for anything the DLL doesn't export
             directly (i.e. most of GL 1.2+).
    """
    if sys.platform == "darwin":
        return cdll.LoadLibrary(
            "/System/Library/Frameworks/OpenGL.framework/OpenGL")
    if sys.platform.startswith("linux"):
        return cdll.LoadLibrary("libGL.so.1")
    if sys.platform == "win32":
        return _Win32GLShim()
    raise NotImplementedError(f"unsupported platform: {sys.platform}")


class _Win32GLFunc:
    """Lazily-bound Windows GL function. Supports argtypes/restype
    assignment like a ctypes function object; the real WINFUNCTYPE
    binding happens on first call, by which point a GL context must be
    current (which is true inside VTK's EndEvent handler).
    """
    __slots__ = ("_shim", "_name", "argtypes", "restype", "_bound")

    def __init__(self, shim, name):
        self._shim = shim
        self._name = name
        self.argtypes = None
        self.restype = ctypes.c_int  # ctypes default
        self._bound = None

    def __call__(self, *args):
        if self._bound is None:
            addr = self._shim._resolve(self._name)
            at = list(self.argtypes or [])
            proto = ctypes.WINFUNCTYPE(self.restype, *at)
            self._bound = proto(addr)
        return self._bound(*args)


class _Win32GLShim:
    """Adapter over opengl32.dll. On Windows, opengl32.dll only exports
    GL 1.0 / 1.1 symbols; every framebuffer / shader / VAO / multitexture
    entry point must be queried per-context via wglGetProcAddress. This
    shim makes `shim.glBindFramebuffer(...)` etc. just work.

    GL context must be current when an unknown function is first called
    (not when argtypes/restype are assigned). VTK's EndEvent fires with
    the render-window context current, and all bridge GL calls happen
    inside that handler, so this is transparently satisfied.
    """

    def __init__(self):
        # WinDLL -> __stdcall, which is what Windows GL uses (APIENTRY).
        self._gl = ctypes.WinDLL("opengl32")
        self._gl.wglGetProcAddress.argtypes = [ctypes.c_char_p]
        self._gl.wglGetProcAddress.restype = ctypes.c_void_p
        self._cache = {}

    def __getattr__(self, name):
        # Private attributes raise normally so __init__'s "self._gl = ..."
        # etc. doesn't recurse into wglGetProcAddress.
        if name.startswith("_"):
            raise AttributeError(name)
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        try:
            fn = getattr(self._gl, name)         # GL 1.0 / 1.1 direct export
        except AttributeError:
            fn = _Win32GLFunc(self, name)        # GL 1.2+ via wglGetProcAddress
        self._cache[name] = fn
        return fn

    def _resolve(self, name):
        addr = self._gl.wglGetProcAddress(name.encode("ascii"))
        if not addr:
            raise RuntimeError(
                f"wglGetProcAddress({name}) returned NULL -- either no GL "
                "context is current or this GL version doesn't support it. "
                "All bridge GL calls must happen inside the EndEvent "
                "callback where VTK has made its context current.")
        return addr


# ---------------------------------------------------------------------------
# GL compositor: hardware premultiplied-alpha blend of our wgpu output over
# VTK's framebuffer color attachment. Replaces CPU numpy composite + VTK
# readback. Per frame: one GPU->CPU readback of OUR output + one upload to
# a GL texture + one fullscreen draw with blending. No VTK readback.
# ---------------------------------------------------------------------------

# GLSL 150 is the minimum for Core 3.2 (macOS 4.1 Core, Linux 3.2+).
_COMP_VS = b"""#version 150
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_COMP_FS = b"""#version 150
uniform sampler2D u_tex;
out vec4 FragColor;
void main() {
    FragColor = texelFetch(u_tex, ivec2(gl_FragCoord.xy), 0);
}
"""


class _GLCompositor:
    """Premultiplied-alpha over-blend of a CPU RGBA8 buffer onto a bound FBO's
    color attachment, using GL hardware blending."""

    # GL constants (subset we use)
    GL_TEXTURE_2D              = 0x0DE1
    GL_RGBA                    = 0x1908
    GL_RGBA8                   = 0x8058
    GL_UNSIGNED_BYTE           = 0x1401
    GL_TEXTURE_MIN_FILTER      = 0x2801
    GL_TEXTURE_MAG_FILTER      = 0x2800
    GL_NEAREST                 = 0x2600
    GL_VERTEX_SHADER           = 0x8B31
    GL_FRAGMENT_SHADER         = 0x8B30
    GL_COMPILE_STATUS          = 0x8B81
    GL_LINK_STATUS             = 0x8B82
    GL_INFO_LOG_LENGTH         = 0x8B84
    GL_DRAW_FRAMEBUFFER        = 0x8CA9
    GL_BLEND                   = 0x0BE2
    GL_ONE                     = 1
    GL_ONE_MINUS_SRC_ALPHA     = 0x0303
    GL_DEPTH_TEST              = 0x0B71
    GL_SCISSOR_TEST            = 0x0C11
    GL_CULL_FACE               = 0x0B44
    GL_TEXTURE0                = 0x84C0
    GL_TRIANGLES               = 0x0004
    GL_CURRENT_PROGRAM         = 0x8B8D
    GL_TEXTURE_BINDING_2D      = 0x8069
    GL_ACTIVE_TEXTURE          = 0x84E0
    GL_VERTEX_ARRAY_BINDING    = 0x85B5
    GL_DRAW_FRAMEBUFFER_BINDING= 0x8CA6
    GL_VIEWPORT                = 0x0BA2
    GL_DEPTH_WRITEMASK         = 0x0B72
    GL_TRUE                    = 1
    GL_FALSE                   = 0

    def __init__(self, gl):
        self.gl = gl
        self._inited = False
        self._tex = 0
        self._size = (0, 0)
        self._prog = 0
        self._vao = 0
        self._loc_tex = -1
        self._declare_signatures()

    def _declare_signatures(self):
        """Set ctypes arg/return types for the GL entry points we use.
        Missing on some platforms if the symbol isn't exported — we let
        the first call raise so the error points at the missing function."""
        from ctypes import (c_int, c_uint, c_char_p, c_void_p, c_size_t,
                            POINTER, c_char, c_ubyte, c_ulong)
        g = self.gl
        g.glGenTextures.argtypes    = [c_int, POINTER(c_uint)]
        g.glBindTexture.argtypes    = [c_uint, c_uint]
        g.glTexImage2D.argtypes     = [c_uint, c_int, c_int, c_int, c_int,
                                       c_int, c_uint, c_uint, c_void_p]
        g.glTexSubImage2D.argtypes  = [c_uint, c_int, c_int, c_int, c_int, c_int,
                                       c_uint, c_uint, c_void_p]
        g.glTexParameteri.argtypes  = [c_uint, c_uint, c_int]
        g.glCreateShader.argtypes   = [c_uint]
        g.glCreateShader.restype    = c_uint
        g.glShaderSource.argtypes   = [c_uint, c_int, POINTER(c_char_p), POINTER(c_int)]
        g.glCompileShader.argtypes  = [c_uint]
        g.glGetShaderiv.argtypes    = [c_uint, c_uint, POINTER(c_int)]
        g.glGetShaderInfoLog.argtypes = [c_uint, c_int, POINTER(c_int), c_char_p]
        g.glDeleteShader.argtypes   = [c_uint]
        g.glCreateProgram.argtypes  = []
        g.glCreateProgram.restype   = c_uint
        g.glAttachShader.argtypes   = [c_uint, c_uint]
        g.glLinkProgram.argtypes    = [c_uint]
        g.glGetProgramiv.argtypes   = [c_uint, c_uint, POINTER(c_int)]
        g.glGetProgramInfoLog.argtypes = [c_uint, c_int, POINTER(c_int), c_char_p]
        g.glUseProgram.argtypes     = [c_uint]
        g.glGetUniformLocation.argtypes = [c_uint, c_char_p]
        g.glGetUniformLocation.restype  = c_int
        g.glUniform1i.argtypes      = [c_int, c_int]
        g.glGenVertexArrays.argtypes= [c_int, POINTER(c_uint)]
        g.glBindVertexArray.argtypes= [c_uint]
        g.glBindFramebuffer.argtypes= [c_uint, c_uint]
        g.glViewport.argtypes       = [c_int, c_int, c_int, c_int]
        g.glEnable.argtypes         = [c_uint]
        g.glDisable.argtypes        = [c_uint]
        g.glBlendFunc.argtypes      = [c_uint, c_uint]
        g.glActiveTexture.argtypes  = [c_uint]
        g.glDrawArrays.argtypes     = [c_uint, c_int, c_int]
        g.glGetIntegerv.argtypes    = [c_uint, POINTER(c_int)]
        g.glGetBooleanv.argtypes    = [c_uint, POINTER(ctypes.c_ubyte)]
        g.glDepthMask.argtypes      = [ctypes.c_ubyte]

    def _compile(self, kind, src):
        from ctypes import c_int, c_char_p, POINTER, byref
        g = self.gl
        s = g.glCreateShader(kind)
        arr = (c_char_p * 1)(src)
        g.glShaderSource(s, 1, arr, None)
        g.glCompileShader(s)
        ok = c_int(0)
        g.glGetShaderiv(s, self.GL_COMPILE_STATUS, byref(ok))
        if not ok.value:
            log_len = c_int(0)
            g.glGetShaderiv(s, self.GL_INFO_LOG_LENGTH, byref(log_len))
            buf = ctypes.create_string_buffer(max(log_len.value, 1))
            g.glGetShaderInfoLog(s, log_len.value, None, buf)
            raise RuntimeError(
                f"GL shader compile failed: {buf.value.decode(errors='replace')}")
        return s

    def _lazy_init(self):
        if self._inited:
            return
        from ctypes import c_uint, byref
        g = self.gl
        vs = self._compile(self.GL_VERTEX_SHADER, _COMP_VS)
        fs = self._compile(self.GL_FRAGMENT_SHADER, _COMP_FS)
        self._prog = g.glCreateProgram()
        g.glAttachShader(self._prog, vs)
        g.glAttachShader(self._prog, fs)
        g.glLinkProgram(self._prog)
        linked = ctypes.c_int(0)
        g.glGetProgramiv(self._prog, self.GL_LINK_STATUS, ctypes.byref(linked))
        if not linked.value:
            raise RuntimeError("GL program link failed")
        g.glDeleteShader(vs)
        g.glDeleteShader(fs)
        self._loc_tex = g.glGetUniformLocation(self._prog, b"u_tex")
        vao = c_uint(0)
        g.glGenVertexArrays(1, byref(vao))
        self._vao = vao.value
        self._inited = True

    def _ensure_texture(self, w, h):
        from ctypes import c_uint, byref
        g = self.gl
        if self._tex and self._size == (w, h):
            return
        if not self._tex:
            t = c_uint(0)
            g.glGenTextures(1, byref(t))
            self._tex = t.value
        g.glBindTexture(self.GL_TEXTURE_2D, self._tex)
        g.glTexImage2D(self.GL_TEXTURE_2D, 0, self.GL_RGBA8, w, h, 0,
                       self.GL_RGBA, self.GL_UNSIGNED_BYTE, None)
        g.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_MIN_FILTER,
                          self.GL_NEAREST)
        g.glTexParameteri(self.GL_TEXTURE_2D, self.GL_TEXTURE_MAG_FILTER,
                          self.GL_NEAREST)
        g.glBindTexture(self.GL_TEXTURE_2D, 0)
        self._size = (w, h)

    def composite(self, rgba_gl, fbo_index, w, h):
        """Draw rgba_gl (premultiplied RGBA8, GL Y-up) onto the given FBO's
        color attachment 0 with premultiplied-alpha OVER blending."""
        self._lazy_init()
        self._ensure_texture(w, h)
        g = self.gl

        # Save critical state
        prev_prog = ctypes.c_int(0)
        prev_vao = ctypes.c_int(0)
        prev_tex0 = ctypes.c_int(0)
        prev_active = ctypes.c_int(0)
        prev_fbo = ctypes.c_int(0)
        prev_vp = (ctypes.c_int * 4)()
        prev_depth_mask = ctypes.c_ubyte(0)
        g.glGetIntegerv(self.GL_CURRENT_PROGRAM, ctypes.byref(prev_prog))
        g.glGetIntegerv(self.GL_VERTEX_ARRAY_BINDING, ctypes.byref(prev_vao))
        g.glGetIntegerv(self.GL_ACTIVE_TEXTURE, ctypes.byref(prev_active))
        g.glGetIntegerv(self.GL_DRAW_FRAMEBUFFER_BINDING, ctypes.byref(prev_fbo))
        g.glGetIntegerv(self.GL_VIEWPORT, prev_vp)
        g.glGetBooleanv(self.GL_DEPTH_WRITEMASK, ctypes.byref(prev_depth_mask))
        g.glActiveTexture(self.GL_TEXTURE0)
        g.glGetIntegerv(self.GL_TEXTURE_BINDING_2D, ctypes.byref(prev_tex0))

        # Upload our wgpu output into the GL texture
        g.glBindTexture(self.GL_TEXTURE_2D, self._tex)
        g.glTexSubImage2D(
            self.GL_TEXTURE_2D, 0, 0, 0, w, h,
            self.GL_RGBA, self.GL_UNSIGNED_BYTE,
            rgba_gl.ctypes.data_as(ctypes.c_void_p))

        # Bind the target FBO + set viewport
        g.glBindFramebuffer(self.GL_DRAW_FRAMEBUFFER, fbo_index)
        g.glViewport(0, 0, w, h)

        # Disable state that would interfere with a fullscreen blend draw
        g.glDisable(self.GL_DEPTH_TEST)
        g.glDisable(self.GL_CULL_FACE)
        g.glDisable(self.GL_SCISSOR_TEST)
        # Critical: don't write to the depth buffer. VTK's later overlay
        # renderers (transform interaction handles, etc.) rely on the
        # depth values VTK wrote earlier. If we clobber them with our
        # fullscreen-triangle z=0, handle depth tests become unreliable
        # and the handles pop in and out as the camera moves.
        g.glDepthMask(self.GL_FALSE)
        # Premultiplied-alpha over: out = src + dst*(1-src.a)
        g.glEnable(self.GL_BLEND)
        g.glBlendFunc(self.GL_ONE, self.GL_ONE_MINUS_SRC_ALPHA)

        # Use our program + VAO; sampler on unit 0
        g.glUseProgram(self._prog)
        g.glBindVertexArray(self._vao)
        if self._loc_tex >= 0:
            g.glUniform1i(self._loc_tex, 0)
        g.glDrawArrays(self.GL_TRIANGLES, 0, 3)

        # Restore state
        g.glDisable(self.GL_BLEND)
        g.glDepthMask(prev_depth_mask.value)
        g.glBindVertexArray(prev_vao.value)
        g.glUseProgram(prev_prog.value)
        g.glBindTexture(self.GL_TEXTURE_2D, prev_tex0.value)
        g.glActiveTexture(prev_active.value)
        g.glBindFramebuffer(self.GL_DRAW_FRAMEBUFFER, prev_fbo.value)
        g.glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3])

from pygfx.renderers.wgpu.engine.shared import get_shared
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object, update_resource


def diagnose_wgpu():
    """Print adapter + device diagnostics for this machine. Call from the
    Slicer Python console if the bridge fails to initialize:

        from SceneRenderingLib import wgpu_vtk_inject
        wgpu_vtk_inject.diagnose_wgpu()
    """
    import pygfx as _pgf
    print(f"pygfx version: {_pgf.__version__}")
    print(f"wgpu version:  {wgpu.__version__}")
    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
    except Exception as e:
        print(f"enumerate_adapters_sync FAILED: {e!r}")
        return
    if not adapters:
        print("No wgpu adapters found!")
        return
    for i, a in enumerate(adapters):
        print(f"\nadapter {i}: {a.summary}")
        feats = sorted(a.features)
        print(f"  features ({len(feats)}): {feats}")
        print(f"  has float32-filterable: "
              f"{'float32-filterable' in a.features}")
    # Check pygfx Shared state
    from pygfx.renderers.wgpu.engine.shared import Shared
    inst = Shared._instance
    print(f"\nShared._instance: {'None' if inst is None else 'set'}")
    if inst is not None:
        print(f"  has _device attr:  {hasattr(inst, '_device')}")
        print(f"  has _adapter attr: {hasattr(inst, '_adapter')}")


def _adapter_backend(adapter):
    """Heuristic: read the backend from the adapter's summary string.
    ('Vulkan', 'Metal', 'DX12', 'OpenGL', or 'Unknown')."""
    s = getattr(adapter, "summary", "") or ""
    for label in ("Vulkan", "Metal", "DX12", "OpenGL"):
        if label in s:
            return label
    return "Unknown"


def _force_vulkan_only_wgpu_instance():
    """On Linux, create the wgpu instance with only the Vulkan backend enabled.

    wgpu enumerates EVERY backend when it first creates its instance / requests an adapter.
    Its OpenGL-ES backend opens an EGL platform display chosen from the environment
    (WAYLAND_DISPLAY -> wayland, else DISPLAY -> X11). On NVIDIA under XWayland (a headless /
    browser-streamed desktop) that EGL probe aborts the whole process -- wgpu-hal panics with
    BadAccess across the C FFI (unrecoverable). Restricting the instance to Vulkan means the
    GL/ES backend is never created, so the probe never runs. The injection bridge's device is
    purely offscreen anyway (compute + render-to-texture; the GL composite uses VTK's own
    already-current context via libGL), so it needs neither X nor Wayland from wgpu.

    Linux-only (macOS=Metal, Windows=DX12/Vulkan, where this would remove the only backend).
    Override with SLICER_WGPU_INSTANCE_BACKENDS. Idempotent; no-op once the instance exists.
    This is normally already done by SceneRendering / slicer_wgpu import, but is repeated here
    so this module stays correct if dropped in and reloaded on its own.
    """
    import os
    import sys
    if not sys.platform.startswith("linux"):
        return
    try:
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
        print(f"wgpu_vtk_inject: could not restrict wgpu instance to Vulkan: {exc}")


def _shared_wgpu_device():
    """Acquire the offscreen wgpu device, with the GL/ES backend disabled on Linux.

    See `_force_vulkan_only_wgpu_instance` for why the GL backend must not be enumerated
    (it aborts the process on NVIDIA under XWayland).
    """
    _force_vulkan_only_wgpu_instance()
    return _shared_wgpu_device_impl()


def _shared_wgpu_device_impl():
    """Return pygfx's shared wgpu device, with fallback paths for partial
    pygfx initialization and OpenGL-only systems.

    Two failure modes are handled:

    1. **Half-constructed Shared**. pygfx's Shared.__init__ sets
       Shared._instance = self BEFORE calling wgpu.get_default_device().
       If that device request raises, the exception propagates but the
       singleton is left half-constructed; every subsequent
       get_shared().device access then reports "'Shared' object has no
       attribute '_device'". We clear the half-construct and retry.

    2. **OpenGL-only systems**. wgpu's OpenGL backend doesn't expose the
       COMPUTE_SHADER + storage-buffer features the bridge's compute
       pipelines (segment smoothing, Colorize bake) depend on. If the
       only adapter wgpu exposes is OpenGL, we can't make it work --
       emit a RuntimeError with per-platform instructions for getting a
       Vulkan / Metal / DX12 ICD installed.
    """
    from pygfx.renderers.wgpu.engine.shared import Shared
    # Happy path
    try:
        return get_shared().device
    except AttributeError:
        pass
    except Exception:
        pass

    # Clear a half-constructed Shared.
    if Shared._instance is not None and not hasattr(Shared._instance, "_device"):
        Shared._instance = None

    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
    except Exception as e:
        raise RuntimeError(
            f"wgpu_vtk_inject: wgpu adapter enumeration failed: {e!r}")
    if not adapters:
        raise RuntimeError(
            "wgpu_vtk_inject: no wgpu adapters available on this system.")

    # Try preferred-backend adapters first (Vulkan / Metal / DX12 all
    # expose compute + storage). Retry request_device with an empty
    # required_features set in case pygfx's default features choked.
    preferred = [a for a in adapters
                 if _adapter_backend(a) in ("Vulkan", "Metal", "DX12")]
    last_error = None
    for adapter in preferred:
        try:
            wgpu.preconfigure_default_device(
                "pygfx", adapter=adapter, required_features=set())
            device = wgpu.get_default_device()
            inst = Shared.__new__(Shared)
            inst._device = device
            inst._adapter = device.adapter
            Shared._instance = inst
            return device
        except Exception as e:
            last_error = e
            continue

    # No preferred backend (or all of them failed to produce a device).
    # The OpenGL backend is not enough for compute + storage buffers,
    # which our bridge needs, so we stop here with a specific message.
    backends = sorted({_adapter_backend(a) for a in adapters})
    lines = [
        "wgpu_vtk_inject: this GPU / driver combination cannot run the bridge.",
        "",
        "The bridge uses WGPU compute shaders + storage-buffer bindings",
        "(segment smoothing, ColorizeVolume bake). Those features are",
        "available on WGPU's Vulkan / Metal / DX12 backends, but not on",
        "the OpenGL backend.",
        "",
        f"Adapters detected ({len(adapters)}):"]
    for a in adapters:
        lines.append(f"  - {a.summary}")
    lines.append("")
    lines.append(f"Backends seen: {backends}")
    if last_error is not None:
        lines += ["", f"Last device-request error: {last_error!r}"]
    lines.append("")
    if sys.platform.startswith("linux"):
        lines += [
            "Linux: a Vulkan adapter is missing. For NVIDIA cards:",
            "  Ubuntu/Debian:  sudo apt install libvulkan1",
            "                  # also make sure nvidia-driver-<ver> exposes",
            "                  # its Vulkan ICD under /usr/share/vulkan/icd.d",
            "  Fedora:         sudo dnf install vulkan-loader",
            "  Arch:           sudo pacman -S vulkan-icd-loader nvidia-utils",
            "",
            "Verify afterwards:",
            "  ls /usr/share/vulkan/icd.d/ | grep -i nvidia   # nvidia_icd.json",
            "  vulkaninfo --summary                           # lists device",
            "",
            "Any Maxwell-era or newer NVIDIA GPU (GTX 7xx/9xx/10xx/RTX)",
            "supports Vulkan with driver version >= 450.",
        ]
    elif sys.platform == "win32":
        lines += [
            "Windows: WGPU normally auto-selects DX12. If only an OpenGL",
            "adapter is visible, update your GPU driver from the vendor's",
            "site (NVIDIA / AMD / Intel).",
        ]
    elif sys.platform == "darwin":
        lines += [
            "macOS: WGPU uses Metal. A Metal adapter is normally always",
            "visible -- the fact that none shows up here likely means the",
            "Metal backend crashed at enumeration time. Reinstall Slicer.",
        ]
    lines += [
        "",
        "Run `from SceneRenderingLib import wgpu_vtk_inject;"
        " wgpu_vtk_inject.diagnose_wgpu()` in Slicer's Python console",
        "for full adapter and feature details.",
    ]
    raise RuntimeError("\n".join(lines))

from slicer_wgpu.fields import ImageField
from slicer_wgpu.displayers.volume import VolumeRenderingDisplayer


# ---------------------------------------------------------------------------
# WGSL generation -- matches ImageField's TF + gradient Phong but stripped
# to what we need here; all pygfx stdlib references replaced with inline
# minimal versions.
# ---------------------------------------------------------------------------

_HEADER = """
struct Varyings { @builtin(position) position: vec4<f32>, };
struct FragmentOutput { @location(0) color: vec4<f32>, };

struct Camera {
    proj_inv: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    // size.xy = (viewport_w, viewport_h)
    // size.zw = (jitter_ndc_x, jitter_ndc_y) -- per-frame sub-pixel
    //          camera jitter for temporal anti-aliasing. Range ~ (-1/w, 1/w).
    size: vec4<f32>,
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    // taa.x = frame_index modulo something, for hash-seeding the per-ray
    //         dt jitter so the noise pattern shifts each frame and TAA
    //         can average it out.
    taa: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u_cam: Camera;

fn ndc_to_world(ndc: vec4<f32>) -> vec3<f32> {
    let clip = u_cam.proj_inv * ndc;
    let eye = clip.xyz / clip.w;
    let world = u_cam.view_inv * vec4<f32>(eye, 1.0);
    return world.xyz / world.w;
}

fn world_depth_01(wp: vec3<f32>) -> f32 {
    let clip = u_cam.proj * u_cam.view * vec4<f32>(wp, 1.0);
    let ndc_z = clip.z / max(clip.w, 1e-6);
    return clamp(ndc_z * 0.5 + 0.5, 0.0, 1.0);
}

fn ray_aabb(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = vec3<f32>(1.0) / d;
    let tb = (bmin - o) * inv;
    let tt = (bmax - o) * inv;
    let tmn = min(tt, tb);
    let tmx = max(tt, tb);
    return vec2<f32>(max(max(tmn.x, tmn.y), tmn.z), min(min(tmx.x, tmx.y), tmx.z));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> Varyings {
    let x = select(-1.0, 3.0, vi == 1u);
    let y = select(-1.0, 3.0, vi == 2u);
    var v: Varyings;
    v.position = vec4<f32>(x, y, 0.0, 1.0);
    return v;
}
"""


def _halton_2_3(i):
    """Return the (Halton-base-2, Halton-base-3) value for index i, in [0, 1).
    Used to drive sub-pixel jitter for TAA. The 2D Halton sequence is the
    standard low-discrepancy choice for camera jitter (Karis 2014 TAA)."""
    def halton(idx, base):
        f = 1.0
        r = 0.0
        while idx > 0:
            f = f / base
            r = r + f * (idx % base)
            idx = idx // base
        return r
    return halton(i + 1, 2), halton(i + 1, 3)


def _mat_struct_wgsl(n, m, k):
    lines = []
    for i in range(n):
        lines += [
            f"    img{i}_p2t: mat4x4<f32>,",
            f"    img{i}_clim: vec4<f32>,",
            f"    img{i}_shade: vec4<f32>,",
            f"    img{i}_step_unit: vec4<f32>,",
        ]
    for j in range(m):
        lines += [
            f"    seg{j}_p2t: mat4x4<f32>,",
            f"    seg{j}_color: vec4<f32>,     // (r, g, b, opacity)",
            f"    seg{j}_shade: vec4<f32>,     // (ka, kd, ks, shin)",
            f"    seg{j}_params: vec4<f32>,    // (band_mm, _, mode, _)",
        ]
    for q in range(k):
        lines += [
            f"    rgba{q}_p2t: mat4x4<f32>,",
            f"    rgba{q}_step_unit: vec4<f32>,  // (step, unit, visible, grad_h_mm)",
            f"    rgba{q}_shade: vec4<f32>,     // (ka, kd, ks, shin)",
            f"    rgba{q}_w2l: mat4x4<f32>,     // world -> label-tex coords",
            f"    rgba{q}_carve_center: vec4<f32>,    // (cx, cy, cz, radius)",
            f"    rgba{q}_carve_ids_lo: vec4<u32>,    // 4 label values, 0 = unused",
            f"    rgba{q}_carve_ids_hi: vec4<u32>,    // 4 label values, 0 = unused",
        ]
    body = "\n".join(lines)
    return f"""
struct Mat {{
    scene_bmin: vec4<f32>,
    scene_bmax: vec4<f32>,
    scene_step: f32,
    _pad0: vec3<f32>,
{body}
    grid_p2t: mat4x4<f32>,
    grid_enabled: vec4<f32>,  // (on, gain, _, _)
    // Clip-plane tail. `clip_count` > 0 enables clipping; each active
    // plane is (nx, ny, nz, offset) in world space and a ray sample is
    // discarded when dot(wp, n) + offset < 0.
    clip_planes: array<vec4<f32>, 8>,
    clip_count: vec4<u32>,         // (count, _, _, _)
}};
@group(0) @binding(1) var<uniform> u_mat: Mat;
"""


def _vtk_depth_wgsl(n):
    """Scene-wide VTK depth texture + sampler. Placed AFTER all per-field
    bindings so the per-field binding indices stay slot*4+2."""
    b = 2 + n * 4
    return f"""
@group(0) @binding({b+0}) var s_vtkdepth: sampler;
@group(0) @binding({b+1}) var t_vtkdepth: texture_2d<f32>;
"""


def _grid_transform_wgsl(n):
    """Scene-wide (one) grid-transform displacement field. Sampled per ray
    step to warp the world position before the volume texture lookup.
    Mirrors slicer_wgpu.fields.transform.TransformField. When no grid is
    active, u_mat.grid_enabled.x < 0.5 and displacement_grid returns zero.
    """
    b = 2 + n * 4 + 2  # after the 2 VTK-depth slots
    return f"""
@group(0) @binding({b+0}) var s_grid: sampler;
@group(0) @binding({b+1}) var t_grid: texture_3d<f32>;

fn displacement_grid(wp: vec3<f32>) -> vec3<f32> {{
    if (u_mat.grid_enabled.x < 0.5) {{ return vec3<f32>(0.0); }}
    let tex4 = u_mat.grid_p2t * vec4<f32>(wp, 1.0);
    let tex = tex4.xyz;
    if (any(tex < vec3<f32>(0.0)) || any(tex > vec3<f32>(1.0))) {{
        return vec3<f32>(0.0);
    }}
    let d = textureSampleLevel(t_grid, s_grid, tex, 0.0).xyz;
    return u_mat.grid_enabled.y * d;
}}

fn warp(wp: vec3<f32>) -> vec3<f32> {{
    return wp + displacement_grid(wp);
}}
"""


def _field_wgsl(slot):
    i = slot
    b0 = 2 + slot * 4
    return f"""
@group(0) @binding({b0+0}) var s_vol{i}: sampler;
@group(0) @binding({b0+1}) var t_vol{i}: texture_3d<f32>;
@group(0) @binding({b0+2}) var s_lut{i}: sampler;
@group(0) @binding({b0+3}) var t_lut{i}: texture_1d<f32>;

fn sample_v_{i}(wp: vec3<f32>) -> f32 {{
    let wpw = warp(wp);
    let t4 = u_mat.img{i}_p2t * vec4<f32>(wpw, 1.0);
    let t = t4.xyz;
    if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {{ return 0.0; }}
    return textureSampleLevel(t_vol{i}, s_vol{i}, t, 0.0).r;
}}
fn sample_v_c_{i}(wp: vec3<f32>) -> f32 {{
    let wpw = warp(wp);
    let t4 = u_mat.img{i}_p2t * vec4<f32>(wpw, 1.0);
    let t = clamp(t4.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(t_vol{i}, s_vol{i}, t, 0.0).r;
}}
fn sample_field_{i}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.img{i}_step_unit.z < 0.5) {{ return vec4<f32>(0.0); }}
    let wpw = warp(wp);
    let t4 = u_mat.img{i}_p2t * vec4<f32>(wpw, 1.0);
    if (any(t4.xyz < vec3<f32>(0.0)) || any(t4.xyz > vec3<f32>(1.0))) {{
        return vec4<f32>(0.0);
    }}
    let s = sample_v_{i}(wp);
    let clim = u_mat.img{i}_clim;
    let li = clamp((s - clim.x) / max(clim.y - clim.x, 1e-6), 0.0, 1.0);
    // textureSample (implicit LOD) rather than textureSampleLevel because
    // naga rejects Exact level on 1D textures -- the LUT has a single mip,
    // so implicit derivatives don't affect the result.
    let tf = textureSample(t_lut{i}, s_lut{i}, li);
    if (tf.a <= 0.0) {{ return vec4<f32>(0.0); }}
    let step = max(u_mat.img{i}_step_unit.x, 1e-3);
    let unit = max(u_mat.img{i}_step_unit.y, 1e-3);
    let op = clamp(tf.a * (step / unit), 0.0, 1.0);

    let h = step;
    let gx = sample_v_c_{i}(wp+vec3<f32>(h,0,0)) - sample_v_c_{i}(wp-vec3<f32>(h,0,0));
    let gy = sample_v_c_{i}(wp+vec3<f32>(0,h,0)) - sample_v_c_{i}(wp-vec3<f32>(0,h,0));
    let gz = sample_v_c_{i}(wp+vec3<f32>(0,0,h)) - sample_v_c_{i}(wp-vec3<f32>(0,0,h));
    let grad = vec3<f32>(gx,gy,gz) / (2.0 * h);
    let glen = length(grad);
    var lit = tf.rgb * u_mat.img{i}_shade.x;
    if (glen > 1e-6) {{
        var n = grad / glen;
        if (dot(n, -rd) < 0.0) {{ n = -n; }}
        let ldn = max(dot(-rd, n), 0.0);
        let r = normalize(2.0 * ldn * n + rd);
        let rdv = max(dot(r, -rd), 0.0);
        lit = lit + tf.rgb * (u_mat.img{i}_shade.y * ldn)
              + vec3<f32>(u_mat.img{i}_shade.z * pow(rdv, max(u_mat.img{i}_shade.w, 1.0)));
    }}
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(lit * op, op);
}}
"""


def _seg_field_wgsl(slot, n):
    """Per-segment bindings + isosurface sampler. Binding layout is after
    per-image (2+4n), VTK depth (+2), and grid transform (+2), so segments
    start at 2 + 4n + 4 and use 2 slots each (sampler + 3D texture).

    The presence field is an r8unorm 3D texture, 255 = inside, 0 = outside.
    Hardware linear filtering gives a smoothed v in [0,1] with a ~1 voxel
    transition zone at segment boundaries.

    Local SDF approximation (first-order Taylor):
        d(x) = (v(x) - 0.5) / |grad v(x)|     (mm, with grad in world units)
    where |grad v| ~= 1/voxel_mm in the boundary region. The opacity transfer
    function peaks at d=0 and falls to 0 over band_mm (stashed in seg_params.x).
    Shading uses the same grad as the surface normal.
    """
    i = slot
    b0 = 2 + n * 4 + 4 + slot * 2
    return f"""
@group(0) @binding({b0+0}) var s_seg{i}: sampler;
@group(0) @binding({b0+1}) var t_seg{i}: texture_3d<f32>;

fn sample_v_seg_{i}(wp: vec3<f32>) -> f32 {{
    let wpw = warp(wp);
    let t4 = u_mat.seg{i}_p2t * vec4<f32>(wpw, 1.0);
    let t = t4.xyz;
    if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {{ return 0.0; }}
    // t_seg{i} is a rgba16float texture pre-blurred by a separable Gaussian
    // compute pass (see _smooth_segment in the bridge). Trilinear fetch
    // only -- the low-pass is baked in.
    return textureSampleLevel(t_seg{i}, s_seg{i}, t, 0.0).r;
}}

fn sample_seg_{i}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.seg{i}_color.a <= 0.0) {{ return vec4<f32>(0.0); }}
    let wpw = warp(wp);
    let t4 = u_mat.seg{i}_p2t * vec4<f32>(wpw, 1.0);
    if (any(t4.xyz < vec3<f32>(0.0)) || any(t4.xyz > vec3<f32>(1.0))) {{
        return vec4<f32>(0.0);
    }}
    let v = sample_v_seg_{i}(wp);
    // Skip deep interior / exterior: |grad| will be ~0 there and we'd
    // emit nothing anyway; this avoids 6 gradient fetches per step.
    if (v <= 0.02 || v >= 0.98) {{ return vec4<f32>(0.0); }}

    let h = max(u_mat.scene_step, 1e-3);
    let gx = sample_v_seg_{i}(wp+vec3<f32>(h,0,0)) - sample_v_seg_{i}(wp-vec3<f32>(h,0,0));
    let gy = sample_v_seg_{i}(wp+vec3<f32>(0,h,0)) - sample_v_seg_{i}(wp-vec3<f32>(0,h,0));
    let gz = sample_v_seg_{i}(wp+vec3<f32>(0,0,h)) - sample_v_seg_{i}(wp-vec3<f32>(0,0,h));
    let grad = vec3<f32>(gx, gy, gz) / (2.0 * h);
    let glen = length(grad);
    if (glen < 1e-5) {{ return vec4<f32>(0.0); }}

    // Local signed distance (mm). abs() is all we need for opacity.
    let d_mm = abs((v - 0.5) / glen);
    let band = max(u_mat.seg{i}_params.x, 1e-3);
    let a = 1.0 - clamp(d_mm / band, 0.0, 1.0);
    if (a <= 0.0) {{ return vec4<f32>(0.0); }}
    let op = clamp(a * u_mat.seg{i}_color.a, 0.0, 1.0);

    // Phong shading; flip normal to face the camera like ImageField does.
    var n = grad / glen;
    if (dot(n, -rd) < 0.0) {{ n = -n; }}
    let ldn = max(dot(-rd, n), 0.0);
    let r = normalize(2.0 * ldn * n + rd);
    let rdv = max(dot(r, -rd), 0.0);
    let color = u_mat.seg{i}_color.rgb;
    let sh = u_mat.seg{i}_shade;
    var lit = color * sh.x
            + color * (sh.y * ldn)
            + vec3<f32>(sh.z * pow(rdv, max(sh.w, 1.0)));
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(lit * op, op);
}}
"""


def _seg_surface_field_wgsl(slot, n):
    """Per-segment bindings + gradient-opacity surface sampler. Same bind-
    ing layout and sample_v_seg_{i} helper as the iso-surface variant; the
    only difference is sample_seg_{i}'s alpha formula.

    Emission rule:
        dα/ds = opacity * |grad v|         (grad in per-mm)
    The integral along a ray crossing the 0->1 presence transition once
    equals opacity * total_variation(v) = opacity, *independent of how
    thick the ray's path through the segment is*. This gives parity with
    Slicer's polydata surface rendering: a 30%-opaque segment always
    accumulates ~0.3 alpha per surface crossing, front + back face add
    like two transparent shells.
    """
    i = slot
    b0 = 2 + n * 4 + 4 + slot * 2
    return f"""
@group(0) @binding({b0+0}) var s_seg{i}: sampler;
@group(0) @binding({b0+1}) var t_seg{i}: texture_3d<f32>;

fn sample_v_seg_{i}(wp: vec3<f32>) -> f32 {{
    let wpw = warp(wp);
    let t4 = u_mat.seg{i}_p2t * vec4<f32>(wpw, 1.0);
    let t = t4.xyz;
    if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {{ return 0.0; }}
    return textureSampleLevel(t_seg{i}, s_seg{i}, t, 0.0).r;
}}

fn sample_seg_{i}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.seg{i}_color.a <= 0.0) {{ return vec4<f32>(0.0); }}
    let wpw = warp(wp);
    let t4 = u_mat.seg{i}_p2t * vec4<f32>(wpw, 1.0);
    if (any(t4.xyz < vec3<f32>(0.0)) || any(t4.xyz > vec3<f32>(1.0))) {{
        return vec4<f32>(0.0);
    }}
    let v = sample_v_seg_{i}(wp);
    // Fast-skip deep interior / exterior: |grad| ~ 0 and there's no
    // surface to shade, so we'd emit nothing anyway.
    if (v <= 0.02 || v >= 0.98) {{ return vec4<f32>(0.0); }}

    let h = max(u_mat.scene_step, 1e-3);
    let gx = sample_v_seg_{i}(wp+vec3<f32>(h,0,0)) - sample_v_seg_{i}(wp-vec3<f32>(h,0,0));
    let gy = sample_v_seg_{i}(wp+vec3<f32>(0,h,0)) - sample_v_seg_{i}(wp-vec3<f32>(0,h,0));
    let gz = sample_v_seg_{i}(wp+vec3<f32>(0,0,h)) - sample_v_seg_{i}(wp-vec3<f32>(0,0,h));
    let grad = vec3<f32>(gx, gy, gz) / (2.0 * h);
    let glen = length(grad);
    if (glen < 1e-5) {{ return vec4<f32>(0.0); }}

    // Gradient-opacity emission: α_step = opacity * |grad v| * step.
    // Integrated over the 0->1 presence transition this sums to
    // opacity * 1 = opacity, regardless of transition thickness.
    let step = max(u_mat.scene_step, 1e-3);
    let op = clamp(u_mat.seg{i}_color.a * glen * step, 0.0, 1.0);
    if (op <= 0.0) {{ return vec4<f32>(0.0); }}

    // Phong from the same gradient used for emission.
    var n = grad / glen;
    if (dot(n, -rd) < 0.0) {{ n = -n; }}
    let ldn = max(dot(-rd, n), 0.0);
    let r = normalize(2.0 * ldn * n + rd);
    let rdv = max(dot(r, -rd), 0.0);
    let color = u_mat.seg{i}_color.rgb;
    let sh = u_mat.seg{i}_shade;
    var lit = color * sh.x
            + color * (sh.y * ldn)
            + vec3<f32>(sh.z * pow(rdv, max(sh.w, 1.0)));
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(lit * op, op);
}}
"""


def _rgba_field_wgsl(slot, n, m, render_mode="density"):
    """A pre-baked RGBA 3D volume.

    Density mode (modulate_by_ct=True): RGB = palette color, A = smoothed
    presence * palette opacity. Surface ramp lives in the alpha texture's
    Gaussian-blurred boundary; the shader reads alpha directly and uses a
    multi-tap gradient to suppress the residual voxel stair-step.
        dα/ds = alpha / unit  (accumulating; thicker path -> more opaque).

    Surface mode (modulate_by_ct=False): RGB = grown palette color, A =
    signed distance to the segment surface, in mm (negative inside,
    positive outside). The ramp lives in the analytic SDF, so the
    alpha gradient is mathematically smooth and a single central
    difference is enough; per-step contribution is constant inside the
    band.
        alpha = clamp(0.5 - sdf/band, 0, 1)
        dα/ds = step / band  (thickness-independent; integrates to 1
                              across each surface crossing).

    Bindings follow the existing (per-image, vtk-depth, grid, per-segment)
    tail: 3 per rgba slot. The third slot is the (uint) labelmap used for
    segment-aware sphere carving (independent of the SDF/density choice).
    """
    q = slot
    b0 = 2 + n * 4 + 4 + m * 2 + slot * 3
    # Out-of-bounds sentinel for sample_rgba_v_q. Density mode wants
    # alpha=0 (no presence), surface mode wants a large positive SDF (far
    # outside any segment) so the band gate culls the sample.
    if render_mode == "surface":
        oob_sentinel = "vec4<f32>(0.0, 0.0, 0.0, 1e3)"
    else:
        oob_sentinel = "vec4<f32>(0.0)"
    head = f"""
@group(0) @binding({b0+0}) var s_rgba{q}: sampler;
@group(0) @binding({b0+1}) var t_rgba{q}: texture_3d<f32>;
@group(0) @binding({b0+2}) var t_label{q}: texture_3d<u32>;

fn carve_rgba_{q}(wp: vec3<f32>) -> bool {{
    let r = u_mat.rgba{q}_carve_center.w;
    if (r <= 0.0) {{ return false; }}
    let d = wp - u_mat.rgba{q}_carve_center.xyz;
    if (dot(d, d) >= r * r) {{ return false; }}
    let lp4 = u_mat.rgba{q}_w2l * vec4<f32>(wp, 1.0);
    let lp = lp4.xyz;
    if (any(lp < vec3<f32>(0.0)) || any(lp > vec3<f32>(1.0))) {{ return false; }}
    let dims = vec3<f32>(textureDimensions(t_label{q}));
    let ijk = vec3<i32>(clamp(lp * dims,
                              vec3<f32>(0.0),
                              dims - vec3<f32>(1.0)));
    let lv = textureLoad(t_label{q}, ijk, 0).r;
    if (lv == 0u) {{ return false; }}
    let lo = u_mat.rgba{q}_carve_ids_lo;
    let hi = u_mat.rgba{q}_carve_ids_hi;
    if (lv == lo.x || lv == lo.y || lv == lo.z || lv == lo.w) {{ return true; }}
    if (lv == hi.x || lv == hi.y || lv == hi.z || lv == hi.w) {{ return true; }}
    return false;
}}

fn sample_rgba_v_{q}(wp: vec3<f32>) -> vec4<f32> {{
    let wpw = warp(wp);
    let t4 = u_mat.rgba{q}_p2t * vec4<f32>(wpw, 1.0);
    let t = t4.xyz;
    if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {{
        return {oob_sentinel};
    }}
    return textureSampleLevel(t_rgba{q}, s_rgba{q}, t, 0.0);
}}
"""

    if render_mode == "surface":
        # SDF-in-alpha. RGB is grown palette color (median-filtered).
        return head + f"""
fn sample_rgba_{q}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.rgba{q}_step_unit.z < 0.5) {{ return vec4<f32>(0.0); }}
    if (carve_rgba_{q}(wp)) {{ return vec4<f32>(0.0); }}

    let band = max(u_mat.rgba{q}_step_unit.w, 1e-3);
    let v4 = sample_rgba_v_{q}(wp);
    let sdf = v4.a;
    // Outside the transition band (with margin) skip everything: no
    // gradient samples, no shading. SDF magnitude > band/2 + scene_step
    // guarantees the central-difference gradient stencil also lies
    // outside the band.
    let step = max(u_mat.scene_step, 1e-3);
    if (sdf > 0.5 * band + step) {{ return vec4<f32>(0.0); }}

    // Single central-difference gradient on the SDF. SDF is mathematically
    // smooth so no multi-tap smoothing is needed. We still need to honor
    // carving inside the gradient stencil so the cut surface gets a clean
    // normal; sample_rgba_v_q is shared and the ray-march loop's outer
    // carve-test covers the center sample.
    let h = step;
    let gx = sample_rgba_v_{q}(wp+vec3<f32>(h,0,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(h,0,0)).a;
    let gy = sample_rgba_v_{q}(wp+vec3<f32>(0,h,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,h,0)).a;
    let gz = sample_rgba_v_{q}(wp+vec3<f32>(0,0,h)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,0,h)).a;
    let grad_sdf = vec3<f32>(gx, gy, gz) / (2.0 * h);
    let glen = length(grad_sdf);

    // Phong using the inward-pointing normal (-grad SDF). Uses the same
    // shade tuple semantics as density mode.
    let color = v4.rgb;
    let shade = u_mat.rgba{q}_shade;
    var lit = color * shade.x;
    if (glen > 1e-6) {{
        var n = -grad_sdf / glen;
        if (dot(n, -rd) < 0.0) {{ n = -n; }}
        let ldn = max(dot(-rd, n), 0.0);
        let r = normalize(2.0 * ldn * n + rd);
        let rdv = max(dot(r, -rd), 0.0);
        lit = lit + color * (shade.y * ldn)
                  + vec3<f32>(shade.z * pow(rdv, max(shade.w, 1.0)));
    }}
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));

    // Per-step opacity for surface-mode compositing. Gradient-opacity:
    // dα/ds = |grad α| = |grad_sdf · rd| / band inside the band, 0
    // outside -- gives the thin-shell "surface" look (contribution
    // concentrated where the alpha ramp lives, not throughout the
    // segment interior). The naive form `op = |grad α| · step` has
    // total integral 1.0 per crossing but only converges to 1 - 1/e
    // ≈ 0.63 under outer over-compositing; we instead apply opacity
    // correction so the total integrated alpha hits alpha_target
    // (≈ fully opaque) per surface crossing, independent of ray
    // angle and step size:
    //     op = 1 - (1 - α_t) ^ (|grad α| · step)
    // Across a band crossing, ∫|grad α| ds = 1, so
    //     ∏ (1 - op_i)  =  (1 - α_t) ^ ∑(|grad α|·step)  =  (1 - α_t)
    // and integrated.a converges to α_t.
    let in_band = step(abs(sdf), 0.5 * band);
    let dalpha_ds = abs(dot(grad_sdf, rd)) / band;
    let alpha_target = 0.95;
    let op = (1.0 - pow(1.0 - alpha_target, dalpha_ds * step)) * in_band;
    return vec4<f32>(lit * op, op);
}}
"""
    else:
        # Density mode (existing). RGB = palette color, A = smoothed
        # presence * palette opacity. Multi-tap gradient smoothing in the
        # screen plane suppresses voxel-axis stair-step.
        return head + f"""
fn grad_rgba_{q}(wp: vec3<f32>, h: f32) -> vec3<f32> {{
    let gx = sample_rgba_v_{q}(wp+vec3<f32>(h,0,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(h,0,0)).a;
    let gy = sample_rgba_v_{q}(wp+vec3<f32>(0,h,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,h,0)).a;
    let gz = sample_rgba_v_{q}(wp+vec3<f32>(0,0,h)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,0,h)).a;
    return vec3<f32>(gx, gy, gz) / (2.0 * h);
}}

fn sample_rgba_{q}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.rgba{q}_step_unit.z < 0.5) {{ return vec4<f32>(0.0); }}
    if (carve_rgba_{q}(wp)) {{ return vec4<f32>(0.0); }}
    let v4 = sample_rgba_v_{q}(wp);
    let alpha = v4.a;
    if (alpha <= 1e-3) {{ return vec4<f32>(0.0); }}

    let h = max(max(u_mat.rgba{q}_step_unit.w, u_mat.scene_step), 1e-3);

    let g0 = grad_rgba_{q}(wp, h);
    var u_axis = cross(rd, vec3<f32>(0.0, 1.0, 0.0));
    if (dot(u_axis, u_axis) < 1e-6) {{
        u_axis = cross(rd, vec3<f32>(1.0, 0.0, 0.0));
    }}
    u_axis = normalize(u_axis);
    let v_axis = normalize(cross(rd, u_axis));
    let g1 = grad_rgba_{q}(wp + u_axis * h, h);
    let g2 = grad_rgba_{q}(wp - u_axis * h, h);
    let g3 = grad_rgba_{q}(wp + v_axis * h, h);
    let g4 = grad_rgba_{q}(wp - v_axis * h, h);
    let grad = (g0 * 2.0 + g1 + g2 + g3 + g4) * (1.0 / 6.0);
    let glen = length(grad);

    let color = v4.rgb;
    let shade = u_mat.rgba{q}_shade;
    var lit = color * shade.x;
    if (glen > 1e-6) {{
        var n = grad / glen;
        if (dot(n, -rd) < 0.0) {{ n = -n; }}
        let ldn = max(dot(-rd, n), 0.0);
        let r = normalize(2.0 * ldn * n + rd);
        let rdv = max(dot(r, -rd), 0.0);
        lit = lit + color * (shade.y * ldn)
                  + vec3<f32>(shade.z * pow(rdv, max(shade.w, 1.0)));
    }}
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));

    let step = max(u_mat.scene_step, 1e-3);
    let op = clamp(alpha *
                   (step / max(u_mat.rgba{q}_step_unit.y, 1e-3)),
                   0.0, 1.0);
    return vec4<f32>(lit * op, op);
}}
"""


def _main_wgsl(n, m, k, has_fragments=False, frag_binding=0, K=32):
    img_calls = "\n".join([
        f"        {{ let c = sample_field_{i}(wp, rd); "
        f"if (c.a > 0.0) {{ sum = sum + c; }} }}"
        for i in range(n)])
    seg_calls = "\n".join([
        f"        {{ let c = sample_seg_{j}(wp, rd); "
        f"if (c.a > 0.0) {{ sum = sum + c; }} }}"
        for j in range(m)])
    rgba_calls = "\n".join([
        f"        {{ let c = sample_rgba_{q}(wp, rd); "
        f"if (c.a > 0.0) {{ sum = sum + c; }} }}"
        for q in range(k)])
    calls_list = [c for c in (img_calls, seg_calls, rgba_calls) if c]
    calls = "\n".join(calls_list)

    # Fragment-integration plumbing. When the bridge has a FragmentField,
    # the main shader pulls per-pixel sorted (depth, packed_rgba) entries
    # from a storage buffer and interleaves them by depth with the volume
    # samples (depth-correct compositing of rasterized strands + volume).
    if has_fragments:
        frag_decls = f"""
@group(0) @binding({frag_binding+0}) var<storage, read> u_frag_counts: array<u32>;
@group(0) @binding({frag_binding+1}) var<storage, read> u_frag_data:   array<u32>;
"""
        frag_init = f"""
    let pixel_idx = u32(v.position.y) * u32(u_cam.size.x) + u32(v.position.x);
    let frag_count = min(u_frag_counts[pixel_idx], {K}u);
    var frag_idx: u32 = 0u;
"""
        frag_consume_in_loop = f"""
        // Consume any A-buffer fragments that lie in front of the current
        // ray sample (depth_01 <= world_depth_01(wp)). Standard front-to-
        // back over-compositing.
        let curr_depth = world_depth_01(wp);
        loop {{
            if (frag_idx >= frag_count) {{ break; }}
            let f_depth = bitcast<f32>(
                u_frag_data[(pixel_idx * {K}u + frag_idx) * 2u]);
            if (f_depth > curr_depth) {{ break; }}
            let pkd = u_frag_data[(pixel_idx * {K}u + frag_idx) * 2u + 1u];
            let fr = f32(pkd & 0xffu) / 255.0;
            let fg = f32((pkd >> 8u) & 0xffu) / 255.0;
            let fb = f32((pkd >> 16u) & 0xffu) / 255.0;
            let fa = f32((pkd >> 24u) & 0xffu) / 255.0;
            integrated = vec4<f32>(
                integrated.rgb + (1.0 - integrated.a) * vec3<f32>(fr, fg, fb),
                integrated.a   + (1.0 - integrated.a) * fa);
            frag_idx = frag_idx + 1u;
        }}
"""
        frag_drain = f"""
    // After the volume ray-march terminates (hit t_far, alpha-saturated,
    // or safety break), drain any remaining sorted fragments that lie
    // behind the volume's far face -- they were never reached but should
    // still composite as background.
    loop {{
        if (frag_idx >= frag_count) {{ break; }}
        if (integrated.a >= 0.999) {{ break; }}
        let pkd = u_frag_data[(pixel_idx * {K}u + frag_idx) * 2u + 1u];
        let fr = f32(pkd & 0xffu) / 255.0;
        let fg = f32((pkd >> 8u) & 0xffu) / 255.0;
        let fb = f32((pkd >> 16u) & 0xffu) / 255.0;
        let fa = f32((pkd >> 24u) & 0xffu) / 255.0;
        integrated = vec4<f32>(
            integrated.rgb + (1.0 - integrated.a) * vec3<f32>(fr, fg, fb),
            integrated.a   + (1.0 - integrated.a) * fa);
        frag_idx = frag_idx + 1u;
    }}
"""
    else:
        frag_decls = ""
        frag_init = ""
        frag_consume_in_loop = ""
        frag_drain = ""

    return f"""
{frag_decls}
@fragment
fn fs_main(v: Varyings) -> FragmentOutput {{
    var out: FragmentOutput;
    // Sub-pixel camera jitter for TAA. size.zw carries the per-frame
    // jitter offset in NDC units (typically a Halton(2,3) sequence
    // scaled by 1/viewport). After the main render, a TAA composite
    // pass blends with a history buffer so the jitter averages out.
    let ndc_x = (v.position.x / u_cam.size.x) * 2.0 - 1.0 + u_cam.size.z;
    let ndc_y = 1.0 - (v.position.y / u_cam.size.y) * 2.0 + u_cam.size.w;
    let wn = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
    let wf = ndc_to_world(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0));
    let ro = wn;
    let rd = normalize(wf - wn);
    let tr = ray_aabb(ro, rd, u_mat.scene_bmin.xyz, u_mat.scene_bmax.xyz);
    var t_near = max(tr.x, 0.0);
    var t_far = tr.y;

    // Clip t_far at VTK's geometry depth so opaque scene props (fiducials,
    // markups, ROI handles) occlude the volume where they lie in front of
    // or inside it. VTK's depth buffer is in [0,1]; a value of 1.0 means
    // "no geometry / far plane" and leaves the volume unobstructed.
    let vd = textureLoad(t_vtkdepth, vec2<i32>(v.position.xy), 0).r;
    if (vd < 1.0) {{
        let gw = ndc_to_world(vec4<f32>(ndc_x, ndc_y, vd, 1.0));
        let tg = dot(gw - ro, rd);
        if (tg > 0.0) {{ t_far = min(t_far, tg); }}
    }}

    let step = max(u_mat.scene_step, 1e-3);
    var integrated = vec4<f32>(0.0);
{frag_init}
    if (t_far <= t_near) {{
        // Volume contributes nothing along this ray, but A-buffer
        // fragments at this pixel may still be visible.
{frag_drain}
        out.color = integrated;
        return out;
    }}
    // dt-jitter the ray-march start by a per-pixel-per-frame hash so
    // each frame samples a different sub-step alignment. TAA averages
    // out the resulting noise and removes the regular banding that
    // appears at TF discontinuities under low sample rate.
    let _seed = v.position.xy + vec2<f32>(u_cam.taa.x * 0.12387, u_cam.taa.x * 0.7351);
    var t = t_near + fract(sin(dot(_seed, vec2<f32>(12.9898,78.233))) * 43758.5453) * step;
    var safety: i32 = 0;
    loop {{
        if (t >= t_far) {{ break; }}
        if (safety >= 2048) {{ break; }}
        if (integrated.a >= 0.99) {{ break; }}
        let wp = ro + rd * t;
{frag_consume_in_loop}

        // Clip-plane test: if any active plane says wp is on its
        // negative side, skip this sample. The wp > n * offset check
        // keeps the positive side of each plane.
        var clipped = false;
        let ccount = u_mat.clip_count.x;
        for (var ci = 0u; ci < ccount; ci = ci + 1u) {{
            let p = u_mat.clip_planes[ci];
            if (dot(wp, p.xyz) + p.w < 0.0) {{
                clipped = true;
                break;
            }}
        }}
        if (clipped) {{ t = t + step; safety = safety + 1; continue; }}

        var sum = vec4<f32>(0.0);
{calls}
        if (sum.a > 0.0) {{
            let op = clamp(sum.a, 0.0, 1.0);
            integrated = vec4<f32>(
                integrated.rgb + (1.0 - integrated.a) * sum.rgb,
                integrated.a   + (1.0 - integrated.a) * op);
        }}
        t = t + step;
        safety = safety + 1;
    }}
{frag_drain}
    out.color = integrated;
    return out;
}}
"""


def _build_wgsl(n, m, k, segment_render_mode="iso",
                rgba_render_modes=None,
                has_fragments=False, frag_binding=0, K=32):
    parts = [_HEADER, _mat_struct_wgsl(n, m, k)]
    # Grid transform needs to come BEFORE the per-field functions (they call
    # warp()); its bindings sit after the VTK depth bindings.
    parts.append(_vtk_depth_wgsl(n))
    parts.append(_grid_transform_wgsl(n))
    for i in range(n):
        parts.append(_field_wgsl(i))
    seg_gen = (_seg_surface_field_wgsl
               if segment_render_mode == "surface"
               else _seg_field_wgsl)
    for j in range(m):
        parts.append(seg_gen(j, n))
    if rgba_render_modes is None:
        rgba_render_modes = ["density"] * k
    for q in range(k):
        mode = (rgba_render_modes[q]
                if q < len(rgba_render_modes) else "density")
        parts.append(_rgba_field_wgsl(q, n, m, render_mode=mode))
    parts.append(_main_wgsl(n, m, k,
                            has_fragments=has_fragments,
                            frag_binding=frag_binding,
                            K=K))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Material UBO packing
# ---------------------------------------------------------------------------

# 64 bytes: vec4 bmin(16) + vec4 bmax(16) + f32 step + vec3 _pad0, rounded
# up to next 16-byte boundary (WGSL aligns the following mat4x4 to 16).
_SCENE_BYTES = 64
_PER_FIELD_BYTES = 112   # mat4(64) + 3 * vec4(48) per image field
_PER_SEG_BYTES = 112     # mat4(64) + 3 * vec4(48) per segment
_PER_RGBA_BYTES = 208    # mat4(64) + 2 vec4(32) + carve: mat4(64) + 3 vec4(48)
# Grid transform tail: mat4x4 (64) + vec4 enabled (16) = 80 bytes
_GRID_TAIL_BYTES = 80
# Clip-plane tail: 8 * vec4 planes + vec4 count = 144 bytes
_CLIP_TAIL_BYTES = 8 * 16 + 16


def _mat_ubo_size(n, m, k):
    total = (_SCENE_BYTES
             + n * _PER_FIELD_BYTES
             + m * _PER_SEG_BYTES
             + k * _PER_RGBA_BYTES
             + _GRID_TAIL_BYTES
             + _CLIP_TAIL_BYTES)
    return (total + 15) & ~15


def _pack_material(fields, segments, rgba_volumes, bmin, bmax, step,
                   grid_p2t=None, grid_gain=1.0,
                   clip_planes=None):
    """Pack scene + per-field + per-segment + grid-transform tail.
    grid_p2t is the world->texture matrix for the displacement grid
    (None = no grid)."""
    n = len(fields)
    m = len(segments)
    k = len(rgba_volumes)
    buf = bytearray(_mat_ubo_size(n, m, k))
    arr = np.frombuffer(buf, dtype=np.float32)
    arr[0:3] = bmin
    arr[4:7] = bmax
    arr[8] = step
    for i, fld in enumerate(fields):
        off = (_SCENE_BYTES // 4) + i * (_PER_FIELD_BYTES // 4)
        p2t = _p2t_for_field(fld)
        arr[off:off+16] = p2t.T.ravel()
        arr[off+16:off+18] = fld.clim
        arr[off+20:off+24] = [fld.k_ambient, fld.k_diffuse, fld.k_specular,
                              max(fld.shininess, 1.0)]
        # Force visible=1 since the bridge hides the VRDN to silence
        # Slicer's native VR mapper; we always want to render our fields.
        arr[off+24:off+28] = [fld.sample_step_mm, fld.opacity_unit_distance,
                              1.0, 0.0]
    for j, s in enumerate(segments):
        off = (_SCENE_BYTES // 4) + n * (_PER_FIELD_BYTES // 4) \
              + j * (_PER_SEG_BYTES // 4)
        p2t = _p2t_for_field(s)
        arr[off:off+16] = p2t.T.ravel()
        arr[off+16:off+20] = [*s.color_rgb, s.opacity]
        arr[off+20:off+24] = [s.k_ambient, s.k_diffuse, s.k_specular,
                              max(s.shininess, 1.0)]
        arr[off+24:off+28] = [s.band_mm, 0.0, float(s.mode), 0.0]
    for q, r in enumerate(rgba_volumes):
        off = ((_SCENE_BYTES // 4)
               + n * (_PER_FIELD_BYTES // 4)
               + m * (_PER_SEG_BYTES // 4)
               + q * (_PER_RGBA_BYTES // 4))
        p2t = _p2t_for_field(r)
        arr[off:off+16] = p2t.T.ravel()
        # .w double-duty: gradient_h_mm in density mode, band_mm in
        # surface mode (the SDF -> alpha transition band).
        if getattr(r, "render_mode", "density") == "surface":
            param_w = float(getattr(r, "band_mm", 0.0))
        else:
            param_w = float(getattr(r, "gradient_h_mm", 0.0))
        arr[off+16:off+20] = [r.sample_step_mm, r.opacity_unit_distance,
                              1.0 if r.visible else 0.0, param_w]
        arr[off+20:off+24] = [r.k_ambient, r.k_diffuse, r.k_specular,
                              max(r.shininess, 1.0)]
        # Carve tail: world->label mat4, then center+radius and 8 packed
        # label-value ids (4 per vec4, 0 = unused slot).
        w2l_world = np.asarray(getattr(r, "_world_to_label_tex",
                                       np.eye(4)), dtype=np.float64)
        # _world_to_label_tex was computed against world (no per-volume
        # parent transform applied). Compose w/ inverse world_from_local so
        # the same wp -> texture chain works after a transform on the field.
        w_from_l = np.asarray(r.world_from_local, dtype=np.float64)
        try:
            l_from_w = np.linalg.inv(w_from_l)
        except np.linalg.LinAlgError:
            l_from_w = np.eye(4, dtype=np.float64)
        w2l = (w2l_world @ l_from_w).astype(np.float32)
        arr[off+24:off+40] = w2l.T.ravel()
        c = np.asarray(getattr(r, "carve_center_world",
                               np.zeros(3)), dtype=np.float32)
        radius = float(getattr(r, "carve_radius_mm", 0.0))
        arr[off+40:off+44] = [float(c[0]), float(c[1]), float(c[2]), radius]
        ids = list(getattr(r, "carve_segment_label_values", []) or [])[:8]
        ids = ids + [0] * (8 - len(ids))
        ids_u32 = np.asarray(ids, dtype=np.uint32).view(np.float32)
        arr[off+44:off+48] = ids_u32[0:4]
        arr[off+48:off+52] = ids_u32[4:8]
    # Grid transform tail
    tail_off = ((_SCENE_BYTES // 4)
                + n * (_PER_FIELD_BYTES // 4)
                + m * (_PER_SEG_BYTES // 4)
                + k * (_PER_RGBA_BYTES // 4))
    if grid_p2t is not None:
        arr[tail_off:tail_off+16] = np.asarray(grid_p2t, dtype=np.float32).T.ravel()
        arr[tail_off+16:tail_off+20] = [1.0, float(grid_gain), 0.0, 0.0]
    else:
        arr[tail_off:tail_off+16] = np.eye(4, dtype=np.float32).T.ravel()
        arr[tail_off+16:tail_off+20] = [0.0, 0.0, 0.0, 0.0]
    # Clip-plane tail: 8 vec4 planes + 1 vec4 (count + pad)
    clip_off = tail_off + 20
    planes_arr = np.asarray(clip_planes, dtype=np.float32) if clip_planes else None
    if planes_arr is None or len(planes_arr) == 0:
        arr[clip_off:clip_off + 32] = 0.0
        # count in the last vec4 (slots 32..36), stored as f32 for the single
        # shared memory view; reinterpretation as u32 in WGSL works because
        # 0.0f and 0u share the same bit pattern.
        arr[clip_off + 32:clip_off + 36] = 0.0
    else:
        arr[clip_off:clip_off + 32] = 0.0
        count = min(len(planes_arr), 8)
        for i in range(count):
            arr[clip_off + 4 * i: clip_off + 4 * i + 4] = planes_arr[i]
        # Write count as u32 into the same float slot via a bit-cast.
        count_bits = np.uint32(count).view(np.float32)
        arr[clip_off + 32] = count_bits
        arr[clip_off + 33: clip_off + 36] = 0.0
    return bytes(buf)


def _p2t_for_field(f):
    """Compose ImageField's patient_to_texture with its world_from_local inverse
    to get the world->texture transform used by the ray-march."""
    w2l = np.linalg.inv(np.asarray(f.world_from_local, dtype=np.float64))
    p2t = np.asarray(f.patient_to_texture, dtype=np.float64) @ w2l
    return p2t.astype(np.float32)


# ---------------------------------------------------------------------------
# SegmentField: per-segment resources for iso-surface rendering via local
# distance-transform approximation. Holds a raw wgpu r8unorm 3D texture of
# the binary presence field + metadata matching the SegBlock uniform layout.
#
# Attribute names (patient_to_texture, world_from_local) are chosen to match
# ImageField so _p2t_for_field() works for both.
# ---------------------------------------------------------------------------

class SegmentField:
    mode = 0  # 0 = DT-based opacity (current); 1 (future) = source-volume-alpha

    def __init__(self, device, segmentation_node_id, segment_id):
        self.device = device
        self.segmentation_node_id = segmentation_node_id
        self.segment_id = segment_id

        # raw_tex:     r8unorm 3D, binary labelmap uploaded from MRML
        # smooth_tex:  rgba16float 3D, Gaussian-blurred presence (compute output)
        # scratch_tex: rgba16float 3D, ping-pong buffer for the separable pass
        self.raw_tex = None
        self.smooth_tex = None
        self.scratch_tex = None
        self.tex_view = None      # view of smooth_tex -- bound to fragment shader
        self.sampler = None
        self.dims = (0, 0, 0)
        self.patient_to_texture = np.eye(4, dtype=np.float32)
        self.world_from_local = np.eye(4, dtype=np.float32)

        self.color_rgb = (1.0, 0.5, 0.0)
        self.opacity = 1.0
        self.band_mm = 1.0
        # Gaussian sigma (in voxels) applied on the GPU via a separable
        # compute shader before render time. Single-tap sampling at render.
        # Default 1.5 voxels. Bumping sigma widens the filter without
        # changing per-frame cost -- only the one-shot compute pass gets
        # a tiny amount longer.
        self.sigma_voxels = 1.5
        self.k_ambient = 0.20
        self.k_diffuse = 0.85
        self.k_specular = 0.30
        self.shininess = 32.0
        self.sample_step_mm = 1.0
        self.visible = True
        self._bounds = (np.array([-100, -100, -100], dtype=np.float32),
                        np.array([100, 100, 100], dtype=np.float32))

    def aabb(self):
        return self._bounds

    def refresh(self):
        """Re-read the binary labelmap for this segment from MRML and
        re-upload to the GPU. Returns True if the texture was reallocated
        (caller must rebuild the bind group)."""
        import vtk.util.numpy_support as vnp

        seg_node = slicer.mrmlScene.GetNodeByID(self.segmentation_node_id)
        if seg_node is None:
            return False
        # Make sure a binary labelmap exists (source rep usually IS this,
        # so this is a no-op on a normal edit, but guards against fresh
        # segmentation nodes where the rep isn't created yet).
        try:
            seg_node.CreateBinaryLabelmapRepresentation()
        except Exception:
            pass
        oimg = seg_node.GetBinaryLabelmapInternalRepresentation(self.segment_id)
        if oimg is None:
            return False
        ext = oimg.GetExtent()
        dx = ext[1] - ext[0] + 1
        dy = ext[3] - ext[2] + 1
        dz = ext[5] - ext[4] + 1
        if dx <= 0 or dy <= 0 or dz <= 0:
            return False
        scalars = oimg.GetPointData().GetScalars()
        if scalars is None or scalars.GetNumberOfTuples() == 0:
            return False

        raw = vnp.vtk_to_numpy(scalars)
        if raw.ndim == 2 and raw.shape[1] > 1:
            raw = raw[:, 0]
        # Extract ONLY voxels that belong to this segment. Slicer uses a
        # shared labelmap by default where each segment is distinguished
        # by its label value (1, 2, 3, ...), so `raw > 0` would lump
        # adjacent segments together and the smoothing filter would bleed
        # neighbors of different labels into each other.
        segment = seg_node.GetSegmentation().GetSegment(self.segment_id)
        try:
            label_value = int(segment.GetLabelValue())
        except Exception:
            label_value = 1
        presence = (raw == label_value).astype(np.uint8) * 255
        presence = np.ascontiguousarray(presence.reshape(dz, dy, dx))

        # Texture (voxel-centered [0,1]^3) -> absolute ijk (including extent
        # offset) -> world. GetImageToWorldMatrix returns absolute-ijk->world
        # in the labelmap's own frame, so this matches ImageField semantics.
        m4 = vtk.vtkMatrix4x4()
        oimg.GetImageToWorldMatrix(m4)
        ijk_abs_to_world = np.array(
            [[m4.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=np.float64)
        tex_to_ijk_abs = np.array([
            [dx, 0.0, 0.0, ext[0] - 0.5],
            [0.0, dy, 0.0, ext[2] - 0.5],
            [0.0, 0.0, dz, ext[4] - 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tex_to_world = ijk_abs_to_world @ tex_to_ijk_abs
        world_to_tex = np.linalg.inv(tex_to_world)
        self.patient_to_texture = world_to_tex.astype(np.float32)

        corners_tex = np.array([
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1],
        ], dtype=np.float64).T
        world_corners = (tex_to_world @ corners_tex)[:3].T
        # Apply the segment's parent transform (world_from_local) to the AABB
        self._update_world_from_local(seg_node)
        wfl = np.asarray(self.world_from_local, dtype=np.float64)
        world_corners_h = np.hstack([world_corners, np.ones((8, 1))])
        world_corners_w = (wfl @ world_corners_h.T).T[:, :3]
        self._bounds = (world_corners_w.min(axis=0).astype(np.float32),
                        world_corners_w.max(axis=0).astype(np.float32))

        spacing = oimg.GetSpacing()
        vox = float(min(spacing))
        self.sample_step_mm = max(vox * 0.5, 0.1)
        # Default band: 1 voxel-worth of thickness gives a ~1-voxel AA band
        self.band_mm = vox

        # Color + opacity from the segmentation display + segment itself
        try:
            segment = seg_node.GetSegmentation().GetSegment(self.segment_id)
            if segment is not None:
                self.color_rgb = tuple(float(c) for c in segment.GetColor())
        except Exception:
            pass
        dn = seg_node.GetDisplayNode()
        if dn is not None:
            try:
                self.opacity = float(dn.GetSegmentOpacity3D(self.segment_id))
            except Exception:
                pass
            try:
                self.visible = bool(dn.GetSegmentVisibility(self.segment_id)
                                    and dn.GetVisibility3D())
            except Exception:
                self.visible = True

        new_dims = (dx, dy, dz)
        realloc = (new_dims != self.dims) or self.raw_tex is None
        if realloc:
            self.raw_tex = self.device.create_texture(
                size=(dx, dy, dz), dimension="3d",
                format=wgpu.TextureFormat.r8unorm,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST),
            )
            # Float16 storage for the Gaussian-blurred output. STORAGE_BINDING
            # so the compute shader can write it; TEXTURE_BINDING so the
            # fragment shader can sample it. rgba16float is mandatorily
            # filterable + storage-compatible in core WebGPU.
            def _mk_storage():
                return self.device.create_texture(
                    size=(dx, dy, dz), dimension="3d",
                    format=wgpu.TextureFormat.rgba16float,
                    usage=(wgpu.TextureUsage.TEXTURE_BINDING
                           | wgpu.TextureUsage.STORAGE_BINDING),
                )
            self.smooth_tex = _mk_storage()
            self.scratch_tex = _mk_storage()
            self.tex_view = self.smooth_tex.create_view()
            if self.sampler is None:
                self.sampler = self.device.create_sampler(
                    mag_filter=wgpu.FilterMode.linear,
                    min_filter=wgpu.FilterMode.linear,
                    address_mode_u=wgpu.AddressMode.clamp_to_edge,
                    address_mode_v=wgpu.AddressMode.clamp_to_edge,
                    address_mode_w=wgpu.AddressMode.clamp_to_edge,
                )
            self.dims = new_dims
        self.device.queue.write_texture(
            {"texture": self.raw_tex, "mip_level": 0, "origin": (0, 0, 0)},
            presence,  # numpy array -- wgpu-py reads via buffer protocol
            {"offset": 0, "bytes_per_row": dx, "rows_per_image": dy},
            (dx, dy, dz),
        )
        return realloc

    def _update_world_from_local(self, seg_node):
        parent = seg_node.GetParentTransformNode()
        if parent is None:
            self.world_from_local = np.eye(4, dtype=np.float32)
            return
        m = vtk.vtkMatrix4x4()
        parent.GetMatrixTransformToWorld(m)
        self.world_from_local = np.array(
            [[m.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=np.float32)


# ---------------------------------------------------------------------------
# RGBAVolumeField: a pre-baked 3D RGBA volume, rendered as-is with RGB =
# emissive color and A = opacity. No TF, no per-segment logic at draw time.
# Created by the bridge's ColorizeVolume-style GPU bake pipeline
# (add_colorize_volume).
# ---------------------------------------------------------------------------

class RGBAVolumeField:
    def __init__(self, device):
        self.device = device
        self.tex = None           # rgba16float 3D (final bake output)
        self.scratch_tex = None   # rgba16float 3D (ping-pong during bake)
        self.tex_view = None
        self.sampler = None
        self.dims = (0, 0, 0)
        # Reuses ImageField's attribute names so _p2t_for_field composes them.
        self.patient_to_texture = np.eye(4, dtype=np.float32)
        self.world_from_local = np.eye(4, dtype=np.float32)
        self.sample_step_mm = 1.0
        self.opacity_unit_distance = 5.0
        self.k_ambient = 0.20
        self.k_diffuse = 0.85
        self.k_specular = 0.30
        self.shininess = 32.0
        self.visible = True
        # Width (in mm) of the central-difference stencil used to compute
        # the alpha gradient at sample time. Used by density-mode rgba
        # volumes (where alpha = smoothed presence). 0 falls back to
        # scene_step.
        self.gradient_h_mm = 0.0
        # Surface-mode SDF -> alpha band width (in mm). The 0->1 alpha
        # transition is centered on sdf=0 and spans this thickness; total
        # alpha across one surface crossing is opacity_scale.
        self.band_mm = 0.0
        # Surface-mode: post-JFA Gaussian sigma (in voxels of the rgba
        # grid) applied to the SDF channel. Knocks down the equidistant
        # ridges where neighboring boundary seeds meet without spreading
        # the surface. ~2.5 voxels approximates Slicer's default
        # closed-surface smoothing on segmentations. 0 disables.
        self.sdf_smooth_sigma_voxels = 2.5
        # "density" (ColorizeVolume-style accumulating alpha * step / unit)
        # or "surface" (gradient-opacity: α_step = |grad alpha| * step, so
        # the total α per 0->opacity boundary crossing equals the baked-in
        # opacity, independent of path length through the segment).
        self.render_mode = "density"
        self._bounds = (np.array([-100, -100, -100], dtype=np.float32),
                        np.array([100, 100, 100], dtype=np.float32))
        # Carving: at sample time, voxels whose label is in carve_segment_label_values
        # AND whose world position is inside the sphere (carve_center_world,
        # carve_radius_mm) are dropped from compositing. radius<=0 disables.
        self.carve_segment_label_values: list[int] = []
        self.carve_center_world = np.zeros(3, dtype=np.float32)
        self.carve_radius_mm = 0.0

    def aabb(self):
        return self._bounds

    def allocate(self, dx, dy, dz):
        def _mk():
            return self.device.create_texture(
                size=(dx, dy, dz), dimension="3d",
                format=wgpu.TextureFormat.rgba16float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.STORAGE_BINDING),
            )
        self.tex = _mk()
        self.scratch_tex = _mk()
        self.tex_view = self.tex.create_view()
        if self.sampler is None:
            self.sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
                address_mode_w=wgpu.AddressMode.clamp_to_edge,
            )
        self.dims = (dx, dy, dz)


# ---------------------------------------------------------------------------
# FragmentField: singleton per-bridge A-buffer for depth-correct compositing
# of rasterized geometry (FiberStrandField, etc.) with the volume ray-march.
#
# Per pixel: a sorted list of K fragments, each (depth f32, packed rgba8 u32).
# Layout in the storage buffer: 2 u32 per fragment, K fragments per pixel,
# pixel-major (row-major over (x, y)). Counts are kept in a separate buffer
# of atomic<u32>, one per pixel.
#
# Lifecycle:
#   - lazily allocated on first add_fiber_strands.
#   - reallocated when viewport size changes (handled by ensure_size).
#   - per-frame: counts cleared, rasterizers append fragments, sort pass
#     reorders by depth, main ray-march reads sorted lists and interleaves
#     fragment compositing with volume samples.
# ---------------------------------------------------------------------------

class FragmentField:
    K = 64          # fragments kept per pixel (overflow handled by
                    # keep-K-shallowest in the rasterizer; deepest is
                    # evicted when the buffer is full)
    FRAG_BYTES = 8  # 4 B depth (f32 bitcast u32) + 4 B packed rgba8

    def __init__(self, device):
        self.device = device
        self.fragments_buffer = None
        self.counts_buffer = None
        self.viewport = (0, 0)

    def ensure_size(self, w, h):
        if self.viewport == (w, h) and self.fragments_buffer is not None:
            return
        self.fragments_buffer = self.device.create_buffer(
            size=w * h * self.K * self.FRAG_BYTES,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.counts_buffer = self.device.create_buffer(
            size=w * h * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.viewport = (w, h)


# ---------------------------------------------------------------------------
# FiberStrandField: per-strand cylinder rasterization data source. Holds
# polyline geometry on the GPU; render-time, an instanced quad expansion +
# cylinder-impostor fragment shader writes hits into the bridge's
# FragmentField. Not directly a Field in the ray-march sense.
# ---------------------------------------------------------------------------

class FiberStrandField:
    def __init__(self, device):
        self.device = device
        self.segments_buffer = None       # storage buffer of (p0, p1, bid, ...)
        self.num_segments = 0
        # Strip-rendering buffers. Per polyline-point we emit 2 vertices
        # (top, bot of the tube). The index buffer wires up adjacent
        # sub-segments via SHARED vertex indices, so the rasterization
        # mesh is watertight by construction (no per-joint billboard
        # boundary).
        self.vertex_buffer = None
        self.index_buffer = None
        self.num_indices = 0
        self.palette_tex = None           # rgba8unorm 1D, 256 entries
        self.palette_view = None
        self.params_ubo = None            # per-field StrandParams UBO
        self.bind_group = None            # per-field bind group for raster pipeline
        self.world_from_local = np.eye(4, dtype=np.float32)
        self.tube_radius_mm = 0.2
        self.k_ambient = 0.20
        self.k_diffuse = 0.65
        self.k_specular = 0.20
        self.shininess = 96.0
        self.visible = True
        self._bounds = (np.array([-100, -100, -100], dtype=np.float32),
                        np.array([100, 100, 100], dtype=np.float32))

    def aabb(self):
        return self._bounds

    def write_palette(self, palette_rgba_u8):
        """palette_rgba_u8: (256, 4) uint8. Bundle 0 reserved (no tube)."""
        arr = np.ascontiguousarray(
            np.asarray(palette_rgba_u8, dtype=np.uint8).reshape(256, 4))
        if self.palette_tex is None:
            self.palette_tex = self.device.create_texture(
                size=(256, 1, 1), dimension="1d",
                format=wgpu.TextureFormat.rgba8unorm,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST),
            )
            self.palette_view = self.palette_tex.create_view()
        self.device.queue.write_texture(
            {"texture": self.palette_tex, "mip_level": 0, "origin": (0, 0, 0)},
            arr,
            {"offset": 0, "bytes_per_row": 256 * 4, "rows_per_image": 1},
            (256, 1, 1))


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

class WgpuVolumeBridge:
    """VTK-injection bridge. Wires VolumeRenderingDisplayer to a raw-wgpu
    ray-march pipeline rendered into Slicer's native 3D view via an
    EndEvent hook + glTexSubImage2D blit."""

    def __init__(self, vtk_renderer, vtk_render_window):
        self.vtk_renderer = vtk_renderer
        self.rw = vtk_render_window
        self.device = _shared_wgpu_device()

        self._displayer = None
        self._claimed_vrdn_ids: set[str] = set()

        self._pipeline = None
        self._bgl = None
        self._bind_group = None
        self._color_tex = None
        self._readback_buf = None
        self._size = (0, 0)
        self._aligned_bpr = 0
        self._fields = []
        # Camera UBO: proj_inv (64) + view_inv (64) + size (16) +
        # proj (64) + view (64) + taa (16) = 288.
        # The forward proj/view are needed by FragmentField's depth
        # comparisons during the ray-march fragment-integration loop.
        # taa carries the TAA frame index for shader-side noise seeding;
        # size.zw carries the sub-pixel jitter offset (NDC units).
        self._cam_ubo = self.device.create_buffer(
            size=288,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._mat_ubo = None

        self._end_tag = None
        self._disposed = False
        self._scene_obs_tags = []

        self._libGL = _load_gl_library()
        # glReadPixels we use to pull VTK's depth buffer each frame
        from ctypes import c_int, c_uint, c_void_p
        self._libGL.glReadPixels.argtypes = [
            c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p]
        self._libGL.glBindFramebuffer.argtypes = [c_uint, c_uint]
        self._gl_compositor = _GLCompositor(self._libGL)

        # VTK depth -> wgpu texture (allocated lazily at viewport size)
        self._vtk_depth_tex = None
        self._vtk_depth_view = None
        self._vtk_depth_sampler = None
        self._vtk_depth_size = (0, 0)
        self._vtk_depth_buf = None  # numpy staging buffer reused across frames

        # TAA state. _taa_frame indexes the Halton(2,3) jitter sequence; it
        # increments each rendered frame and resets to 0 when the camera or
        # scene changes so the history buffer doesn't accumulate stale data.
        # _taa_prev_pv is the previous frame's (proj * view) for change
        # detection. _taa_history_tex is the per-pixel accumulator.
        self._taa_frame = 0
        self._taa_prev_pv = None
        self._taa_history_tex = None
        self._taa_history_view = None
        self._taa_output_tex = None
        self._taa_output_view = None
        self._taa_pipeline = None
        self._taa_bgl = None

        # Grid transform: one scene-wide displacement field. None when no grid
        # is attached -- the shader short-circuits via u_mat.grid_enabled.x.
        # Still needs a placeholder texture so the bind group is well-formed.
        self._grid_tex = None
        self._grid_view = None
        self._grid_sampler = None
        self._grid_dims = (0, 0, 0)     # (W, H, D)
        self._grid_p2t = None           # 4x4 world->texture for displacement grid
        self._grid_gain = 1.0
        self._grid_node = None          # vtkMRMLGridTransformNode we observe
        self._grid_obs_tags = []        # (node, tag) pairs for teardown

        # Segmentation: per-segment iso-surface fields.
        self._segments: list[SegmentField] = []
        self._seg_node = None           # the currently attached vtkMRMLSegmentationNode
        self._seg_obs_tags = []         # (object, tag) pairs for teardown
        # Segment render mode: "iso" (band-based iso-surface, paint demo) or
        # "surface" (gradient-opacity, thickness-independent polydata-like).
        self._segment_render_mode = "iso"

        # Pre-baked RGBA 3D volumes (ColorizeVolume-style). Each entry is an
        # RGBAVolumeField with a rgba16float 3D texture produced by a GPU
        # bake (see add_colorize_volume). Rendered directly: RGB = emissive
        # color, A = opacity. No TF, no per-segment work at draw time.
        self._rgba_volumes: list = []
        # (object, observer_tag, callback) triples for segmentation display
        # nodes driving RGBA volumes. On display change we rebuild the
        # palette UBO and rerun the bake using cached textures.
        self._rgba_obs_tags: list = []

        # Per-strand cylinder rasterization sources. Each entry is a
        # FiberStrandField; rasterization writes hits into self._fragment_field
        # (lazy singleton) for depth-correct compositing in the main shader.
        self._fiber_strand_fields: list = []
        self._fragment_field: FragmentField | None = None
        self._strand_pipeline = None
        self._strand_pipeline_bgl = None
        self._fragment_clear_pipeline = None
        self._fragment_clear_bgl = None
        self._fragment_sort_pipeline = None
        self._fragment_sort_bgl = None
        self._fragment_sort_ubo = None

        # Active clip planes in world space: list of (nx, ny, nz, offset).
        # Observed from vtkMRMLClipModelsNode(s) via set_clip_nodes().
        self._clip_planes: list = []
        self._clip_obs_tags: list = []

        # Segment-smoothing compute pipeline (lazy). Separable 1D Gaussian
        # dispatched three times (X, Y, Z) per segment on paint -- turns the
        # binary r8unorm labelmap into a smoothly low-passed rgba16float
        # field that the fragment shader then reads with a single trilinear
        # tap per sample.
        self._smooth_compute_pipeline = None
        self._smooth_compute_bgl = None
        self._smooth_ubos = None        # 3 x 16-byte UBOs, one per axis

    def install(self):
        self._displayer = VolumeRenderingDisplayer(
            on_structure_changed=self._on_structure_changed,
            on_field_modified=self._on_field_modified,
        )
        self._claim_current_vrdns()
        self._rebuild_pipeline()

        # Compute pipelines are compiled lazily on first use. Eager compile
        # here sounded nice in theory but measured at ~6 s on the MCP Linux
        # box (3 pipelines through naga + NVIDIA driver), which made the
        # bridge install feel slow. Better to pay it once, when the user
        # actually bakes or paints.

        def _on_end(caller, event):
            self._on_end_event(caller)
        self._end_tag = self.vtk_renderer.AddObserver(
            vtk.vtkCommand.EndEvent, _on_end)
        self._on_end_cb = _on_end

        # Tear down automatically when the scene is cleared so we don't
        # keep pointers to deleted MRML nodes and don't leave a pipeline
        # rendering stale content in the 3D view.
        scene = slicer.mrmlScene
        self._scene_obs_tags = [
            (scene, scene.AddObserver(
                scene.StartCloseEvent, self._on_scene_start_close)),
            (scene, scene.AddObserver(
                scene.EndCloseEvent, self._on_scene_end_close)),
        ]

        self.rw.Render()

    def _on_scene_start_close(self, caller, event):
        """Tear down everything that references MRML nodes BEFORE the scene
        actually clears them -- otherwise our observers fire on node-removal
        events with dangling pointers."""
        try:
            self.uninstall()
        except Exception as e:
            print(f"WgpuVolumeBridge scene-close uninstall: {e}")
        try:
            if getattr(slicer.modules, "wgpuVtkBridge", None) is self:
                slicer.modules.wgpuVtkBridge = None
        except Exception:
            pass

    def _on_scene_end_close(self, caller, event):
        # Scene has been cleared; force a redraw so the 3D view no longer
        # shows anything our bridge had previously composited.
        try:
            self.rw.Render()
        except Exception:
            pass

    def uninstall(self):
        # Set this first so any EndEvent still in flight short-circuits.
        self._disposed = True
        for obj, tag in self._scene_obs_tags:
            try: obj.RemoveObserver(tag)
            except Exception: pass
        self._scene_obs_tags = []
        if self._end_tag is not None:
            self.vtk_renderer.RemoveObserver(self._end_tag)
            self._end_tag = None
        if self._displayer is not None:
            self._displayer.cleanup()
            self._displayer = None
        for nid in list(self._claimed_vrdn_ids):
            n = slicer.mrmlScene.GetNodeByID(nid)
            if n is not None:
                try: n.SetVisibility(True)
                except Exception: pass
        self._claimed_vrdn_ids.clear()
        for node, tag in self._grid_obs_tags:
            try: node.RemoveObserver(tag)
            except Exception: pass
        self._grid_obs_tags = []
        self._grid_node = None
        for obj, tag in self._seg_obs_tags:
            try: obj.RemoveObserver(tag)
            except Exception: pass
        self._seg_obs_tags = []
        self._seg_node = None
        self._segments = []
        for obj, tag, _cb in self._rgba_obs_tags:
            try: obj.RemoveObserver(tag)
            except Exception: pass
        self._rgba_obs_tags = []
        self._rgba_volumes = []
        self._fiber_strand_fields = []
        self._fragment_field = None
        for obj, tag in self._clip_obs_tags:
            try: obj.RemoveObserver(tag)
            except Exception: pass
        self._clip_obs_tags = []
        self._clip_planes = []
        self._fields = []
        # Drop wgpu resources so the Python GC can reclaim GPU memory
        # immediately rather than waiting for the bridge object itself
        # to be collected.
        self._pipeline = None
        self._bgl = None
        self._bind_group = None
        self._color_tex = None
        self._readback_buf = None
        self._cam_ubo = None
        self._mat_ubo = None
        self._vtk_depth_tex = None
        self._vtk_depth_view = None
        self._vtk_depth_sampler = None
        self._vtk_depth_buf = None
        self._grid_tex = None
        self._grid_view = None
        self._grid_sampler = None
        self._smooth_compute_pipeline = None
        self._smooth_compute_bgl = None
        self._smooth_ubos = None
        self._gl_compositor = None

    @property
    def images_by_vrdn(self):
        if self._displayer is None:
            return {}
        return dict(self._displayer.fields_by_nid)

    def _claim_current_vrdns(self):
        if self._displayer is None:
            return
        for nid in self._displayer.fields_by_nid:
            if nid in self._claimed_vrdn_ids:
                continue
            node = slicer.mrmlScene.GetNodeByID(nid)
            if node is not None:
                node.SetVisibility(False)
                self._claimed_vrdn_ids.add(nid)

    def _on_structure_changed(self):
        self._claim_current_vrdns()
        self._rebuild_pipeline()
        self.rw.Render()

    def _on_field_modified(self, field):
        self._claim_current_vrdns()
        self.rw.Render()

    def _rebuild_pipeline(self):
        self._fields = [
            f for f in self._displayer.fields()
            if isinstance(f, ImageField)
        ] if self._displayer else []
        n = len(self._fields)
        m = len(self._segments)
        k = len(self._rgba_volumes)
        has_strands = bool(self._fiber_strand_fields)
        if n == 0 and m == 0 and k == 0 and not has_strands:
            self._pipeline = None
            self._bind_group = None
            return

        for img_f in self._fields:
            for tex in (img_f._volume_tex, img_f._lut_tex, img_f._grad_lut_tex):
                if tex is None:
                    continue
                if tex._wgpu_object is None:
                    tex._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
                ensure_wgpu_object(tex)
                update_resource(tex)

        rgba_modes = [getattr(r, "render_mode", "density")
                      for r in self._rgba_volumes]
        # Fragment-buffer bindings sit AFTER all field bindings:
        #   2 + n*4 + 4 (depth+grid) + m*2 + k*3 = frag_base.
        frag_base = 2 + n * 4 + 4 + m * 2 + k * 3
        K_const = FragmentField.K
        wgsl = _build_wgsl(n, m, k,
                           segment_render_mode=self._segment_render_mode,
                           rgba_render_modes=rgba_modes,
                           has_fragments=has_strands,
                           frag_binding=frag_base,
                           K=K_const)
        shader = self.device.create_shader_module(code=wgsl)

        self._mat_ubo = self.device.create_buffer(
            size=_mat_ubo_size(n, m, k),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]
        for i in range(n):
            b0 = 2 + i * 4
            entries += [
                {"binding": b0+0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": b0+1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.float,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
                {"binding": b0+2, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": b0+3, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.float,
                             "view_dimension": wgpu.TextureViewDimension.d1,
                             "multisampled": False}},
            ]
        # Scene-wide VTK depth at binding 2+n*4, 3+n*4
        bd = 2 + n * 4
        entries += [
            {"binding": bd+0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.non_filtering}},
            {"binding": bd+1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d2,
                         "multisampled": False}},
        ]
        # Scene-wide grid transform displacement at binding 4+n*4, 5+n*4
        bg_grid = 2 + n * 4 + 2
        entries += [
            {"binding": bg_grid+0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": wgpu.SamplerBindingType.filtering}},
            {"binding": bg_grid+1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
        ]
        # Per-segment bindings: (sampler, 3D texture) per segment, starting
        # right after the grid transform bindings.
        bs0 = 2 + n * 4 + 4
        for j in range(m):
            bs = bs0 + j * 2
            entries += [
                {"binding": bs+0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": bs+1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.float,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
            ]
        # Per-RGBA-volume bindings (sampler + rgba16float 3D texture +
        # uint labelmap 3D), starting right after per-segment bindings.
        # The labelmap is bound for segment-aware carving; sampled with
        # textureLoad (no sampler needed for uint formats).
        br0 = bs0 + m * 2
        for q in range(k):
            br = br0 + q * 3
            entries += [
                {"binding": br+0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": br+1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.float,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
                {"binding": br+2, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.uint,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
            ]
        if has_strands:
            entries += [
                {"binding": frag_base + 0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
                {"binding": frag_base + 1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            ]
        self._bgl = self.device.create_bind_group_layout(entries=entries)
        pl = self.device.create_pipeline_layout(bind_group_layouts=[self._bgl])
        self._pipeline = self.device.create_render_pipeline(
            layout=pl,
            vertex={"module": shader, "entry_point": "vs_main"},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        # Depth texture is sized lazily on first render; ensure it exists
        # (1x1 placeholder if we don't yet know the viewport size) so the
        # bind group is well-formed.
        if self._vtk_depth_tex is None:
            self._ensure_vtk_depth_tex(1, 1)
        # Grid transform texture always exists -- placeholder when no grid
        # is attached; u_mat.grid_enabled.x guards sampling.
        if self._grid_tex is None:
            self._ensure_grid_placeholder()
        self._rebuild_bind_group()

    def _rebuild_bind_group(self):
        """Assemble bind group from current fields + current depth texture.
        Called when pipeline rebuilds OR when the depth texture reallocates."""
        if self._bgl is None or (not self._fields and not self._segments
                                 and not self._rgba_volumes
                                 and not self._fiber_strand_fields):
            return
        bg = [
            {"binding": 0, "resource": {
                "buffer": self._cam_ubo, "offset": 0, "size": self._cam_ubo.size}},
            {"binding": 1, "resource": {
                "buffer": self._mat_ubo, "offset": 0, "size": self._mat_ubo.size}},
        ]
        n = len(self._fields)
        for i, f in enumerate(self._fields):
            b0 = 2 + i * 4
            bg += [
                {"binding": b0+0, "resource": self._linear_sampler()},
                {"binding": b0+1, "resource": f._volume_tex._wgpu_object.create_view()},
                {"binding": b0+2, "resource": self._linear_sampler()},
                {"binding": b0+3, "resource": f._lut_tex._wgpu_object.create_view()},
            ]
        bd = 2 + n * 4
        bg += [
            {"binding": bd+0, "resource": self._vtk_depth_sampler},
            {"binding": bd+1, "resource": self._vtk_depth_view},
        ]
        bg_grid = bd + 2
        bg += [
            {"binding": bg_grid+0, "resource": self._grid_sampler},
            {"binding": bg_grid+1, "resource": self._grid_view},
        ]
        bs0 = bg_grid + 2
        for j, s in enumerate(self._segments):
            bs = bs0 + j * 2
            bg += [
                {"binding": bs+0, "resource": s.sampler},
                {"binding": bs+1, "resource": s.tex_view},
            ]
        br0 = bs0 + len(self._segments) * 2
        for q, r in enumerate(self._rgba_volumes):
            br = br0 + q * 3
            bg += [
                {"binding": br+0, "resource": r.sampler},
                {"binding": br+1, "resource": r.tex_view},
                {"binding": br+2, "resource": r._label_tex_view},
            ]
        if self._fiber_strand_fields and self._fragment_field is not None \
                and self._fragment_field.fragments_buffer is not None:
            frag_base = br0 + len(self._rgba_volumes) * 3
            bg += [
                {"binding": frag_base + 0, "resource": {
                    "buffer": self._fragment_field.counts_buffer,
                    "offset": 0,
                    "size": self._fragment_field.counts_buffer.size}},
                {"binding": frag_base + 1, "resource": {
                    "buffer": self._fragment_field.fragments_buffer,
                    "offset": 0,
                    "size": self._fragment_field.fragments_buffer.size}},
            ]
        self._bind_group = self.device.create_bind_group(
            layout=self._bgl, entries=bg)

    def _ensure_vtk_depth_tex(self, w, h):
        """Allocate (or reallocate) the wgpu texture that mirrors VTK's
        depth attachment. Also create a numpy staging buffer."""
        if self._vtk_depth_size == (w, h) and self._vtk_depth_tex is not None:
            return
        self._vtk_depth_tex = self.device.create_texture(
            size=(w, h, 1), dimension="2d",
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._vtk_depth_view = self._vtk_depth_tex.create_view()
        self._vtk_depth_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        self._vtk_depth_buf = np.empty((h, w), dtype=np.float32)
        self._vtk_depth_size = (w, h)

    def _upload_vtk_depth(self, w, h):
        """Read VTK's depth attachment via glReadPixels + upload to our
        wgpu depth texture. Flips Y so texelFetch with wgpu Y-down frag
        coords maps to the right pixel."""
        GL_READ_FRAMEBUFFER = 0x8CA8
        GL_DEPTH_COMPONENT  = 0x1902
        GL_FLOAT            = 0x1406
        rfb = self.rw.GetRenderFramebuffer()
        self._libGL.glBindFramebuffer(GL_READ_FRAMEBUFFER, rfb.GetFBOIndex())
        self._libGL.glReadPixels(
            0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT,
            self._vtk_depth_buf.ctypes.data_as(ctypes.c_void_p))
        self._libGL.glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        # glReadPixels returns Y-up (row 0 = bottom); wgpu fragcoord is
        # Y-down (row 0 = top). Flip once on CPU before upload.
        flipped = np.ascontiguousarray(self._vtk_depth_buf[::-1])
        self.device.queue.write_texture(
            {"texture": self._vtk_depth_tex, "mip_level": 0, "origin": (0, 0, 0)},
            flipped,
            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1),
        )

    def _linear_sampler(self):
        return self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

    # ---------------- Grid transform support ----------------

    def _ensure_grid_placeholder(self):
        """Allocate a 1x1x1 rgba32float texture + sampler used when no grid
        transform is attached. The shader's u_mat.grid_enabled.x gate skips
        sampling, so the contents are irrelevant; the binding just needs to
        be well-formed."""
        if self._grid_tex is not None:
            return
        self._grid_tex = self.device.create_texture(
            size=(1, 1, 1), dimension="3d",
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._grid_view = self._grid_tex.create_view()
        self._grid_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )
        self._grid_dims = (1, 1, 1)
        self._grid_p2t = None

    def _patient_to_texture_from_grid(self, grid_image_data):
        """4x4 world RAS -> [0,1]^3 tex coords for a vtkImageData.
        Mirrors slicer_wgpu.fields.transform._patient_to_texture_from_grid."""
        origin = np.asarray(grid_image_data.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(grid_image_data.GetSpacing(), dtype=np.float64)
        dim = grid_image_data.GetDimensions()
        if hasattr(grid_image_data, "GetDirectionMatrix"):
            direction = grid_image_data.GetDirectionMatrix()
            D = np.array([[direction.GetElement(i, j) for j in range(3)]
                          for i in range(3)], dtype=np.float64)
        else:
            D = np.eye(3, dtype=np.float64)
        ijk_to_ras = np.eye(4, dtype=np.float64)
        ijk_to_ras[:3, :3] = D @ np.diag(spacing)
        ijk_to_ras[:3, 3] = origin
        ras_to_ijk = np.linalg.inv(ijk_to_ras)
        ijk_to_tex = np.eye(4, dtype=np.float64)
        for axis in range(3):
            ijk_to_tex[axis, axis] = 1.0 / dim[axis]
            ijk_to_tex[axis, 3] = 0.5 / dim[axis]
        return (ijk_to_tex @ ras_to_ijk).astype(np.float32)

    def _upload_grid_from_node(self, grid_node):
        """Extract displacement grid from a vtkMRMLGridTransformNode and
        upload to the GPU. (Re)allocates the texture if dims changed.
        Returns True if a bind-group rebuild is needed (texture reallocated)."""
        import vtk.util.numpy_support as vnp

        core = grid_node.GetTransformFromParent()
        dgrid = core.GetDisplacementGrid() if hasattr(
            core, "GetDisplacementGrid") else None
        if dgrid is None:
            core = grid_node.GetTransformToParent()
            dgrid = core.GetDisplacementGrid() if hasattr(
                core, "GetDisplacementGrid") else None
        if dgrid is None or dgrid.GetNumberOfPoints() <= 1:
            return False

        dims = dgrid.GetDimensions()  # (nx, ny, nz)
        pd = dgrid.GetPointData()
        vec_arr = pd.GetScalars() if pd is not None else None
        if vec_arr is None and pd is not None and pd.GetNumberOfArrays() > 0:
            vec_arr = pd.GetArray(0)
        if vec_arr is None:
            return False

        raw = vnp.vtk_to_numpy(vec_arr).astype(np.float32, copy=False)
        n_comp = raw.shape[1] if raw.ndim == 2 else 1
        n_vox = dims[0] * dims[1] * dims[2]
        raw = raw.reshape(n_vox, n_comp)
        rgba = np.zeros((n_vox, 4), dtype=np.float32)
        rgba[:, 0:min(n_comp, 3)] = raw[:, 0:min(n_comp, 3)]
        # (D, H, W, 4) for wgpu write_texture with I fastest
        rgba = rgba.reshape(dims[2], dims[1], dims[0], 4)

        p2t = self._patient_to_texture_from_grid(dgrid)
        self._grid_p2t = p2t

        realloc = (self._grid_dims != dims)
        if realloc:
            self._grid_tex = self.device.create_texture(
                size=(dims[0], dims[1], dims[2]), dimension="3d",
                format=wgpu.TextureFormat.rgba32float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST),
            )
            self._grid_view = self._grid_tex.create_view()
            self._grid_dims = tuple(dims)

        # Upload: bytes_per_row = W * 4ch * 4bytes; rows_per_image = H.
        self.device.queue.write_texture(
            {"texture": self._grid_tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba,
            {"offset": 0,
             "bytes_per_row": dims[0] * 16,
             "rows_per_image": dims[1]},
            (dims[0], dims[1], dims[2]),
        )
        return realloc

    def set_clip_nodes(self, clip_nodes):
        """Install observers on a list of vtkMRMLClipModelsNode / equivalent
        plane-like clip nodes. The shader receives up to 8 active clip
        planes; a sample is discarded when it lies on the negative side
        of any active plane. Pass None or [] to disable clipping.

        MRML's `vtkMRMLClipNode` exposes GetClippingNodeID + an implicit
        function; for simplicity we accept objects implementing
        `GetOrigin() -> (x, y, z)` and `GetNormal() -> (nx, ny, nz)` (the
        common case for a plane-based clip node / `vtkMRMLMarkupsPlaneNode`).
        """
        # Tear down old observers
        for obj, tag in self._clip_obs_tags:
            try: obj.RemoveObserver(tag)
            except Exception: pass
        self._clip_obs_tags = []
        self._clip_nodes = list(clip_nodes) if clip_nodes else []
        self._refresh_clip_planes()
        # Observe each for updates so dragging / moving the plane
        # triggers a re-render.
        for n in self._clip_nodes:
            try:
                tag = n.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                    self._on_clip_modified)
                self._clip_obs_tags.append((n, tag))
            except Exception:
                pass
        self.rw.Render()

    def _on_clip_modified(self, caller, event):
        if self._disposed:
            return
        try:
            self._refresh_clip_planes()
            self.rw.Render()
        except Exception as e:
            print(f"clip modified: {e}")

    def _refresh_clip_planes(self):
        """Read origin + normal from each clip node; rebuild the
        (nx, ny, nz, offset) tuples where offset = -dot(origin, normal)
        (shader uses `dot(wp, n) + offset >= 0` for 'keep side')."""
        planes = []
        for n in getattr(self, "_clip_nodes", []):
            try:
                origin = list(n.GetOrigin())
                normal = list(n.GetNormal())
                nlen = (normal[0]**2 + normal[1]**2 + normal[2]**2) ** 0.5
                if nlen < 1e-6:
                    continue
                normal = [x / nlen for x in normal]
                offset = -(origin[0] * normal[0]
                           + origin[1] * normal[1]
                           + origin[2] * normal[2])
                planes.append((normal[0], normal[1], normal[2], offset))
                if len(planes) >= 8:
                    break
            except Exception:
                continue
        self._clip_planes = planes

    def set_grid_transform(self, grid_node):
        """Attach a vtkMRMLGridTransformNode as the scene-wide warp. Pass
        None to detach. The bridge observes the node for Modified + transform
        mutations and re-uploads the displacement texture."""
        # Tear down old observers
        for node, tag in self._grid_obs_tags:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._grid_obs_tags = []
        self._grid_node = grid_node

        if grid_node is None:
            # Revert to placeholder + disable shader path
            self._ensure_grid_placeholder()
            self._rebuild_bind_group()
            self.rw.Render()
            return

        realloc = self._upload_grid_from_node(grid_node)
        # Observe both ModifiedEvent (re-parenting, node-level edits) and
        # TransformModifiedEvent (displacement mutations).
        for ev in (vtk.vtkCommand.ModifiedEvent,
                   slicer.vtkMRMLTransformNode.TransformModifiedEvent):
            tag = grid_node.AddObserver(ev, self._on_grid_modified)
            self._grid_obs_tags.append((grid_node, tag))

        if realloc:
            self._rebuild_bind_group()
        self.rw.Render()

    def _on_grid_modified(self, caller, event):
        if self._disposed or self._grid_node is None:
            return
        try:
            realloc = self._upload_grid_from_node(self._grid_node)
            if realloc:
                self._rebuild_bind_group()
            self.rw.Render()
        except Exception as e:
            print(f"WgpuVolumeBridge grid-modified: {e}")

    # ---------------- Segmentation support ----------------

    def set_rgba_carve(self, rgba_field, segment_label_values,
                       point_world, radius_mm):
        """Configure segment-aware sphere carving for an RGBAVolumeField.

        At sample time, voxels whose label value is in
        `segment_label_values` AND whose world position lies inside the
        sphere (`point_world`, `radius_mm`) drop out of compositing.
        Up to 8 label values supported. radius_mm <= 0 disables carving.

        Pure UBO update -- no pipeline rebuild, so this is cheap to call
        on every markup-point drag.
        """
        if rgba_field is None:
            return
        ids = [int(v) for v in (segment_label_values or [])][:8]
        rgba_field.carve_segment_label_values = ids
        rgba_field.carve_center_world = np.asarray(
            point_world, dtype=np.float32).reshape(3)
        rgba_field.carve_radius_mm = float(max(radius_mm, 0.0))
        self.rw.Render()

    def set_segment_render_mode(self, mode):
        """Switch between the two segment shaders:
          - "iso"     : band-based isosurface. Paint demo default.
          - "surface" : gradient-opacity. Thickness-independent alpha --
                        emulates Slicer's polydata closed-surface
                        rendering when a segment is semi-opaque.
        Rebuilds the render pipeline immediately so the next frame picks
        up the new shader. Safe to call before or after
        set_segmentation_node().
        """
        mode = str(mode).lower()
        if mode not in ("iso", "surface"):
            raise ValueError(
                f"segment_render_mode must be 'iso' or 'surface', got {mode!r}")
        if mode == self._segment_render_mode:
            return
        self._segment_render_mode = mode
        self._rebuild_pipeline()
        self.rw.Render()

    def set_segmentation_node(self, seg_node):
        """Attach a vtkMRMLSegmentationNode. One SegmentField is created per
        visible segment; the bridge observes the internal vtkSegmentation for
        SourceRepresentationModified events so paint strokes in the Segment
        Editor trigger immediate GPU re-uploads. Pass None to detach."""
        # Tear down old observers
        for obj, tag in self._seg_obs_tags:
            try:
                obj.RemoveObserver(tag)
            except Exception:
                pass
        self._seg_obs_tags = []
        self._segments = []
        self._seg_node = seg_node

        if seg_node is None:
            self._rebuild_pipeline()
            self.rw.Render()
            return

        self._refresh_segments(rebuild=False)

        # Mirror what vtkMRMLSegmentationsDisplayableManager3D observes
        # (Modules/Loadable/Segmentations/MRMLDM/
        #  vtkMRMLSegmentationsDisplayableManager3D.cxx ~L784-815):
        #   on the vtkSegmentation: SegmentModified (paint strokes!),
        #     RepresentationModified, SegmentAdded, SegmentRemoved
        #   on the MRML node: ModifiedEvent, TransformModifiedEvent,
        #     DisplayModifiedEvent
        #   on the display node: ModifiedEvent (fall-through for display
        #     nodes not repackaged via DisplayModifiedEvent)
        try:
            import vtkSegmentationCorePython as vtkSegCore
            segmentation = seg_node.GetSegmentation()
            VSEG = vtkSegCore.vtkSegmentation
            content_events = [
                # Fires continuously during Segment Editor paint / erase.
                VSEG.SegmentModified,
                # Fires when any representation is rebuilt / converted.
                VSEG.RepresentationModified,
            ]
            structure_events = [
                VSEG.SegmentAdded,
                VSEG.SegmentRemoved,
            ]
            for ev in content_events:
                self._seg_obs_tags.append((segmentation, segmentation.AddObserver(
                    ev, self._on_segmentation_content_modified)))
            for ev in structure_events:
                self._seg_obs_tags.append((segmentation, segmentation.AddObserver(
                    ev, self._on_segmentation_structure_modified)))
        except Exception as e:
            print(f"WgpuVolumeBridge segmentation observers: {e}")
        # Node-level events: ModifiedEvent covers the segmentation object
        # being swapped out; TransformModifiedEvent covers re-parenting;
        # DisplayModifiedEvent is re-fired from the display node.
        self._seg_obs_tags.append((seg_node, seg_node.AddObserver(
            vtk.vtkCommand.ModifiedEvent,
            self._on_segmentation_content_modified)))
        try:
            ev_tx = slicer.vtkMRMLDisplayableNode.TransformModifiedEvent
            self._seg_obs_tags.append((seg_node, seg_node.AddObserver(
                ev_tx, self._on_segmentation_content_modified)))
        except Exception:
            pass
        try:
            ev_disp = slicer.vtkMRMLDisplayableNode.DisplayModifiedEvent
            self._seg_obs_tags.append((seg_node, seg_node.AddObserver(
                ev_disp, self._on_segmentation_display_modified)))
        except Exception:
            pass
        # Display node: color / opacity / visibility changes just re-pack
        # uniforms (no texture upload).
        dn = seg_node.GetDisplayNode()
        if dn is not None:
            self._seg_obs_tags.append((dn, dn.AddObserver(
                vtk.vtkCommand.ModifiedEvent,
                self._on_segmentation_display_modified)))

        self._rebuild_pipeline()
        self.rw.Render()

    def _visible_segment_ids(self):
        if self._seg_node is None:
            return []
        segmentation = self._seg_node.GetSegmentation()
        dn = self._seg_node.GetDisplayNode()
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(ids)
        result = []
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            if dn is None or dn.GetSegmentVisibility(sid):
                result.append(sid)
        return result

    def _refresh_segments(self, rebuild=True):
        """Re-sync the per-segment field list with the node's visible segments.
        Uploads each segment's current labelmap. Returns True if the segment
        count changed (or any texture was reallocated) -- caller decides
        whether to rebuild the pipeline or just the bind group."""
        if self._seg_node is None:
            return False
        want = self._visible_segment_ids()
        seg_node_id = self._seg_node.GetID()

        existing = {s.segment_id: s for s in self._segments}
        new_segments = []
        realloc = False
        for sid in want:
            s = existing.get(sid)
            if s is None:
                s = SegmentField(self.device, seg_node_id, sid)
                realloc = True
            if s.refresh():
                realloc = True
            # Separable Gaussian on the GPU -- turns the binary r8unorm
            # labelmap into a smoothly low-passed rgba16float field.
            self._smooth_segment(s)
            new_segments.append(s)
        structure_changed = (len(new_segments) != len(self._segments))
        self._segments = new_segments
        if rebuild:
            if structure_changed:
                self._rebuild_pipeline()
            elif realloc:
                self._rebuild_bind_group()
        return structure_changed or realloc

    def _on_segmentation_content_modified(self, caller, event):
        """Segment voxels changed -- typically a Segment Editor brush
        stroke (SegmentModified) or a representation rebuild
        (RepresentationModified). Re-upload affected segments. We don't
        have per-segment granularity without parsing callData, so we
        re-refresh all visible segments; full-texture upload is cheap at
        typical segmentation sizes."""
        if self._disposed:
            return
        try:
            self._refresh_segments(rebuild=True)
            self.rw.Render()
        except Exception as e:
            print(f"WgpuVolumeBridge segmentation content-modified: {e}")

    def _on_segmentation_structure_modified(self, caller, event):
        """Segment added/removed. Rebuild pipeline + bind group against the
        new segment list."""
        if self._disposed:
            return
        try:
            self._refresh_segments(rebuild=False)
            self._rebuild_pipeline()
            self.rw.Render()
        except Exception as e:
            print(f"WgpuVolumeBridge segmentation structure-modified: {e}")

    def _on_segmentation_display_modified(self, caller, event):
        """Display props (color/opacity/visibility) changed. We treat a
        visibility flip as a structural change (number of visible segments
        changes) and everything else as a uniform repack by calling
        _refresh_segments."""
        if self._disposed:
            return
        try:
            before = len(self._segments)
            self._refresh_segments(rebuild=False)
            if len(self._segments) != before:
                self._rebuild_pipeline()
            self.rw.Render()
        except Exception as e:
            print(f"WgpuVolumeBridge segmentation display-modified: {e}")

    # ---------------- GPU-side Gaussian smoothing of segments ----------------

    _SMOOTH_COMPUTE_WGSL = """
@group(0) @binding(0) var t_src: texture_3d<f32>;
@group(0) @binding(1) var t_dst: texture_storage_3d<rgba16float, write>;

struct Params {
    sigma: f32,
    axis: u32,
    radius: u32,
    _pad: u32,
};
@group(0) @binding(2) var<uniform> u_params: Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let dims_i = vec3<i32>(dims);
    let radius = i32(u_params.radius);
    let sigma = max(u_params.sigma, 1e-3);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let base = vec3<i32>(gid);

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var k = -radius; k <= radius; k = k + 1) {
        var coord = base;
        if (u_params.axis == 0u) {
            coord.x = clamp(coord.x + k, 0, dims_i.x - 1);
        } else if (u_params.axis == 1u) {
            coord.y = clamp(coord.y + k, 0, dims_i.y - 1);
        } else {
            coord.z = clamp(coord.z + k, 0, dims_i.z - 1);
        }
        let v = textureLoad(t_src, coord, 0).r;
        let w = exp(-f32(k * k) * inv_two_sigma_sq);
        sum = sum + v * w;
        wsum = wsum + w;
    }
    let out = sum / max(wsum, 1e-6);
    textureStore(t_dst, base, vec4<f32>(out, 0.0, 0.0, 0.0));
}
"""

    def _ensure_smooth_compute(self):
        if self._smooth_compute_pipeline is not None:
            return
        shader = self.device.create_shader_module(
            code=self._SMOOTH_COMPUTE_WGSL)
        entries = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]
        self._smooth_compute_bgl = self.device.create_bind_group_layout(
            entries=entries)
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._smooth_compute_bgl])
        self._smooth_compute_pipeline = self.device.create_compute_pipeline(
            layout=layout,
            compute={"module": shader, "entry_point": "main"},
        )
        # One UBO per axis so writes don't race the dispatches that consume
        # them (a single UBO overwritten three times would collapse).
        self._smooth_ubos = [
            self.device.create_buffer(
                size=16,
                usage=(wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            ) for _ in range(3)
        ]

    # ---------------- GPU-side ColorizeVolume bake ----------------

    _BAKE_INIT_WGSL = """
// Pass 1: resample the labelmap (in its own native grid) into the output
// grid, then apply the palette to produce RGBA. alpha is binary-presence
// scaled by the segment's opacity (both come from the palette). The
// resample is nearest-neighbor -- labels don't interpolate.
@group(0) @binding(0) var t_label: texture_3d<u32>;
@group(0) @binding(1) var t_dst: texture_storage_3d<rgba16float, write>;

struct Palette {
    entries: array<vec4<f32>, 256>,
};
@group(0) @binding(2) var<uniform> u_palette: Palette;

// output_to_world:    output [0,1]^3 tex coord  -> world (RAS) mm
// world_to_label_tex: world (RAS) mm            -> label [0,1]^3 tex coord
struct ResampleParams {
    output_to_world: mat4x4<f32>,
    world_to_label_tex: mat4x4<f32>,
};
@group(0) @binding(3) var<uniform> u_rs: ResampleParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }

    let out_tex = (vec3<f32>(gid) + vec3<f32>(0.5)) / vec3<f32>(dims);
    let world4 = u_rs.output_to_world * vec4<f32>(out_tex, 1.0);
    let world = world4.xyz / world4.w;
    let lt4 = u_rs.world_to_label_tex * vec4<f32>(world, 1.0);
    let lt = lt4.xyz / lt4.w;

    var label: u32 = 0u;
    if (all(lt >= vec3<f32>(0.0)) && all(lt <= vec3<f32>(1.0))) {
        let ldims = textureDimensions(t_label);
        let lijk_f = lt * vec3<f32>(ldims) - vec3<f32>(0.5);
        let lijk = clamp(vec3<i32>(round(lijk_f)),
                         vec3<i32>(0),
                         vec3<i32>(ldims) - vec3<i32>(1));
        label = textureLoad(t_label, lijk, 0).r;
    }
    let li = min(label, 255u);
    let pal = u_palette.entries[li];
    let present = select(0.0, 1.0, label > 0u);
    textureStore(t_dst, vec3<i32>(gid), vec4<f32>(pal.rgb, present * pal.a));
}
"""

    _BAKE_SMOOTH_WGSL = """
// Pass 2/3/4: separable Gaussian on the alpha channel. RGB is copied from
// the center tap so it propagates unchanged through the smoothing passes.
@group(0) @binding(0) var t_src: texture_3d<f32>;
@group(0) @binding(1) var t_dst: texture_storage_3d<rgba16float, write>;

struct SmoothParams {
    sigma: f32,
    axis: u32,
    radius: u32,
    _pad: u32,
};
@group(0) @binding(2) var<uniform> u_sm: SmoothParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let dims_i = vec3<i32>(dims);
    let radius = i32(u_sm.radius);
    let sigma = max(u_sm.sigma, 1e-3);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let base = vec3<i32>(gid);
    let center = textureLoad(t_src, base, 0);

    var asum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var q = -radius; q <= radius; q = q + 1) {
        var c = base;
        if (u_sm.axis == 0u) { c.x = clamp(c.x + q, 0, dims_i.x - 1); }
        else if (u_sm.axis == 1u) { c.y = clamp(c.y + q, 0, dims_i.y - 1); }
        else { c.z = clamp(c.z + q, 0, dims_i.z - 1); }
        let v = textureLoad(t_src, c, 0).a;
        let w = exp(-f32(q * q) * inv_two_sigma_sq);
        asum = asum + v * w;
        wsum = wsum + w;
    }
    let a = asum / max(wsum, 1e-6);
    textureStore(t_dst, base, vec4<f32>(center.rgb, a));
}
"""

    # ---- Surface-mode bake (Jump Flooding SDF + grown palette color) ----
    #
    # Three passes total: init seeds inside-boundary voxels (label > 0 with
    # at least one label-0 neighbor) with their own coord + label; the JFA
    # step pings ping-pong with offsets 2^(L-1), 2^(L-2), ..., 1 so each
    # voxel ends up with its nearest inside-boundary seed; compose reads
    # that seed back, computes the world-space distance, signs it by the
    # voxel's own label, and looks up the palette to write (rgb, sdf_mm)
    # into the rgba16float output. Total cost: O(log N) compute passes.
    _BAKE_JFA_INIT_WGSL = """
@group(0) @binding(0) var t_label: texture_3d<u32>;
@group(0) @binding(1) var t_seed_dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_seed_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let v = vec3<i32>(gid);
    let dims_i = vec3<i32>(dims);
    let label_v = textureLoad(t_label, v, 0).r;

    var seed = vec4<f32>(0.0, 0.0, 0.0, 0.0);  // w==0 means "no seed"
    if (label_v > 0u) {
        // Inside voxel. Boundary if any 6-face neighbor is label-0 (or OOB).
        var is_b = false;
        if (v.x == 0 ||
            textureLoad(t_label, v + vec3<i32>(-1,0,0), 0).r == 0u) { is_b = true; }
        if (!is_b && (v.x == dims_i.x - 1 ||
            textureLoad(t_label, v + vec3<i32>(1,0,0), 0).r == 0u)) { is_b = true; }
        if (!is_b && (v.y == 0 ||
            textureLoad(t_label, v + vec3<i32>(0,-1,0), 0).r == 0u)) { is_b = true; }
        if (!is_b && (v.y == dims_i.y - 1 ||
            textureLoad(t_label, v + vec3<i32>(0,1,0), 0).r == 0u)) { is_b = true; }
        if (!is_b && (v.z == 0 ||
            textureLoad(t_label, v + vec3<i32>(0,0,-1), 0).r == 0u)) { is_b = true; }
        if (!is_b && (v.z == dims_i.z - 1 ||
            textureLoad(t_label, v + vec3<i32>(0,0,1), 0).r == 0u)) { is_b = true; }
        if (is_b) { seed = vec4<f32>(vec3<f32>(v), f32(label_v)); }
    }
    textureStore(t_seed_dst, v, seed);
}
"""

    _BAKE_JFA_STEP_WGSL = """
@group(0) @binding(0) var t_seed_src: texture_3d<f32>;
@group(0) @binding(1) var t_seed_dst: texture_storage_3d<rgba16float, write>;

struct StepParams {
    step_k: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(2) var<uniform> u_params: StepParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_seed_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let v = vec3<i32>(gid);
    let v_pos = vec3<f32>(v);
    let k = i32(u_params.step_k);
    let dims_i = vec3<i32>(dims);

    var best = textureLoad(t_seed_src, v, 0);
    var best_d2 = 1e30;
    if (best.w > 0.5) {
        let d = best.xyz - v_pos;
        best_d2 = dot(d, d);
    }

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                let nv = v + vec3<i32>(dx, dy, dz) * k;
                if (any(nv < vec3<i32>(0)) || any(nv >= dims_i)) { continue; }
                let s = textureLoad(t_seed_src, nv, 0);
                if (s.w < 0.5) { continue; }
                let d = s.xyz - v_pos;
                let d2 = dot(d, d);
                if (d2 < best_d2) {
                    best = s;
                    best_d2 = d2;
                }
            }
        }
    }
    textureStore(t_seed_dst, v, best);
}
"""

    _BAKE_JFA_COMPOSE_WGSL = """
@group(0) @binding(0) var t_seed: texture_3d<f32>;
@group(0) @binding(1) var t_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var t_label: texture_3d<u32>;

struct Palette {
    entries: array<vec4<f32>, 256>,
};
@group(0) @binding(3) var<uniform> u_palette: Palette;

struct ComposeParams {
    // tex (0..1) -> world (mm). Maps voxel-center tex coord to a world point.
    output_to_world: mat4x4<f32>,
};
@group(0) @binding(4) var<uniform> u_cp: ComposeParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let v = vec3<i32>(gid);
    let v_pos = vec3<f32>(v);
    let dims_f = vec3<f32>(dims);

    let seed = textureLoad(t_seed, v, 0);
    let label_v = textureLoad(t_label, v, 0).r;
    let v_tex = (v_pos + vec3<f32>(0.5)) / dims_f;
    let v_world4 = u_cp.output_to_world * vec4<f32>(v_tex, 1.0);
    let v_world = v_world4.xyz / v_world4.w;

    var sdf_mm: f32 = 0.0;
    var rgb = vec3<f32>(0.0);

    if (seed.w > 0.5) {
        let li = min(u32(seed.w), 255u);
        let pal = u_palette.entries[li];
        if (pal.a < 0.5) {
            // Nearest segment is hidden (display opacity 0 / invisible).
            // Push SDF outside any reasonable band so the surface shader's
            // band gate culls this voxel; the JFA still propagates the
            // seed so neighboring visible segments aren't disturbed.
            sdf_mm = select(1e3, -1e3, label_v > 0u);
        } else {
            let s_tex = (seed.xyz + vec3<f32>(0.5)) / dims_f;
            let s_world4 = u_cp.output_to_world * vec4<f32>(s_tex, 1.0);
            let s_world = s_world4.xyz / s_world4.w;
            let dist_mm = length(v_world - s_world);
            let sgn = select(1.0, -1.0, label_v > 0u);
            sdf_mm = sgn * dist_mm;
            rgb = pal.rgb;
        }
    } else {
        // No seed found anywhere -- volume has no segments, or JFA passes
        // didn't reach this voxel. Push SDF beyond any reasonable band so
        // the shader's outside-band fast-path culls this sample.
        sdf_mm = select(1e3, -1e3, label_v > 0u);
    }
    textureStore(t_dst, v, vec4<f32>(rgb, sdf_mm));
}
"""

    _BAKE_MODULATE_WGSL = """
// Final pass: multiply alpha by normalized source-volume intensity.
// Produces the same end-result as ColorizeVolume's CPU pipeline step
//   rgbaVoxels[:,:,:,3] = smoothedAlpha * shiftScaleArray / 255
// but in one compute invocation, reading the raw CT directly.
@group(0) @binding(0) var t_src: texture_3d<f32>;
@group(0) @binding(1) var t_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var t_ct: texture_3d<f32>;

struct ModParams {
    ct_min: f32,
    ct_range: f32,  // (ct_max - ct_min); guarded against 0
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(3) var<uniform> u_mod: ModParams;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(t_dst);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) { return; }
    let base = vec3<i32>(gid);
    let src = textureLoad(t_src, base, 0);
    let ct_raw = textureLoad(t_ct, base, 0).r;
    let ct_n = clamp((ct_raw - u_mod.ct_min)
                     / max(u_mod.ct_range, 1e-6), 0.0, 1.0);
    textureStore(t_dst, base, vec4<f32>(src.rgb, src.a * ct_n));
}
"""

    def _ensure_bake_pipelines(self):
        if getattr(self, "_bake_init_pipeline", None) is not None:
            return
        # Pipeline 1: init (labelmap resample + palette lookup -> RGBA)
        shader = self.device.create_shader_module(code=self._BAKE_INIT_WGSL)
        bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.uint,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._bake_init_bgl = bgl
        self._bake_init_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[bgl]),
            compute={"module": shader, "entry_point": "main"})

        # Pipeline 2: separable smooth (uses the same Smooth WGSL as the
        # segment path but a dedicated layout that reads RGBA -- reuses
        # the per-axis UBO).
        sshader = self.device.create_shader_module(
            code=self._BAKE_SMOOTH_WGSL)
        sbgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._bake_smooth_bgl = sbgl
        self._bake_smooth_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[sbgl]),
            compute={"module": sshader, "entry_point": "main"})

        # Pipeline 3: modulate by CT intensity
        mshader = self.device.create_shader_module(
            code=self._BAKE_MODULATE_WGSL)
        mbgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._bake_modulate_bgl = mbgl
        self._bake_modulate_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[mbgl]),
            compute={"module": mshader, "entry_point": "main"})

        # UBOs: palette (4 KB), 3 smooth-axis, 1 CT modulate, 1 resample.
        self._bake_palette_ubo = self.device.create_buffer(
            size=4096,  # array<vec4<f32>, 256>
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._bake_smooth_ubos = [
            self.device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            for _ in range(3)]
        self._bake_mod_ubo = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._bake_resample_ubo = self.device.create_buffer(
            size=128,  # two mat4x4<f32> = 2 * 64
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # ---- JFA pipelines (surface-mode SDF + grown color) ----
        # Pipeline: JFA init -- read t_label, write seeds where label>0 has
        # a label-0 face neighbor.
        jfa_init_shader = self.device.create_shader_module(
            code=self._BAKE_JFA_INIT_WGSL)
        self._bake_jfa_init_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.uint,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
        ])
        self._bake_jfa_init_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._bake_jfa_init_bgl]),
            compute={"module": jfa_init_shader, "entry_point": "main"})

        # Pipeline: JFA step -- ping-pong propagation of seeds. Reads
        # rgba16float seed source, writes rgba16float seed dest.
        jfa_step_shader = self.device.create_shader_module(
            code=self._BAKE_JFA_STEP_WGSL)
        self._bake_jfa_step_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._bake_jfa_step_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._bake_jfa_step_bgl]),
            compute={"module": jfa_step_shader, "entry_point": "main"})

        # Pipeline: JFA compose -- read final seed + label + palette,
        # write (rgb, sdf_mm) into the rgba volume.
        jfa_compose_shader = self.device.create_shader_module(
            code=self._BAKE_JFA_COMPOSE_WGSL)
        self._bake_jfa_compose_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.unfilterable_float,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                 "format": wgpu.TextureFormat.rgba16float,
                                 "view_dimension": wgpu.TextureViewDimension.d3}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.uint,
                         "view_dimension": wgpu.TextureViewDimension.d3,
                         "multisampled": False}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._bake_jfa_compose_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._bake_jfa_compose_bgl]),
            compute={"module": jfa_compose_shader, "entry_point": "main"})

        self._bake_jfa_step_ubo = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._bake_jfa_compose_ubo = self.device.create_buffer(
            size=64,  # mat4x4<f32>
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    def _run_bake(self, rgba_field):
        """Dispatch the bake into rgba_field.tex.
        Assumes rgba_field holds cached `_label_tex`, `_output_to_world`,
        `_world_to_label_tex`, `sigma_voxels`, and that the palette UBO
        is already up to date.

        Density mode (`rgba_field._ct_tex is not None`): the existing
        ColorizeVolume schedule -- resample on output grid, separable
        Gaussian on alpha, multiply by normalized CT.

            init  -> tex
            smX   -> scratch
            smY   -> tex
            smZ   -> scratch
            mod   -> tex                (alpha *= ct_norm)

        Surface mode (CT not loaded): Jump-Flooding-Algorithm SDF +
        grown palette color, run on the labelmap's own grid (rgba volume
        is allocated to (lx, ly, lz) in this mode). One init pass seeds
        every inside-boundary voxel with its own coord + label, then
        ceil(log2(max_dim)) ping-pong steps propagate seeds; a final
        compose writes (rgb, sdf_mm) into rgba_field.tex. Storage cost:
        the existing `tex` and `scratch_tex` are reused as JFA seed
        ping-pong; no extra allocation.

            init      -> scratch
            steps...  -> tex/scratch (ping-pong, parity controlled
                                      to land in `scratch`)
            compose   -> tex
        """
        self._ensure_bake_pipelines()
        import math, struct as _st
        dx, dy, dz = rgba_field.dims
        surface_mode = rgba_field._ct_tex is None

        wg_x, wg_y, wg_z = 8, 8, 4
        groups = ((dx + wg_x - 1) // wg_x,
                  (dy + wg_y - 1) // wg_y,
                  (dz + wg_z - 1) // wg_z)

        if surface_mode:
            self._run_bake_surface(rgba_field, groups)
            return

        # ---------------------- Density-mode bake ----------------------
        modulate = True

        # Resample params UBO: output_to_world then world_to_label_tex,
        # both column-major so the transpose matches WGSL.
        rparams = np.zeros(32, dtype=np.float32)
        rparams[0:16] = np.asarray(
            rgba_field._output_to_world, dtype=np.float32).T.ravel()
        rparams[16:32] = np.asarray(
            rgba_field._world_to_label_tex, dtype=np.float32).T.ravel()
        self.device.queue.write_buffer(
            self._bake_resample_ubo, 0, rparams.tobytes())

        sigma = max(float(rgba_field.sigma_voxels), 0.25)
        radius = max(int(math.ceil(3.0 * sigma)), 1)
        for axis in range(3):
            sbuf = bytearray(16)
            _st.pack_into("<f", sbuf, 0, sigma)
            _st.pack_into("<I", sbuf, 4, axis)
            _st.pack_into("<I", sbuf, 8, radius)
            self.device.queue.write_buffer(
                self._bake_smooth_ubos[axis], 0, bytes(sbuf))

        mbuf = bytearray(16)
        _st.pack_into("<f", mbuf, 0, float(rgba_field.window_min))
        _st.pack_into("<f", mbuf, 4,
                      float(rgba_field.window_max - rgba_field.window_min))
        self.device.queue.write_buffer(self._bake_mod_ubo, 0, bytes(mbuf))

        encoder = self.device.create_command_encoder()
        init_dst = rgba_field.tex
        smooth_steps = [
            (rgba_field.tex,         rgba_field.scratch_tex, 0),
            (rgba_field.scratch_tex, rgba_field.tex,         1),
            (rgba_field.tex,         rgba_field.scratch_tex, 2),
        ]

        bg = self.device.create_bind_group(
            layout=self._bake_init_bgl, entries=[
                {"binding": 0, "resource": rgba_field._label_tex.create_view()},
                {"binding": 1, "resource": init_dst.create_view()},
                {"binding": 2, "resource": {
                    "buffer": self._bake_palette_ubo, "offset": 0, "size": 4096}},
                {"binding": 3, "resource": {
                    "buffer": self._bake_resample_ubo, "offset": 0, "size": 128}},
            ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._bake_init_pipeline)
        p.set_bind_group(0, bg, [], 0, 0)
        p.dispatch_workgroups(*groups)
        p.end()

        for src_tex, dst_tex, axis in smooth_steps:
            bg = self.device.create_bind_group(
                layout=self._bake_smooth_bgl, entries=[
                    {"binding": 0, "resource": src_tex.create_view()},
                    {"binding": 1, "resource": dst_tex.create_view()},
                    {"binding": 2, "resource": {
                        "buffer": self._bake_smooth_ubos[axis],
                        "offset": 0, "size": 16}},
                ])
            p = encoder.begin_compute_pass()
            p.set_pipeline(self._bake_smooth_pipeline)
            p.set_bind_group(0, bg, [], 0, 0)
            p.dispatch_workgroups(*groups)
            p.end()

        bg = self.device.create_bind_group(
            layout=self._bake_modulate_bgl, entries=[
                {"binding": 0, "resource": rgba_field.scratch_tex.create_view()},
                {"binding": 1, "resource": rgba_field.tex.create_view()},
                {"binding": 2, "resource": rgba_field._ct_tex.create_view()},
                {"binding": 3, "resource": {
                    "buffer": self._bake_mod_ubo, "offset": 0, "size": 16}},
            ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._bake_modulate_pipeline)
        p.set_bind_group(0, bg, [], 0, 0)
        p.dispatch_workgroups(*groups)
        p.end()

        self.device.queue.submit([encoder.finish()])

    def _run_bake_surface(self, rgba_field, groups):
        """Surface-mode bake: GPU JFA over rgba_field._label_tex on the
        labelmap grid, producing (rgb, sdf_mm) in rgba_field.tex.

        Pass schedule:
          1. Init: seeds every inside-boundary voxel into rgba_field.scratch.
          2. JFA steps (ceil(log2(N)) + 1 + parity, ODD count chosen so
             the final seed write lands in `tex` -- compose then reads
             tex and writes scratch.
          3. Compose: read seeds, write (rgb, sdf_mm) into scratch.
          4. Three separable Gaussian passes on the .a channel (rgb is
             copied through unchanged) to round off the equidistant
             ridges where neighboring boundary seeds meet:
                 scratch -> tex   (X)
                 tex     -> scratch (Y)
                 scratch -> tex   (Z)
        Final result lands in rgba_field.tex.
        """
        import math, struct as _st
        dx, dy, dz = rgba_field.dims

        # Compose params UBO: output_to_world (tex 0..1 -> world mm).
        # In surface mode rgba_field.patient_to_texture maps world -> tex,
        # so output_to_world is its inverse.
        p2t = np.asarray(rgba_field.patient_to_texture, dtype=np.float64)
        try:
            output_to_world = np.linalg.inv(p2t)
        except np.linalg.LinAlgError:
            output_to_world = np.eye(4, dtype=np.float64)
        cp = np.asarray(output_to_world, dtype=np.float32).T.ravel()
        self.device.queue.write_buffer(
            self._bake_jfa_compose_ubo, 0, cp.tobytes())

        # Smooth params: separable Gaussian on the SDF .a channel. Sigma
        # in voxels of the rgba grid (= labelmap grid in surface mode).
        sdf_sigma = max(float(getattr(rgba_field,
                                      "sdf_smooth_sigma_voxels", 1.0)),
                        0.25)
        sdf_radius = max(int(math.ceil(3.0 * sdf_sigma)), 1)
        for axis in range(3):
            sbuf = bytearray(16)
            _st.pack_into("<f", sbuf, 0, sdf_sigma)
            _st.pack_into("<I", sbuf, 4, axis)
            _st.pack_into("<I", sbuf, 8, sdf_radius)
            self.device.queue.write_buffer(
                self._bake_smooth_ubos[axis], 0, bytes(sbuf))

        encoder = self.device.create_command_encoder()

        # JFA init -- seed every inside-boundary voxel into scratch.
        bg = self.device.create_bind_group(
            layout=self._bake_jfa_init_bgl, entries=[
                {"binding": 0, "resource": rgba_field._label_tex.create_view()},
                {"binding": 1, "resource": rgba_field.scratch_tex.create_view()},
            ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._bake_jfa_init_pipeline)
        p.set_bind_group(0, bg, [], 0, 0)
        p.dispatch_workgroups(*groups)
        p.end()

        # JFA steps. Ping-pong scratch <-> tex. We want final seeds in
        # `tex` so compose reads tex and writes scratch (different
        # textures -> no aliasing in the bind group). Init wrote to
        # scratch and each step swaps, so an ODD step count lands the
        # final write in tex.
        max_dim = max(dx, dy, dz)
        ks = []
        k = max(max_dim // 2, 1)
        while k >= 1:
            ks.append(k)
            if k == 1:
                break
            k = max(k // 2, 1)
        ks.append(1)  # extra k=1 pass: cleans up missed corners
        if len(ks) % 2 == 0:
            ks.append(1)  # parity pad: enforce odd count so seeds end in tex

        step_ubos = []
        for k_val in ks:
            ubo = self.device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            sbuf = bytearray(16)
            _st.pack_into("<I", sbuf, 0, int(k_val))
            self.device.queue.write_buffer(ubo, 0, bytes(sbuf))
            step_ubos.append(ubo)

        src = rgba_field.scratch_tex
        dst = rgba_field.tex
        for k_val, ubo in zip(ks, step_ubos):
            bg = self.device.create_bind_group(
                layout=self._bake_jfa_step_bgl, entries=[
                    {"binding": 0, "resource": src.create_view()},
                    {"binding": 1, "resource": dst.create_view()},
                    {"binding": 2, "resource": {
                        "buffer": ubo, "offset": 0, "size": 16}},
                ])
            p = encoder.begin_compute_pass()
            p.set_pipeline(self._bake_jfa_step_pipeline)
            p.set_bind_group(0, bg, [], 0, 0)
            p.dispatch_workgroups(*groups)
            p.end()
            src, dst = dst, src
        final_seed_tex = src  # = rgba_field.tex after odd N

        # Compose: read final seeds + labelmap + palette, write (rgb, sdf)
        # into scratch_tex (so the smoothing chain ends in tex).
        bg = self.device.create_bind_group(
            layout=self._bake_jfa_compose_bgl, entries=[
                {"binding": 0, "resource": final_seed_tex.create_view()},
                {"binding": 1, "resource": rgba_field.scratch_tex.create_view()},
                {"binding": 2, "resource": rgba_field._label_tex.create_view()},
                {"binding": 3, "resource": {
                    "buffer": self._bake_palette_ubo, "offset": 0,
                    "size": 4096}},
                {"binding": 4, "resource": {
                    "buffer": self._bake_jfa_compose_ubo,
                    "offset": 0, "size": 64}},
            ])
        p = encoder.begin_compute_pass()
        p.set_pipeline(self._bake_jfa_compose_pipeline)
        p.set_bind_group(0, bg, [], 0, 0)
        p.dispatch_workgroups(*groups)
        p.end()

        # Smooth the SDF: 3 separable passes (X, Y, Z). The smooth shader
        # copies rgb from the center tap and writes the smoothed alpha,
        # so the grown segment color is preserved.
        smooth_steps = [
            (rgba_field.scratch_tex, rgba_field.tex,         0),
            (rgba_field.tex,         rgba_field.scratch_tex, 1),
            (rgba_field.scratch_tex, rgba_field.tex,         2),
        ]
        for src_tex, dst_tex, axis in smooth_steps:
            bg = self.device.create_bind_group(
                layout=self._bake_smooth_bgl, entries=[
                    {"binding": 0, "resource": src_tex.create_view()},
                    {"binding": 1, "resource": dst_tex.create_view()},
                    {"binding": 2, "resource": {
                        "buffer": self._bake_smooth_ubos[axis],
                        "offset": 0, "size": 16}},
                ])
            p = encoder.begin_compute_pass()
            p.set_pipeline(self._bake_smooth_pipeline)
            p.set_bind_group(0, bg, [], 0, 0)
            p.dispatch_workgroups(*groups)
            p.end()

        self.device.queue.submit([encoder.finish()])

    def _build_palette_array(self, segmentation_node):
        """Build a 256-entry palette (RGBA float32) from the current segment
        colors and 3D visibility / opacity. Invisible segments get alpha=0
        so they disappear from the bake without re-uploading the labelmap."""
        palette = np.zeros((256, 4), dtype=np.float32)
        seg = segmentation_node.GetSegmentation()
        dn = segmentation_node.GetDisplayNode()
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            segment = seg.GetSegment(sid)
            try:
                lv = int(segment.GetLabelValue())
            except Exception:
                lv = i + 1
            if not (0 < lv < 256):
                continue
            rgb = segment.GetColor()
            op = 1.0
            if dn is not None:
                try:
                    visible = bool(dn.GetSegmentVisibility(sid)
                                   and dn.GetVisibility3D())
                except Exception:
                    visible = True
                try:
                    op = float(dn.GetSegmentOpacity3D(sid))
                except Exception:
                    op = 1.0
                if not visible:
                    op = 0.0
            palette[lv, 0] = rgb[0]
            palette[lv, 1] = rgb[1]
            palette[lv, 2] = rgb[2]
            palette[lv, 3] = op
        palette[0, :] = (0.0, 0.0, 0.0, 0.0)
        return palette

    @staticmethod
    def _extract_shared_labelmap(segmentation_node, visible_ids):
        """Return the vtkOrientedImageData shared by all visible segments,
        or None if they span multiple layers (caller should fall back).
        """
        if not visible_ids:
            return None
        first = segmentation_node.GetBinaryLabelmapInternalRepresentation(
            visible_ids[0])
        if first is None:
            return None
        for sid in visible_ids[1:]:
            other = segmentation_node.GetBinaryLabelmapInternalRepresentation(sid)
            if other is not first:
                return None
        return first

    # ---------------------------------------------------------------------
    # FragmentField + strand rasterization pipelines (lazy)
    # ---------------------------------------------------------------------

    _STRAND_RASTER_WGSL = """
// Quadratic-Bezier-tube rasterization. Each polyline kink p_i (interior
// vertex) becomes ONE rendered piece, parameterized as a quadratic
// Bezier with control points:
//     B0 = midpoint(p_{i-1}, p_i)
//     B1 = p_i
//     B2 = midpoint(p_i, p_{i+1})
// The tangent at B0 is 2(B1-B0) = p_i - p_{i-1} (incoming edge dir);
// the tangent at B2 is 2(B2-B1) = p_{i+1} - p_i (outgoing edge dir).
// Adjacent pieces share both their endpoint and their tangent direction
// at the joint midpoint -- the rendered strand is C0 + tangent-C1
// continuous across EVERY original polyline kink, even very sharp ones.
//
// Heads/tails of strands use degenerate Beziers with B0=B1 (head) or
// B1=B2 (tail), which collapses the tube to a hemispherical cap at the
// strand endpoint.
//
// Fragment shader does a true analytic ray vs swept-tube intersection:
// 1) Solve the cubic d/dt |u(t)|^2 = 0 (where u(t) is the curve point
//    projected perpendicular to the ray) via Cardano to find t* where
//    the centerline is closest to the ray.
// 2) Approximate the tube near c(t*) as a local cylinder along c'(t*)
//    for the surface-point + entry-depth resolution.
// 3) Use c'(t*) as tangent and (hit - c(t*))_perp_c'(t*) as the normal
//    -- both are smooth functions of t, with no per-segment jumps.
//
// Reference: Reshetov & Luebke, "Phantom Ray-Hair Intersector"
// (SIGGRAPH 2018) uses the same construction for hair rendering.

struct Camera {
    proj_inv: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    // size.xy = (w, h); size.zw = TAA jitter in NDC units
    size: vec4<f32>,
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    taa: vec4<f32>,           // .x = frame index for hash seeding
};

struct StrandParams {
    world_from_local: mat4x4<f32>,
    tube_radius_mm: f32,
    pad_world_mm: f32,       // billboard padding for sub-pixel coverage
    K: u32,                  // FragmentField K (max frags / pixel)
    _pad0: u32,
    shade: vec4<f32>,        // (ka, kd, ks, shin)
};

@group(0) @binding(0) var<uniform> u_cam: Camera;
@group(0) @binding(1) var<uniform> u_field: StrandParams;
@group(0) @binding(3) var t_pal: texture_1d<f32>;
@group(0) @binding(4) var<storage, read_write> u_counts: array<atomic<u32>>;
// Atomic view of the fragment buffer so keep-K-shallowest can do a
// compare-exchange on the deepest stored slot. Sort + main shaders read
// the same memory non-atomically (no concurrent writes after raster).
@group(0) @binding(5) var<storage, read_write> u_fragments: array<atomic<u32>>;

struct VsIn {
    @location(0) B0_local: vec3<f32>,
    @location(1) B1_local: vec3<f32>,
    @location(2) B2_local: vec3<f32>,
    @location(3) side_sign: f32,    // -1 or +1 (perpendicular sign)
    @location(4) end_t: f32,        // 0.0 = anchor at B0, 1.0 = anchor at B2
    @location(5) bundle_id: f32,
    @location(6) is_head: f32,      // 1.0 if this piece is a strand head/tail
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // The 3 Bezier control points (world space), flat-interpolated so
    // the fragment shader gets the same piece data regardless of which
    // triangle vertex the rasterizer uses for the provoking vertex.
    @location(0) @interpolate(flat) B0_world: vec3<f32>,
    @location(1) @interpolate(flat) B1_world: vec3<f32>,
    @location(2) @interpolate(flat) B2_world: vec3<f32>,
    @location(3) @interpolate(flat) bundle_id: f32,
    @location(4) @interpolate(flat) is_head: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let bid_whole = floor(in.bundle_id);
    let B0w = (u_field.world_from_local * vec4<f32>(in.B0_local, 1.0)).xyz;
    let B1w = (u_field.world_from_local * vec4<f32>(in.B1_local, 1.0)).xyz;
    let B2w = (u_field.world_from_local * vec4<f32>(in.B2_local, 1.0)).xyz;

    // Anchor at B0 or B2 (the curve's endpoints).
    let anchor = mix(B0w, B2w, in.end_t);

    // Endpoint tangents.
    let tan0 = 2.0 * (B1w - B0w);
    let tan1 = 2.0 * (B2w - B1w);
    let tan_raw = mix(tan0, tan1, in.end_t);
    var tan_n = vec3<f32>(1.0, 0.0, 0.0);
    let tan_l = length(tan_raw);
    if (tan_l > 1e-6) { tan_n = tan_raw / tan_l; }

    // Screen-perpendicular side direction (in world space).
    let cam_pos = u_cam.view_inv[3].xyz;
    let view_v = anchor - cam_pos;
    let side_raw = cross(tan_n, view_v);
    var side = vec3<f32>(0.0, 1.0, 0.0);
    let side_l = length(side_raw);
    if (side_l > 1e-6) { side = side_raw / side_l; }

    let r = u_field.tube_radius_mm + max(u_field.pad_world_mm, 0.0);

    // Bezier convex hull bulge perpendicular to the chord. The curve
    // stays inside the convex hull of (B0, B1, B2), so the maximum
    // side excursion is the perpendicular distance from B1 to the
    // B0-B2 chord. Extending the billboard sideways by this amount
    // ensures we cover the entire visible tube even for sharp kinks.
    let chord = B2w - B0w;
    let chord_l = length(chord);
    var chord_n = tan_n;
    if (chord_l > 1e-6) { chord_n = chord / chord_l; }
    let b1_off = B1w - B0w;
    let b1_along = dot(b1_off, chord_n);
    let b1_perp_vec = b1_off - b1_along * chord_n;
    let b1_perp = length(b1_perp_vec);

    // Axial extension at the endpoint side (by r along the endpoint
    // tangent direction, outward from the curve). Plus an extra chord-
    // axis push at the B0 end to also cover B1 when the kink is sharp.
    let axial = mix(-r * tan_n, r * tan_n, in.end_t);
    let r_side = r + b1_perp;
    let p_world = anchor + axial + (r_side * in.side_sign) * side;

    var out: VsOut;
    out.clip = u_cam.proj * u_cam.view * vec4<f32>(p_world, 1.0);
    out.B0_world = B0w;
    out.B1_world = B1w;
    out.B2_world = B2w;
    out.bundle_id = bid_whole;
    out.is_head = in.is_head;
    return out;
}

fn ndc_to_world_strand(ndc: vec4<f32>) -> vec3<f32> {
    let v = u_cam.proj_inv * ndc;
    let w = u_cam.view_inv * vec4<f32>(v.xyz / v.w, 1.0);
    return w.xyz;
}

struct FsOut {
    @location(0) dummy: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    var out: FsOut;
    out.dummy = vec4<f32>(0.0);

    // Reconstruct world-space ray for this pixel from frag_coord (in.clip).
    let ndc_x = (in.clip.x / u_cam.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.clip.y / u_cam.size.y) * 2.0;
    let wn = ndc_to_world_strand(vec4<f32>(ndc_x, ndc_y, 0.0, 1.0));
    let wf = ndc_to_world_strand(vec4<f32>(ndc_x, ndc_y, 1.0, 1.0));
    let ro = wn;
    let rd = normalize(wf - wn);

    // Ray vs quadratic-Bezier tube of radius r.
    //
    // Bezier centerline (offset by ro for the per-pixel ray test):
    //   c(t) - ro = alpha + beta * t + gamma * t^2
    // with alpha = B0 - ro, beta = 2(B1-B0), gamma = B0 - 2 B1 + B2.
    //
    // Project these onto the plane perpendicular to rd:
    //   u(t) = c(t)_perp - ro_perp = a_p + b_p * t + g_p * t^2
    // where x_p = x - (x . rd) rd. The closest point on the centerline
    // to the ray is the t* that minimizes |u(t)|^2.
    //
    // d/dt |u|^2 = 2 u . u' = 0 -> a cubic in t. Closed form via
    // Cardano's formula.
    let r = u_field.tube_radius_mm;
    let alpha_v = in.B0_world - ro;
    let beta_v = 2.0 * (in.B1_world - in.B0_world);
    let gamma_v = in.B0_world - 2.0 * in.B1_world + in.B2_world;
    let a_p = alpha_v - dot(alpha_v, rd) * rd;
    let b_p = beta_v - dot(beta_v, rd) * rd;
    let g_p = gamma_v - dot(gamma_v, rd) * rd;
    let gg = dot(g_p, g_p);
    let gb = dot(g_p, b_p);
    let ga = dot(g_p, a_p);
    let bb = dot(b_p, b_p);
    let ba = dot(b_p, a_p);

    // Strict ownership across joints. Adjacent pieces share endpoint
    // midpoints, so we'd otherwise see duplicate fragments on every
    // joint (bubbles). The clean rule:
    //   - Interior pieces accept ONLY natural cubic roots in [0, 1).
    //     Strict on the right (1.0 excluded) means the next piece's
    //     t=0 owns the joint uniquely. Clamping the cubic root to a
    //     boundary is NOT accepted -- that's a hit for the neighbor.
    //   - Head pieces (B0==B1) additionally accept t=0 unconditionally
    //     (the spherical-cap tip).
    //   - Tail pieces (B1==B2) accept t=1 unconditionally and extend
    //     the upper bound to inclusive 1.0.
    let is_head = length(in.B1_world - in.B0_world) < 1e-6;
    let is_tail = length(in.B2_world - in.B1_world) < 1e-6;
    let t_upper = select(1.0 - 1e-5, 1.0, is_tail);

    var best_d2: f32 = 1e30;
    var t_star: f32 = 0.0;
    var found: bool = false;

    if (gg < 1e-12) {
        // Degenerate (linear) Bezier -- the cubic collapses to a
        // linear equation in t.
        if (bb > 1e-12) {
            let t_lin = -ba / bb;
            if (t_lin >= 0.0 && t_lin <= t_upper) {
                let u_lin = a_p + b_p * t_lin;
                let d2 = dot(u_lin, u_lin);
                if (d2 < best_d2) {
                    best_d2 = d2; t_star = t_lin; found = true;
                }
            }
        }
    } else {
        // Real cubic: A t^3 + B t^2 + C t + D = 0, with
        //   A = 2 |g|^2, B = 3 g.b, C = 2 g.a + |b|^2, D = b.a.
        let A = 2.0 * gg;
        let B = 3.0 * gb;
        let C = 2.0 * ga + bb;
        let D = ba;
        // Depress: substitute t = u - B/(3A) so the t^2 term vanishes.
        let inv_A = 1.0 / A;
        let b_over_a = B * inv_A;
        let c_over_a = C * inv_A;
        let d_over_a = D * inv_A;
        let p_dep = c_over_a - b_over_a * b_over_a / 3.0;
        let q_dep = (2.0 / 27.0) * b_over_a * b_over_a * b_over_a
                  - (b_over_a * c_over_a) / 3.0 + d_over_a;
        let disc = -4.0 * p_dep * p_dep * p_dep - 27.0 * q_dep * q_dep;
        let shift = -b_over_a / 3.0;
        if (disc > 0.0) {
            // Three real roots: trigonometric form.
            let m = 2.0 * sqrt(max(-p_dep / 3.0, 0.0));
            let arg = clamp(3.0 * q_dep / (p_dep * m), -1.0, 1.0);
            let theta = acos(arg) / 3.0;
            let tau = 2.0943951;     // 2*pi/3
            let u0 = m * cos(theta);
            let u1 = m * cos(theta - tau);
            let u2 = m * cos(theta - 2.0 * tau);
            for (var i = 0; i < 3; i = i + 1) {
                var u_root = u0;
                if (i == 1) { u_root = u1; }
                if (i == 2) { u_root = u2; }
                let t_root = u_root + shift;
                if (t_root >= 0.0 && t_root <= t_upper) {
                    let u_c = a_p + b_p * t_root + g_p * (t_root * t_root);
                    let d2 = dot(u_c, u_c);
                    if (d2 < best_d2) {
                        best_d2 = d2; t_star = t_root; found = true;
                    }
                }
            }
        } else {
            // One real root: Cardano.
            let half_q = q_dep * 0.5;
            let third_p = p_dep / 3.0;
            let sq_arg = half_q * half_q + third_p * third_p * third_p;
            let sq = sqrt(max(sq_arg, 0.0));
            let uu = -half_q + sq;
            let vv = -half_q - sq;
            let cu = sign(uu) * pow(abs(uu), 1.0 / 3.0);
            let cv = sign(vv) * pow(abs(vv), 1.0 / 3.0);
            let u_root = cu + cv;
            let t_root = u_root + shift;
            if (t_root >= 0.0 && t_root <= t_upper) {
                let u_c = a_p + b_p * t_root + g_p * (t_root * t_root);
                let d2 = dot(u_c, u_c);
                if (d2 < best_d2) {
                    best_d2 = d2; t_star = t_root; found = true;
                }
            }
        }
    }

    // Head/tail cap endpoint candidates -- always allowed for the
    // actual strand ends, never for interior pieces.
    if (is_head) {
        let u0 = a_p;
        let d2 = dot(u0, u0);
        if (d2 < best_d2) {
            best_d2 = d2; t_star = 0.0; found = true;
        }
    }
    if (is_tail) {
        let u1 = a_p + b_p + g_p;
        let d2 = dot(u1, u1);
        if (d2 < best_d2) {
            best_d2 = d2; t_star = 1.0; found = true;
        }
    }

    if (!found || best_d2 > r * r) { discard; }

    // Centerline point + tangent at t*.
    let ts = t_star;
    let c_t = in.B0_world + beta_v * ts + gamma_v * (ts * ts);
    let c_prime = beta_v + 2.0 * gamma_v * ts;
    let c_prime_l = length(c_prime);
    var tangent = vec3<f32>(0.0, 0.0, 1.0);
    if (c_prime_l > 1e-6) { tangent = c_prime / c_prime_l; }

    // Resolve the surface entry point via a local-cylinder
    // approximation around c(t*) along c'(t*). This is exact when
    // the curve is locally straight (which it is, to first order, in
    // any neighborhood of t*); the second-order error is O(curvature
    // * r^2), which is well below a pixel for typical fiber radii.
    // For the cap region (t* = 0 or t* = 1 with c'(t*) -> 0 at a
    // degenerate Bezier endpoint), the local-cylinder degenerates to
    // a sphere and gives the correct hemispherical cap automatically.
    let oc = ro - c_t;
    let rd_along_t = dot(rd, tangent);
    let oc_along_t = dot(oc, tangent);
    let rd_perp_t = rd - rd_along_t * tangent;
    let oc_perp_t = oc - oc_along_t * tangent;
    let cyl_a = dot(rd_perp_t, rd_perp_t);
    var hit = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);
    if (cyl_a > 1e-9) {
        let cyl_b = 2.0 * dot(oc_perp_t, rd_perp_t);
        let cyl_c = dot(oc_perp_t, oc_perp_t) - r * r;
        let cyl_disc = cyl_b * cyl_b - 4.0 * cyl_a * cyl_c;
        if (cyl_disc < 0.0) { discard; }
        let sd = sqrt(cyl_disc);
        let s_hit = (-cyl_b - sd) / (2.0 * cyl_a);
        if (s_hit < 0.0) { discard; }
        hit = ro + s_hit * rd;
        let perp = hit - c_t;
        let perp_along = dot(perp, tangent);
        let perp_n_v = perp - perp_along * tangent;
        let perp_n_l = length(perp_n_v);
        if (perp_n_l > 1e-6) { normal = perp_n_v / perp_n_l; } else { discard; }
    } else {
        // Ray nearly parallel to tangent: fall back to sphere at c(t*).
        let oc_l = ro - c_t;
        let bs = 2.0 * dot(rd, oc_l);
        let cs = dot(oc_l, oc_l) - r * r;
        let ds = bs * bs - 4.0 * cs;
        if (ds < 0.0) { discard; }
        let s_hit = (-bs - sqrt(ds)) * 0.5;
        if (s_hit < 0.0) { discard; }
        hit = ro + s_hit * rd;
        normal = normalize(hit - c_t);
    }

    // Shading: same Lambertian + Kajiya-Kay as before, but now
    // tangent + normal are smooth functions of t along the strand,
    // so the per-segment shading discontinuity vanishes.
    let bid_clamped = clamp(i32(in.bundle_id), 0, 255);
    let pal = textureLoad(t_pal, bid_clamped, 0);
    let base_color = pal.rgb;
    let bundle_alpha = pal.a;
    let nl = max(dot(-rd, normal), 0.0);
    let ct = dot(rd, tangent);
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let shade = u_field.shade;
    var lit = base_color * (shade.x + shade.y * nl)
            + vec3<f32>(shade.z * pow(st, max(shade.w, 1.0)) * nl);
    lit = clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0));

    let alpha = bundle_alpha;
    let cr = u32(clamp(lit.r * alpha, 0.0, 1.0) * 255.0);
    let cg = u32(clamp(lit.g * alpha, 0.0, 1.0) * 255.0);
    let cb = u32(clamp(lit.b * alpha, 0.0, 1.0) * 255.0);
    let ca = u32(clamp(alpha, 0.0, 1.0) * 255.0);
    let packed = cr | (cg << 8u) | (cb << 16u) | (ca << 24u);

    let clip_hit = u_cam.proj * u_cam.view * vec4<f32>(hit, 1.0);
    let ndc_z = clip_hit.z / max(clip_hit.w, 1e-6);
    let depth_01 = clamp(ndc_z * 0.5 + 0.5, 0.0, 1.0);

    let pixel_idx = u32(in.clip.y) * u32(u_cam.size.x) + u32(in.clip.x);
    let slot = atomicAdd(&u_counts[pixel_idx], 1u);
    let K = u_field.K;
    let new_d_bits = bitcast<u32>(depth_01);
    if (slot < K) {
        // Fast path: append at the next free slot.
        let buf_idx = (pixel_idx * K + slot) * 2u;
        atomicStore(&u_fragments[buf_idx], new_d_bits);
        atomicStore(&u_fragments[buf_idx + 1u], packed);
    } else {
        // Slow path: buffer full. Find the deepest stored fragment and,
        // if our depth is shallower, atomic-replace it. Read-then-CAS is
        // racy under concurrent writes from other threads at the same
        // pixel, but the worst case is "approximate" K-shallowest --
        // visually far better than first-come-wins.
        var worst_idx: u32 = 0u;
        var worst_d_bits: u32 = atomicLoad(&u_fragments[pixel_idx * K * 2u]);
        var worst_d: f32 = bitcast<f32>(worst_d_bits);
        for (var i = 1u; i < K; i = i + 1u) {
            let d_bits = atomicLoad(&u_fragments[(pixel_idx * K + i) * 2u]);
            let d = bitcast<f32>(d_bits);
            if (d > worst_d) {
                worst_d = d;
                worst_d_bits = d_bits;
                worst_idx = i;
            }
        }
        if (depth_01 < worst_d) {
            let buf_idx = (pixel_idx * K + worst_idx) * 2u;
            let res = atomicCompareExchangeWeak(
                &u_fragments[buf_idx], worst_d_bits, new_d_bits);
            if (res.exchanged) {
                // We won the race; commit the colour. There is a brief
                // window where another thread can read (new depth, old
                // colour) -- accepted: races resolve before the strand
                // pass ends and the sort/main shader fire.
                atomicStore(&u_fragments[buf_idx + 1u], packed);
            }
        }
    }
    return out;
}
"""

    _FRAG_CLEAR_WGSL = """
@group(0) @binding(0) var<storage, read_write> u_counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&u_counts)) { return; }
    atomicStore(&u_counts[idx], 0u);
}
"""

    _FRAG_SORT_WGSL = """
struct SortParams {
    width: u32,
    height: u32,
    K: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> u_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> u_fragments: array<u32>;
@group(0) @binding(2) var<uniform> u_p: SortParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = u_p.width * u_p.height;
    let pixel_idx = gid.x;
    if (pixel_idx >= total) { return; }
    let count = min(u_counts[pixel_idx], u_p.K);
    if (count <= 1u) { return; }

    // Per-pixel insertion sort by depth (ascending: front-to-back).
    let base = pixel_idx * u_p.K * 2u;
    for (var i = 1u; i < count; i = i + 1u) {
        let key_d_bits = u_fragments[base + i * 2u];
        let key_c = u_fragments[base + i * 2u + 1u];
        let key_d = bitcast<f32>(key_d_bits);
        var j = i;
        loop {
            if (j == 0u) { break; }
            let prev_d = bitcast<f32>(u_fragments[base + (j - 1u) * 2u]);
            if (prev_d <= key_d) { break; }
            u_fragments[base + j * 2u]      = u_fragments[base + (j - 1u) * 2u];
            u_fragments[base + j * 2u + 1u] = u_fragments[base + (j - 1u) * 2u + 1u];
            j = j - 1u;
        }
        u_fragments[base + j * 2u] = key_d_bits;
        u_fragments[base + j * 2u + 1u] = key_c;
    }
}
"""

    def _ensure_strand_pipelines(self):
        if self._strand_pipeline is not None:
            return
        # ---- Strand raster pipeline (vertex + fragment) ----
        shader = self.device.create_shader_module(
            code=self._STRAND_RASTER_WGSL)
        # Strip pipeline: per-vertex polyline neighborhood data + index
        # buffer that shares vertices at sub-segment joints. No segment
        # storage buffer -- the LEFT vertex of each segment carries
        # p0/p1/prev_p0 forward to the fragment shader via @flat.
        self._strand_pipeline_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": wgpu.TextureSampleType.float,
                         "view_dimension": wgpu.TextureViewDimension.d1,
                         "multisampled": False}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
        ])
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._strand_pipeline_bgl])
        # Render pass writes a dummy output; the actual visible work is the
        # atomic-append into the A-buffer. Color target format must match
        # whatever begin_render_pass binds, but we use a 1x1 throwaway view.
        # Vertex layout (52 bytes): B0_local(12) + B1_local(12)
        # + B2_local(12) + side_sign(4) + end_t(4) + bundle_id(4)
        # + is_head(4). Each piece is one quadratic Bezier in 3D and
        # one of 4 corners of a billboard quad around it.
        vertex_buffer_layout = {
            "array_stride": 52,
            "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {"format": wgpu.VertexFormat.float32x3, "offset":  0, "shader_location": 0},
                {"format": wgpu.VertexFormat.float32x3, "offset": 12, "shader_location": 1},
                {"format": wgpu.VertexFormat.float32x3, "offset": 24, "shader_location": 2},
                {"format": wgpu.VertexFormat.float32,   "offset": 36, "shader_location": 3},
                {"format": wgpu.VertexFormat.float32,   "offset": 40, "shader_location": 4},
                {"format": wgpu.VertexFormat.float32,   "offset": 44, "shader_location": 5},
                {"format": wgpu.VertexFormat.float32,   "offset": 48, "shader_location": 6},
            ],
        }
        self._strand_pipeline = self.device.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main",
                    "buffers": [vertex_buffer_layout]},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": wgpu.TextureFormat.rgba8unorm,
                                   "write_mask": 0}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list,
                       "cull_mode": wgpu.CullMode.none},
        )

        # ---- Clear-counts compute pipeline ----
        clear_shader = self.device.create_shader_module(
            code=self._FRAG_CLEAR_WGSL)
        self._fragment_clear_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
        ])
        self._fragment_clear_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._fragment_clear_bgl]),
            compute={"module": clear_shader, "entry_point": "main"})

        # ---- Sort compute pipeline ----
        sort_shader = self.device.create_shader_module(
            code=self._FRAG_SORT_WGSL)
        self._fragment_sort_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._fragment_sort_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._fragment_sort_bgl]),
            compute={"module": sort_shader, "entry_point": "main"})
        self._fragment_sort_ubo = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # ---- TAA composite compute pipeline ----
        # The 7-yr-old AMD adapter on the dev machine doesn't expose the
        # read_write storage-texture access mode for rgba8unorm, so we
        # split the history into read (sampled texture_2d) and write
        # (write-only storage_texture) bindings and copy the output back
        # to the history texture after the pass.
        taa_shader = self.device.create_shader_module(code=self._TAA_WGSL)
        self._taa_bgl = self.device.create_bind_group_layout(entries=[
            # current frame, sampled as texture
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {
                "sample_type": wgpu.TextureSampleType.float,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "multisampled": False,
             }},
            # history accumulator (previous frame), sampled as texture
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {
                "sample_type": wgpu.TextureSampleType.float,
                "view_dimension": wgpu.TextureViewDimension.d2,
                "multisampled": False,
             }},
            # output texture, write-only storage image
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {
                "access": wgpu.StorageTextureAccess.write_only,
                "format": wgpu.TextureFormat.rgba8unorm,
                "view_dimension": wgpu.TextureViewDimension.d2,
             }},
            # params UBO: blend factor + viewport size
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._taa_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._taa_bgl]),
            compute={"module": taa_shader, "entry_point": "main"})
        self._taa_ubo = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    _TAA_WGSL = """
struct TaaParams {
    // .x = blend factor for current frame (1.0 = pure current = reset);
    //      typical sustained value ~0.1 .
    // .y, .z = viewport (w, h) so we can bounds-check.
    params: vec4<f32>,
};

@group(0) @binding(0) var t_current: texture_2d<f32>;
@group(0) @binding(1) var t_history: texture_2d<f32>;
@group(0) @binding(2) var t_output:  texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> u_p: TaaParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(u_p.params.y);
    let h = u32(u_p.params.z);
    if (gid.x >= w || gid.y >= h) { return; }
    let xy = vec2<i32>(i32(gid.x), i32(gid.y));
    let cur = textureLoad(t_current, xy, 0);
    let hist = textureLoad(t_history, xy, 0);
    let mixf = u_p.params.x;
    let blended = mix(hist, cur, mixf);
    textureStore(t_output, xy, blended);
}
"""

    def add_colorize_volume(self, volume_node, segmentation_node,
                            sigma_voxels=1.5, window_level=None,
                            modulate_by_ct=True,
                            carve_dilate_voxels=None):
        """Bake a ColorizeVolume-style RGBA 3D texture on the GPU and install
        it as a new RGBAVolumeField. All steps (labelmap resample + palette
        LUT, separable Gaussian on alpha, alpha * normalized-CT) run in
        compute shaders. Output is a single rgba16float 3D texture read at
        render time with RGB = color, A = opacity.

        Fast path: if all visible segments share one labelmap (typical
        Slicer segmentation), no CPU ExportSegmentsToLabelmapNode call --
        the shared labelmap is uploaded as-is and resampled to the CT grid
        inside the init compute pass. Fallback to Export only when segments
        span multiple layers.

        Also installs a display-node observer: toggling segment visibility
        or changing color / opacity only re-packs the palette UBO and re-runs
        the compute passes; the CT and labelmap textures are cached.
        """
        import vtk.util.numpy_support as vnp

        dn_seg = segmentation_node.GetDisplayNode()
        visible_ids = []
        if dn_seg is not None:
            raw = dn_seg.GetVisibleSegmentIDs()
            if hasattr(raw, "GetNumberOfValues"):
                # Older API shape: vtkStringArray
                for i in range(raw.GetNumberOfValues()):
                    visible_ids.append(raw.GetValue(i))
            elif raw is not None:
                # Newer API shape: plain list of strings
                try:
                    visible_ids = [str(x) for x in raw]
                except TypeError:
                    visible_ids = []
        if not visible_ids:
            raise RuntimeError("segmentation has no visible segments")

        # Try the fast path: one shared labelmap across visible segments.
        label_oimg = self._extract_shared_labelmap(
            segmentation_node, visible_ids)
        label_node_to_delete = None
        if label_oimg is None:
            # Fallback: Slicer logic merges per-segment labelmaps into one
            # aligned to the source volume's grid. The Export API takes a
            # vtkStringArray (or compatible wrapper) regardless of the API
            # shape of GetVisibleSegmentIDs, so marshal into one here.
            sa = vtk.vtkStringArray()
            for sid in visible_ids:
                sa.InsertNextValue(sid)
            label_node_to_delete = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLabelMapVolumeNode", "__bake_label__")
            ok = slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentation_node, sa, label_node_to_delete,
                volume_node)
            if not ok:
                slicer.mrmlScene.RemoveNode(label_node_to_delete)
                raise RuntimeError("ExportSegmentsToLabelmapNode failed")
            label_oimg = label_node_to_delete.GetImageData()
            # Compose the node's IJK-to-RAS onto the image so extractors
            # below treat it like a vtkOrientedImageData.
            m4 = vtk.vtkMatrix4x4()
            label_node_to_delete.GetIJKToRASMatrix(m4)

        # Labelmap geometry (absolute-ijk -> world) -------------------------
        label_ext = label_oimg.GetExtent()
        lx = label_ext[1] - label_ext[0] + 1
        ly = label_ext[3] - label_ext[2] + 1
        lz = label_ext[5] - label_ext[4] + 1
        lbl_scalars = label_oimg.GetPointData().GetScalars()
        label_arr = vnp.vtk_to_numpy(lbl_scalars).reshape(lz, ly, lx)
        # If the labelmap is already uint8 with values in [0,255], upload as
        # is -- skip an allocating copy-and-clip. Slicer's binary labelmaps
        # almost always are uint8 already.
        if label_arr.dtype == np.uint8:
            label_u8 = np.ascontiguousarray(label_arr)
        else:
            label_u8 = np.ascontiguousarray(
                np.clip(label_arr, 0, 255).astype(np.uint8))

        if label_node_to_delete is not None:
            # labelmap_node path: use its IJK-to-RAS directly (no extent shift).
            lbl_mat = vtk.vtkMatrix4x4()
            label_node_to_delete.GetIJKToRASMatrix(lbl_mat)
            label_ijk_to_world = np.array(
                [[lbl_mat.GetElement(i, j) for j in range(4)]
                 for i in range(4)], dtype=np.float64)
        else:
            # Oriented image: GetImageToWorldMatrix is absolute-ijk -> world.
            lbl_mat = vtk.vtkMatrix4x4()
            label_oimg.GetImageToWorldMatrix(lbl_mat)
            label_ijk_to_world = np.array(
                [[lbl_mat.GetElement(i, j) for j in range(4)]
                 for i in range(4)], dtype=np.float64)
        label_tex_to_ijk = np.array([
            [lx, 0.0, 0.0, label_ext[0] - 0.5],
            [0.0, ly, 0.0, label_ext[2] - 0.5],
            [0.0, 0.0, lz, label_ext[4] - 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        label_tex_to_world = label_ijk_to_world @ label_tex_to_ijk
        world_to_label_tex = np.linalg.inv(label_tex_to_world)

        # CT geometry -------------------------------------------------------
        ct_img = volume_node.GetImageData()
        ct_ext = ct_img.GetExtent()
        dx = ct_ext[1] - ct_ext[0] + 1
        dy = ct_ext[3] - ct_ext[2] + 1
        dz = ct_ext[5] - ct_ext[4] + 1
        ct_arr = slicer.util.arrayFromVolume(volume_node)
        # astype(copy=False) returns the input unchanged when dtype already
        # matches -- no allocation, no memcpy. Only actually casts when the
        # CT is stored in a non-float32 scalar type.
        ct_f32 = np.ascontiguousarray(
            ct_arr.astype(np.float32, copy=False))

        ct_ijk_to_ras_mat = vtk.vtkMatrix4x4()
        volume_node.GetIJKToRASMatrix(ct_ijk_to_ras_mat)
        ct_ijk_to_world = np.array(
            [[ct_ijk_to_ras_mat.GetElement(i, j) for j in range(4)]
             for i in range(4)], dtype=np.float64)
        ct_tex_to_ijk = np.array([
            [dx, 0.0, 0.0, ct_ext[0] - 0.5],
            [0.0, dy, 0.0, ct_ext[2] - 0.5],
            [0.0, 0.0, dz, ct_ext[4] - 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        output_to_world = ct_ijk_to_world @ ct_tex_to_ijk  # used by init shader
        world_to_output_tex = np.linalg.inv(output_to_world)

        # Window / level for alpha modulation -------------------------------
        if window_level is not None:
            ct_min, ct_max = float(window_level[0]), float(window_level[1])
        else:
            d = volume_node.GetScalarVolumeDisplayNode()
            if d is not None:
                ct_min = float(d.GetWindowLevelMin())
                ct_max = float(d.GetWindowLevelMax())
            else:
                rng = ct_img.GetScalarRange()
                ct_min, ct_max = float(rng[0]), float(rng[1])

        # Upload CT + labelmap (cached on the field for re-bake). Pass the
        # numpy arrays in directly -- wgpu-py reads them via the buffer
        # protocol, so skipping a .tobytes() call saves a full copy of the
        # volume on the way in. For surface mode (modulate_by_ct=False) the
        # CT is never needed -- skip its upload entirely to save ~685 MB of
        # GPU memory and ~2 s of upload time.
        if modulate_by_ct:
            ct_tex = self.device.create_texture(
                size=(dx, dy, dz), dimension="3d",
                format=wgpu.TextureFormat.r32float,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture(
                {"texture": ct_tex, "mip_level": 0, "origin": (0, 0, 0)},
                ct_f32,
                {"offset": 0, "bytes_per_row": dx * 4, "rows_per_image": dy},
                (dx, dy, dz))
        else:
            ct_tex = None
        label_tex = self.device.create_texture(
            size=(lx, ly, lz), dimension="3d",
            format=wgpu.TextureFormat.r8uint,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": label_tex, "mip_level": 0, "origin": (0, 0, 0)},
            label_u8,
            {"offset": 0, "bytes_per_row": lx, "rows_per_image": ly},
            (lx, ly, lz))
        if label_node_to_delete is not None:
            slicer.mrmlScene.RemoveNode(label_node_to_delete)

        # Dilated labelmap for carving. The bake's Gaussian spreads each
        # segment's alpha ~2*sigma voxels into nominally-blank (label-0)
        # territory; without dilation, the carve sphere only nulls samples
        # whose exact labelmap voxel is in the carve set, so the soft
        # halo at segment boundaries survives the cut. Grow each non-zero
        # label into adjacent zero voxels by dilate_voxels (max-filter,
        # preserving existing labels) so the carve check picks up that
        # halo too. The bake's init pass keeps reading the un-dilated
        # labelmap so segment shapes in the rendered output don't grow.
        if carve_dilate_voxels is None:
            carve_dilate_voxels = int(round(2.0 * float(sigma_voxels)))
        carve_dilate_voxels = max(int(carve_dilate_voxels), 0)
        if carve_dilate_voxels > 0:
            from scipy.ndimage import maximum_filter
            size = 2 * carve_dilate_voxels + 1
            grown = maximum_filter(label_u8, size=size)
            carve_u8 = np.ascontiguousarray(
                np.where(label_u8 == 0, grown, label_u8).astype(np.uint8))
            label_carve_tex = self.device.create_texture(
                size=(lx, ly, lz), dimension="3d",
                format=wgpu.TextureFormat.r8uint,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture(
                {"texture": label_carve_tex, "mip_level": 0,
                 "origin": (0, 0, 0)},
                carve_u8,
                {"offset": 0, "bytes_per_row": lx, "rows_per_image": ly},
                (lx, ly, lz))
        else:
            label_carve_tex = label_tex

        # Palette + RGBA field ----------------------------------------------
        palette = self._build_palette_array(segmentation_node)
        self._ensure_bake_pipelines()
        self.device.queue.write_buffer(
            self._bake_palette_ubo, 0,
            np.ascontiguousarray(palette).tobytes())

        # Surface mode allocates the rgba volume on the labelmap's own
        # grid: the JFA SDF reads t_label directly, so resampling to the
        # CT grid would just add a pass and consume more memory. Density
        # mode keeps the CT grid because the modulate pass needs the CT.
        rgba_field = RGBAVolumeField(self.device)
        if modulate_by_ct:
            rgba_field.allocate(dx, dy, dz)
            rgba_field.patient_to_texture = world_to_output_tex.astype(np.float32)
            field_corner_to_world = output_to_world
        else:
            rgba_field.allocate(lx, ly, lz)
            rgba_field.patient_to_texture = world_to_label_tex.astype(np.float32)
            field_corner_to_world = label_tex_to_world
        parent = volume_node.GetParentTransformNode()
        if parent is not None:
            pm = vtk.vtkMatrix4x4()
            parent.GetMatrixTransformToWorld(pm)
            rgba_field.world_from_local = np.array(
                [[pm.GetElement(i, j) for j in range(4)] for i in range(4)],
                dtype=np.float32)
        corners = np.array([
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1],
        ], dtype=np.float64).T
        world_corners = (field_corner_to_world @ corners)[:3].T
        wfl = rgba_field.world_from_local.astype(np.float64)
        world_corners_h = np.hstack([world_corners, np.ones((8, 1))])
        world_corners_w = (wfl @ world_corners_h.T).T[:, :3]
        rgba_field._bounds = (world_corners_w.min(axis=0).astype(np.float32),
                              world_corners_w.max(axis=0).astype(np.float32))

        spacing = ct_img.GetSpacing()
        vox = float(min(spacing))
        rgba_field.sample_step_mm = max(vox * 0.5, 0.1)
        rgba_field.opacity_unit_distance = max(vox * 5.0, 1.0)
        # Density mode: gradient stencil width for the multi-tap smoothing.
        # Surface mode: alpha-from-SDF transition band thickness in mm.
        rgba_field.gradient_h_mm = max(vox * 1.5, rgba_field.sample_step_mm)
        rgba_field.band_mm = max(vox * 1.5, rgba_field.sample_step_mm)

        # Cache on the field so rebakes don't re-upload CT + label.
        # For surface mode _ct_tex is None -- _run_bake uses its presence
        # as the switch between the 4-pass (colorize) and 3-pass (surface)
        # schedules and leaves the result in rgba_field.tex either way.
        rgba_field._ct_tex = ct_tex
        rgba_field._label_tex = label_tex
        rgba_field._label_carve_tex = label_carve_tex
        # Bound for the carve check; either the dilated copy or, when
        # carve_dilate_voxels=0, an alias of the bake labelmap.
        rgba_field._label_tex_view = label_carve_tex.create_view()
        rgba_field._output_to_world = output_to_world
        rgba_field._world_to_label_tex = world_to_label_tex
        rgba_field.sigma_voxels = float(sigma_voxels)
        rgba_field.window_min = ct_min
        rgba_field.window_max = ct_max
        rgba_field.volume_node_id = volume_node.GetID()
        rgba_field.segmentation_node_id = segmentation_node.GetID()
        rgba_field.render_mode = "density" if modulate_by_ct else "surface"

        # Run the bake
        self._run_bake(rgba_field)

        # Register + rebuild pipeline so the fragment shader picks up the
        # new RGBA binding.
        self._rgba_volumes.append(rgba_field)
        self._rebuild_pipeline()

        # Observe the segmentation display node so visibility / opacity /
        # color toggles trigger a cheap rebake.
        if dn_seg is not None:
            cb = lambda caller, event, f=rgba_field: \
                self._on_rgba_display_modified(f)
            tag = dn_seg.AddObserver(vtk.vtkCommand.ModifiedEvent, cb)
            self._rgba_obs_tags.append((dn_seg, tag, cb))

        self.rw.Render()
        return rgba_field

    # ---------------------------------------------------------------------
    # FiberStrandField (per-strand cylinder rasterization, A-buffer)
    # ---------------------------------------------------------------------

    def add_fiber_strands(self, polydata, bundle_palette,
                          tube_radius_mm=0.2,
                          pad_world_mm=None,
                          bundle_id_array_name=None,
                          parent_transform_node=None,
                          k_ambient=0.20, k_diffuse=0.65,
                          k_specular=0.20, shininess=96.0,
                          subdivisions=4):
        """Add a per-strand cylinder rasterization source. Each polyline
        segment becomes an instanced quad billboard; the fragment shader
        does an analytic ray-cylinder intersection and atomic-appends
        (depth, premultiplied rgba8) into the bridge's FragmentField for
        depth-correct compositing with the volume ray-march.

        polydata: vtkPolyData with Lines (one polyline per cell). Optionally
            a cell-data int array `bundle_id_array_name` mapping each
            polyline to a bundle id in [1, 255].
        bundle_palette: numpy (256, 4) uint8 RGBA per bundle id. Index 0
            reserved (no tube). The .a channel doubles as per-bundle
            opacity.
        tube_radius_mm: rendered tube radius in world mm.
        pad_world_mm: extra billboard padding (defaults to tube_radius).
        subdivisions: each polyline segment is replaced by N short linear
            sub-segments interpolated along a Catmull-Rom cubic Hermite
            curve through the polyline points. N=1 = original linear
            polyline. N>=4 makes the kink between adjacent sub-segments
            sub-pixel for typical viewing scales (eliminating joint
            wedge gaps) and gives a smooth tube look at zoom. Cost
            scales linearly: N x more rasterization + memory.
        """
        import vtk.util.numpy_support as vnp

        # --- Extract segments + bundle ids. ---
        pts = polydata.GetPoints()
        if pts is None or pts.GetNumberOfPoints() == 0:
            raise ValueError("polydata has no points")
        all_pts = vnp.vtk_to_numpy(pts.GetData()).astype(np.float64)
        cell_bundles = None
        if bundle_id_array_name is not None:
            arr = polydata.GetCellData().GetArray(bundle_id_array_name)
            if arr is not None:
                cell_bundles = vnp.vtk_to_numpy(arr).astype(np.int32)
        lines = polydata.GetLines()
        if lines is None or lines.GetNumberOfCells() == 0:
            raise ValueError("polydata has no Lines")
        lines.InitTraversal()
        id_list = vtk.vtkIdList()
        # Per-vertex strip data, accumulated across all strands.
        # Each polyline-point produces 2 vertices (top, bot). Adjacent
        # sub-segments share their joint vertex INDEX, so the
        # rasterization mesh is watertight by construction (no per-
        # joint billboard tiling, no edge precision artifacts).
        verts_b0: list = []            # vec3 per vertex
        verts_b1: list = []            # vec3
        verts_b2: list = []            # vec3
        verts_side: list = []          # f32: -1 or +1 (billboard side sign)
        verts_end_t: list = []         # f32: 0 = anchor at B0, 1 = anchor at B2
        verts_bid: list = []           # f32: bundle id
        verts_head: list = []          # f32: 1 if this piece is a strand head, else 0
        indices: list = []             # u32 triangle list, 6 indices per sub-segment
        # AABB tracking.
        all_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        all_max = -all_min.copy()
        cell_idx = 0
        n_sub = max(1, int(subdivisions))
        num_segments = 0

        while lines.GetNextCell(id_list):
            n_pts = id_list.GetNumberOfIds()
            bid = 1
            if cell_bundles is not None and cell_idx < len(cell_bundles):
                bid = max(1, min(255, int(cell_bundles[cell_idx])))
            cell_idx += 1
            if n_pts < 2:
                continue
            # Pull all points for this polyline, drop zero-length runs.
            pts = np.asarray(
                [all_pts[id_list.GetId(i)] for i in range(n_pts)],
                dtype=np.float64)
            keep = np.ones(len(pts), dtype=bool)
            for i in range(1, len(pts)):
                if np.linalg.norm(pts[i] - pts[i - 1]) < 1e-6:
                    keep[i] = False
            pts = pts[keep]
            if len(pts) < 2:
                continue
            # Catmull-Rom subdivision: cubic Hermite through every point
            # with centered-difference tangents. n_sub samples per
            # original polyline-segment; n_sub=1 disables subdivision.
            tans = np.zeros_like(pts)
            tans[1:-1] = 0.5 * (pts[2:] - pts[:-2])
            tans[0] = pts[1] - pts[0]
            tans[-1] = pts[-1] - pts[-2]
            ts = np.linspace(0.0, 1.0, n_sub + 1)
            tcol = ts[:, None]
            t2 = tcol * tcol
            t3 = t2 * tcol
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + tcol
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            # Densify the polyline to (len(pts)-1)*n_sub + 1 points.
            dense_pts = [pts[0]]
            for i in range(len(pts) - 1):
                sub_pts = (h00 * pts[i] + h10 * tans[i]
                           + h01 * pts[i + 1] + h11 * tans[i + 1])
                # Skip the first sample of each segment (it equals the
                # last sample of the previous segment / pts[i]).
                for k in range(1, n_sub + 1):
                    dense_pts.append(sub_pts[k])
            dense_pts = np.asarray(dense_pts, dtype=np.float64)
            # Drop sub-segments of zero length (shouldn't happen with
            # smooth Catmull-Rom, but defensive).
            keep_dense = np.ones(len(dense_pts), dtype=bool)
            for i in range(1, len(dense_pts)):
                if np.linalg.norm(
                        dense_pts[i] - dense_pts[i - 1]) < 1e-6:
                    keep_dense[i] = False
            dense_pts = dense_pts[keep_dense]
            if len(dense_pts) < 2:
                continue
            all_min = np.minimum(all_min, dense_pts.min(axis=0))
            all_max = np.maximum(all_max, dense_pts.max(axis=0))

            # One quadratic-Bezier piece per polyline vertex i. The
            # piece's control points are:
            #   B0 = midpoint(pts[i-1], pts[i])
            #   B1 = pts[i]
            #   B2 = midpoint(pts[i], pts[i+1])
            # Adjacent pieces share B0/B2 (the midpoints) and have
            # matching tangent directions there by construction
            # (Bezier tangent at B0 = 2(B1-B0) = pts[i]-pts[i-1],
            # at B2 = 2(B2-B1) = pts[i+1]-pts[i]) -- so the rendered
            # tube is C0 + C1-direction smooth across every joint
            # regardless of the kink angle.
            #
            # Strand head (i=0) and tail (i=n-1) use degenerate
            # Beziers with B0=B1 (head) or B1=B2 (tail), which the
            # ray-tube intersection naturally reduces to a sphere at
            # the strand endpoint -> hemispherical cap.
            n_dense = len(dense_pts)
            for i in range(n_dense):
                p_cur = dense_pts[i]
                if i > 0:
                    mid_prev = 0.5 * (dense_pts[i - 1] + p_cur)
                else:
                    mid_prev = p_cur          # head: degenerate
                if i + 1 < n_dense:
                    mid_next = 0.5 * (p_cur + dense_pts[i + 1])
                else:
                    mid_next = p_cur          # tail: degenerate
                B0 = mid_prev
                B1 = p_cur
                B2 = mid_next
                is_head = 1.0 if (i == 0 or i == n_dense - 1) else 0.0
                # 4 vertices per piece quad:
                # Vertex 0: side=+1 anchor@B0
                # Vertex 1: side=-1 anchor@B0
                # Vertex 2: side=+1 anchor@B2
                # Vertex 3: side=-1 anchor@B2
                base_v = len(verts_b0)
                for sign, ep in [(+1.0, 0.0), (-1.0, 0.0),
                                 (+1.0, 1.0), (-1.0, 1.0)]:
                    verts_b0.append(B0)
                    verts_b1.append(B1)
                    verts_b2.append(B2)
                    verts_side.append(sign)
                    verts_end_t.append(ep)
                    verts_bid.append(float(bid))
                    verts_head.append(is_head)
                # Triangle indices: standard quad split.
                indices.extend([base_v + 0, base_v + 1, base_v + 2])
                indices.extend([base_v + 1, base_v + 3, base_v + 2])
                num_segments += 1

        if not verts_b0:
            raise ValueError("polydata has Lines but no usable segments")

        # --- Build the FiberStrandField + GPU resources. ---
        sf = FiberStrandField(self.device)
        sf.tube_radius_mm = float(tube_radius_mm)
        if pad_world_mm is None:
            pad_world_mm = float(tube_radius_mm) * 1.0
        sf.k_ambient = float(k_ambient)
        sf.k_diffuse = float(k_diffuse)
        sf.k_specular = float(k_specular)
        sf.shininess = float(shininess)
        if parent_transform_node is not None:
            pm = vtk.vtkMatrix4x4()
            parent_transform_node.GetMatrixTransformToWorld(pm)
            sf.world_from_local = np.array(
                [[pm.GetElement(i, j) for j in range(4)] for i in range(4)],
                dtype=np.float32)

        # World AABB from accumulated min/max in local space.
        wfl = np.asarray(sf.world_from_local, dtype=np.float64)
        local_corners = np.array([
            [all_min[0], all_min[1], all_min[2]],
            [all_max[0], all_min[1], all_min[2]],
            [all_min[0], all_max[1], all_min[2]],
            [all_max[0], all_max[1], all_min[2]],
            [all_min[0], all_min[1], all_max[2]],
            [all_max[0], all_min[1], all_max[2]],
            [all_min[0], all_max[1], all_max[2]],
            [all_max[0], all_max[1], all_max[2]],
        ], dtype=np.float64)
        ch = np.hstack([local_corners, np.ones((8, 1))])
        cw = (wfl @ ch.T).T[:, :3]
        pad = float(tube_radius_mm) * 4.0
        sf._bounds = ((cw.min(axis=0) - pad).astype(np.float32),
                      (cw.max(axis=0) + pad).astype(np.float32))

        # Vertex buffer: per-vertex Bezier-piece data (52 bytes each).
        # Layout matches the pipeline's vertex_buffer_layout.
        n_verts = len(verts_b0)
        vbuf = np.zeros((n_verts, 13), dtype=np.float32)
        vbuf[:, 0:3]  = np.asarray(verts_b0, dtype=np.float32)
        vbuf[:, 3:6]  = np.asarray(verts_b1, dtype=np.float32)
        vbuf[:, 6:9]  = np.asarray(verts_b2, dtype=np.float32)
        vbuf[:, 9]    = np.asarray(verts_side,  dtype=np.float32)
        vbuf[:, 10]   = np.asarray(verts_end_t, dtype=np.float32)
        vbuf[:, 11]   = np.asarray(verts_bid,   dtype=np.float32)
        vbuf[:, 12]   = np.asarray(verts_head,  dtype=np.float32)
        sf.vertex_buffer = self.device.create_buffer(
            size=vbuf.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(
            sf.vertex_buffer, 0, np.ascontiguousarray(vbuf).tobytes())

        # Index buffer (u32).
        ibuf = np.asarray(indices, dtype=np.uint32)
        sf.index_buffer = self.device.create_buffer(
            size=ibuf.nbytes,
            usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(
            sf.index_buffer, 0, np.ascontiguousarray(ibuf).tobytes())
        sf.num_indices = int(ibuf.size)
        sf.num_segments = int(num_segments)

        # Palette.
        sf.write_palette(np.asarray(bundle_palette, dtype=np.uint8))

        # Per-field StrandParams UBO (96 bytes: mat4 + 4 floats + 4 padding floats + vec4).
        sf.params_ubo = self.device.create_buffer(
            size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._update_strand_params_ubo(sf, pad_world_mm)

        # Compile pipelines + create bind group. Bind group references
        # FragmentField buffers, so the FragmentField must exist first.
        self._ensure_strand_pipelines()
        if self._fragment_field is None:
            self._fragment_field = FragmentField(self.device)
        # Allocate fragment buffers at a sensible default size; will be
        # reallocated to match the actual viewport on first render.
        try:
            w, h = self.rw.GetSize()
            if w > 0 and h > 0:
                self._fragment_field.ensure_size(w, h)
        except Exception:
            pass
        if self._fragment_field.fragments_buffer is None:
            # Pre-allocate at 1080p so the bind group can resolve. Will
            # reallocate at first render if the actual viewport differs.
            self._fragment_field.ensure_size(1920, 1080)

        sf.bind_group = self._make_strand_bind_group(sf)
        self._fiber_strand_fields.append(sf)
        self._rebuild_pipeline()
        self.rw.Render()
        return sf

    def _update_strand_params_ubo(self, sf, pad_world_mm):
        import struct as _st
        buf = bytearray(128)
        wfl = np.asarray(sf.world_from_local,
                         dtype=np.float32).T.ravel()
        for i in range(16):
            _st.pack_into("<f", buf, i * 4, float(wfl[i]))
        _st.pack_into("<f", buf, 64, float(sf.tube_radius_mm))
        _st.pack_into("<f", buf, 68, float(pad_world_mm))
        _st.pack_into("<I", buf, 72, FragmentField.K)
        _st.pack_into("<I", buf, 76, 0)  # _pad0
        _st.pack_into("<f", buf, 80, float(sf.k_ambient))
        _st.pack_into("<f", buf, 84, float(sf.k_diffuse))
        _st.pack_into("<f", buf, 88, float(sf.k_specular))
        _st.pack_into("<f", buf, 92, max(float(sf.shininess), 1.0))
        # 96..128 padding (struct rounded to 16-byte alignment by WGSL).
        self.device.queue.write_buffer(sf.params_ubo, 0, bytes(buf))

    def _make_strand_bind_group(self, sf):
        return self.device.create_bind_group(
            layout=self._strand_pipeline_bgl, entries=[
                {"binding": 0, "resource": {
                    "buffer": self._cam_ubo, "offset": 0,
                    "size": self._cam_ubo.size}},
                {"binding": 1, "resource": {
                    "buffer": sf.params_ubo, "offset": 0,
                    "size": sf.params_ubo.size}},
                {"binding": 3, "resource": sf.palette_view},
                {"binding": 4, "resource": {
                    "buffer": self._fragment_field.counts_buffer,
                    "offset": 0,
                    "size": self._fragment_field.counts_buffer.size}},
                {"binding": 5, "resource": {
                    "buffer": self._fragment_field.fragments_buffer,
                    "offset": 0,
                    "size": self._fragment_field.fragments_buffer.size}},
            ])

    def _on_rgba_display_modified(self, rgba_field):
        """Segmentation display changed (visibility / color / opacity).
        Rebuild the palette UBO and re-run the bake passes using cached
        CT + labelmap textures. No CPU export, no texture re-upload.
        """
        if self._disposed:
            return
        try:
            seg = slicer.mrmlScene.GetNodeByID(
                rgba_field.segmentation_node_id)
            if seg is None:
                return
            palette = self._build_palette_array(seg)
            self._ensure_bake_pipelines()
            self.device.queue.write_buffer(
                self._bake_palette_ubo, 0,
                np.ascontiguousarray(palette).tobytes())
            self._run_bake(rgba_field)
            self.rw.Render()
        except Exception as e:
            print(f"_on_rgba_display_modified: {e}")

    def _smooth_segment(self, segment):
        """Run a separable 3D Gaussian on the segment's raw presence texture,
        writing the result into segment.smooth_tex. Ping-pongs through
        segment.scratch_tex for the Y pass. Final result is in smooth_tex.

        Three compute passes with read-after-write synchronization between
        them (each in its own begin_compute_pass / end)."""
        if segment.raw_tex is None or segment.smooth_tex is None:
            return
        self._ensure_smooth_compute()
        import math
        sigma = max(float(segment.sigma_voxels), 0.25)
        radius = max(int(math.ceil(3.0 * sigma)), 1)

        # Pack params: sigma (f32) + axis (u32) + radius (u32) + pad
        for axis in range(3):
            buf = np.zeros(16, dtype=np.uint8)
            # sigma as f32 bytes at offset 0
            buf[0:4] = np.frombuffer(
                np.float32(sigma).tobytes(), dtype=np.uint8)
            buf[4:8] = np.frombuffer(
                np.uint32(axis).tobytes(), dtype=np.uint8)
            buf[8:12] = np.frombuffer(
                np.uint32(radius).tobytes(), dtype=np.uint8)
            # last 4 bytes stay zero
            self.device.queue.write_buffer(
                self._smooth_ubos[axis], 0, buf.tobytes())

        raw_view = segment.raw_tex.create_view()
        smooth_view = segment.smooth_tex.create_view()
        scratch_view = segment.scratch_tex.create_view()

        # X: raw -> smooth     Y: smooth -> scratch     Z: scratch -> smooth
        passes = [
            (raw_view,     smooth_view,  self._smooth_ubos[0]),
            (smooth_view,  scratch_view, self._smooth_ubos[1]),
            (scratch_view, smooth_view,  self._smooth_ubos[2]),
        ]

        dx, dy, dz = segment.dims
        wg_x, wg_y, wg_z = 8, 8, 4
        groups = (
            (dx + wg_x - 1) // wg_x,
            (dy + wg_y - 1) // wg_y,
            (dz + wg_z - 1) // wg_z,
        )

        encoder = self.device.create_command_encoder()
        for src, dst, ubo in passes:
            bg = self.device.create_bind_group(
                layout=self._smooth_compute_bgl,
                entries=[
                    {"binding": 0, "resource": src},
                    {"binding": 1, "resource": dst},
                    {"binding": 2, "resource": {
                        "buffer": ubo, "offset": 0, "size": 16}},
                ])
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self._smooth_compute_pipeline)
            cpass.set_bind_group(0, bg, [], 0, 0)
            cpass.dispatch_workgroups(*groups)
            cpass.end()
        self.device.queue.submit([encoder.finish()])

    def _ensure_target(self, w, h):
        if self._size == (w, h) and self._color_tex is not None:
            return
        # _color_tex is the main pipeline's render target; we also bind it
        # as a sampled texture in the TAA composite pass, so it needs
        # TEXTURE_BINDING and COPY_DST (to receive history afterward).
        self._color_tex = self.device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC
                   | wgpu.TextureUsage.COPY_DST),
        )
        # TAA history accumulator (sampled read-only). Written each frame
        # via copy_texture_to_texture from _taa_output_tex so we never
        # rely on the optional read_write storage access mode.
        self._taa_history_tex = self.device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_DST),
        )
        self._taa_history_view = self._taa_history_tex.create_view()
        # TAA compute output (write-only storage). One copy out to the
        # history texture (for next frame's input) and one copy out to
        # _color_tex (for the VTK readback path).
        self._taa_output_tex = self.device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(wgpu.TextureUsage.STORAGE_BINDING
                   | wgpu.TextureUsage.COPY_SRC),
        )
        self._taa_output_view = self._taa_output_tex.create_view()
        self._taa_frame = 0          # reset accumulator on resize
        bpr = w * 4
        self._aligned_bpr = (bpr + 255) & ~255
        self._readback_buf = self.device.create_buffer(
            size=self._aligned_bpr * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        self._size = (w, h)

    def _wgpu_render(self, w, h, proj_inv, view_inv,
                     proj_fwd=None, view_fwd=None):
        if self._pipeline is None or (not self._fields and not self._segments
                                      and not self._rgba_volumes
                                      and not self._fiber_strand_fields):
            return np.zeros((h, w, 4), dtype=np.uint8)
        self._ensure_target(w, h)

        # Cam UBO layout (288 bytes):
        #   proj_inv (mat4 col-major)  bytes   0..64
        #   view_inv (mat4)            bytes  64..128
        #   size      (vec4 w,h,jx,jy) bytes 128..144  (jx/jy = TAA NDC jitter)
        #   proj      (mat4)           bytes 144..208
        #   view      (mat4)           bytes 208..272
        #   taa       (vec4 frame,_,_,_) bytes 272..288
        cbuf = np.zeros(72, dtype=np.float32)
        cbuf[0:16] = proj_inv.T.ravel()
        cbuf[16:32] = view_inv.T.ravel()
        cbuf[32] = float(w); cbuf[33] = float(h)
        # TAA jitter: Halton(2, 3) sequence scaled so the offset is one
        # pixel in NDC (= 2/viewport). Cycled every 8 frames. If the
        # camera changed since the last render, reset the index so the
        # history buffer (which is keyed on the jitter pattern) starts
        # fresh and we don't ghost across a camera move.
        pv = (proj_fwd @ view_fwd) if (proj_fwd is not None
                                       and view_fwd is not None) else None
        if (self._taa_prev_pv is None or pv is None
                or not np.allclose(pv, self._taa_prev_pv, atol=1e-4)):
            self._taa_frame = 0
        else:
            self._taa_frame = (self._taa_frame + 1) & 0x7FFFFFFF
        self._taa_prev_pv = pv
        taa_frame = self._taa_frame
        jx_n, jy_n = _halton_2_3(taa_frame % 8)
        cbuf[34] = (jx_n - 0.5) * 2.0 / float(w)
        cbuf[35] = (jy_n - 0.5) * 2.0 / float(h)
        if proj_fwd is not None:
            cbuf[36:52] = np.asarray(proj_fwd, dtype=np.float32).T.ravel()
        else:
            cbuf[36:52] = np.eye(4, dtype=np.float32).T.ravel()
        if view_fwd is not None:
            cbuf[52:68] = np.asarray(view_fwd, dtype=np.float32).T.ravel()
        else:
            cbuf[52:68] = np.eye(4, dtype=np.float32).T.ravel()
        cbuf[68] = float(taa_frame)   # taa.x
        self.device.queue.write_buffer(self._cam_ubo, 0, cbuf.tobytes())

        visible_fields = [f for f in self._fields if f.visible]
        visible_segs = [s for s in self._segments if s.visible]
        visible_rgba = [r for r in self._rgba_volumes if r.visible]
        visible_strands = [t for t in self._fiber_strand_fields if t.visible]
        # Strand fields contribute to scene bounds (so the volume AABB
        # ray-march reaches them), but only volumetric fields contribute
        # ray-march sample step.
        vol_src = (visible_fields or self._fields) \
                  + (visible_segs or self._segments) \
                  + (visible_rgba or self._rgba_volumes)
        all_src = vol_src \
                  + (visible_strands or self._fiber_strand_fields)
        boxes = [r.aabb() for r in all_src]
        boxes = [b for b in boxes if b is not None]
        if boxes:
            bmin = np.min(np.stack([b[0] for b in boxes]), axis=0)
            bmax = np.max(np.stack([b[1] for b in boxes]), axis=0)
            pad = float((bmax - bmin).max() * 0.01)
            bmin = (bmin - pad).astype(np.float32)
            bmax = (bmax + pad).astype(np.float32)
        else:
            bmin = np.array([-100, -100, -100], dtype=np.float32)
            bmax = np.array([100, 100, 100], dtype=np.float32)
        if vol_src:
            step = float(min(r.sample_step_mm for r in vol_src))
        else:
            # Strand-only scene: pick a step that gives reasonable depth
            # resolution along the AABB. ~256 samples across the diagonal.
            diag = float(np.linalg.norm(bmax - bmin))
            step = max(diag / 256.0, 0.1)
        self.device.queue.write_buffer(
            self._mat_ubo, 0, _pack_material(
                self._fields, self._segments, self._rgba_volumes,
                bmin, bmax, step,
                grid_p2t=self._grid_p2t, grid_gain=self._grid_gain,
                clip_planes=self._clip_planes))

        enc = self.device.create_command_encoder()

        # Strand rasterization → A-buffer pass. Only when at least one
        # FiberStrandField is present and visible. Uses a separate render
        # pipeline whose fragment shader atomic-appends into the
        # FragmentField's per-pixel sorted list.
        if self._fiber_strand_fields and self._fragment_field is not None:
            old_vp = self._fragment_field.viewport
            self._fragment_field.ensure_size(w, h)
            if self._fragment_field.viewport != old_vp:
                # Buffers reallocated -- strand bind groups + main bind
                # group still hold references to the old buffers; rebuild.
                for sf in self._fiber_strand_fields:
                    sf.bind_group = self._make_strand_bind_group(sf)
                self._rebuild_bind_group()
            # 1. Clear counts buffer.
            num_pixels = w * h
            cgroups = (num_pixels + 63) // 64
            cbg = self.device.create_bind_group(
                layout=self._fragment_clear_bgl, entries=[
                    {"binding": 0, "resource": {
                        "buffer": self._fragment_field.counts_buffer,
                        "offset": 0,
                        "size": self._fragment_field.counts_buffer.size}},
                ])
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self._fragment_clear_pipeline)
            cp.set_bind_group(0, cbg, [], 0, 0)
            cp.dispatch_workgroups(cgroups, 1, 1)
            cp.end()
            # 2. Rasterize each strand field.
            # The strand pipeline writes only to the A-buffer; its color
            # attachment is masked off (write_mask=0). Use load_op=clear so
            # the color_tex is in a defined state even when this is the
            # first pass of the frame; the main ray-march clears again.
            stub_color = self._color_tex.create_view()
            srp = enc.begin_render_pass(color_attachments=[{
                "view": stub_color,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0, 0, 0, 0),
            }])
            srp.set_pipeline(self._strand_pipeline)
            for sf in (visible_strands or []):
                if (sf.bind_group is None or sf.vertex_buffer is None
                        or sf.num_indices == 0):
                    continue
                srp.set_bind_group(0, sf.bind_group, [], 0, 0)
                srp.set_vertex_buffer(0, sf.vertex_buffer)
                srp.set_index_buffer(sf.index_buffer,
                                     wgpu.IndexFormat.uint32)
                srp.draw_indexed(sf.num_indices)
            srp.end()
            # 3. Per-pixel sort.
            import struct as _st
            sbuf = bytearray(16)
            _st.pack_into("<I", sbuf, 0, w)
            _st.pack_into("<I", sbuf, 4, h)
            _st.pack_into("<I", sbuf, 8, FragmentField.K)
            self.device.queue.write_buffer(
                self._fragment_sort_ubo, 0, bytes(sbuf))
            sortbg = self.device.create_bind_group(
                layout=self._fragment_sort_bgl, entries=[
                    {"binding": 0, "resource": {
                        "buffer": self._fragment_field.counts_buffer,
                        "offset": 0,
                        "size": self._fragment_field.counts_buffer.size}},
                    {"binding": 1, "resource": {
                        "buffer": self._fragment_field.fragments_buffer,
                        "offset": 0,
                        "size": self._fragment_field.fragments_buffer.size}},
                    {"binding": 2, "resource": {
                        "buffer": self._fragment_sort_ubo,
                        "offset": 0, "size": 16}},
                ])
            spp = enc.begin_compute_pass()
            spp.set_pipeline(self._fragment_sort_pipeline)
            spp.set_bind_group(0, sortbg, [], 0, 0)
            spp.dispatch_workgroups(cgroups, 1, 1)
            spp.end()

        # Main ray-march (volume + interleaved A-buffer fragments).
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self._color_tex.create_view(),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 0),
        }])
        rp.set_pipeline(self._pipeline)
        rp.set_bind_group(0, self._bind_group, [], 0, 0)
        rp.draw(3)
        rp.end()

        # TAA composite: blend the just-rendered _color_tex with the
        # previous frame's history into _taa_output_tex, then copy that
        # output into both _taa_history_tex (for next frame's input) and
        # _color_tex (for the existing readback path). On a camera/scene
        # reset (_taa_frame == 0) the blend factor is 1.0 which discards
        # stale history -- the still-uninitialized history texture is
        # then overwritten by the first frame's output via the copy.
        import struct as _st2
        taa_buf = bytearray(16)
        mixf = 1.0 if self._taa_frame == 0 else 0.10
        _st2.pack_into("<f", taa_buf, 0, mixf)
        _st2.pack_into("<f", taa_buf, 4, float(w))
        _st2.pack_into("<f", taa_buf, 8, float(h))
        self.device.queue.write_buffer(self._taa_ubo, 0, bytes(taa_buf))
        taa_bg = self.device.create_bind_group(
            layout=self._taa_bgl, entries=[
                {"binding": 0, "resource": self._color_tex.create_view()},
                {"binding": 1, "resource": self._taa_history_view},
                {"binding": 2, "resource": self._taa_output_view},
                {"binding": 3, "resource": {
                    "buffer": self._taa_ubo, "offset": 0, "size": 16}},
            ])
        taa_pass = enc.begin_compute_pass()
        taa_pass.set_pipeline(self._taa_pipeline)
        taa_pass.set_bind_group(0, taa_bg, [], 0, 0)
        taa_pass.dispatch_workgroups((w + 7) // 8, (h + 7) // 8, 1)
        taa_pass.end()
        # Update history for the next frame.
        enc.copy_texture_to_texture(
            {"texture": self._taa_output_tex,
             "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._taa_history_tex,
             "mip_level": 0, "origin": (0, 0, 0)},
            (w, h, 1))
        # And expose the composited result to the readback path.
        enc.copy_texture_to_texture(
            {"texture": self._taa_output_tex,
             "mip_level": 0, "origin": (0, 0, 0)},
            {"texture": self._color_tex,
             "mip_level": 0, "origin": (0, 0, 0)},
            (w, h, 1))

        enc.copy_texture_to_buffer(
            {"texture": self._color_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": self._readback_buf, "offset": 0,
             "bytes_per_row": self._aligned_bpr, "rows_per_image": h},
            (w, h, 1))
        self.device.queue.submit([enc.finish()])
        raw = np.frombuffer(self.device.queue.read_buffer(self._readback_buf),
                            dtype=np.uint8)
        raw = raw.reshape(h, self._aligned_bpr)[:, :w*4].reshape(h, w, 4)
        return raw.copy()

    def _on_end_event(self, caller):
        if self._disposed:
            return
        try:
            w, h = self.rw.GetSize()
            if w <= 0 or h <= 0 or (not self._fields
                                    and not self._segments
                                    and not self._rgba_volumes
                                    and not self._fiber_strand_fields):
                return
            cam = caller.GetActiveCamera()
            aspect = w / h
            pm = cam.GetProjectionTransformMatrix(aspect, -1.0, 1.0)
            P = np.array([[pm.GetElement(i, j) for j in range(4)] for i in range(4)],
                         dtype=np.float64)
            P[2, :] = 0.5 * (P[2, :] + P[3, :])
            vm = cam.GetViewTransformMatrix()
            V = np.array([[vm.GetElement(i, j) for j in range(4)] for i in range(4)],
                         dtype=np.float64)
            proj_inv = np.linalg.inv(P).astype(np.float32)
            view_inv = np.linalg.inv(V).astype(np.float32)
            proj_fwd = P.astype(np.float32)
            view_fwd = V.astype(np.float32)

            self.rw.MakeCurrent()
            rfb = self.rw.GetRenderFramebuffer()
            ct = rfb.GetColorAttachmentAsTextureObject(0)
            if ct.GetWidth() != w or ct.GetHeight() != h:
                return

            # Read VTK's depth attachment + upload to wgpu so the ray-march
            # can clip t_far at geometry that VTK already drew (fiducials,
            # ROI handles, etc). This depth is sampled per-pixel inside
            # the fragment shader.
            old_size = self._vtk_depth_size
            self._ensure_vtk_depth_tex(w, h)
            if old_size != self._vtk_depth_size:
                self._rebuild_bind_group()
            self._upload_vtk_depth(w, h)

            rgba = self._wgpu_render(w, h, proj_inv, view_inv,
                                     proj_fwd, view_fwd)
            rgba_gl = np.ascontiguousarray(rgba[::-1])

            # GL-side composite: hardware premultiplied-alpha blend of our
            # wgpu output over VTK's color attachment. No VTK color readback.
            self._gl_compositor.composite(rgba_gl, rfb.GetFBOIndex(), w, h)
        except Exception as e:
            print(f"WgpuVolumeBridge._on_end_event error: {e}")
            import traceback; traceback.print_exc()


def install_default_bridge():
    lm = slicer.app.layoutManager()
    tw = lm.threeDWidget(0)
    rw = tw.threeDView().renderWindow()
    renderer = rw.GetRenderers().GetFirstRenderer()
    bridge = WgpuVolumeBridge(renderer, rw)
    bridge.install()
    return bridge
