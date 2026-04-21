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
    size: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u_cam: Camera;

fn ndc_to_world(ndc: vec4<f32>) -> vec3<f32> {
    let clip = u_cam.proj_inv * ndc;
    let eye = clip.xyz / clip.w;
    let world = u_cam.view_inv * vec4<f32>(eye, 1.0);
    return world.xyz / world.w;
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
            f"    rgba{q}_step_unit: vec4<f32>,  // (step, unit, visible, _)",
            f"    rgba{q}_shade: vec4<f32>,     // (ka, kd, ks, shin)",
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


def _rgba_field_wgsl(slot, n, m):
    """A pre-baked RGBA 3D volume. RGB is emissive color, A is opacity
    (already Gaussian-smoothed on the GPU during bake). No TF lookup,
    no per-segment pickup at render time. Gradient for Phong is taken
    from the alpha channel since that's where the iso-surface is.

    Bindings follow the existing (per-image, vtk-depth, grid, per-segment)
    tail: 2 per rgba slot starting at 2 + 4n + 4 + 2m.
    """
    q = slot
    b0 = 2 + n * 4 + 4 + m * 2 + slot * 2
    return f"""
@group(0) @binding({b0+0}) var s_rgba{q}: sampler;
@group(0) @binding({b0+1}) var t_rgba{q}: texture_3d<f32>;

fn sample_rgba_v_{q}(wp: vec3<f32>) -> vec4<f32> {{
    let wpw = warp(wp);
    let t4 = u_mat.rgba{q}_p2t * vec4<f32>(wpw, 1.0);
    let t = t4.xyz;
    if (any(t < vec3<f32>(0.0)) || any(t > vec3<f32>(1.0))) {{
        return vec4<f32>(0.0);
    }}
    return textureSampleLevel(t_rgba{q}, s_rgba{q}, t, 0.0);
}}

fn sample_rgba_{q}(wp: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {{
    if (u_mat.rgba{q}_step_unit.z < 0.5) {{ return vec4<f32>(0.0); }}
    let v4 = sample_rgba_v_{q}(wp);
    let alpha = v4.a;
    if (alpha <= 1e-3) {{ return vec4<f32>(0.0); }}

    // Central differences on alpha for surface normal.
    let h = max(u_mat.scene_step, 1e-3);
    let gx = sample_rgba_v_{q}(wp+vec3<f32>(h,0,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(h,0,0)).a;
    let gy = sample_rgba_v_{q}(wp+vec3<f32>(0,h,0)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,h,0)).a;
    let gz = sample_rgba_v_{q}(wp+vec3<f32>(0,0,h)).a
           - sample_rgba_v_{q}(wp-vec3<f32>(0,0,h)).a;
    let grad = vec3<f32>(gx, gy, gz) / (2.0 * h);
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
    let unit = max(u_mat.rgba{q}_step_unit.y, 1e-3);
    let op = clamp(alpha * (step / unit), 0.0, 1.0);
    return vec4<f32>(lit * op, op);
}}
"""


def _main_wgsl(n, m, k):
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
    return f"""
@fragment
fn fs_main(v: Varyings) -> FragmentOutput {{
    var out: FragmentOutput;
    let ndc_x = (v.position.x / u_cam.size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (v.position.y / u_cam.size.y) * 2.0;
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

    if (t_far <= t_near) {{ out.color = vec4<f32>(0.0); return out; }}
    let step = max(u_mat.scene_step, 1e-3);
    var t = t_near + fract(sin(dot(v.position.xy, vec2<f32>(12.9898,78.233))) * 43758.5453) * step;
    var integrated = vec4<f32>(0.0);
    var safety: i32 = 0;
    loop {{
        if (t >= t_far) {{ break; }}
        if (safety >= 2048) {{ break; }}
        if (integrated.a >= 0.99) {{ break; }}
        let wp = ro + rd * t;
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
    out.color = integrated;
    return out;
}}
"""


def _build_wgsl(n, m, k):
    parts = [_HEADER, _mat_struct_wgsl(n, m, k)]
    # Grid transform needs to come BEFORE the per-field functions (they call
    # warp()); its bindings sit after the VTK depth bindings.
    parts.append(_vtk_depth_wgsl(n))
    parts.append(_grid_transform_wgsl(n))
    for i in range(n):
        parts.append(_field_wgsl(i))
    for j in range(m):
        parts.append(_seg_field_wgsl(j, n))
    for q in range(k):
        parts.append(_rgba_field_wgsl(q, n, m))
    parts.append(_main_wgsl(n, m, k))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Material UBO packing
# ---------------------------------------------------------------------------

# 64 bytes: vec4 bmin(16) + vec4 bmax(16) + f32 step + vec3 _pad0, rounded
# up to next 16-byte boundary (WGSL aligns the following mat4x4 to 16).
_SCENE_BYTES = 64
_PER_FIELD_BYTES = 112   # mat4(64) + 3 * vec4(48) per image field
_PER_SEG_BYTES = 112     # mat4(64) + 3 * vec4(48) per segment
_PER_RGBA_BYTES = 96     # mat4(64) + 2 * vec4(32) per rgba volume
# Grid transform tail: mat4x4 (64) + vec4 enabled (16) = 80 bytes
_GRID_TAIL_BYTES = 80


def _mat_ubo_size(n, m, k):
    total = (_SCENE_BYTES
             + n * _PER_FIELD_BYTES
             + m * _PER_SEG_BYTES
             + k * _PER_RGBA_BYTES
             + _GRID_TAIL_BYTES)
    return (total + 15) & ~15


def _pack_material(fields, segments, rgba_volumes, bmin, bmax, step,
                   grid_p2t=None, grid_gain=1.0):
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
    for i, f in enumerate(fields):
        off = (_SCENE_BYTES // 4) + i * (_PER_FIELD_BYTES // 4)
        p2t = _p2t_for_field(f)
        arr[off:off+16] = p2t.T.ravel()
        arr[off+16:off+18] = f.clim
        arr[off+20:off+24] = [f.k_ambient, f.k_diffuse, f.k_specular,
                              max(f.shininess, 1.0)]
        # Force visible=1 since the bridge hides the VRDN to silence
        # Slicer's native VR mapper; we always want to render our fields.
        arr[off+24:off+28] = [f.sample_step_mm, f.opacity_unit_distance,
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
        arr[off+16:off+20] = [r.sample_step_mm, r.opacity_unit_distance,
                              1.0 if r.visible else 0.0, 0.0]
        arr[off+20:off+24] = [r.k_ambient, r.k_diffuse, r.k_specular,
                              max(r.shininess, 1.0)]
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
        self._bounds = (np.array([-100, -100, -100], dtype=np.float32),
                        np.array([100, 100, 100], dtype=np.float32))

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
# Bridge class
# ---------------------------------------------------------------------------

class WgpuVolumeBridge:
    """VTK-injection bridge. Wires VolumeRenderingDisplayer to a raw-wgpu
    ray-march pipeline rendered into Slicer's native 3D view via an
    EndEvent hook + glTexSubImage2D blit."""

    def __init__(self, vtk_renderer, vtk_render_window):
        self.vtk_renderer = vtk_renderer
        self.rw = vtk_render_window
        self.device = get_shared().device

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
        self._cam_ubo = self.device.create_buffer(
            size=144,
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

        # Pre-baked RGBA 3D volumes (ColorizeVolume-style). Each entry is an
        # RGBAVolumeField with a rgba16float 3D texture produced by a GPU
        # bake (see add_colorize_volume). Rendered directly: RGB = emissive
        # color, A = opacity. No TF, no per-segment work at draw time.
        self._rgba_volumes: list = []
        # (object, observer_tag, callback) triples for segmentation display
        # nodes driving RGBA volumes. On display change we rebuild the
        # palette UBO and rerun the bake using cached textures.
        self._rgba_obs_tags: list = []

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
        if n == 0 and m == 0 and k == 0:
            self._pipeline = None
            self._bind_group = None
            return

        for f in self._fields:
            for tex in (f._volume_tex, f._lut_tex, f._grad_lut_tex):
                if tex is None:
                    continue
                if tex._wgpu_object is None:
                    tex._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
                ensure_wgpu_object(tex)
                update_resource(tex)

        wgsl = _build_wgsl(n, m, k)
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
        # Per-RGBA-volume bindings (sampler + rgba16float 3D texture),
        # starting right after per-segment bindings.
        br0 = bs0 + m * 2
        for q in range(k):
            br = br0 + q * 2
            entries += [
                {"binding": br+0, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "sampler": {"type": wgpu.SamplerBindingType.filtering}},
                {"binding": br+1, "visibility": wgpu.ShaderStage.FRAGMENT,
                 "texture": {"sample_type": wgpu.TextureSampleType.float,
                             "view_dimension": wgpu.TextureViewDimension.d3,
                             "multisampled": False}},
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
                                 and not self._rgba_volumes):
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
            br = br0 + q * 2
            bg += [
                {"binding": br+0, "resource": r.sampler},
                {"binding": br+1, "resource": r.tex_view},
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

    def _run_bake(self, rgba_field):
        """Dispatch the full ColorizeVolume-style bake into rgba_field.tex.
        Assumes rgba_field holds cached `_ct_tex`, `_label_tex`,
        `_output_to_world` (4x4), `_world_to_label_tex` (4x4), `sigma_voxels`,
        `window_min`, `window_max`, and that `_bake_palette_ubo` is already
        up to date with the current palette.

        Passes: resample-init (label->RGBA), smooth X/Y/Z on alpha,
        modulate alpha by CT."""
        self._ensure_bake_pipelines()
        import math, struct as _st
        dx, dy, dz = rgba_field.dims

        # Resample params UBO: output_to_world then world_to_label_tex,
        # both column-major so the transpose matches WGSL.
        rparams = np.zeros(32, dtype=np.float32)
        rparams[0:16] = np.asarray(
            rgba_field._output_to_world, dtype=np.float32).T.ravel()
        rparams[16:32] = np.asarray(
            rgba_field._world_to_label_tex, dtype=np.float32).T.ravel()
        self.device.queue.write_buffer(
            self._bake_resample_ubo, 0, rparams.tobytes())

        # Smooth params (one UBO per axis)
        sigma = max(float(rgba_field.sigma_voxels), 0.25)
        radius = max(int(math.ceil(3.0 * sigma)), 1)
        for axis in range(3):
            sbuf = bytearray(16)
            _st.pack_into("<f", sbuf, 0, sigma)
            _st.pack_into("<I", sbuf, 4, axis)
            _st.pack_into("<I", sbuf, 8, radius)
            self.device.queue.write_buffer(
                self._bake_smooth_ubos[axis], 0, bytes(sbuf))

        # Modulate params
        mbuf = bytearray(16)
        _st.pack_into("<f", mbuf, 0, float(rgba_field.window_min))
        _st.pack_into("<f", mbuf, 4,
                      float(rgba_field.window_max - rgba_field.window_min))
        self.device.queue.write_buffer(self._bake_mod_ubo, 0, bytes(mbuf))

        wg_x, wg_y, wg_z = 8, 8, 4
        groups = ((dx + wg_x - 1) // wg_x,
                  (dy + wg_y - 1) // wg_y,
                  (dz + wg_z - 1) // wg_z)

        encoder = self.device.create_command_encoder()

        # Pass 1: resample labelmap + palette, writes rgba_field.tex
        bg = self.device.create_bind_group(
            layout=self._bake_init_bgl, entries=[
                {"binding": 0, "resource": rgba_field._label_tex.create_view()},
                {"binding": 1, "resource": rgba_field.tex.create_view()},
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

        # Smooth X: tex -> scratch    Y: scratch -> tex    Z: tex -> scratch
        smooth_steps = [
            (rgba_field.tex,         rgba_field.scratch_tex, 0),
            (rgba_field.scratch_tex, rgba_field.tex,         1),
            (rgba_field.tex,         rgba_field.scratch_tex, 2),
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

        # Modulate: scratch -> tex (final result lives in rgba_field.tex)
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

    def add_colorize_volume(self, volume_node, segmentation_node,
                            sigma_voxels=1.5, window_level=None):
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
        # volume on the way in.
        ct_tex = self.device.create_texture(
            size=(dx, dy, dz), dimension="3d",
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": ct_tex, "mip_level": 0, "origin": (0, 0, 0)},
            ct_f32,
            {"offset": 0, "bytes_per_row": dx * 4, "rows_per_image": dy},
            (dx, dy, dz))
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

        # Palette + RGBA field ----------------------------------------------
        palette = self._build_palette_array(segmentation_node)
        self._ensure_bake_pipelines()
        self.device.queue.write_buffer(
            self._bake_palette_ubo, 0,
            np.ascontiguousarray(palette).tobytes())

        rgba_field = RGBAVolumeField(self.device)
        rgba_field.allocate(dx, dy, dz)
        rgba_field.patient_to_texture = world_to_output_tex.astype(np.float32)
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
        world_corners = (output_to_world @ corners)[:3].T
        wfl = rgba_field.world_from_local.astype(np.float64)
        world_corners_h = np.hstack([world_corners, np.ones((8, 1))])
        world_corners_w = (wfl @ world_corners_h.T).T[:, :3]
        rgba_field._bounds = (world_corners_w.min(axis=0).astype(np.float32),
                              world_corners_w.max(axis=0).astype(np.float32))

        spacing = ct_img.GetSpacing()
        vox = float(min(spacing))
        rgba_field.sample_step_mm = max(vox * 0.5, 0.1)
        rgba_field.opacity_unit_distance = max(vox * 5.0, 1.0)

        # Cache on the field so rebakes don't re-upload CT + label.
        rgba_field._ct_tex = ct_tex
        rgba_field._label_tex = label_tex
        rgba_field._output_to_world = output_to_world
        rgba_field._world_to_label_tex = world_to_label_tex
        rgba_field.sigma_voxels = float(sigma_voxels)
        rgba_field.window_min = ct_min
        rgba_field.window_max = ct_max
        rgba_field.volume_node_id = volume_node.GetID()
        rgba_field.segmentation_node_id = segmentation_node.GetID()

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
        self._color_tex = self.device.create_texture(
            size=(w, h, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        bpr = w * 4
        self._aligned_bpr = (bpr + 255) & ~255
        self._readback_buf = self.device.create_buffer(
            size=self._aligned_bpr * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        self._size = (w, h)

    def _wgpu_render(self, w, h, proj_inv, view_inv):
        if self._pipeline is None or (not self._fields and not self._segments
                                      and not self._rgba_volumes):
            return np.zeros((h, w, 4), dtype=np.uint8)
        self._ensure_target(w, h)

        cbuf = np.zeros(36, dtype=np.float32)
        cbuf[0:16] = proj_inv.T.ravel()
        cbuf[16:32] = view_inv.T.ravel()
        cbuf[32] = float(w); cbuf[33] = float(h)
        self.device.queue.write_buffer(self._cam_ubo, 0, cbuf.tobytes())

        visible_fields = [f for f in self._fields if f.visible]
        visible_segs = [s for s in self._segments if s.visible]
        visible_rgba = [r for r in self._rgba_volumes if r.visible]
        src = (visible_fields or self._fields) \
              + (visible_segs or self._segments) \
              + (visible_rgba or self._rgba_volumes)
        boxes = [r.aabb() for r in src]
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
        step = float(min(r.sample_step_mm for r in src))
        self.device.queue.write_buffer(
            self._mat_ubo, 0, _pack_material(
                self._fields, self._segments, self._rgba_volumes,
                bmin, bmax, step,
                grid_p2t=self._grid_p2t, grid_gain=self._grid_gain))

        enc = self.device.create_command_encoder()
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
                                    and not self._rgba_volumes):
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

            rgba = self._wgpu_render(w, h, proj_inv, view_inv)
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
