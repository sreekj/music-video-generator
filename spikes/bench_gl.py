"""Benchmark headless GL throughput for the Stage B renderer.

Stage B has to sustain well above realtime for the live-preview UX to work.
This measures both raw shading and the pixel readback that feeds ffmpeg, since
readback is usually the actual bottleneck in an offscreen pipeline.

    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GALLIUM_DRIVER=d3d12 \
        python spikes/bench_gl.py
"""

from __future__ import annotations

import time

import moderngl
import numpy as np

VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Deliberately representative of a real shot: parallax sample, bloom-ish blur,
# chromatic aberration, vignette, grain.
FRAG = """
#version 330
uniform sampler2D tex;
uniform float t;
uniform float energy;
in vec2 uv;
out vec4 frag;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
    vec2 c = uv - 0.5;
    float zoom = 1.0 - 0.08 * energy;
    vec2 suv = c * zoom + 0.5 + vec2(sin(t * 0.3), cos(t * 0.21)) * 0.01;

    float ca = 0.004 * energy;
    vec3 col;
    col.r = texture(tex, suv + vec2(ca, 0.0)).r;
    col.g = texture(tex, suv).g;
    col.b = texture(tex, suv - vec2(ca, 0.0)).b;

    vec3 blur = vec3(0.0);
    for (int i = 0; i < 12; i++) {
        float a = float(i) * 0.5236;
        blur += texture(tex, suv + vec2(cos(a), sin(a)) * 0.006).rgb;
    }
    col = mix(col, col + blur / 12.0 * 0.5, 0.35 * energy);

    col *= 1.0 - dot(c, c) * 0.9;
    col += (hash(uv * 1000.0 + t) - 0.5) * 0.03;
    frag = vec4(col, 1.0);
}
"""


def bench(w: int, h: int, n: int = 120) -> None:
    ctx = moderngl.create_standalone_context()
    print(f"RENDERER: {ctx.info['GL_RENDERER']}")
    print(f"GL      : {ctx.info['GL_VERSION']}\n")

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])

    rng = np.random.default_rng(0)
    tex = ctx.texture((w, h), 3, rng.integers(0, 255, (h, w, 3), dtype=np.uint8).tobytes())
    tex.build_mipmaps()
    tex.use(0)

    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    # warm up (shader compile, first-draw allocation)
    for _ in range(5):
        prog["t"].value = 0.0
        prog["energy"].value = 0.5
        vao.render()
    ctx.finish()

    t0 = time.perf_counter()
    for i in range(n):
        prog["t"].value = i / 30.0
        prog["energy"].value = 0.5 + 0.5 * np.sin(i / 10.0)
        vao.render()
    ctx.finish()
    t_render = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n):
        prog["t"].value = i / 30.0
        prog["energy"].value = 0.5
        vao.render()
        _buf = fbo.read(components=3)
    ctx.finish()
    t_full = time.perf_counter() - t0

    print(f"{w}x{h}, {n} frames")
    print(f"  render only     : {n / t_render:7.1f} fps  ({t_render * 1000 / n:5.2f} ms/frame)")
    print(f"  render + readback: {n / t_full:7.1f} fps  ({t_full * 1000 / n:5.2f} ms/frame)")
    print(f"  -> 4-min video   : {(4 * 60 * 30) / (n / t_full):6.1f}s of render time")
    print(f"  -> realtime ratio: {(n / t_full) / 30:6.2f}x\n")


if __name__ == "__main__":
    bench(1920, 1080)
    bench(854, 480)
