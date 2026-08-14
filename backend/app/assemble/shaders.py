"""GLSL for the Stage B renderer.

One uber-shader handles every shot: camera transform is computed on the CPU and
passed as uniforms, effects are toggled by a bitmask. The GL benchmark showed
shading costs ~0.03ms against ~2.1ms of readback at 1080p, so branching in the
fragment shader is effectively free -- there is no reason to specialise.
"""

from __future__ import annotations

# Effect name -> bit in u_fx. Must stay in sync with schema.Effect.
FX_FLAGS: dict[str, int] = {
    "bloom": 1 << 0,
    "god_rays": 1 << 1,
    "chroma": 1 << 2,
    "grain": 1 << 3,
    "fog_drift": 1 << 4,
    "vignette_pulse": 1 << 5,
    "kaleidoscope": 1 << 6,
    "glitch": 1 << 7,
    "halation": 1 << 8,
    "ripple": 1 << 9,
}

VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SHOT_FRAG = """
#version 330

uniform sampler2D u_tex;
uniform vec2  u_tex_aspect;   // scale to letterbox/cover-fit the source
uniform int   u_fx;

// camera (CPU-computed)
uniform float u_zoom;
uniform vec2  u_offset;
uniform float u_rot;

// reactive drives, all pre-mixed on the CPU into [0, ~1]
uniform float u_pulse;      // zoom pulse
uniform float u_shake;
uniform float u_bloom;
uniform float u_chroma;
uniform float u_vig_pulse;
uniform float u_bright;
uniform float u_sat_drive;
uniform float u_ripple;

// grade
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_vignette;
uniform float u_grain;
uniform vec3  u_tint;

uniform float u_time;

in  vec2 uv;
out vec4 frag;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
    return v;
}

bool has(int flag) { return (u_fx & flag) != 0; }

// Sample the source with cover-fit and clamping, so camera moves never
// reveal the edge of the image.
vec3 src(vec2 p) {
    vec2 c = clamp((p - 0.5) * u_tex_aspect + 0.5, 0.0015, 0.9985);
    return texture(u_tex, c).rgb;
}

void main() {
    vec2 c = uv - 0.5;

    if (has(%KALEIDOSCOPE%)) {
        float a = atan(c.y, c.x), r = length(c);
        float seg = 3.14159265 / 3.0;
        a = abs(mod(a, seg * 2.0) - seg);
        c = vec2(cos(a), sin(a)) * r;
    }

    // camera: rotate, zoom, translate
    float ca = cos(u_rot), sa = sin(u_rot);
    c = mat2(ca, -sa, sa, ca) * c;
    float zoom = max(1.0, u_zoom * (1.0 + u_pulse));
    c = c / zoom + u_offset;

    if (has(%RIPPLE%) && u_ripple > 0.001) {
        float r = length(c);
        c += normalize(c + 1e-6) * sin(r * 30.0 - u_time * 3.0) * 0.006 * u_ripple;
    }

    if (u_shake > 0.001) {
        c += vec2(hash(vec2(u_time, 1.7)) - 0.5, hash(vec2(u_time, 3.1)) - 0.5) * 0.02 * u_shake;
    }

    if (has(%GLITCH%) && u_shake > 0.35) {
        float band = floor(uv.y * 24.0);
        float j = (hash(vec2(band, floor(u_time * 18.0))) - 0.5);
        c.x += j * 0.05 * u_shake;
    }

    vec2 suv = c + 0.5;

    // chromatic aberration, scaled by distance from centre
    vec3 col;
    if (has(%CHROMA%) && u_chroma > 0.001) {
        vec2 dir = (suv - 0.5) * u_chroma * 0.02;
        col.r = src(suv + dir).r;
        col.g = src(suv).g;
        col.b = src(suv - dir).b;
    } else {
        col = src(suv);
    }

    // bloom: 12-tap ring blur mixed back into highlights
    if (has(%BLOOM%) && u_bloom > 0.001) {
        vec3 b = vec3(0.0);
        for (int i = 0; i < 12; i++) {
            float a = float(i) * 0.5235988;
            b += src(suv + vec2(cos(a), sin(a)) * 0.008);
        }
        b /= 12.0;
        vec3 hi = max(b - 0.55, 0.0);
        col += hi * u_bloom * 1.6;
    }

    if (has(%HALATION%)) {
        vec3 h = vec3(0.0);
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.7853982;
            h += src(suv + vec2(cos(a), sin(a)) * 0.018);
        }
        h /= 8.0;
        col += max(h - 0.6, 0.0) * vec3(0.9, 0.35, 0.2) * 0.5;
    }

    // god rays: radial accumulation toward centre
    if (has(%GOD_RAYS%)) {
        vec3 g = vec3(0.0);
        vec2 d = (suv - 0.5) / 20.0;
        vec2 p = suv;
        float w = 1.0;
        for (int i = 0; i < 20; i++) {
            p -= d;
            g += max(src(p) - 0.62, 0.0) * w;
            w *= 0.92;
        }
        col += g / 20.0 * (0.6 + u_bloom);
    }

    if (has(%FOG_DRIFT%)) {
        float f = fbm(suv * 3.0 + vec2(u_time * 0.05, u_time * 0.03));
        col = mix(col, col + vec3(0.06, 0.07, 0.09), f * 0.5);
    }

    // ---- grade ----
    col *= pow(2.0, u_exposure + u_bright * 0.5);
    col = (col - 0.5) * u_contrast + 0.5;

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation + u_sat_drive);
    col *= u_tint;

    float vig = u_vignette + u_vig_pulse * 0.35;
    col *= 1.0 - clamp(dot(c, c) * vig * 2.2, 0.0, 1.0);

    if (has(%GRAIN%) && u_grain > 0.0) {
        col += (hash(uv * 997.0 + fract(u_time) * 131.0) - 0.5) * u_grain;
    }

    frag = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# Bake the bit values into the shader source so GLSL sees literals.
for _name, _bit in FX_FLAGS.items():
    SHOT_FRAG = SHOT_FRAG.replace(f"%{_name.upper()}%", str(_bit))


COMPOSITE_FRAG = """
#version 330

uniform sampler2D u_a;
uniform sampler2D u_b;
uniform float u_mix;      // 0 = all A, 1 = all B
uniform int   u_mode;     // 0 cut/crossfade, 1 dip-to-colour, 2 whip
uniform vec3  u_dip;

in  vec2 uv;
out vec4 frag;

void main() {
    // No flip: the shot pass samples a top-down texture into a bottom-up
    // framebuffer, and fbo.read() returns bottom-up rows. Those two inversions
    // cancel, so the buffer handed to ffmpeg is already correctly oriented.
    vec2 t = uv;

    vec3 a = texture(u_a, t).rgb;
    vec3 b = texture(u_b, t).rgb;
    vec3 col;

    if (u_mode == 1) {
        // First half fades A into the dip colour, second half brings B out.
        col = u_mix < 0.5 ? mix(a, u_dip, u_mix * 2.0)
                          : mix(u_dip, b, (u_mix - 0.5) * 2.0);
    } else if (u_mode == 2) {
        // Whip: directional smear on both sides, then blend.
        vec3 sa = vec3(0.0), sb = vec3(0.0);
        float amt = sin(u_mix * 3.14159265) * 0.06;
        for (int i = 0; i < 6; i++) {
            float o = float(i) / 6.0 * amt;
            sa += texture(u_a, t + vec2(o, 0.0)).rgb;
            sb += texture(u_b, t - vec2(amt - o, 0.0)).rgb;
        }
        col = mix(sa / 6.0, sb / 6.0, u_mix);
    } else {
        col = mix(a, b, u_mix);
    }

    frag = vec4(col, 1.0);
}
"""
