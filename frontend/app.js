/* VideoCreator UI -- dependency-free by design: no build step, no node_modules.
 *
 * The whole UX rests on the Stage A / Stage B split. Every edit here mutates
 * shotlist.json and re-runs Stage B only, which is ~46x realtime at preview
 * resolution -- fast enough that "Preview shot" on a 10s shot returns in about
 * a second. Nothing in this file ever triggers generation.
 */

const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(async (r) => {
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.status === 204 ? null : r.json();
});

// Effects the shader supports. Kept in sync with assemble/shaders.py FX_FLAGS.
const EFFECTS = ["bloom", "god_rays", "chroma", "grain", "fog_drift",
                 "vignette_pulse", "kaleidoscope", "glitch", "halation", "ripple"];
const CAMERAS = ["static", "push_in", "pull_out", "pan_left", "pan_right",
                 "orbit", "drift", "sway"];
const TARGETS = ["zoom_pulse", "vignette_pulse", "bloom", "shake", "chroma",
                 "rotate", "brightness", "saturation", "ripple"];
const TRANSITIONS = ["cut", "crossfade", "dip_to_black", "dip_to_white", "whip"];

const S = { pid: null, timeline: null, shotlist: null, sel: -1, busy: false, saveTimer: null };

/* ---------------- health ---------------- */

async function health() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    el.textContent = `${h.gl_renderer.replace(/^D3D12 \(|\)$/g, "")} · ${h.codec}`;
    el.className = "health ok";
  } catch (e) {
    el.textContent = "render env unavailable";
    el.className = "health bad";
    el.title = String(e.message || e);
  }
}

/* ---------------- project setup ---------------- */

function wireFile(input, label) {
  $(input).addEventListener("change", (e) => {
    const f = e.target.files[0];
    const n = $(label);
    n.textContent = f ? f.name : "no file";
    n.classList.toggle("set", !!f);
  });
}

async function listProjects() {
  const sel = $("#existing");
  try {
    const rows = await api("/api/projects");
    for (const r of rows) {
      const o = document.createElement("option");
      o.value = r.id;
      o.textContent = `${r.id} · ${Math.round(r.duration)}s · ${Math.round(r.tempo)} BPM`;
      sel.appendChild(o);
    }
  } catch { /* listing is optional; the page still works without it */ }
}

async function createProject() {
  const audio = $("#f-audio").files[0], theme = $("#f-theme").files[0];
  if (!audio || !theme) { setup("Pick both an audio file and a theme image."); return; }

  const fd = new FormData();
  fd.append("audio", audio);
  fd.append("theme", theme);
  fd.append("prompt", $("#f-prompt").value);

  $("#create").disabled = true;
  setup("Analysing audio — beat tracking and structural segmentation…");
  try {
    const d = await api("/api/projects", { method: "POST", body: fd });
    load(d.id, d.timeline, d.shotlist);
    setup("");
  } catch (e) {
    setup(`Failed: ${e.message}`);
  } finally {
    $("#create").disabled = false;
  }
}

const setup = (m) => { $("#setup-status").textContent = m; };
const status = (m) => { $("#render-status").textContent = m; };

async function openProject(pid) {
  setup("Loading…");
  const d = await api(`/api/projects/${pid}`);
  load(d.id, d.timeline, d.shotlist);
  setup("");
}

function load(pid, timeline, shotlist) {
  S.pid = pid; S.timeline = timeline; S.shotlist = shotlist; S.sel = -1;
  $("#editor").hidden = false;
  $("#poster").src = `/api/projects/${pid}/theme`;
  $("#tl-meta").textContent =
    `${timeline.duration.toFixed(1)}s · ${Math.round(timeline.tempo)} BPM · ` +
    `${timeline.sections.length} sections · ${shotlist.shots.length} shots`;
  drawWave();
  drawShots();
  selectShot(0);
}

/* ---------------- timeline ---------------- */

function drawWave() {
  const c = $("#wave"), tl = S.timeline;
  const w = c.clientWidth, h = 96;
  const dpr = window.devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  // Section bands first, so the waveform reads on top of them.
  const looks = lookIndexBySection();
  const colors = ["#f0883e", "#58a6ff", "#8b949e"];
  for (const s of tl.sections) {
    const x0 = (s.start / tl.duration) * w, x1 = (s.end / tl.duration) * w;
    g.fillStyle = colors[looks[s.id]] + "14";
    g.fillRect(x0, 0, x1 - x0, h);
    g.fillStyle = "#2a333f";
    g.fillRect(x0, 0, 1, h);
    g.fillStyle = "#8b98a5";
    g.font = "10px ui-sans-serif, system-ui";
    g.fillText(s.label, x0 + 4, 12);
  }

  // Beat ticks, thinned so they stay legible on long songs.
  const step = Math.max(1, Math.round(tl.beats.length / (w / 6)));
  g.fillStyle = "#ffffff10";
  for (let i = 0; i < tl.beats.length; i += step) {
    g.fillRect((tl.beats[i] / tl.duration) * w, h - 12, 1, 12);
  }

  // Waveform envelope.
  const peaks = tl.waveform_peaks, mid = h / 2;
  g.fillStyle = "#3fb95088";
  for (let x = 0; x < w; x++) {
    const p = peaks[Math.floor((x / w) * peaks.length)] || 0;
    const a = p * (h * 0.42);
    g.fillRect(x, mid - a, 1, a * 2);
  }
}

/** Mirror of builder._looks_by_energy: rank sections by energy percentile. */
function lookIndexBySection() {
  const secs = [...S.timeline.sections].sort((a, b) => a.energy - b.energy);
  const out = {}, n = secs.length;
  secs.forEach((s, i) => {
    const pct = i / Math.max(n - 1, 1);
    out[s.id] = pct >= 2 / 3 ? 0 : (pct >= 1 / 3 ? 1 : 2);
  });
  return out;
}

function shotLook(shot) {
  const sec = S.timeline.sections.find((s) => s.label === shot.section);
  return sec ? lookIndexBySection()[sec.id] : 2;
}

function drawShots() {
  const host = $("#shots"), dur = S.timeline.duration;
  host.innerHTML = "";
  S.shotlist.shots.forEach((sh, i) => {
    const d = document.createElement("div");
    d.className = `shot look${shotLook(sh)}${i === S.sel ? " sel" : ""}`;
    d.style.left = `${(sh.start / dur) * 100}%`;
    d.style.width = `${((sh.end - sh.start) / dur) * 100}%`;
    d.textContent = `${i + 1}·${sh.section}`;
    d.title = `Shot ${i + 1} [${sh.section}] ${sh.start.toFixed(1)}–${sh.end.toFixed(1)}s`;
    d.onclick = () => selectShot(i);
    host.appendChild(d);
  });
}

$("#wave").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const t = ((e.clientX - r.left) / r.width) * S.timeline.duration;
  const i = S.shotlist.shots.findIndex((s) => t >= s.start && t < s.end);
  if (i >= 0) selectShot(i);
});

/* ---------------- inspector ---------------- */

function selectShot(i) {
  S.sel = i;
  drawShots();
  const sh = S.shotlist.shots[i];
  $("#shot-n").textContent = `${i + 1} / ${S.shotlist.shots.length}`;
  $("#insp-body").innerHTML = "";
  $("#insp-body").className = "";
  $("#insp-body").append(
    el("div", "shot-time",
       `[${sh.section}] ${sh.start.toFixed(2)}s – ${sh.end.toFixed(2)}s ` +
       `(${(sh.end - sh.start).toFixed(1)}s)`),
    selectField("Camera path", CAMERAS, sh.camera.path, (v) => { sh.camera.path = v; }),
    sliderField("Camera amount", sh.camera.amplitude, 0, 1, .01, (v) => { sh.camera.amplitude = v; }),
    selectField("Transition in", TRANSITIONS, sh.transition_in.type, (v) => {
      sh.transition_in.type = v;
      // Mirrors the schema validator: a cut has no duration, anything else needs one.
      sh.transition_in.duration = v === "cut" ? 0 : (sh.transition_in.duration || 0.6);
    }),
    effectsField(sh),
    sliderField("Exposure", sh.grade.exposure, -1, 1, .01, (v) => { sh.grade.exposure = v; }),
    sliderField("Contrast", sh.grade.contrast, 0, 2, .01, (v) => { sh.grade.contrast = v; }),
    sliderField("Saturation", sh.grade.saturation, 0, 2, .01, (v) => { sh.grade.saturation = v; }),
    sliderField("Vignette", sh.grade.vignette, 0, 1, .01, (v) => { sh.grade.vignette = v; }),
    sliderField("Grain", sh.grade.grain, 0, .2, .005, (v) => { sh.grade.grain = v; }),
    bindingsField(sh),
  );
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function field(label) {
  const f = el("div", "field");
  f.append(el("span", "lbl", label));
  return f;
}

function selectField(label, opts, value, onChange) {
  const f = field(label), s = el("select");
  for (const o of opts) {
    const op = el("option", null, o.replace(/_/g, " "));
    op.value = o;
    if (o === value) op.selected = true;
    s.appendChild(op);
  }
  s.onchange = () => { onChange(s.value); save(); };
  f.appendChild(s);
  return f;
}

function sliderField(label, value, min, max, step, onChange) {
  const f = field(label), wrap = el("div", "slider");
  const r = el("input"), out = el("span", "val", (+value).toFixed(2));
  r.type = "range"; r.min = min; r.max = max; r.step = step; r.value = value;
  r.oninput = () => { out.textContent = (+r.value).toFixed(2); onChange(+r.value); save(); };
  wrap.append(r, out);
  f.appendChild(wrap);
  return f;
}

function effectsField(sh) {
  const f = field("Effects"), box = el("div", "chips");
  for (const fx of EFFECTS) {
    const c = el("span", `chip${sh.fx.includes(fx) ? " on" : ""}`, fx.replace(/_/g, " "));
    c.onclick = () => {
      const i = sh.fx.indexOf(fx);
      if (i >= 0) sh.fx.splice(i, 1); else sh.fx.push(fx);
      c.classList.toggle("on");
      save();
    };
    box.appendChild(c);
  }
  f.appendChild(box);
  return f;
}

function bindingsField(sh) {
  const f = field("Audio reactive"), box = el("div", "binds");
  const curves = Object.keys(S.timeline.curves);

  const redraw = () => {
    box.innerHTML = "";
    sh.reactive.forEach((b, i) => {
      const row = el("div", "bind");
      const cs = el("select"), ts = el("select");
      for (const c of curves) {
        const o = el("option", null, c); o.value = c;
        if (c === b.curve) o.selected = true; cs.appendChild(o);
      }
      for (const t of TARGETS) {
        const o = el("option", null, t.replace(/_/g, " ")); o.value = t;
        if (t === b.target) o.selected = true; ts.appendChild(o);
      }
      cs.onchange = () => { b.curve = cs.value; save(); };
      ts.onchange = () => { b.target = ts.value; save(); };
      const del = el("button", null, "×");
      del.title = "remove binding";
      del.onclick = () => { sh.reactive.splice(i, 1); redraw(); save(); };
      row.append(cs, el("span", "muted", "→"), ts, del);
      box.appendChild(row);
    });
    const add = el("button", null, "+ binding");
    add.onclick = () => {
      sh.reactive.push({ curve: curves[0], target: "zoom_pulse", amount: 0.5 });
      redraw(); save();
    };
    box.appendChild(add);
  };
  redraw();
  f.appendChild(box);
  return f;
}

/* ---------------- persist + render ---------------- */

/** Debounced: dragging a slider fires continuously, one PUT per settle. */
function save() {
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(async () => {
    try {
      await api(`/api/projects/${S.pid}/shotlist`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(S.shotlist),
      });
      status("Saved.");
    } catch (e) {
      // The backend validates coverage and enum membership; surface rejections
      // rather than letting the UI drift out of sync with the saved shotlist.
      status(`Rejected: ${e.message}`);
    }
  }, 350);
}

async function render({ preview, start, end, label }) {
  if (S.busy) return;
  S.busy = true;
  ["#prev-shot", "#prev-all", "#render-full"].forEach((s) => { $(s).disabled = true; });
  $("#prog-wrap").hidden = false;
  $("#prog-bar").style.width = "0%";
  $("#prog-txt").textContent = "starting…";
  status(label);

  try {
    const body = { preview, start: start ?? 0 };
    if (end != null) body.end = end;
    const job = await api(`/api/projects/${S.pid}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const done = await follow(job.id);
    if (done.state === "done") {
      const v = $("#video");
      v.hidden = false;
      v.src = `${done.result.video}?t=${Date.now()}`;  // bust the cache per render
      v.load();
      v.play().catch(() => { /* autoplay policy -- user can hit play */ });
      status(`Rendered in ${label.toLowerCase()}.`);
    } else {
      status(`Failed: ${done.error}`);
      if (done.traceback) console.error(done.traceback);
    }
  } catch (e) {
    status(`Failed: ${e.message}`);
  } finally {
    S.busy = false;
    ["#prev-shot", "#prev-all", "#render-full"].forEach((s) => { $(s).disabled = false; });
    $("#prog-wrap").hidden = true;
  }
}

/** Follow a job over the WebSocket the backend already exposes. */
function follow(jobId) {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}`);
    let last = null;
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.error && !m.state) { reject(new Error(m.error)); ws.close(); return; }
      last = m;
      $("#prog-bar").style.width = `${(m.progress * 100).toFixed(1)}%`;
      $("#prog-txt").textContent = m.message || m.state;
      if (["done", "failed", "cancelled"].includes(m.state)) { ws.close(); resolve(m); }
    };
    ws.onerror = () => reject(new Error("progress socket failed"));
    ws.onclose = () => { if (last && !["done", "failed", "cancelled"].includes(last.state))
      reject(new Error("progress socket closed early")); };
  });
}

/* ---------------- wiring ---------------- */

$("#create").onclick = createProject;
$("#existing").onchange = (e) => { if (e.target.value) openProject(e.target.value); };

$("#prev-shot").onclick = () => {
  const sh = S.shotlist.shots[S.sel];
  if (!sh) { status("Select a shot first."); return; }
  // Start a beat early so the incoming transition is visible in the preview.
  const pad = Math.min(1.0, sh.start);
  render({ preview: true, start: sh.start - pad, end: sh.end,
           label: `Shot ${S.sel + 1} preview` });
};
$("#prev-all").onclick = () => render({ preview: true, label: "Full preview" });
$("#render-full").onclick = () => render({ preview: false, label: "1080p render" });

wireFile("#f-audio", "#n-audio");
wireFile("#f-theme", "#n-theme");

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (S.timeline) { drawWave(); drawShots(); } }, 120);
});

health();
listProjects();
