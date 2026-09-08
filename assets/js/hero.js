// Interactive hero: a 2D polycrystalline 6H-SiC microstructure on a thin slab, with a hexagonal
// prism floating over every grain showing that grain's crystal orientation (prism axis = c-axis).
//
// Everything shown is one realization from my thermo-mechanics training set (assets/data/sample.json):
// real grain geometry, real Haar-uniform orientations, real loading, and the finite-element answers
// for per-grain energy, per-grain thermal dissipation and per-boundary Kapitza dissipation.
//
//   Heat flow   NOT simulation output. Steady conduction is solved here, in the browser, on the grain
//               graph (one temperature per grain) with the material constants the finite-element run
//               used. The temperature field, the fluxes and the effective conductivity are all from
//               that reduced model; the finite-element interface dissipation is the yardstick it is
//               checked against. Drag a crystal and it re-solves.
//   Energy /    finite-element output, next to an ordinary-least-squares baseline fitted on the whole
//   Dissipation training set, and the residual between them.
//
import * as THREE from 'three';

const el = document.getElementById('hero');
if (el) init(el);

async function init(container) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    container.textContent = 'Interactive visualisation needs WebGL.';
    return;
  }

  // ---- constants --------------------------------------------------------------------------
  const DEPTH = 0.08;              // slab thickness, world units
  const C_OVER_A = 1.6;            // prism height / circumradius (real 6H-SiC is ~4.9, too spindly)
  const DRAG_SPEED = 0.008;        // radians per pixel
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- renderer / DOM ---------------------------------------------------------------------
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.textContent = '';
  container.classList.add('hero--live');
  container.setAttribute('aria-label',
    'Interactive 3D polycrystalline silicon carbide microstructure from a finite-element training set. Drag to rotate. Buttons switch between a heat-flow solve run in the page, the simulated fields, a least-squares baseline and their residual.');
  container.appendChild(renderer.domElement);

  const ui = document.createElement('div'); ui.className = 'hero-ui'; container.appendChild(ui);
  const rowView = document.createElement('div'); rowView.className = 'hero-ui__row'; ui.appendChild(rowView);
  const rowSource = document.createElement('div'); rowSource.className = 'hero-ui__row'; ui.appendChild(rowSource);
  const srcGroup = document.createElement('span'); srcGroup.className = 'hero-ui__group'; rowSource.appendChild(srcGroup);
  const texGroup = document.createElement('span'); texGroup.className = 'hero-ui__group'; rowSource.appendChild(texGroup);
  const readout = document.createElement('div'); readout.className = 'hero-readout'; container.appendChild(readout);
  const hint = document.createElement('span'); hint.className = 'hero-hint'; container.appendChild(hint);

  function button(row, label, onClick, cls) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = label; if (cls) b.className = cls;
    b.addEventListener('click', onClick);
    row.appendChild(b);
    return b;
  }
  const setPressed = (buttons, active) => buttons.forEach(([k, b]) => b.setAttribute('aria-pressed', String(k === active)));

  // ---- scene ------------------------------------------------------------------------------
  let needsRender = true;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 50);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x6f6f6f, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(2, 3, 4);
  scene.add(sun);
  const sample = new THREE.Group();
  scene.add(sample);
  const restPose = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.95, 0, 0));
  sample.quaternion.copy(restPose);

  const yAxis = new THREE.Vector3(0, 1, 0), zAxis = new THREE.Vector3(0, 0, 1), xAxis = new THREE.Vector3(1, 0, 0);
  const boundaryMat = new THREE.LineBasicMaterial({ color: 0x2b2b2b });
  const prismEdgeMat = new THREE.LineBasicMaterial({ color: 0x1c1c1c });
  const coneMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.6 });
  const boxGeo = new THREE.BoxGeometry(1, 0.026, 0.012);
  const coneGeo = new THREE.ConeGeometry(0.028, 0.085, 12);
  const sharedGeo = new Set([boxGeo, coneGeo]), sharedMat = new Set([boundaryMat, prismEdgeMat, coneMat]);

  // ---- world: slab, prisms and per-boundary markers built from a spec ------------------------
  // spec = { W, H, cells: [[x,y],...][], quats: Quaternion[], edges: [{ i, j, a: [x,y], b: [x,y] }] }
  let world = null;
  function dispose(o) {
    o.children.slice().forEach(dispose);
    if (o.geometry && !sharedGeo.has(o.geometry)) o.geometry.dispose();
    if (o.material && !sharedMat.has(o.material)) o.material.dispose();
  }
  function buildWorld(spec) {
    if (world) world.objects.forEach((o) => { sample.remove(o); dispose(o); });
    const objects = [], grains = [], prisms = [];
    spec.cells.forEach((poly, idx) => {
      const shape = new THREE.Shape(poly.map((p) => new THREE.Vector2(p[0], p[1])));
      const geo = new THREE.ExtrudeGeometry(shape, { depth: DEPTH, bevelEnabled: false });
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0 }));
      const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), boundaryMat);
      sample.add(mesh, lines); objects.push(mesh, lines);

      const c = centroid(poly);
      const r = 0.62 * inradius(poly, c);
      const h = C_OVER_A * r;
      const pGeo = new THREE.CylinderGeometry(r, r, h, 6);      // local Y = crystal c-axis
      const prism = new THREE.Mesh(pGeo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }));
      prism.add(new THREE.LineSegments(new THREE.EdgesGeometry(pGeo, 20), prismEdgeMat));
      const clearance = Math.sqrt(r * r + (h / 2) * (h / 2)) * 1.03;   // never intersects the slab
      prism.position.set(c[0], c[1], DEPTH + clearance);
      prism.quaternion.copy(spec.quats[idx]);
      sample.add(prism); objects.push(prism); prisms.push(prism);
      grains.push({
        mesh, prism, c: new THREE.Vector3(), ipf: new THREE.Color(), centroid: c,
      });
    });
    const edges = spec.edges.map((e) => {
      const ci = grains[e.i].centroid, cj = grains[e.j].centroid;   // arrow direction: centroid to centroid
      const dx = cj[0] - ci[0], dy = cj[1] - ci[1], d = Math.hypot(dx, dy) || 1;
      const mid = [(e.a[0] + e.b[0]) / 2, (e.a[1] + e.b[1]) / 2];
      const len = Math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1]);
      const box = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({ roughness: 0.8 }));
      box.scale.x = len;
      box.position.set(mid[0], mid[1], DEPTH + 0.006);
      box.quaternion.setFromAxisAngle(zAxis, Math.atan2(e.b[1] - e.a[1], e.b[0] - e.a[0]));
      box.visible = false;
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.set(mid[0], mid[1], DEPTH + 0.03);
      cone.visible = false;
      sample.add(box, cone); objects.push(box, cone);
      return { i: e.i, j: e.j, mid, len, n: [dx / d, dy / d], box, cone };
    });
    world = { W: spec.W, H: spec.H, grains, prisms, edges, objects, particles: null };
    resize();
  }

  // ---- colours ----------------------------------------------------------------------------
  const white = new THREE.Color(0xffffff);
  const cold = new THREE.Color(0x3a6fd8), mild = new THREE.Color(0xf2f0ea), warm = new THREE.Color(0xd8402c);
  const seqLow = new THREE.Color(0xf4f1ea), seqHigh = new THREE.Color(0xb5301f);
  const edgeLow = new THREE.Color(0xbdbdbd), edgeHigh = new THREE.Color(0x7a1010);
  const tmpV = new THREE.Vector3();
  function ipfColour(g) {
    g.c.copy(yAxis).applyQuaternion(g.prism.quaternion);      // c-axis in the sample frame
    const x = Math.abs(g.c.x), y = Math.abs(g.c.y), z = Math.abs(g.c.z);
    const m = Math.max(x, y, z) || 1;
    g.ipf.setRGB(z / m, x / m, y / m);                         // IPF-style: c parallel to slab normal -> red
  }
  const diverging = (t, out) => (t < 0 ? out.copy(mild).lerp(cold, -t) : out.copy(mild).lerp(warm, t));
  const sequential = (t, out, lo = seqLow, hi = seqHigh) => out.copy(lo).lerp(hi, Math.max(0, Math.min(1, t)));
  const paintPrisms = () => world.grains.forEach((g) => { ipfColour(g); g.prism.material.color.copy(g.ipf).lerp(white, 0.35); });
  const paintOrientation = () => world.grains.forEach((g) => g.mesh.material.color.copy(g.ipf).lerp(white, 0.12));
  // Sequential when `range` is [lo, hi], diverging about zero when it is a single max-magnitude.
  function paintScalars(gv, range, ev, erange, stretch = (t) => t) {
    world.grains.forEach((g, k) => {
      if (Array.isArray(range)) sequential(stretch((gv[k] - range[0]) / ((range[1] - range[0]) || 1)), g.mesh.material.color);
      else diverging(gv[k] / (range || 1), g.mesh.material.color);
    });
    world.edges.forEach((e, k) => {
      e.box.visible = !!ev;
      if (!ev) return;
      if (Array.isArray(erange)) sequential(stretch((ev[k] - erange[0]) / ((erange[1] - erange[0]) || 1)), e.box.material.color, edgeLow, edgeHigh);
      else diverging(ev[k] / (erange || 1), e.box.material.color);
    });
  }

  // ---- the dataset sample -------------------------------------------------------------------
  // resolved against this module's own URL, so the page also works under a project subpath
  const dataUrl = new URL('../data/sample.json', import.meta.url);
  const data = await fetch(dataUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (!data) {
    container.textContent = 'Could not load the simulation sample.';
    return;
  }
  const dataset = buildDataset(data);

  function buildDataset(d) {
    const L = d.source.domain_um, s = 2 / L, o = L / 2;
    const toWorld = (p) => [(p[0] - o) * s, (p[1] - o) * s];
    const yToZ = new THREE.Quaternion().setFromUnitVectors(yAxis, zAxis);
    const kept = d.edges.filter((e) => e.seg);                 // only boundaries we could place geometrically
    const spec = {
      W: 2, H: 2,
      cells: d.grains.map((g) => g.poly.map(toWorld)),
      quats: d.grains.map((g) => {
        const rv = new THREE.Vector3(...g.rv), th = rv.length();
        const q = th > 1e-9 ? new THREE.Quaternion().setFromAxisAngle(rv.normalize(), th) : new THREE.Quaternion();
        return q.multiply(yToZ);                                // prism +Y -> crystal c-axis
      }),
      edges: kept.map((e) => ({ i: e.i, j: e.j, a: toWorld(e.seg[0]), b: toWorld(e.seg[1]) })),
    };
    // Physics geometry in SI: the conduction model mixes a bulk resistance (scales with length) with
    // a Kapitza resistance (which does not), so it needs real metres, not world units.
    const um = 1e-6;
    const cents = d.grains.map((g) => centroid(g.poly).map((v) => v * um));
    const pedges = kept.map((e) => {
      const a = e.seg[0].map((v) => v * um), b = e.seg[1].map((v) => v * um);
      const ex = b[0] - a[0], ey = b[1] - a[1], len = Math.hypot(ex, ey) || 1e-12;
      const n = [-ey / len, ex / len];                          // boundary normal
      const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const di = Math.max(Math.abs((mid[0] - cents[e.i][0]) * n[0] + (mid[1] - cents[e.i][1]) * n[1]), 1e-11);
      const dj = Math.max(Math.abs((mid[0] - cents[e.j][0]) * n[0] + (mid[1] - cents[e.j][1]) * n[1]), 1e-11);
      return { i: e.i, j: e.j, len, n, di, dj };
    });
    // Dirichlet segments: polygon edges lying on the domain boundary carry T = g . x.
    const outer = [];
    d.grains.forEach((g, i) => {
      const p = g.poly;
      for (let k = 0; k < p.length; k++) {
        const a = p[k], b = p[(k + 1) % p.length];
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        if (Math.min(mx, my, L - mx, L - my) > 2e-3) continue;
        const ex = (b[0] - a[0]) * um, ey = (b[1] - a[1]) * um, len = Math.hypot(ex, ey);
        if (len < 1e-12) continue;
        const n = [-ey / len, ex / len];
        const mid = [mx * um, my * um];
        const di = Math.max(Math.abs((mid[0] - cents[i][0]) * n[0] + (mid[1] - cents[i][1]) * n[1]), 1e-11);
        outer.push({ i, len, n, di, mid });
      }
    });
    return {
      spec, cents, pedges, outer,
      E: d.grains.map((g) => g.E), E_ols: d.grains.map((g) => g.E_ols),
      D: d.grains.map((g) => g.D), D_ols: d.grains.map((g) => g.D_ols),
      I: kept.map((e) => e.I), I_ols: kept.map((e) => e.I_ols),
      qref: null, tref: null,
    };
  }

  // ---- live conduction solve on the real grain graph -----------------------------------------
  // k(n) = ka + (kc - ka)(c . n)^2 for hexagonal 6H-SiC; every boundary adds a Kapitza resistance.
  // Edge conductance = boundary length / (half-resistance of each grain + Kapitza), in series.
  const r2 = (y, p) => {
    const m = y.reduce((s, v) => s + v, 0) / y.length;
    let ss = 0, st = 0;
    for (let k = 0; k < y.length; k++) { ss += (y[k] - p[k]) ** 2; st += (y[k] - m) ** 2; }
    return st > 0 ? 1 - ss / st : 0;
  };
  function solveHeat() {
    const ph = data.physics, g = [data.loading.g1, data.loading.g2];
    const n = dataset.cents.length;
    const A = Array.from({ length: n }, () => new Float64Array(n)), rhs = new Float64Array(n);
    const kOf = (grain, nv) => { const cn = grain.c.x * nv[0] + grain.c.y * nv[1]; return ph.ka + (ph.kc - ph.ka) * cn * cn; };
    const G = new Float64Array(dataset.pedges.length);
    dataset.pedges.forEach((e, k) => {
      G[k] = e.len / (e.di / kOf(world.grains[e.i], e.n) + e.dj / kOf(world.grains[e.j], e.n) + 1 / ph.h_kapitza);
      A[e.i][e.i] += G[k]; A[e.i][e.j] -= G[k];
      A[e.j][e.j] += G[k]; A[e.j][e.i] -= G[k];
    });
    const Gout = dataset.outer.map((o) => o.len / (o.di / kOf(world.grains[o.i], o.n)));
    dataset.outer.forEach((o, k) => {
      A[o.i][o.i] += Gout[k];
      rhs[o.i] += Gout[k] * (g[0] * o.mid[0] + g[1] * o.mid[1]);
    });
    const T = gauss(A, rhs);
    const q = dataset.pedges.map((e, k) => G[k] * (T[e.i] - T[e.j]));            // W per metre of depth, i -> j
    const Ipred = dataset.pedges.map((e, k) => ph.scale_interface * q[k] * q[k] / (ph.h_kapitza * e.len));
    // Homogenized conductivity: heat entering through the hot part of the boundary, over |grad T| L.
    let Qin = 0;
    dataset.outer.forEach((o, k) => {
      const flow = Gout[k] * ((g[0] * o.mid[0] + g[1] * o.mid[1]) - T[o.i]);
      if (flow > 0) Qin += flow;
    });
    const kEff = Qin / (Math.hypot(g[0], g[1]) * data.source.domain_um * 1e-6);
    if (dataset.qref === null) {          // reference scales, so later edits visibly change the picture
      dataset.qref = Math.max(1e-30, ...q.map(Math.abs));
      dataset.tref = [Math.min(...T), Math.max(...T)];
    }
    return { T, q, Ipred, kEff, r2: r2(dataset.I, Ipred) };
  }

  // ---- animated heat particles ----------------------------------------------------------------
  const MAX_PARTICLES = 520;
  function ensureParticles() {
    if (world.particles) return world.particles;
    const pos = new Float32Array(MAX_PARTICLES * 3).fill(-99);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const points = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0x15130f, size: 0.05, sizeAttenuation: true, transparent: true, opacity: 0.88 }));
    points.frustumCulled = false; points.visible = false;
    sample.add(points); world.objects.push(points);
    world.particles = { points, geo, pos, items: [] };
    return world.particles;
  }
  // One stream of dots per boundary, denser and faster where more heat crosses.
  function seedParticles(q, qref) {
    const p = ensureParticles();
    const qmax = Math.max(1e-30, qref);
    p.items = [];
    for (let k = 0; k < world.edges.length && p.items.length < MAX_PARTICLES; k++) {
      const f = Math.min(1.4, Math.abs(q[k]) / qmax);
      const count = Math.min(5, Math.round(0.6 + 3.4 * Math.sqrt(f)));
      const e = world.edges[k];
      const from = q[k] >= 0 ? e.i : e.j, to = q[k] >= 0 ? e.j : e.i;
      for (let m = 0; m < count && p.items.length < MAX_PARTICLES; m++) {
        p.items.push({ a: world.grains[from].centroid, b: world.grains[to].centroid, speed: 0.12 + 0.5 * f, phase: m / count });
      }
    }
    for (let k = p.items.length; k < MAX_PARTICLES; k++) p.pos[k * 3 + 2] = -99;   // park unused dots
    p.geo.attributes.position.needsUpdate = true;
    p.points.visible = true;
    return p;
  }
  function stepParticles(t) {
    const p = world && world.particles;
    if (!p || !p.points.visible) return;
    for (let k = 0; k < p.items.length; k++) {
      const it = p.items[k];
      const u = (it.phase + t * it.speed) % 1;
      p.pos[k * 3] = it.a[0] + (it.b[0] - it.a[0]) * u;
      p.pos[k * 3 + 1] = it.a[1] + (it.b[1] - it.a[1]) * u;
      p.pos[k * 3 + 2] = DEPTH + 0.05;
    }
    p.geo.attributes.position.needsUpdate = true;
    needsRender = true;
  }

  // ---- readout helpers ------------------------------------------------------------------------
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const minmax = (...arrs) => { const all = arrs.flat(); return [Math.min(...all), Math.max(...all)]; };
  const maxAbs = (a) => Math.max(...a.map(Math.abs));
  const sub = (a, b) => a.map((v, k) => v - b[k]);
  const fmt = (v, d = 2) => v.toExponential(d).replace(/e([+-])(\d+)/, (m, sg, nn) => ` × 10<sup>${sg === '-' ? '−' : ''}${nn}</sup>`);

  // ---- dataset mode ----------------------------------------------------------------------------
  let view = 'flow', source = 'sim', texture = 'dataset', modified = false;
  const datasetViews = [['flow', 'Heat flow'], ['energy', 'Energy'], ['dissipation', 'Dissipation'], ['orientation', 'Orientation']];
  const sources = [['sim', 'Simulation'], ['ols', 'OLS baseline'], ['res', 'Residual']];
  const textures = [['dataset', 'As simulated'], ['gradient', 'all c ∥ ∇T'], ['normal', 'all c ⊥ slab']];

  function applyDataset() {
    paintPrisms();
    const d = data, L = d.loading, r2s = d.ols.this_sample_r2, tr2 = d.ols.test_r2;
    const head = `<span class="hero-readout__note">6H-SiC, ${d.source.ngrains} grains, ${d.source.domain_um} µm square · ` +
      `${texture === 'dataset' ? `test-set realization #${d.source.index}`
        : texture === 'edited' ? 'orientations edited by you'
        : texture === 'gradient' ? 'every c-axis aligned with ∇T' : 'every c-axis normal to the slab'} · ` +
      `∇T (${L.g1.toFixed(2)}, ${L.g2.toFixed(2)}) K/m` +
      `${view === 'flow' ? '' : `, ΔT ${L.T.toFixed(0)} K, ε (${(L.exx * 1e5).toFixed(1)}, ${(L.exy * 1e5).toFixed(1)}, ${(L.eyy * 1e5).toFixed(1)}) × 10<sup>−5</sup>`}</span><br>`;
    let body = '';
    const showParticles = view === 'flow' && !reduceMotion;
    if (world.particles) world.particles.points.visible = showParticles;

    if (view === 'flow') {
      const h = solveHeat();
      const [lo, hi] = dataset.tref;
      const span = Math.max(hi - lo, 1e-30);
      world.grains.forEach((g, k) => diverging(Math.max(-1, Math.min(1, 2 * (h.T[k] - lo) / span - 1)), g.mesh.material.color));
      const qmax = dataset.qref;
      world.edges.forEach((e, k) => {
        e.box.visible = false;
        const f = Math.min(1.4, Math.abs(h.q[k]) / qmax);
        e.cone.visible = !showParticles && f > 0.08;            // static arrows when motion is reduced
        if (e.cone.visible) {
          const s = h.q[k] > 0 ? 1 : -1;
          tmpV.set(s * e.n[0], s * e.n[1], 0);
          e.cone.quaternion.setFromUnitVectors(yAxis, tmpV);
          e.cone.scale.setScalar(0.45 + 0.8 * f);
        }
      });
      if (showParticles) seedParticles(h.q, dataset.qref);
      body = `effective conductivity <b>${h.kEff.toFixed(1)}</b> W/(m·K) ` +
        `<span class="hero-readout__note">(crystal 390 ⊥c, 273 ∥c; Kapitza 2.5 × 10<sup>8</sup> W/(m²·K))</span><br>` +
        (modified
          ? `Σ interface dissipation <b>${fmt(sum(h.Ipred), 3)}</b> W/m <span class="hero-readout__note">· a prediction for orientations the solver never saw</span>`
          : `Σ interface dissipation · graph <b>${fmt(sum(h.Ipred), 3)}</b> W/m · finite element <b>${fmt(sum(dataset.I), 3)}</b> W/m<br>` +
            `per-boundary R² against the finite element <b>${h.r2.toFixed(2)}</b> <span class="hero-readout__note">(${d.physics.validation})</span>`) +
        `<br><span class="hero-readout__note">grain colour: temperature · ${showParticles ? 'moving dots: heat crossing each boundary' : 'arrows: heat flux'} · drag a crystal to re-solve</span>`;
    } else if (view === 'orientation') {
      paintOrientation();
      world.edges.forEach((e) => { e.box.visible = false; e.cone.visible = false; });
      body = `<span class="hero-readout__note">grains coloured by c-axis direction, inverse-pole-figure style</span>`;
    } else {
      world.edges.forEach((e) => { e.cone.visible = false; });
      const g = view === 'energy' ? [dataset.E, dataset.E_ols] : [dataset.D, dataset.D_ols];
      const e = view === 'energy' ? null : [dataset.I, dataset.I_ols];
      let legend;
      if (source === 'res') {
        const gr = sub(g[0], g[1]), er = e && sub(e[0], e[1]);
        const st = view === 'energy' ? (t) => t : (t) => Math.sign(t) * Math.sqrt(Math.abs(t));
        paintScalars(gr.map(st), st(maxAbs(gr)), er && er.map(st), er && st(maxAbs(er)));
        legend = 'residual = simulation − OLS, blue below / red above, symmetric scale';
      } else {
        const k = source === 'sim' ? 0 : 1;
        paintScalars(g[k], minmax(g[0], g[1]), e && e[k], e && minmax(e[0], e[1]), view === 'energy' ? (t) => t : Math.sqrt);
        legend = view === 'energy' ? 'grain colour: mechanical energy, one scale for simulation and OLS'
                                   : 'grain colour: thermal dissipation; boundary bars: interface dissipation';
      }
      body = view === 'dissipation'
        ? `Σ dissipation · simulation <b>${fmt(sum(dataset.D) + sum(dataset.I))}</b> W/m · OLS <b>${fmt(sum(dataset.D_ols) + sum(dataset.I_ols))}</b> W/m<br>` +
          `OLS R² here: grains <b>${r2s.D.toFixed(2)}</b>, interfaces <b>${r2s.I.toFixed(2)}</b> <span class="hero-readout__note">(whole test set ${tr2.D.toFixed(2)} / ${tr2.I.toFixed(2)})</span><br>` +
          `<span class="hero-readout__note">${legend}</span>`
        : `Σ energy · simulation <b>${fmt(sum(dataset.E))}</b> J · OLS <b>${fmt(sum(dataset.E_ols))}</b> J<br>` +
          `OLS R² here <b>${r2s.E.toFixed(3)}</b> <span class="hero-readout__note">(whole test set ${tr2.E.toFixed(3)})</span><br>` +
          `<span class="hero-readout__note">${legend}</span>`;
      if (modified) body += `<br><span class="hero-readout__note">orientations edited: the simulated fields belong to the original ones</span>`;
    }
    readout.innerHTML = head + body;
    needsRender = true;
  }

  // ---- mode switching ---------------------------------------------------------------------------
  let viewButtons = [], sourceButtons = [], textureButtons = [];
  function apply() {
    applyDataset();
    setPressed(viewButtons, view); setPressed(sourceButtons, source); setPressed(textureButtons, texture);
    const fields = view === 'energy' || view === 'dissipation';
    srcGroup.hidden = !fields;
    texGroup.hidden = view !== 'flow';
    rowSource.hidden = srcGroup.hidden && texGroup.hidden;
  }
  // Orientation presets. 'dataset' restores the simulated texture; the others give every grain the
  // same c-axis, a single-crystal limit that shows how much texture moves the homogenized answer.
  const texQ = new THREE.Quaternion(), texV = new THREE.Vector3();
  function applyTexture(name) {
    texture = name;
    if (name === 'dataset') {
      dataset.spec.quats.forEach((q, k) => world.grains[k].prism.quaternion.copy(q));
    } else {
      const gx = data.loading.g1, gy = data.loading.g2, m = Math.hypot(gx, gy) || 1;
      if (name === 'gradient') texV.set(gx / m, gy / m, 0); else texV.set(0, 0, 1);
      texQ.setFromUnitVectors(yAxis, texV);
      world.grains.forEach((g) => g.prism.quaternion.copy(texQ));
    }
    modified = name !== 'dataset';
    apply();
  }
  function start() {
    buildWorld(dataset.spec);
    viewButtons = datasetViews.map(([k, l]) => [k, button(rowView, l, () => { view = k; apply(); })]);
    sourceButtons = sources.map(([k, l]) => [k, button(srcGroup, l, () => { source = k; apply(); })]);
    textureButtons = textures.map(([k, l]) => [k, button(texGroup, l, () => applyTexture(k))]);
    hint.textContent = 'drag to rotate · drag a crystal to reorient its grain';
    apply();
  }

  // ---- sizing -----------------------------------------------------------------------------------
  function resize() {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const W = world ? world.W : 2, H = world ? world.H : 2;
    const R = Math.hypot(W / 2, H / 2);
    const vHalf = THREE.MathUtils.degToRad(camera.fov / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
    camera.position.set(0, 0, R / Math.sin(Math.min(vHalf, hHalf)) * (W === H ? 0.88 : 0.72));
    sample.position.y = W === H ? 0.1 : 0.05;
    camera.updateProjectionMatrix();
    needsRender = true;
  }
  new ResizeObserver(resize).observe(container);

  // ---- interaction ------------------------------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let dragging = null, hovered = null, lastX = 0, lastY = 0, interacted = false;
  const crystalsDraggable = () => world && view === 'flow';

  function pick(ev) {
    if (!crystalsDraggable()) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(world.prisms, false)[0];
    return hit ? hit.object : null;
  }
  function setHover(obj) {
    if (hovered === obj) return;
    if (hovered) hovered.material.emissive.setHex(0x000000);
    hovered = obj;
    if (hovered) hovered.material.emissive.setHex(0x333333);
    container.style.cursor = hovered ? 'grab' : '';
    needsRender = true;
  }
  const qx = new THREE.Quaternion(), qy = new THREE.Quaternion();
  const parentQ = new THREE.Quaternion(), parentInv = new THREE.Quaternion();
  function rotate(obj, dx, dy) {
    qy.setFromAxisAngle(yAxis, dx * DRAG_SPEED);
    qx.setFromAxisAngle(xAxis, dy * DRAG_SPEED);
    qy.multiply(qx);
    if (obj === sample) {
      sample.quaternion.premultiply(qy);
      needsRender = true;
    } else {
      parentQ.copy(sample.quaternion); parentInv.copy(parentQ).invert();
      obj.quaternion.premultiply(parentQ).premultiply(qy).premultiply(parentInv);
      modified = true; texture = 'edited';
      apply();
    }
  }
  const canvas = renderer.domElement;
  canvas.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0 && ev.pointerType === 'mouse') return;
    dragging = pick(ev) || sample;
    lastX = ev.clientX; lastY = ev.clientY;
    interacted = true;
    hint.classList.add('hero-hint--hidden');
    container.style.cursor = 'grabbing';
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (dragging) { rotate(dragging, ev.clientX - lastX, ev.clientY - lastY); lastX = ev.clientX; lastY = ev.clientY; }
    else if (ev.pointerType === 'mouse') setHover(pick(ev));
  });
  const release = (ev) => {
    if (!dragging) return;
    dragging = null;
    container.style.cursor = hovered ? 'grab' : '';
    if (ev && canvas.hasPointerCapture && canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { if (!dragging) setHover(null); });

  // ---- start + render loop ------------------------------------------------------------------------
  start();
  let visible = true;
  new IntersectionObserver((entries) => { visible = entries[0].isIntersecting; }).observe(container);
  const idle = new THREE.Quaternion();
  const clock = new THREE.Clock();
  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    const t = clock.getElapsedTime();
    if (!interacted && !reduceMotion) {
      idle.setFromAxisAngle(yAxis, 0.35 * Math.sin(t * 0.45));
      sample.quaternion.copy(idle).multiply(restPose);
      needsRender = true;
    }
    stepParticles(t);
    if (needsRender) { renderer.render(scene, camera); needsRender = false; }
  }
  frame();
}

// ---- geometry helpers -----------------------------------------------------------------------

function centroid(poly) {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const cross = p[0] * q[1] - q[0] * p[1];
    a += cross; cx += (p[0] + q[0]) * cross; cy += (p[1] + q[1]) * cross;
  }
  a *= 0.5;
  return [cx / (6 * a), cy / (6 * a)];
}

// Distance from `c` to the nearest polygon edge.
function inradius(poly, c) {
  let best = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const ex = q[0] - p[0], ey = q[1] - p[1];
    const len = Math.hypot(ex, ey) || 1;
    best = Math.min(best, Math.abs(ex * (c[1] - p[1]) - ey * (c[0] - p[0])) / len);
  }
  return best;
}

// Dense Gaussian elimination with partial pivoting (a few dozen grains, solved on every drag frame).
function gauss(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const p = M[col][col] || 1e-12;
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / p;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / (M[r][r] || 1e-12);
  }
  return x;
}
