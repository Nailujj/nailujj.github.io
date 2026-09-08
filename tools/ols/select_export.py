"""Pick a showable test realization and export it (with reconstructed polygons) as JSON for the hero."""
import numpy as np, json, sys
S = sys.argv[1]; OUT = sys.argv[2]; PICK = int(sys.argv[3]) if len(sys.argv) > 3 else -1
# Calibration and validation of the browser-side graph conduction model; see graph_physics.py.
SCALE_INTERFACE = 0.4030
VALIDATION_NOTE = '0.89 pooled over 24 held-out realizations, 1965 boundaries'
d = np.load(S + '/data.npz'); f = np.load(S + '/fit.npz')
phys, ng, seed = d['phys'], d['ng'], d['seed']
g_real, area, cx, cy, rv, E, Dt = (d[k] for k in ['g_real', 'g_area', 'g_cx', 'g_cy', 'g_rv', 'g_E', 'g_D'])
e_real, ei, ej, Di = (d[k] for k in ['e_real', 'e_i', 'e_j', 'e_D'])
predE, predD, predI, r2E, r2D, r2I, C = (f[k] for k in ['predE', 'predD', 'predI', 'r2E', 'r2D', 'r2I', 'c_axis'])
R = len(ng); g_off = np.concatenate([[0], np.cumsum(ng)])
e_off = np.concatenate([[0], np.cumsum(np.bincount(e_real, minlength=R))])
test = np.arange(R) % 5 == 0

# ---- candidates ------------------------------------------------------------------------------
T = phys[:, 3]; gmag = np.hypot(phys[:, 4], phys[:, 5])
sel = test & (ng >= 25) & (ng <= 45)
medD, medI = np.median(r2D[sel]), np.median(r2I[sel])
cand = np.where(sel & (gmag > 0.7) & (np.abs(T) > 20) & (np.abs(r2D - medD) < 0.15) & (np.abs(r2I - medI) < 0.08) & (r2E > 0.97))[0]
rows = []
for r in cand:
    s = slice(g_off[r], g_off[r + 1])
    rows.append((r, ng[r], T[r], phys[r, 4], phys[r, 5], r2E[r], r2D[r], r2I[r], np.std(Dt[s]) / np.mean(Dt[s]), area[s].min() / area[s].max()))
rows.sort(key=lambda t: -t[8])
print(f'candidates {len(cand)} (typical per-realization R2: thermal {medD:.2f}, interface {medI:.2f})')
print('  idx  ng      T     g1     g2   r2E   r2D   r2I  cvD  Amin/Amax')
for t in rows[:12]: print('%5d %3d %6.1f %6.2f %6.2f %5.2f %5.2f %5.2f %4.2f %6.3f' % t)
r = rows[0][0] if PICK < 0 else PICK
print('chosen', r)

# ---- power-diagram reconstruction of the grain shapes -------------------------------------
s = slice(g_off[r], g_off[r + 1]); n = ng[r]
P = np.stack([cx[s], cy[s]], 1) * 1e6; At = area[s] * 1e12          # um, um^2
L = 10.0
def clip(poly, f):
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]; fa, fb = f(a), f(b)
        if fa <= 0: out.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            t = fa / (fa - fb); out.append(a + t * (b - a))
    return out
def cells(seeds, w):
    box = [np.array(p, float) for p in [(0, 0), (L, 0), (L, L), (0, L)]]
    res = []
    for i in range(n):
        poly = box
        for j in range(n):
            if j == i or not poly: continue
            dseed = seeds[j] - seeds[i]
            # power bisector: |p-si|^2 - wi <= |p-sj|^2 - wj  <=>  2 p.(sj-si) <= |sj|^2-|si|^2 + wi - wj
            rhs = seeds[j] @ seeds[j] - seeds[i] @ seeds[i] + w[i] - w[j]
            poly = clip(poly, lambda p, dseed=dseed, rhs=rhs: 2 * p @ dseed - rhs)
        res.append(np.array(poly) if poly else np.zeros((0, 2)))
    return res
def poly_area_centroid(poly):
    if len(poly) < 3: return 0.0, np.zeros(2)
    x, y = poly[:, 0], poly[:, 1]; xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y; a = cr.sum() / 2
    if abs(a) < 1e-12: return 0.0, poly.mean(0)
    return a, np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * a)
seeds = P.copy(); w = np.zeros(n)
for it in range(3000):
    cs = cells(seeds, w)
    ac = np.array([poly_area_centroid(c)[0] for c in cs]); cc = np.array([poly_area_centroid(c)[1] for c in cs])
    step = 0.35 if it < 800 else 0.15
    w += step * (At - ac)
    seeds += 0.5 * (P - cc) * (ac > 0)[:, None]        # pull cell centroids onto the true centroids
    w -= w.mean()
    if it > 200 and np.max(np.abs(ac - At) / At) < 0.01: break
err = np.abs(ac - At) / At
print('reconstruction: max area error %.1f%%, mean %.1f%%, max centroid offset %.2f um' % (100 * err.max(), 100 * err.mean(), np.linalg.norm(cc - P, axis=1).max()))

# adjacency of the reconstruction vs the dataset edge list: polygons that share a segment
es = slice(e_off[r], e_off[r + 1]); pairs = set(map(tuple, np.sort(np.stack([ei[es], ej[es]], 1), 1)))
def segset(poly):
    out = {}
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        if np.hypot(*(a - b)) > 1e-4: out[frozenset((tuple(np.round(a, 4)), tuple(np.round(b, 4))))] = (a, b)
    return out
segsets = [segset(c) for c in cs]
recon = {}
for i in range(n):
    for j in range(i + 1, n):
        common_keys = segsets[i].keys() & segsets[j].keys()
        if common_keys: recon[(i, j)] = segsets[i][next(iter(common_keys))]
common = pairs & set(recon)
print(f'dataset edges {len(pairs)}, reconstructed {len(recon)}, in common {len(common)}  ({100*len(common)/len(pairs):.0f}% of dataset edges recovered)')

# ---- export ---------------------------------------------------------------------------------
gi = np.arange(g_off[r], g_off[r + 1])
grains = []
for k, i in enumerate(gi):
    grains.append({'poly': np.round(cs[k], 3).tolist(), 'area': round(float(area[i] * 1e12), 3),
                   'rv': np.round(rv[i], 4).tolist(), 'c': np.round(C[i], 4).tolist(),
                   'E': float(E[i]), 'E_ols': float(predE[i]), 'D': float(Dt[i]), 'D_ols': float(predD[i])})
edges = []
for k in range(e_off[r], e_off[r + 1]):
    key = tuple(sorted((int(ei[k]), int(ej[k])))); seg = recon.get(key)
    edges.append({'i': key[0], 'j': key[1], 'I': float(Di[k]), 'I_ols': float(predI[k]),
                  'seg': None if seg is None else [np.round(seg[0], 3).tolist(), np.round(seg[1], 3).tolist()]})
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
tg = test[g_real]; te = test[e_real]
out = {
  'source': {'index': int(r), 'neper_seed': int(seed[r]), 'ngrains': int(n), 'domain_um': L, 'material': '6H-SiC, hexagonal, Haar-uniform orientations',
             'note': 'c-axis = R(rv) z. Polygons are a power-diagram reconstruction from centroids and areas (not the Neper tessellation).'},
  'loading': dict(zip(['exx', 'exy', 'eyy', 'T', 'g1', 'g2'], map(float, phys[r]))),
  'units': {'E': 'J', 'D': 'W/m', 'I': 'W/m', 'area': 'um^2', 'T': 'K', 'g': 'K/m'},
  'ols': {'description': 'Ordinary least squares on mean-field features: grain area x quadratic monomials of the loading x polynomial in the c-axis; interfaces additionally use the boundary normal.',
          'test_r2': {'E': r2(E[tg], predE[tg]), 'D': r2(Dt[tg], predD[tg]), 'I': r2(Di[te], predI[te])},
          'this_sample_r2': {'E': float(r2E[r]), 'D': float(r2D[r]), 'I': float(r2I[r])},
          'split': 'realization index % 5 == 0 is test; this sample is from the test set', 'n_realizations': int(R)},
  'physics': {'ka': 390.0, 'kc': 273.0, 'h_kapitza': 2.5e8,
              'scale_interface': SCALE_INTERFACE,
              'validation': VALIDATION_NOTE,
              'note': 'Constants taken from summit_driver.py. The browser solves steady conduction on this grain '
                      'graph (one temperature per grain, anisotropic bulk conductivity in series with the Kapitza '
                      'resistance) and predicts interface dissipation as q^2 / (h L), times one global constant '
                      'calibrated on other realizations.'},
  'grains': grains, 'edges': edges,
}
json.dump(out, open(OUT, 'w'), separators=(',', ':'))
print('wrote', OUT, 'bytes', len(json.dumps(out, separators=(',', ':'))))
