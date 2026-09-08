"""Does a one-temperature-per-grain graph solve reproduce the FE thermal results?

Physics (exactly the constants summit_driver.py hands the solver):
  anisotropic 6H-SiC conduction, k = ka I + (kc - ka) c c^T,  ka = 390, kc = 273 W/(m K)
  Kapitza interface conductance h = 2.5e8 W/(m^2 K) on every grain boundary
  imposed boundary temperature T(x) = T0 + g . x  on the 10 um square

Graph model: one unknown temperature per grain; edge conductance = bulk half-resistances of the
two grains in series with the Kapitza resistance, all times the boundary length.

Predicted outputs, up to one global constant each (fitted on training realizations):
  interface dissipation ~ L h (dT)^2       (jump across the boundary)
  grain dissipation     ~ A (grad T . k . grad T)   with grad T from a least-squares fit
"""
import numpy as np, sys, time
S = sys.argv[1]; NREAL = int(sys.argv[2]) if len(sys.argv) > 2 else 40
KA, KC, HK, L_DOM = 390.0, 273.0, 2.5e8, 1e-5

d = np.load(S + '/data.npz'); f = np.load(S + '/fit.npz')
phys, ng = d['phys'], d['ng']
g_real, area, cx, cy, Dt = (d[k] for k in ['g_real', 'g_area', 'g_cx', 'g_cy', 'g_D'])
e_real, ei, ej, Di = (d[k] for k in ['e_real', 'e_i', 'e_j', 'e_D'])
C = f['c_axis']
R = len(ng); g_off = np.concatenate([[0], np.cumsum(ng)])
e_off = np.concatenate([[0], np.cumsum(np.bincount(e_real, minlength=R))])

# ---- power-diagram reconstruction ------------------------------------------------------------
def reconstruct(P, At, iters=2500):
    """Cells whose areas match At and whose centroids match P, on the [0,L_DOM]^2 square."""
    n = len(P); L = L_DOM
    box = [np.array(p, float) for p in [(0, 0), (L, 0), (L, L), (0, L)]]
    seeds = P.copy(); w = np.zeros(n); cs = None
    def clip(poly, nrm, rhs):
        out = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            fa, fb = 2 * a @ nrm - rhs, 2 * b @ nrm - rhs
            if fa <= 0: out.append(a)
            if (fa < 0 < fb) or (fb < 0 < fa):
                out.append(a + fa / (fa - fb) * (b - a))
        return out
    def area_centroid(poly):
        if len(poly) < 3: return 0.0, np.zeros(2)
        p = np.asarray(poly); x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1); cr = x * yn - xn * y; a = cr.sum() / 2
        if abs(a) < 1e-30: return 0.0, p.mean(0)
        return a, np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * a)
    nb = [np.array([j for j in range(n) if j != i]) for i in range(n)]
    for it in range(iters):
        cs = []
        for i in range(n):
            poly = box
            for j in nb[i]:
                if not poly: break
                poly = clip(poly, seeds[j] - seeds[i], seeds[j] @ seeds[j] - seeds[i] @ seeds[i] + w[i] - w[j])
            cs.append(np.array(poly) if poly else np.zeros((0, 2)))
        ac = np.array([area_centroid(c)[0] for c in cs]); cc = np.array([area_centroid(c)[1] for c in cs])
        step = 0.35 if it < 800 else 0.15
        w += step * (At - ac)
        seeds += 0.5 * (P - cc) * (ac > 0)[:, None]
        w -= w.mean()
        if it > 200 and np.max(np.abs(ac - At) / At) < 0.01: break
    return cs, np.max(np.abs(ac - At) / At)

def facets(cs):
    """Shared segments between cells: {(i,j): (a, b)}."""
    def segset(poly):
        out = {}
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            if np.hypot(*(a - b)) > 1e-12: out[frozenset((tuple(np.round(a, 12)), tuple(np.round(b, 12))))] = (a, b)
        return out
    ss = [segset(c) for c in cs]; out = {}
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            common = ss[i].keys() & ss[j].keys()
            if common: out[(i, j)] = ss[i][next(iter(common))]
    return out

def outer_segments(cs):
    """Polygon segments lying on the domain boundary: list of (cell, a, b)."""
    out = []
    for i, poly in enumerate(cs):
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            m = (a + b) / 2
            on = min(m[0], m[1], L_DOM - m[0], L_DOM - m[1])
            if on < 1e-9 and np.hypot(*(a - b)) > 1e-12: out.append((i, a, b))
    return out

# ---- graph conduction solve --------------------------------------------------------------
def kn(c2d, nvec):
    """Conductivity along `nvec` for a grain whose c-axis is `c` (3D, in-plane part used)."""
    cn = c2d[0] * nvec[0] + c2d[1] * nvec[1]
    return KA + (KC - KA) * cn * cn

def solve(cs, fac, outer, Cg, g):
    n = len(cs); A = np.zeros((n, n)); rhs = np.zeros(n)
    cents = np.array([polycentroid(c) for c in cs])
    edata = []
    for (i, j), (a, b) in fac.items():
        seg = b - a; L = np.hypot(*seg); t = seg / L; nvec = np.array([-t[1], t[0]])
        mid = (a + b) / 2
        di = max(np.abs((mid - cents[i]) @ nvec), 1e-9); dj = max(np.abs((mid - cents[j]) @ nvec), 1e-9)
        ki, kj = kn(Cg[i], nvec), kn(Cg[j], nvec)
        G = L / (di / ki + dj / kj + 1.0 / HK)                     # W/K per unit depth
        A[i, i] += G; A[i, j] -= G; A[j, j] += G; A[j, i] -= G
        edata.append((i, j, L, G, nvec, mid))
    for i, a, b in outer:
        seg = b - a; L = np.hypot(*seg); t = seg / L; nvec = np.array([-t[1], t[0]])
        mid = (a + b) / 2
        di = max(np.abs((mid - cents[i]) @ nvec), 1e-9)
        G = L / (di / kn(Cg[i], nvec))                              # Dirichlet on the domain edge
        A[i, i] += G; rhs[i] += G * (g[0] * mid[0] + g[1] * mid[1])
    T = np.linalg.solve(A, rhs)
    return T, edata, cents

def polycentroid(poly):
    p = np.asarray(poly)
    if len(p) < 3: return p.mean(0) if len(p) else np.zeros(2)
    x, y = p[:, 0], p[:, 1]; xn, yn = np.roll(x, -1), np.roll(y, -1); cr = x * yn - xn * y; a = cr.sum() / 2
    return np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * a)

def predict(cs, fac, outer, Cg, g, At):
    T, edata, cents = solve(cs, fac, outer, Cg, g)
    n = len(cs)
    # interface: heat flux q = G dT (W per unit depth); Kapitza jump dT_K = q / (h L); D ~ L h dT_K^2 = q^2/(h L)
    Dint = {}
    for (i, j, L, G, nvec, mid) in edata:
        q = G * (T[i] - T[j])
        Dint[(i, j)] = q * q / (HK * L)
    # grain: least-squares gradient from neighbours, then A * grad.k.grad
    num = np.zeros((n, 2, 2)); vec = np.zeros((n, 2))
    for (i, j, L, G, nvec, mid) in edata:
        dv = cents[j] - cents[i]; dT = T[j] - T[i]
        wgt = L
        num[i] += wgt * np.outer(dv, dv); vec[i] += wgt * dv * dT
        num[j] += wgt * np.outer(dv, dv); vec[j] += wgt * dv * dT
    Dg = np.zeros(n)
    for i in range(n):
        try: gr = np.linalg.solve(num[i] + 1e-30 * np.eye(2), vec[i])
        except np.linalg.LinAlgError: gr = np.zeros(2)
        c = Cg[i]; kk = KA * np.eye(2) + (KC - KA) * np.outer(c[:2], c[:2])
        Dg[i] = At[i] * (gr @ kk @ gr)
    return Dint, Dg, T

# ---- run over realizations ------------------------------------------------------------------
sel = np.where((ng >= 25) & (ng <= 45))[0]
rng = np.random.default_rng(0); pick = rng.choice(sel, size=min(NREAL, len(sel)), replace=False)
rows_i, rows_g = [], []
t0 = time.time()
for cnt, r in enumerate(pick):
    s = slice(g_off[r], g_off[r + 1]); n = ng[r]
    P = np.stack([cx[s], cy[s]], 1); At = area[s]
    cs, err = reconstruct(P, At)
    if err > 0.05: print('  skip', r, 'area err %.2f' % err); continue
    fac = facets(cs); outer = outer_segments(cs)
    ds_pairs = set(map(tuple, np.sort(np.stack([ei[e_off[r]:e_off[r+1]], ej[e_off[r]:e_off[r+1]]], 1), 1)))
    if len(ds_pairs & set(fac)) < 0.9 * len(ds_pairs):
        print('  skip', r, 'edges %d/%d' % (len(ds_pairs & set(fac)), len(ds_pairs))); continue
    g = phys[r, 4:6]
    Dint, Dg, T = predict(cs, fac, outer, C[s], g, At)
    for k in range(e_off[r], e_off[r + 1]):
        key = tuple(sorted((int(ei[k]), int(ej[k]))))
        if key in Dint: rows_i.append((r, Di[k], Dint[key]))
    for k in range(n): rows_g.append((r, Dt[g_off[r] + k], Dg[k]))
    if cnt % 10 == 0: print('  %d/%d realizations (%.0fs)' % (cnt + 1, len(pick), time.time() - t0), flush=True)
rows_i = np.array(rows_i); rows_g = np.array(rows_g)
np.savez(S + '/graphphys.npz', rows_i=rows_i, rows_g=rows_g)

def report(rows, name):
    real, y, p = rows[:, 0], rows[:, 1], rows[:, 2]
    a = np.sum(y * p) / np.sum(p * p)                  # single global scale
    pred = a * p
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    cors = [np.corrcoef(y[real == r], p[real == r])[0, 1] for r in np.unique(real) if (real == r).sum() > 5]
    print(f'{name:24s} n={len(y):6d}  global-scale R2 {r2:.3f}  Pearson {np.corrcoef(y, p)[0,1]:.3f}  median within-realization corr {np.nanmedian(cors):.3f}  scale {a:.4g}')
report(rows_i, 'interface dissipation'); report(rows_g, 'grain thermal dissipation')
