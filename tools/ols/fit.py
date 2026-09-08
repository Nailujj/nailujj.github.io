"""OLS baselines on the polycrystal dataset.

Targets:  per-grain energy (J), per-grain thermal dissipation (W/m), per-interface Kapitza dissipation (W/m).
Features: mean-field physics. 6H-SiC is transversely isotropic, so a grain's response depends on its c-axis
only; energy is quadratic in the loading (strain, temperature) and a polynomial in the c-axis components.
Split:    realization index % 5 == 0 -> test.
"""
import numpy as np, sys, time
S = sys.argv[1]
d = np.load(S + '/data.npz')
phys, ng = d['phys'], d['ng']
g_real, area, cx, cy, rv, E, Dt = (d[k] for k in ['g_real', 'g_area', 'g_cx', 'g_cy', 'g_rv', 'g_E', 'g_D'])
e_real, ei, ej, Di = (d[k] for k in ['e_real', 'e_i', 'e_j', 'e_D'])
R = len(ng)
g_off = np.concatenate([[0], np.cumsum(ng)])          # grain offset per realization

# ---- c-axis: rotate z by the rotation vector (Rodrigues) ------------------------------------
th = np.linalg.norm(rv, axis=1); ax = rv / np.maximum(th, 1e-12)[:, None]
s, c = np.sin(th), np.cos(th)
# R z = z cos + (a x z) sin + a (a.z)(1-cos);  a x z = (ay, -ax, 0)
c_axis = np.stack([ax[:, 1] * s + ax[:, 0] * ax[:, 2] * (1 - c),
                   -ax[:, 0] * s + ax[:, 1] * ax[:, 2] * (1 - c),
                   c + ax[:, 2] * ax[:, 2] * (1 - c)], axis=1)
Cx, Cy, Cz = c_axis.T

# ---- feature builders ----------------------------------------------------------------------
def quad(cols):
    """All monomials of degree <= 2 in the given columns (plus constant)."""
    out = [np.ones_like(cols[0])] + list(cols)
    for a in range(len(cols)):
        for b in range(a, len(cols)):
            out.append(cols[a] * cols[b])
    return out

def outer(A, B):
    return [a * b for a in A for b in B]

def grain_features(idx):
    p = phys[g_real[idx]]
    exx, exy, eyy, T, g1, g2 = p.T
    Tg = g1 * cx[idx] + g2 * cy[idx]                     # temperature shift at the centroid
    load = quad([exx * 1e4, exy * 1e4, eyy * 1e4, T / 50, Tg / 5e-6])   # rescaled O(1)
    x, y = Cx[idx], Cy[idx]
    orient = [np.ones_like(x), x * x, y * y, x * y, x ** 4, y ** 4, x * x * y * y, x ** 3 * y, x * y ** 3]
    A = area[idx] * 1e12
    return np.stack([A * v for v in outer(load, orient)], axis=1)

def therm_features(idx):
    p = phys[g_real[idx]]
    g1, g2 = p[:, 4], p[:, 5]
    x, y = Cx[idx], Cy[idx]
    grad = [np.ones_like(x), g1 * g1, g2 * g2, g1 * g2]
    orient = [np.ones_like(x), x * x, y * y, x * y]
    A = area[idx] * 1e12
    return np.stack([A * v for v in outer(grad, orient)], axis=1)

def edge_features(idx):
    r = e_real[idx]; a = g_off[r] + ei[idx]; b = g_off[r] + ej[idx]
    p = phys[r]; g1, g2 = p[:, 4], p[:, 5]
    dx, dy = cx[b] - cx[a], cy[b] - cy[a]
    L = np.hypot(dx, dy); nx, ny = dx / L, dy / L; tx, ty = -ny, nx
    size = (area[a] * area[b]) ** 0.25 / 1e-6                    # ~ boundary length proxy, in um
    gn, gt = g1 * nx + g2 * ny, g1 * tx + g2 * ty
    cin, cjn = Cx[a] * nx + Cy[a] * ny, Cx[b] * nx + Cy[b] * ny
    cit, cjt = Cx[a] * tx + Cy[a] * ty, Cx[b] * tx + Cy[b] * ty
    grad = [gn * gn, gt * gt, gn * gt]
    orient = [np.ones_like(gn), cin ** 2 + cjn ** 2, cit ** 2 + cjt ** 2, (cin ** 2 - cjn ** 2) ** 2,
              cin * cjn, cit * cjt, cin * cit + cjn * cjt, Cz[a] ** 2 + Cz[b] ** 2]
    f = outer(grad, orient)
    return np.stack([np.ones_like(gn), size] + [size * v for v in f] + f, axis=1)

# ---- chunked OLS ---------------------------------------------------------------------------
def fit(features, y, real_of, name, chunk=1_000_000):
    n = len(y); test = (real_of % 5 == 0)
    k = features(np.arange(min(2, n))).shape[1]
    XtX = np.zeros((k, k)); Xty = np.zeros(k)
    t0 = time.time()
    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(n, s0 + chunk)); idx = idx[~test[idx]]
        X = features(idx); XtX += X.T @ X; Xty += X.T @ y[idx]
    beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    pred = np.empty(n)
    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(n, s0 + chunk)); pred[idx] = features(idx) @ beta
    def r2(m): return 1 - np.sum((y[m] - pred[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)
    print(f'{name:28s} features {k:4d}  train R2 {r2(~test):.4f}  test R2 {r2(test):.4f}  ({time.time()-t0:.0f}s)')
    return beta, pred

bE, pE = fit(grain_features, E, g_real, 'grain energy')
bD, pD = fit(therm_features, Dt, g_real, 'grain thermal dissipation')
bI, pI = fit(edge_features, Di, e_real, 'interface dissipation')

# Back to absolute per-grain quantities and per-realization test scores (for picking a sample)
predE, predD = pE, pD
def per_real(y, pred, real_of, n_real):
    ss_res = np.bincount(real_of, (y - pred) ** 2, n_real)
    mean = np.bincount(real_of, y, n_real) / np.maximum(np.bincount(real_of, minlength=n_real), 1)
    ss_tot = np.bincount(real_of, (y - mean[real_of]) ** 2, n_real)
    return 1 - ss_res / np.maximum(ss_tot, 1e-300)
r2E, r2D, r2I = per_real(E, predE, g_real, R), per_real(Dt, predD, g_real, R), per_real(Di, pI, e_real, R)
test_r = np.arange(R) % 5 == 0
print('per-realization test-set median R2: energy %.3f, thermal %.3f, interface %.3f' %
      tuple(np.median(v[test_r]) for v in (r2E, r2D, r2I)))
# absolute-quantity test R2 (what the plots will show)
tg = test_r[g_real]; te = test_r[e_real]
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
print('absolute test R2: energy %.4f, thermal %.4f, interface %.4f' % (r2(E[tg], predE[tg]), r2(Dt[tg], predD[tg]), r2(Di[te], pI[te])))
np.savez(S + '/fit.npz', bE=bE, bD=bD, bI=bI, predE=predE, predD=predD, predI=pI, r2E=r2E, r2D=r2D, r2I=r2I, c_axis=c_axis)
