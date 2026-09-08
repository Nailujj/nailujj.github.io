"""Reference implementation of the browser-side solve, run on the exported sample.json.

Mirrors exactly what hero.js does, so the JS can be checked against it: same geometry source,
same conductance formula, same boundary conditions, same homogenized conductivity.
"""
import numpy as np, json, sys
d = json.load(open(sys.argv[1]))
ph, L = d['physics'], d['source']['domain_um']
KA, KC, HK = ph['ka'], ph['kc'], ph['h_kapitza']
g = np.array([d['loading']['g1'], d['loading']['g2']])
um = 1e-6

def centroid(poly):
    p = np.asarray(poly); x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1); cr = x * yn - xn * y; a = cr.sum() / 2
    return np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * a)

polys = [np.array(gr['poly']) for gr in d['grains']]
cents = np.array([centroid(p) for p in polys]) * um
C = np.array([gr['c'] for gr in d['grains']])
kept = [e for e in d['edges'] if e['seg']]
n = len(polys)

def kof(c, nv):
    cn = c[0] * nv[0] + c[1] * nv[1]
    return KA + (KC - KA) * cn * cn

def solve(C):
    A = np.zeros((n, n)); rhs = np.zeros(n); G = []
    for e in kept:
        a = np.array(e['seg'][0]) * um; b = np.array(e['seg'][1]) * um
        ev = b - a; ln = np.hypot(*ev); nv = np.array([-ev[1] / ln, ev[0] / ln]); mid = (a + b) / 2
        di = max(abs((mid - cents[e['i']]) @ nv), 1e-11); dj = max(abs((mid - cents[e['j']]) @ nv), 1e-11)
        Ge = ln / (di / kof(C[e['i']], nv) + dj / kof(C[e['j']], nv) + 1 / HK)
        G.append(Ge)
        A[e['i'], e['i']] += Ge; A[e['i'], e['j']] -= Ge
        A[e['j'], e['j']] += Ge; A[e['j'], e['i']] -= Ge
    outer = []
    for i, p in enumerate(polys):
        for k in range(len(p)):
            a, b = p[k], p[(k + 1) % len(p)]
            m = (a + b) / 2
            if min(m[0], m[1], L - m[0], L - m[1]) > 2e-3: continue
            ev = (b - a) * um; ln = np.hypot(*ev)
            if ln < 1e-12: continue
            nv = np.array([-ev[1] / ln, ev[0] / ln]); mid = m * um
            di = max(abs((mid - cents[i]) @ nv), 1e-11)
            Go = ln / (di / kof(C[i], nv))
            A[i, i] += Go; rhs[i] += Go * (g @ mid)
            outer.append((i, Go, mid))
    T = np.linalg.solve(A, rhs)
    q = np.array([G[k] * (T[e['i']] - T[e['j']]) for k, e in enumerate(kept)])
    Ipred = np.array([ph['scale_interface'] * q[k] ** 2 / (HK * np.hypot(*((np.array(e['seg'][1]) - np.array(e['seg'][0])) * um)))
                      for k, e in enumerate(kept)])
    Qin = sum(Go * ((g @ mid) - T[i]) for (i, Go, mid) in outer if Go * ((g @ mid) - T[i]) > 0)
    kEff = Qin / (np.linalg.norm(g) * L * um)
    return T, q, Ipred, kEff

I_fe = np.array([e['I'] for e in kept])
T, q, Ipred, kEff = solve(C)
r2 = 1 - np.sum((I_fe - Ipred) ** 2) / np.sum((I_fe - I_fe.mean()) ** 2)
print('grains %d  boundaries %d' % (n, len(kept)))
print('k_eff        %.1f W/(m K)' % kEff)
print('sum I graph  %.4e   FE %.4e   ratio %.3f' % (Ipred.sum(), I_fe.sum(), Ipred.sum() / I_fe.sum()))
print('per-boundary R2 vs FE  %.3f   Pearson %.3f' % (r2, np.corrcoef(I_fe, Ipred)[0, 1]))
print('T range %.3e K  (imposed drop over the box %.3e K)' % (T.max() - T.min(), np.linalg.norm(g) * L * um))

# sensitivity: rotate one grain's c-axis by 90 degrees about z
for gi in [0, 5, 20]:
    C2 = C.copy()
    c = C2[gi]; C2[gi] = np.array([-c[1], c[0], c[2]])
    _, _, Ip2, k2 = solve(C2)
    print('rotate grain %2d in-plane by 90 deg -> k_eff %.1f (%+.2f%%), sum I %+.2f%%'
          % (gi, k2, 100 * (k2 / kEff - 1), 100 * (Ip2.sum() / Ipred.sum() - 1)))

# Extreme textures: every grain given the same c-axis (a single-crystal limit).
gh = g / np.linalg.norm(g)
for name, c in [('c parallel to grad T', [gh[0], gh[1], 0]), ('c perpendicular to grad T', [-gh[1], gh[0], 0]), ('c normal to the slab', [0, 0, 1])]:
    C3 = np.tile(np.array(c, float), (n, 1))
    _, _, Ip3, k3 = solve(C3)
    print('%-28s k_eff %6.1f   sum I %.3e' % (name, k3, Ip3.sum()))
