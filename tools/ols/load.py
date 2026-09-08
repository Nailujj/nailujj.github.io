"""Parse the ragged training CSVs once into flat numpy arrays (cached as .npz)."""
import numpy as np, sys, time
D = sys.argv[1].rstrip('/') + '/'      # training-data directory with the master CSVs
OUT = sys.argv[2]

def rows(name, skip=0):
    with open(D + name) as f:
        lines = f.read().split('\n')[skip:]
    return [l for l in lines if l.strip()]

t0 = time.time()
phys = np.loadtxt(D + 'physical_inputs.csv', delimiter=',', skiprows=1)
meta = np.loadtxt(D + 'realization_metadata.csv', delimiter=',', skiprows=1)
ng = meta[:, 1].astype(int)
R = len(ng)
ori = rows('grain_orientations.csv'); cen = rows('grain_centroids.csv'); edg = rows('edges.csv')
ene = rows('mechanical_energies.csv'); dis = rows('thermal_dissipation.csv'); idis = rows('interface_dissipation.csv')
r2m = rows('region_to_material.csv')
assert len(ori) == len(cen) == len(edg) == len(ene) == len(dis) == len(idis) == R, (len(ori), R)

g_real, g_area, g_cx, g_cy, g_rv, g_E, g_D = [], [], [], [], [], [], []
e_real, e_i, e_j, e_D = [], [], [], []
bad = 0
for r in range(R):
    n = ng[r]
    o = np.fromstring(ori[r], sep=','); c = np.fromstring(cen[r], sep=',').reshape(-1, 3)
    E = np.fromstring(ene[r], sep=','); Dt = np.fromstring(dis[r], sep=',')
    ed = np.fromstring(edg[r], sep=',', dtype=int).reshape(-1, 2)
    idv = np.fromstring(idis[r], sep=',').reshape(-1, 3)
    q = np.argsort(np.fromstring(r2m[r], sep=',', dtype=int))
    if not (o.size == 3 * n and len(c) == n and len(E) == n and len(Dt) == n and len(idv) == len(ed) and len(q) == n):
        bad += 1; continue
    # Energies and thermal dissipation are written in material order; centroids, edges and orientations
    # are in mesh-region order. Verified empirically: this permutation makes energy ~ area (corr 0.99).
    E = E[q]; Dt = Dt[q]
    g_real.append(np.full(n, r)); g_area.append(c[:, 0]); g_cx.append(c[:, 1]); g_cy.append(c[:, 2])
    g_rv.append(o.reshape(n, 3)); g_E.append(E); g_D.append(Dt)
    e_real.append(np.full(len(ed), r)); e_i.append(ed[:, 0]); e_j.append(ed[:, 1]); e_D.append(idv[:, 2])
print('parsed', R, 'realizations,', bad, 'malformed skipped, in %.1fs' % (time.time() - t0))
np.savez(OUT, phys=phys, ng=ng, seed=meta[:, 0],
         g_real=np.concatenate(g_real), g_area=np.concatenate(g_area), g_cx=np.concatenate(g_cx), g_cy=np.concatenate(g_cy),
         g_rv=np.concatenate(g_rv), g_E=np.concatenate(g_E), g_D=np.concatenate(g_D),
         e_real=np.concatenate(e_real), e_i=np.concatenate(e_i), e_j=np.concatenate(e_j), e_D=np.concatenate(e_D))
print('grains', sum(len(x) for x in g_E), 'edges', sum(len(x) for x in e_D))
