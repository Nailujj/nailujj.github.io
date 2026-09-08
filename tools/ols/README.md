# Baselines and the hero sample

Regenerates `assets/data/sample.json` from the training-data CSVs (the dataset is not in this repo).

    python3 load.py <training_data_dir> work/data.npz   # parse the ragged CSVs once (~15 s)
    python3 fit.py work                                  # mean-field OLS: energy, thermal, interface
    python3 graph_physics.py work 40                     # graph conduction model vs the finite element
    python3 select_export.py work ../../assets/data/sample.json 42250
    python3 check_json.py ../../assets/data/sample.json  # reference for the browser-side solve

## Row alignment (important)

`mechanical_energies.csv` and `thermal_dissipation.csv` are written in **material** order, while
centroids, edges and orientations are in **mesh-region** order. `load.py` applies
`argsort(region_to_material)` to line them up. Without it, within a realization the correlation
between grain energy and grain area is 0.00; with it, 0.99. `data_loader.py` in the dataset returns
`region_to_material` but never applies it.

## What each script does

- `load.py` — flattens the ragged per-realization CSVs into arrays, applying the permutation above.
- `fit.py` — ordinary least squares on mean-field features (grain area x quadratic loading monomials
  x polynomial in the c-axis; interfaces also use the boundary normal). Test split is
  `realization index % 5 == 0`.
- `graph_physics.py` — the reduced model the hero runs in the browser: one temperature per grain,
  anisotropic 6H-SiC conduction (ka = 390, kc = 273 W/(m K)) in series with the Kapitza conductance
  (2.5e8 W/(m^2 K)) from `summit_driver.py`, Dirichlet boundary temperature `T = g . x`. Predicts
  interface dissipation as `q^2 / (h L)` and grain dissipation as `A grad(T) . k . grad(T)`, each up
  to one global constant fitted across realizations. Prints the R2 against the finite element.
- `select_export.py` — picks a test realization and writes the JSON the page loads. Grain polygons
  are a power-diagram reconstruction from centroids and areas (areas within 1%, every dataset edge
  recovered), not the original Neper tessellation.
- `check_json.py` — recomputes the browser solve in NumPy from the exported JSON, so the JavaScript
  can be checked number for number.

The reconstruction does not always converge; `graph_physics.py` skips realizations whose areas are
off by more than 5%, so its statistics cover the ones that reconstruct cleanly.
