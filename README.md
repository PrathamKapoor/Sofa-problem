# The Moving Sofa Problem — solved (Baek 2024), reconstructed here

A complete, from-first-principles reconstruction of **Gerver's sofa**, the
solution of Leo Moser's *moving sofa problem* (1966), whose optimality was
proved by **Jineon Baek** in November 2024 (arXiv:2411.19826).

**Answer:** the maximum area is

> **2.21953166887196…** (OEIS A128463; 40-dps value
> 2.21953166887196742554628),

attained by Gerver's 1992 sofa — a bulky L-corner body with a deep dome
bite carved into its underside; at the critical 45° pose the hallway's
inner walls pass through the bite, missing the sofa by only **0.0012**.

* **SOLUTION.md** — the full mathematical write-up (problem, history,
  Romik's ODE construction, the 22-equation closure, the area integral,
  the shape, and an outline of Baek's optimality proof).
* **references.md** — verified bibliography with primary-source details.
* **notes.md** — formula-to-code map, debug trail, pitfalls, tolerances.

## Files

| file | what it is |
|---|---|
| `gerver_sofa.py` | core module: paths x(t) and A,B,C,D(t), constant re-derivation, exact-area integrals, polygon assembly, self-check battery |
| `mp_area_check.py` | independent 40-decimal-digit mpmath evaluation of the area integral |
| `verify_sofas.py` | 11-check verification battery → `verification_report.json` |
| `plot_sofa.py` / `animate_sofa.py` | figure `sofa_solution.png`; 90-frame animation `gerver_sofa.gif` |
| `bisect_y.py`, `diag_x.py` | debug scripts from the development trail (kept for traceability) |
| `results.json`, `verification_report.json` | machine-readable outputs |
| `run_gerver.log`, `coordinates.csv`, `sofa_polygon_xy.npy`, `sofa_polygon_analytic_xy.npy`, `hallway_angles.npy` | raw run artifacts |

## Quick start

```bash
python -u gerver_sofa.py      # ~5–10 min (n=3600 intersection loop dominates)
python -u mp_area_check.py
python -u verify_sofas.py
python -u plot_sofa.py
python -u animate_sofa.py
```

## Results

| computation | value | diff vs A128463 |
|---|---|---|
| Gerver/MathWorld area integral (closed-form antiderivatives) | 2.219531668871968 | +7.6e−15 |
| mpmath 40-dps evaluation | 2.21953166887196742554628 | +7.4e−15 |
| hallway intersection (eq. (8)), shapely | n=900: 2.220186231623423; n=1800: 2.219858876258156; n=3600: 2.219695244876797 | +6.55e−4 / +3.27e−4 / +1.64e−4, error O(1/n) |
| 1st-order Richardson of the intersections | 2.219531520892889 / 2.219531613495437 | −1.48e−7 / −5.54e−8 |
| analytic boundary polygon (shoelace) | n=20000: 2.219531669656383; n=40000: 2.219531669068064 | +7.84e−10 / +1.96e−10 (chord error O(1/n²)) |
| 2nd-order Richardson of the polygon | 2.219531668871958 | −1.78e−15 (float noise) |

Verification battery (`verify_sofas.py`, all PASS):

* exact endpoints and flat edges, C¹ hand-offs at machine precision;
* Table-1 constants re-derived from scratch, residual ≤ 4.4e−16;
* feasibility: 3458 sofa-frame samples inside hallway copies at 61 poses,
  0 violations;
* pinch clearance at t = π/4: **0.001238** on each side
  (Gerver's 0.0012, recast in Baek's Remark 8.4.1);
* comparison table: square 1 < half-disk π/2 < Hammersley π/2 + 2/π ≈
  2.2074160999 (A086118) < Gerver 2.21953166887196 (A128463); upper
  bounds 2√2 (Hammersley 1968), 2.37 (Kallus–Romik 2018), 2.3337
  (Leng et al. 2024).

## Key landmarks (sofa frame, unit hallway)

* rotation path: `x(0) = (0,0)`, `x'(0) = (0, 1.4206…)`, symmetry
  `x'(π/2) = −x'(0)`; five phases switching at
  `0, φ≈0.039177, θ≈0.681302, π/2−θ, π/2−φ, π/2`;
* boundary: right outer arm A, outer left swoosh C (bulges to
  x = −2.108 — *not* a flat back), niche-ceiling core `x([φ, π/2−φ])`
  with apex `x(π/4) = (−0.6138, 0.6643)`, funnel walls D and B,
  top edge y = 1, bottom strips y = 0 in `[−2.2275, −1.4206]` and
  `[0.1931, 1]`.

Environment: Python 3.12, numpy, scipy, shapely, matplotlib, mpmath.
See `notes.md` §5 for console/precision caveats.
