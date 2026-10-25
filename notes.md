# Implementation notes, traceability, debugging trail

This file documents how every formula in `gerver_sofa.py` connects to a
primary source, the numerical pitfalls observed, and the bugs that were
found and fixed while building it.

## 1. Formula-to-code map

| math object (Romik arXiv:1606.08111) | code |
|---|---|
| hallway `L_horiz`/`L_vert`/`L`, `R_t` CCW (sec. 2) | docstring; `hallway_copy()` |
| sofa `S_x = L_horiz ∩ ⋂ (x(t)+R_t L) ∩ …` (eq. (8)) | `sofa_by_intersection()` |
| contact paths A,B,C,D (Thm 1, eqs. (9)–(12)) | `contact("A"/"B"/"C"/"D", t)` |
| rotation path `x = R_t p + κ_j`, `x' = R_t(p' + Jp)`, `x'' = R_t(2Jp' + p'' − p)` (eq. (12) of Thm 1 / sec. 3) | `x(t, der)` |
| 5-phase breaks `0, φ, θ, π/2−θ, π/2−φ, π/2` (eqs. (24)–(25), Table 1) | `PHI, THETA, PHASE_EDGES, _phase_index()` |
| SOL1–SOL5 piecewise `p_j(t)`, `p'_j`, `p''_j` (Thm 3, eqs. (14)-(16) family) | `_P`, `_dP`, `_ddP` |
| symmetry identities (27)–(34); κ table; endpoints A(0)=(1,0) (23), x'(π/2) (26) | constants block; `main()` checks [0]–[1] |
| gluing/closure equations (43)–(44): `x(φ) = B(π/2−θ)`, `x(π/2−φ) = D(θ)` | `main()` check [3]; `sofa_polygon()` assembly |
| [MathWorld "Gerver Sofa" eqs. (1)–(4)] 4×4 system for A,B,φ,θ | `gerver_constants()` |
| [MathWorld eqs. (5)–(17)] area integral with r(α), y₁, y₂ | `_r`, `_u_du`, `_r_coeffs`, `_Fprime`, `_running_int`, `gerver_area()` |

The identity checks in `main()` [0]–[4] re-derive the Table-1 constants from
the equations (residuals ≤ 4.4e−16), so the module is *not* just hard-coded
numbers: the constants `PHI, THETA, A1..E2, KP` are confirmed by re-solving
the constraints numerically.

## 2. Bugs found and fixed (with evidence trail)

1. **Sign error in `_dP` phases 1 and 5** (the big one).
   `p₂(t) = −A₂cos t + A₁sin t − ½` (with A₂ = −¼), so differentiation gives
   `p₂'(t) = +A₂ sin t + A₁ cos t`. An earlier version had
   `−A₂ sin t + A₁ cos t`, which corrupted x'(t) and x''(t) on phases 1/5.
   Symptoms that traced it down:
   * spurious "0.5-scale jumps" in every quantity at the phase edges;
   * the symmetry identity (26) `x'(π/2) = −x'(0)` failed with O(1) error;
   * contact endpoints did not match Romik's published example.
   `diag_x.py` (plots/diffs of x' across phases) and `bisect_y.py`
   (bisecting the y-piece mismatch) were the diagnostic scripts; both are
   kept as an evidence trail.

2. **Dense-grid quadrature across the jump of r(α).**
   The MathWorld/Gerver area integrand has a jump discontinuity at
   `α = π/2 − θ` (r drops by ≈ 0.034). A uniform grid straddling the jump
   introduced ≈ 1.6e−7 error into the `y₁`/`y₂` pieces. Fix: exact
   closed-form antiderivatives (`_r_coeffs`, `_Fprime`, `_running_int`)
   valid on each interval; the closed form also encodes the identity
   `y₁(π/2−φ) = 0` as a built-in sanity gate. No quadrature grid is needed
   any more.

3. **Intersection error is first order in 1/n.**
   Measured order ≈ 1.00 at n = 900/1800/3600, so the correct refinement is
   the *first-order* Richardson estimate `2·a(2n) − a(n)` at the
   intersections (−1.48e−7 at (900,1800); −5.54e−8 at (1800,3600)),
   **not** the second-order `(4a₄−a₂)/3` formula, which assumes a
   tangential-quadrature error model that does not apply here. *Caveat:*
   because the intersection error is only O(1/n), the Richardson digits
   are trustworthy to about 7 digits — the high-precision value comes
   from the analytic-area integrals, not from extrapolating the
   intersection.

4. **Printed typo inside Romik's paper.** The rows of the printed
   Hammersley-example `C^(r)`, `D^(r)` columns are swapped relative to his
   own formulas (11)/(12). Following the formulas (not the pasted example
   rows) reproduces the correct geometry (D the inner-wall contact,
   C the outer), and all other identities of the paper confirm the formulas.

5. **[V5] half-line side test sign error.** The first pinch-distance table
   read ~0.863 for the "parallel-side" rows, which is the *perpendicular*
   distance from the inner walls' base lines — wrong pairing. Two issues:
   (a) `halfline_dist(P, base, u)` tested `(P−base)·u ≤ 0` for "point is
   on the arm"; an inverted sign turned every reading into a perpendicular
   distance to the base line instead of a distance along the arm toward
   the sofa. (b) even with the sign right, the B-arm must be paired with
   `x(φ)` and the D-arm with `x(π/2−φ)` (cross-pairing them hides the
   pinch because it compares the wrong corner of the niche). With the
   corrected test the readings are 0.001238 on both sides at t = π/4,
   matching Gerver's/Baek's 0.0012 exactly in the first four digits.

6. **[V6] frame mismatch (sofa frame vs hallway frame).** `pose()` returns
   sofa placements *in the hallway frame*, while `hallway_copy(t)` returns
   hallway copies *in the sofa frame*. The first containment test fed
   sofa-frame samples to posed copies; at t = 0 the two frames coincide,
   so the test "half-worked" (74 violations at t≈0 scale, 2647 later) —
   a failure pattern that scaled with t, the tell that the frames were
   rotated against each other. Fix: test sofa-frame samples directly
   against `hallway_copy(t)` (no pose needed); 61 poses, 3458 samples,
   0 violations.

7. **[7] Polygon validity: glue-endpoint noise, not just slivers.** The
   first analytic polygon contained dead edges and still failed GEOS
   validation. Two distinct causes, fixed in `sofa_polygon()`:
   (a) *slivers* — consecutive vertices at the glue hand-offs differing by
   ~1e−16; handled by dropping edges shorter than 1e−9. (b) *the real
   culprit* — the junction constants evaluate with ~1e−17 y-noise
   (`contact("A",φ)` returned y = −2.08e−17, `contact("B",π/2)` returned
   y = −3.15e−17), which poked the ring *below* the exact bottom flats
   y = 0; near (1,0) the first A segment then overlapped the ring closure
   and GEOS reported `Self-intersection[1 -3.19e-35]`. Fix: snap the six
   flat-edge junction constants (A0, A5, C0, C5, D0, B5) to exact
   ordinates y ∈ {0, 1}, overwrite the curve pieces' endpoint rows with
   those snapped constants, and append the explicit closure vertex only
   when the ring does not already end at A(0) (a repeated vertex would
   itself read as a self-intersection). After the fix: `is_valid` True,
   99999 vertices, first == last == (1, 0), y bounds exactly {0, 1}, and
   all [7] area numbers unchanged (2.219531669656383 / 2.219531669068064 /
   Richardson 2.219531668871958). Lesson: *geometric* identities that hold
   to machine epsilon at junctions must still be enforced bit-for-bit when
   those junctions sit exactly on an axis-aligned flat; tiny noise is
   harmless mid-curve but lethal on flats.

## 3. Geometry landmarks (all verified in `run_gerver.log`)

* `x(0) = (0,0)`; `x'(0) = (0, 1.420644844145377)`; `x(π/2) = (−1.227526458860503, 0)`;
  `x'(π/2) = (0, −1.420644844145377)` — symmetry (26) holds.
* Contact endpoints: `A(0) = (1,0)`; `C(π/2) = (−2.2275…, 0)`;
  `C(0) − D(0) = (0,1)` (the two *back corners* are vertically aligned at
  x = −1.420644844… — an interior alignment, **not** a boundary edge);
  `B(π/2) = (0.1931…, 0)`; glue hand-offs (43)/(44) at machine precision.
* The flat edges of the sofa: top `y=1` from `C(0)` to `A(π/2)`, and the
  two bottom strips `y=0` spanning `x ∈ [−2.2275, −1.4206]` and
  `x ∈ [0.1931, 1]`. There is **no flat back**: the C-swoosh bows out to
  `x = −2.108` at mid-height `y = ½`. Everything else on the boundary is
  curvature: A (right outer), D and B (the two funnel walls), the niche
  ceiling `x([φ, π/2−φ])`, C (outer left).
* The niche is a **deep dome bite open to the floor between
  D(0) = (−1.4206, 0) and B(π/2) = (0.1931, 0)**: its ceiling starts at
  height 0.05519 at both ends — `x(φ) = (−0.00293, 0.05519)` and
  `x(π/2−φ) = (−1.2246, 0.05519)` — and bulges *upward*, peaking at the
  apex `x(π/4) = (−0.6138, 0.6643)`; the cavity covers most of the
  footprint. The funnel curves D and B flare from the ceiling ends down
  to the floor. At the t = π/4 pose the two inner walls of the hallway
  squeeze through the dome, coming within **0.0012** of the ceiling
  endpoints `x(φ)`, `x(π/2−φ)` (Gerver's closing remark; recast in Baek,
  Remark 8.4.1) — checked in `verify_sofas.py` [V5] as 0.001238.
* A is frozen (constant) on `[0, φ]`, C on `[π/2−φ, π/2]` — so those
  curvature pieces degenerate to the corner points `(1,0)` and `(−2.2275…,0)`
  respectively; `sofa_polygon()` skips the degenerate spans.

## 4. Numerical tolerances / what is exact vs. approximate

* Constants, glue identities, endpoints: machine-precision (≤ 5e−16).
* Analytic derivative checks vs. central differences: ~1e−10 (fd error
  itself ~ε²).
* Area integral (closed-form antiderivatives): agrees with OEIS A128463 to
  +7.6e−15; independent 40-dps mpmath computation
  2.21953166887196742554628, i.e. +7.4e−15 vs the 14-digit truncation
  (float noise level, ~9.4e−15 below the 40-dps constant).
* Hallway-intersection areas: converges at O(1/n); at n=3600 the error is
  1.64e−4 — this is *evidence of shape agreement*, not a precision claim;
  the exact shape definition is enforced analytically by the boundary
  polygon and the area integral.
* Boundary polygon (n = 20000/40000 per curve): chord error O(1/n²)
  (measured ratio 4.0 ⇒ the (4a₄−a₂)/3 Richardson is the right
  extrapolation here); 2nd-order Richardson gives 2.219531668871958,
  −1.78e−15 vs the 14-digit reference — float noise.
* Max gap between the analytic boundary and the (independent) n=3600
  intersection boundary: **1.8e−4** — expected and understood: the
  n=3600 copy algebra itself has only ~1.64e−4 area accuracy and its
  boundary is discretized to a (coarser) polygon; the gap matches that
  sampling slack. Not an error in the analytic construction.

## 5. Environment notes

* Python 3.12.10; numpy 2.5.4; scipy 1.18.1; shapely 2.2.0; matplotlib;
  mpmath 1.4.1. Run scripts with `python -u`.
* numpy 2.x prints `np.float64(...)` reprs; cast to plain `float` before
  feeding values into `mpmath.mpf(`.
* The console is cp1252: keep scripts' `print` output ASCII (no π, °, ×);
  unicode is fine in file contents and matplotlib text.
* The intersection loop (`sofa_by_intersection`, 900+1800+3600 shapely
  intersections) takes several minutes; don't raise n beyond 3600 without
  optimizing `hallway_copy` (each copy is a union of two big rotated
  rectangles, buffered by 1e−12 so the intersection never collapses).
* The paper-typo in Romik's printed example (item 2.4 above) means any
  future re-derivation should validate `contact()` against the *formulas*
  section and against published constants, not against his sample table.

## 6. Regeneration

```bash
python -u gerver_sofa.py        # ~5-10 min  -> run_gerver.log, results.json,
                                #             *.npy, coordinates.csv
python -u mp_area_check.py
python -u verify_sofas.py       # -> verification_report.json
python -u plot_sofa.py          # -> sofa_solution.png
python -u animate_sofa.py       # -> gerver_sofa.gif
```
