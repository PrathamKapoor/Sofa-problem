# The Moving Sofa Problem — solved

**Answer (brief).** The planar shape of maximal area that can move around a
right-angled unit-width corner is **Gerver's sofa** (Joseph L. Gerver, 1992),
with area

> **2.21953166887196...** (OEIS A128463)

Its optimality was proved by **Jineon Baek**, *"Optimality of Gerver's Sofa"*,
arXiv:2411.19826 (Nov 2024, 119 pages), resolving the problem Leo Moser posed
in 1966. This folder contains an exact reconstruction of Gerver's sofa built
from Dan Romik's ODE theory, three independent verifications of its area, a
verification battery (including the famous 0.0012 clearance "pinch"), figures,
an animation, and a full write-up.

## Files

| file | what it is |
|---|---|
| `gerver_sofa.py` | core module: rotation path `x(t)` (Romik SOL1–SOL5), contact paths `A,B,C,D`, boundary polygon `sofa_polygon()`, area integral, hallway-intersection computation, `main()` cross-check battery |
| `run_gerver.log` | complete output of the verification run (`python -u gerver_sofa.py`) |
| `results.json` | machine-readable summary of the run |
| `mp_area_check.py` | independent 40-decimal-digit (mpmath) computation of the area integral |
| `sofa_polygon_xy.npy` | n=3600 hallway-intersection polygon (shapely, eq. 8) |
| `sofa_polygon_analytic_xy.npy` | analytic boundary polygon from the contact curves |
| `hallway_angles.npy` | the angle grid used by the intersection computation |
| `coordinates.csv` | boundary coordinates of the sofa (for CAD/printing) |
| `verify_sofas.py` → `verification_report.json` | checks: endpoints, frozen contacts, hand-offs, Baek 0.0012 pinch clearance, pose-containment sampling, comparison table |
| `plot_sofa.py` → `sofa_solution.png` | 3-panel figure (shape + pinch inset, five hallway poses, area comparison) |
| `animate_sofa.py` → `gerver_sofa.gif` | animation of the sofa passing the corner |
| `SOLUTION.md` | the full solution: statement, history, bounds, construction from first principles, optimality proof outline, verification results |
| `references.md` | full bibliography |
| `notes.md` | implementation notes, pitfalls, traceability |
| `bisect_y.py`, `diag_x.py` | debug scripts kept as an evidence trail (see notes.md) |

## Quickstart

```bash
python -u gerver_sofa.py      # full verification battery (~5-10 min; the
                              # n=3600 shapely intersection loop dominates)
python -u mp_area_check.py    # 40-decimal-digit mpmath cross-check
python -u verify_sofas.py     # battery -> verification_report.json
python -u plot_sofa.py        # figure under a minute
python -u animate_sofa.py     # GIF, under a minute
```

Requires Python 3.12+ with `numpy`, `scipy`, `matplotlib`, `shapely`
(`mpmath` only for `mp_area_check.py`).

## Conventions

Hallway width 1. In the sofa's frame (Romik, §2 of arXiv:1606.08111):

* `L_horiz = {(x,y): x <= 1, 0 <= y <= 1}`, `L_vert = {(x,y): 0 <= x <= 1, y <= 1}`,
  the hallway is `L = L_horiz ∪ L_vert`;
* the shape at angle `t` must fit inside the copy `x(t) + R_t L`, where `R_t`
  is CCW rotation and `x: [0, pi/2] -> R^2` is the *rotation path* traced by
  the hallway's inner ("reflex") corner in the sofa frame, `x(0) = (0,0)`;
* the answer is  `S_x = L_horiz ∩ ⋂_t (x(t)+R_t L) ∩ (x(π/2)+R_{π/2} L_vert)`
  (eq. (8) of Romik's paper) — exactly what `sofa_by_intersection()` computes;
* the boundary of `S_x` is: the contact curves `A, B, C, D` (Romik Thm 1) on
  their domains, the ceiling `x([φ, π/2−φ])` of the niche, and three straight
  edges — assembled by `sofa_polygon()`.

## Results at a glance

| computation | area | diff vs A128463 |
|---|---|---|
| Gerver's analytic integral (eq. 17), re-solved constants | 2.219531668871968 | +7.5e−15 |
| hallway intersection n=900 / 1800 / 3600 (shapely) | converges, error ∝ 1/n | ≤ 1.6e−4 |
| first-order Richardson of the above | 2.21953152089… / 2.21953161349… | −1.5e−7 / −5.5e−8 |
| analytic boundary polygon (n=20000/40000, shoelace) | 2.21953166887… | ~1e−9 |
| mpmath 40-decimal evaluation of the integral | 2.21953166887196742554628 | +7.4e−15 |
</think><tool_call>write<arg_key>content</arg_key><arg_value># References — the Moving Sofa Problem

**Primary sources used to build and verify this folder's results** are marked
★ (all formulas/numbers here were checked against them directly).

## Problem statement and early history

1. **Leo Moser**, "Problem 66-11", *SIAM Review* **8**(3):381, 1966. — The
   posing of the problem (unbounded sofa? then the bounded-area version).
2. **J. H. Hammersley**, "On the enfeeblement of mathematical skills by
   'modern mathematics' and by similar soft intellectual trash in schools
   and universities", *Bulletin of the Institute of Mathematics and its
   Applications* **4**:66–85, 1968. — Contains the upper bound 2√2 and the
   half-disk sofa of area π/2 + 2/π ≈ 2.2074160999 (OEIS A086118). The same
   lower bound is sometimes credited to Logan (1976) in Ian Stewart's
   account.
3. **N. R. Wagner**, "The sofa problem", *American Mathematical Monthly*
   **83**(3):188–189, 1976. — A short formulation/overview of the problem.
4. **K. Maruyama** (1973) — early computer search producing a near-Gerver
   shape (cited via Wikipedia and D. Romik's moving-sofa page).
5. **P. Gibbs** (2014) — numerical studies of the sofa and the
   ambidextrous variant (cited via D. Romik's page).

## Gerver's construction

6. ★ **Joseph L. Gerver**, "On moving a sofa around a corner",
   *Geometriae Dedicata* **42**(3):267–283, 1992. — The 18-curve-section
   sofa, area 2.219531... (OEIS A128463); Theorem 1 (an optimal sofa can be
   assumed monotone/symmetric), whose logical gap Baek's proof closes.
7. **M. Finch** (2003), "A geometric analysis of Gerver's couch" —
   first-principles rederivation of Gerver's constants and area (source of
   the MathWorld presentation we cross-checked, eqs. (1)–(17)).

## Exact ODE theory (the basis of this folder's reconstruction)

8. ★ **Dan Romik**, "Differential equations and exact solutions in the
   moving sofa problem", *Experimental Mathematics* **27**(3):316–330, 2018;
   arXiv:1606.08111 (math.DG), DOI 10.48550/arXiv.1606.08111. — Theorems 1–3:
   contact paths (eqs. (9)–(12)), the rotation-path ODE, solutions SOL1–SOL5,
   and the gluing equations (27)–(44). Companion Mathematica package *MovingSofas.nb* on
   Romik's page: https://www.math.ucdavis.edu/~romik/movingsofa/ . Also
   contains the **ambidextrous sofa** (~1.64495), area = X + arctan Y with
   x²(x+3)=8, x(4x²+3)=1.

## Upper and lower bounds

9. **Yoav Kallus, Dan Romik**, "Improved upper bounds in the moving sofa
   problem", *Advances in Mathematics* **340**:960–982, 2018;
   arXiv:1706.06630, DOI 10.1016/j.aim.2018.10.022. — Upper bound 2.37 by a
   certified computer-assisted scheme (finite rotated corridor copies);
   also proves any largest sofa must rotate ≥ arcsin(84/85) ≈ 81.203°.
10. **Kuangdai Leng, Jia Bi, Jaehoon Cha, Samuel Pinilla, Jeyan
    Thiyagalingam**, "Deep Learning Evidence for Global Optimality of
    Gerver's Sofa", arXiv:2407.11106 (2024). — Improves the five-angle
    Kallus–Romik bound 2.37 → 2.3337; neural-network evidence for
    Gerver-optimality.
11. **Zhipeng Deng**, "Solving moving sofa problem using calculus of
    variations", arXiv:2407.02587 (2024). — Euler–Lagrange treatment; Gerver's
    and Romik's shapes satisfy the necessary conditions; area 2.2195316
    re-derived.

## The proof of optimality

12. ★ **Jineon Baek**, "Optimality of Gerver's Sofa", arXiv:2411.19826
    (math.MG), 29 Nov 2024, 119 pages, DOI 10.48550/arXiv.2411.19826. —
    Theorem: Gerver's 18-section sofa attains the maximum area 2.2195...;
    resolves the problem. (Remark 8.4.1: the sofa clears the t=π/4 pinch by
    0.0012.)

## Encyclopedic / data / recent

13. **OEIS**: [A086118](https://oeis.org/A086118) (π/2 + 2/π, Hammersley's
    sofa) and [A128463](https://oeis.org/A128463) (Gerver's constant).
14. **Erich's Packing Center / Wikipedia**, "Moving sofa problem" — overview
    of bounds and history.
15. ★ **E. W. Weisstein**, "Gerver Sofa", *MathWorld* — equations (1)–(17)
    used here; and "Moving Sofa Problem", *MathWorld*.
16. **Quanta Magazine**, "Mathematician Solves the Moving Sofa Problem"
    (Feb 2025); *Phys.org* staff, "Mathematician solves the moving sofa
    problem" (11 Dec 2024). — Popular accounts of Baek's proof.
17. **B. Georgiev, J. Gómez-Serrano, T. Tao, A. Z. Wagner**, "Mathematical
    exploration and discovery at scale", arXiv:2511.02864 (2025) — includes
    AI-assisted study of sofa-problem variants.
18. **X. He**, "A gas-driven algorithm for variants of the moving sofa
    problem", arXiv:2608.11206 (2026) — recent independent numerical
    reproduction of the Gerver area via a dynamical-systems model.
19. **Croft, Falconer, Guy**, *Unsolved Problems in Geometry* (Springer,
    1994) — Problem G1 history (Moser's "loading a sofa").
