"""
verify_sofas.py -- final verification battery for Gerver's sofa.

Runs after gerver_sofa.py has produced its artifacts (sofa_polygon_xy.npy,
sofa_polygon_analytic_xy.npy, results.json).  Checks:

  [V1] A(0) = (1, 0)                       (Romik eq. (23))
  [V2] frozen contacts: A on [0, phi], C on [pi/2-phi, pi/2] are constant
  [V3] hand-offs (Romik eqs. (43)-(44)):
           x(phi) = B(pi/2 - theta),   x(pi/2 - phi) = D(theta)
  [V4] straight edges implicated by endpoints:
           C(0) - D(0) = (0,1)          flat vertical back edge
           y(A(0)) = y(B(pi/2)) = 0     bottom edge,
           y(A(pi/2)) = y(C(0)) = 1     top edge
  [V5] Baek (arXiv:2411.19826, Rem. 8.4.1) pinch clearance: at t = pi/4 the
       sofa's niche-ceiling endpoints x(phi), x(pi/2-phi) pass within
       0.0012 of the two inner walls of the hallway copy.
  [V6] feasibility sampling: the analytic sofa polygon, posed at a dense
       grid of times t in [0, pi/2], lies inside the hallway copy x(t)+R_t L
       (boundary samples + interior grid kept by a shapely containment test).
  [V7] comparison table of maximal-area shapes (OEIS A086118, A128463).

Writes verification_report.json.
"""
from __future__ import annotations

import json
import numpy as np
from numpy import pi, cos, sin

import gerver_sofa as g

REPORT = []


def add(id_, desc, value, expected, ok, units=""):
    REPORT.append({"id": id_, "description": desc, "value": value,
                   "expected": expected, "pass": bool(ok), "units": units})
    flag = "PASS" if ok else "FAIL"
    val = value if not isinstance(value, (list, np.ndarray)) else "see JSON"
    print(f"  [{id_}] {flag}: {val}   (expected ~ {expected})")
    return ok


def halfline_dist(P, base, direction_neg):
    """Distance from P to the half-line {base - a*direction_neg, a >= 0}.

    Perpendicular distance when P lies beyond the vertex along the arm
    (i.e. (P-base) has negative part along direction_neg), otherwise the
    distance to the vertex itself.
    """
    P, base = np.asarray(P, float), np.asarray(base, float)
    d = P - base
    u = np.asarray(direction_neg, float)
    proj = float(u @ d)                        # >0: P on the far side of vertex
    perp = abs(float(d[0]*u[1] - d[1]*u[0]))   # perpendicular to the wall line
    return perp if proj <= 0 else float(np.linalg.norm(d))


def main():
    print("=" * 76)
    print("GERVER'S SOFA -- verification battery (verify_sofas.py)")
    print("=" * 76)

    # ---- [V1] A(0) = (1,0) ---------------------------------------------------
    a0 = g.contact("A", 0.0)
    add("V1", "A(0) = (1,0) (Romik eq.(23))", list(map(float, a0)), "(1,0)",
        np.allclose(a0, [1.0, 0.0], atol=1e-12))

    # ---- [V2] frozen contacts -------------------------------------------------
    ts1 = np.linspace(0.0, g.PHI, 101)
    dA = np.max(np.abs(g.contact("A", ts1) - a0))
    ts2 = np.linspace(pi/2 - g.PHI, pi/2, 101)
    c_end = g.contact("C", pi/2)
    dC = np.max(np.abs(g.contact("C", ts2) - c_end))
    add("V2", "A frozen on [0, phi]: max|A(t)-A(0)|", float(dA), "~<=1e-12",
        dA < 1e-12)
    add("V2", "C frozen on [pi/2-phi, pi/2]: max|C(t)-C(pi/2)|", float(dC),
        "~<=1e-12", dC < 1e-12)

    # ---- [V3] hand-offs (43)-(44) ---------------------------------------------
    e43 = float(np.max(np.abs(g.x(g.PHI) - g.contact("B", g.T3))))
    e44 = float(np.max(np.abs(g.x(g.T4) - g.contact("D", g.T2))))
    add("V3", "(43) |x(phi) - B(pi/2-theta)|", e43, "~<=1e-14", e43 < 1e-13)
    add("V3", "(44) |x(pi/2-phi) - D(theta)|", e44, "~<=1e-14", e44 < 1e-13)

    # ---- [V4] straight edges --------------------------------------------------
    back = float(np.max(np.abs(g.contact("C", 0.0) - g.contact("D", 0.0)
                               - np.array([0.0, 1.0]))))
    add("V4", "|C(0)-D(0)-(0,1)| (vertical alignment of back corners; not a boundary edge)", back, "0", back < 1e-13)
    yA0, yB5 = float(g.contact("A", 0.0)[1]), float(g.contact("B", pi/2)[1])
    yA5, yC0 = float(g.contact("A", pi/2)[1]), float(g.contact("C", 0.0)[1])
    add("V4", "bottom edge y(A0)=y(B(pi/2))=0", [yA0, yB5], "0,0",
        abs(yA0) < 1e-12 and abs(yB5) < 1e-12)
    add("V4", "top edge y(A(pi/2))=y(C(0))=1", [yA5, yC0], "1,1",
        abs(yA5 - 1) < 1e-12 and abs(yC0 - 1) < 1e-12)

    # ---- [V5] pinch clearance at t = pi/4 (Baek Rem. 8.4.1) -------------------
    Xq = g.x(pi/4)                     # reflex corner position, sofa frame
    X0 = g.x(g.PHI)                    # niche ceiling right end (= B(t3))
    D2 = g.x(g.T4)                     # niche ceiling left  end (= D(theta))
    mu = np.array([cos(pi/4), sin(pi/4)])   # mu-wall (horizontal corridor)
    nu = np.array([-sin(pi/4), cos(pi/4)])  # nu-wall (vertical corridor)
    # Baek's wall naming (arXiv:2411.19826, line-level Def.): b = right inner
    # wall (the x=0 family, direction +-nu), d = left inner wall (the y=0
    # family, direction +-mu).  The walls emanate from the reflex corner
    # x(t) toward the contact points B(t)/D(t), i.e. arms in -nu and -mu.
    vals = {
        "x(phi)      -> b-wall (x=0 family) arm dist": halfline_dist(X0, Xq, nu),
        "x(pi/2-phi) -> d-wall (y=0 family) arm dist": halfline_dist(D2, Xq, mu),
        "x(phi)      -> d-wall (parallel side)": halfline_dist(X0, Xq, mu),
        "x(pi/2-phi) -> b-wall (parallel side)": halfline_dist(D2, Xq, nu),
    }
    print("  [V5] wall-clearance values (Baek Rem. 8.4.1 quotes 0.0012):")
    for k, v in vals.items():
        print(f"        {k:42s} = {v:.6f}")
    valstr = {k: float(v) for k, v in vals.items()}
    valstr["x(pi/4)"] = [float(Xq[0]), float(Xq[1])]
    hit = min(float(vals["x(phi)      -> b-wall (x=0 family) arm dist"]),
              float(vals["x(pi/2-phi) -> d-wall (y=0 family) arm dist"]))
    ok5 = abs(hit - 0.0012) < 1e-4
    add("V5", "niche-core endpoint clearance to the inner walls at t=pi/4 "
        "(Baek: 0.0012)", float(hit), 0.0012, ok5)

    # ---- [V6] feasibility sampling ------------------------------------------
    import shapely as sh
    Pan = np.load("sofa_polygon_analytic_xy.npy")
    poly = sh.Polygon(Pan)
    assert poly.is_valid, "[V6] analytic polygon invalid: re-run gerver_sofa.py (dedupe) first"
    xmin, ymin, xmax, ymax = poly.bounds
    gx, gy = np.meshgrid(np.arange(xmin, xmax, 0.02),
                         np.arange(ymin + 0.01, ymax, 0.02))
    gxy = np.column_stack([gx.ravel(), gy.ravel()])
    inside = sh.contains(poly, sh.points(gxy))
    interior = gxy[inside]
    interior = interior[:: max(1, len(interior) // 600)][:600]
    boundary = np.asarray(Pan[::35], dtype=float)
    samples = np.vstack([interior.astype(float), boundary])
    print(f"  [V6] {len(samples)} sofa sample points "
          f"({len(interior)} interior + {len(boundary)} boundary); "
          f"tested contained in the hallway copy x(t)+R_t L at 61 poses")
    fails = 0
    for t in np.linspace(0.0, pi/2, 61):
        hall = g.hallway_copy(t, big=14.0, buf=1e-9)
        # NOTE: samples and hallway_copy are both in the sofa frame; posing
        # the samples (R_{-t}(p - x(t)), hallway frame) must NOT be used here.
        okpts = sh.contains(hall, sh.points(samples))
        nbad = int(np.sum(~np.asarray(okpts)))
        fails += nbad
        if nbad:
            print(f"        t={t:.4f}: {nbad} sample(s) outside the hallway copy!")
    add("V6", "sofa samples outside hallway copy x(t)+R_t L (61 poses)", fails,
        "0", fails == 0)

    # ---- [V7] comparison table ----------------------------------------------
    rows = [
        {"shape": "unit square", "area": 1.0,
         "note": "trivial lower bound (Moser's problem, 1966)"},
        {"shape": "semicircle radius 1", "area": pi/2, "area_s": f"{pi/2:.9f}",
         "note": "classic early lower bound"},
        {"shape": "Hammersley's half-disk sofa", "area": pi/2 + 2/pi,
         "area_s": f"{pi/2 + 2/pi:.9f}", "oeis": "A086118",
         "note": "Hammersley 1968 lower bound"},
        {"shape": "Gerver's sofa (18 curve sections)", "area": g.GERVER_AREA,
         "area_s": f"{g.GERVER_AREA:.14f}", "oeis": "A128463",
         "note": "Gerver 1992; PROVED OPTIMAL by Baek 2024 "
                 "(arXiv:2411.19826)"},
    ]
    ubs = [
        {"bound": "Hammersley", "value": 2*np.sqrt(2.0),
         "value_s": f"{2*np.sqrt(2.0):.9f}", "note": "upper bound 2 sqrt(2)"},
        {"bound": "Kallus-Romik", "value": 2.37, "value_s": "2.37",
         "note": "upper bound (Adv. Math. 340, 2018; arXiv:1706.06630)"},
        {"bound": "Leng et al. (2024)", "value": 2.3337, "value_s": "2.3337",
         "note": "5-angle upper bound 2.37 -> 2.3337, arXiv:2407.11106"},
    ]
    print("  [V7] comparison table written to report (4 shapes, 3 bounds)")
    add("V7", "comparison table", "see comparison_table in JSON", "n/a", True)

    # ---- save ------------------------------------------------------------
    with open("verification_report.json", "w") as f:
        json.dump({
            "reference_area": g.GERVER_AREA,
            "pinch_clearance_wall_distances": valstr,
            "checks": REPORT,
            "comparison_table": rows,
            "upper_bounds": ubs,
        }, f, indent=2)
    n_fail = sum(1 for r in REPORT if not r["pass"])
    print(f"\n  {len(REPORT)} checks, {n_fail} failed. "
          f"Wrote verification_report.json")


if __name__ == "__main__":
    main()
