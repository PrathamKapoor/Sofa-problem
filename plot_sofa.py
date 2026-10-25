"""
plot_sofa.py -- static figure of Gerver's sofa solution.

Panel 1: the sofa (analytic boundary polygon), the rotation path x(t) of the
         hallway's reflex corner inside it, and an inset zoom on the niche:
         the two inner walls at the t = pi/4 pose pass within 0.0012 of the
         niche-ceiling endpoints (Baek, arXiv:2411.19826, Rem. 8.4.1).
Panel 2: the sofa at five poses inside the L-shaped hallway.
Panel 3: area comparison with known shapes and best upper bounds.

Requires the artifacts from gerver_sofa.py.  Writes sofa_solution.png.
"""
from __future__ import annotations

import numpy as np
from numpy import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Rectangle

import gerver_sofa as g


def main():
    Pan = np.load("sofa_polygon_analytic_xy.npy")     # (N, 2) closed CCW loop
    ts = np.linspace(0.0, pi/2, 4000)
    path = g.x(ts)

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.2),
                            gridspec_kw={"width_ratios": [1.25, 1.0, 0.9]})

    # ---- panel 1: the sofa ----------------------------------------------------
    ax = axs[0]
    ax.add_patch(MPoly(Pan, closed=True, facecolor="#f5deb3",
                       edgecolor="black", lw=1.4, zorder=2))
    # each boundary piece in its own color (lines drawn atop the outline)
    tA = np.linspace(g.PHI, np.pi/2, 900)
    tC = np.linspace(0.0, np.pi/2 - g.PHI, 900)
    tD = np.linspace(0.0, g.THETA, 900)
    tb = np.linspace(np.pi/2 - g.THETA, np.pi/2, 900)
    txc = np.linspace(np.pi/2 - g.PHI, g.PHI, 900)
    ax.plot(g.contact("A", tA)[:, 0], g.contact("A", tA)[:, 1], color="#2e7d32",
            lw=2.6, zorder=4, label="A(t): right outer curve")
    ax.plot(g.contact("C", tC)[:, 0], g.contact("C", tC)[:, 1], color="#8e44ad",
            lw=2.6, zorder=4, label="C(t): lower-left outer curve")
    ax.plot(g.contact("D", tD)[:, 0], g.contact("D", tD)[:, 1], color="#16a085",
            lw=2.6, zorder=4, label="D(t): left funnel of the niche")
    ax.plot(g.x(txc)[:, 0], g.x(txc)[:, 1], color="#e67e22", lw=2.6, zorder=4,
            label="x(t): niche ceiling (dome, apex at t=pi/4)")
    ax.plot(g.contact("B", tb)[:, 0], g.contact("B", tb)[:, 1], color="#c0392b",
            lw=2.6, zorder=4, label="B(t): right funnel of the niche")
    ax.plot(g.x(np.linspace(0.0, np.pi/2, 1200))[:, 0],
            g.x(np.linspace(0.0, np.pi/2, 1200))[:, 1], color="#1f6feb",
            lw=1.3, ls="--", zorder=5, label="rotation path x(t) (whole)")
    for t_, lab, dx, dy in [(0.0, "A(0)=(1,0)", -4, -16),
                            (np.pi/2, "x(pi/2)", 4, 10),
                            (np.pi/4, "x(pi/4)=(−0.61, 0.66): dome apex", 8, 6)]:
        p = g.x(t_)
        ax.plot(*p, "o", color="#1f6feb", ms=4.5, zorder=6)
        if lab:
            ax.annotate(lab, p, textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color="#1f6feb")
    ax.set_title("Gerver's sofa:  area = 2.21953166887196...  (OEIS A128463)",
                 fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # inset: the famous 0.0012 pinch, walls at the t = pi/4 pose
    Xq = g.x(pi/4)
    mu = np.array([np.cos(pi/4), np.sin(pi/4)])
    nu = np.array([-np.sin(pi/4), np.cos(pi/4)])
    s = np.linspace(-1.05, 1.05, 120)
    axi = ax.inset_axes([0.60, 0.10, 0.37, 0.42])
    axi.add_patch(MPoly(Pan, closed=True, facecolor="#f5deb3",
                        edgecolor="k", lw=1.0, zorder=2))
    axi.plot(g.contact("B", tb)[:, 0], g.contact("B", tb)[:, 1],
             color="#c0392b", lw=2.0, zorder=3)
    axi.plot(g.contact("D", tD)[:, 0], g.contact("D", tD)[:, 1],
             color="#16a085", lw=2.0, zorder=3)
    axi.plot(g.x(txc)[:, 0], g.x(txc)[:, 1], color="#e67e22", lw=2.0, zorder=3)
    axi.plot(Xq[0] + s*mu[0], Xq[1] + s*mu[1], color="#8e44ad", ls="--",
             lw=1.2, zorder=4, label="inner walls at the pi/4 pose")
    axi.plot(Xq[0] + s*nu[0], Xq[1] + s*nu[1], color="#8e44ad", ls="--",
             lw=1.2, zorder=4)
    for P in (g.x(g.PHI), g.x(pi/2 - g.PHI)):
        axi.plot(*P, "o", ms=4, color="#c0392b", zorder=5)
    axi.set_xlim(-1.05, 0.25); axi.set_ylim(-0.02, 0.12)
    axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([])
    axi.set_title(r"pinch: clearance 0.0012 at $t=\pi/4$", fontsize=8)
    ax.indicate_inset_zoom(axi, edgecolor="#555")

    # ---- panel 2: five poses in the hallway ------------------------------------
    ax = axs[1]
    # wall mass (gray) with white corridor interiors; the gray square's top
    # and right edges are exactly the corridor outer walls y = 1 and x = 1
    ax.add_patch(MPoly([[-2.7, -2.7], [1.0, -2.7], [1.0, 1.0], [-2.7, 1.0]],
                       closed=True, facecolor="#e8e8e8", edgecolor="#444",
                       lw=1.5, zorder=0))
    for strip in ([[-2.7, 0.0], [1.0, 0.0], [1.0, 1.0], [-2.7, 1.0]],   # horiz
                  [[0.0, -2.7], [1.0, -2.7], [1.0, 1.0], [0.0, 1.0]]):  # vert
        ax.add_patch(MPoly(strip, closed=True, facecolor="white",
                           edgecolor="none", zorder=1))
    ax.plot([-2.7, 0.0, 0.0], [0.0, 0.0, -2.7], color="#444", lw=2.0,
            zorder=2)                  # inner wall faces at the corner (0,0)
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, 5))
    for i, t_ in enumerate(np.linspace(0.0, pi/2, 5)):
        ps = g.pose(Pan, t_)
        ax.add_patch(MPoly(ps, closed=True, facecolor=cols[i],
                           edgecolor="k", lw=0.6, alpha=0.55, zorder=3))
        cen = Pan.mean(axis=0)      # label each silhouette at its centroid;
                                    # pose(x(t), t) would be (0,0) for every t
        ax.annotate(f"t={np.degrees(t_):g} deg", g.pose(cen, t_),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=8, ha="center", zorder=6)
    ax.set_xlim(-2.45, 1.05)        # show the whole L and posed sofas
    ax.set_ylim(-2.45, 1.05)        # (patches post-draw don't grow datalim)
    ax.set_title("feasible rotation through the L-corner", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)

    # ---- panel 3: area comparison ----------------------------------------------
    ax = axs[2]
    names = ["square", "semicircle", "Hammersley 1968", "Gerver 1992"]
    vals = [1.0, pi/2, pi/2 + 2/pi, g.GERVER_AREA]
    names_out = ["sq", "semic", "Hamm", "Gerv"]
    bars = ax.bar(names_out, vals, color=["#9e9e9e", "#9e9e9e",
                                          "#ef8a17", "#c0392b"])
    lab_map = {"sq": "1", "semic": "pi/2 = 1.571",
               "Hamm": "pi/2 + 2/pi\n= 2.2074160999",
               "Gerv": "2.21953166887196\n(OEIS A128463)"}
    for b, nm, vv in zip(bars, names_out, vals):
        ax.text(b.get_x() + b.get_width()/2, vv + 0.02, lab_map[nm],
                ha="center", va="bottom", fontsize=8)
    for ub, col, lab in [(2*np.sqrt(2), "#2980b9", "2 sqrt(2)  (Hammersley 1968)"),
                         (2.37, "#7f8c8d", "2.37  (Kallus-Romik 2018)"),
                         (2.3337, "#2980b9", "2.3337  (Leng et al. 2024)")]:
        ax.axhline(ub, color=col, ls=":", lw=1.2, alpha=0.8, label=lab)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.set_ylim(0, 3.05)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8, rotation=14)
    ax.set_title("maximal-area shapes (sofa problem)", fontsize=11)
    ax.set_ylabel("area  (1x1-corner L-hallway)")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("The Moving Sofa Problem (Moser, 1966) -- solution: "
                 "Gerver's sofa, proved optimal by Jineon Baek (2024)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig("sofa_solution.png", dpi=160)
    print("wrote sofa_solution.png")


if __name__ == "__main__":
    main()
