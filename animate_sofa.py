"""
animate_sofa.py -- animate Gerver's sofa rotating through the L-corner.

Hallway frame: the L-shaped corridor is drawn statically; the sofa moves and
rotates as posed by g.pose(pts, t) (hallway copy = x(t) + R_t L fixed).

Requires sofa_polygon_analytic_xy.npy from gerver_sofa.py.
Writes gerver_sofa.gif.
"""
from __future__ import annotations

import numpy as np
from numpy import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MPoly

import gerver_sofa as g


def main(frames=90, fps=15):
    Pan = np.load("sofa_polygon_analytic_xy.npy")

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    wall = [(-2.7, 0.0), (0.0, 0.0), (0.0, -2.7), (1.0, -2.7),
            (1.0, 1.0), (-2.7, 1.0)]
    ax.add_patch(MPoly(wall, closed=True, facecolor="#e9e9e9",
                       edgecolor="#333", lw=2.2, zorder=1))
    ax.text(-1.55, 1.14, "horizontal corridor", fontsize=10, color="#333")
    ax.text(0.06, -1.7, "vertical corridor\n(sofa exits downward)",
            fontsize=10, color="#333")

    sofa = MPoly(Pan, closed=True, facecolor="#e67e22", edgecolor="k",
                 lw=1.0, alpha=0.95, zorder=3)
    ax.add_patch(sofa)
    ax.plot([0.0], [0.0], "ro", ms=6, zorder=4)   # the rotating reflex corner
    txt = ax.text(-2.6, -2.25, "", fontsize=11, ha="left")

    ax.set_xlim(-2.75, 1.35); ax.set_ylim(-2.75, 1.35)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Gerver's sofa passing through the L-corner\n"
                 "area 2.21953166887196... -- optimal (Baek 2024)",
                 fontsize=12)

    ts = np.linspace(0.0, pi/2, frames)

    def update(i):
        yy = g.pose(Pan, ts[i])
        sofa.set_xy(yy)
        deg = int(round(np.degrees(ts[i])))
        txt.set_text(("t = %3d deg" % deg) + ("\n(done!)" if deg >= 90 else ""))
        return (sofa, txt)

    anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=50)
    anim.save("gerver_sofa.gif", writer=PillowWriter(fps=fps))
    print("wrote gerver_sofa.gif (%d frames, %d fps)" % (frames, fps))


if __name__ == "__main__":
    main()
