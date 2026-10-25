"""Pinpoint the x'(t) inconsistency: fd vs analytic, glue eqs, direct ODE check."""
import numpy as np
from numpy import cos, sin, pi
import gerver_sofa as g
import gerver_sofa as G

# phase sample points (strictly interior)
samples = {
    1: [0.005, 0.01, 0.02, G.PHI - 0.002],
    2: [G.PHI + 0.002, 0.2, 0.4, G.THETA - 0.002],
    3: [G.THETA + 0.002, 1.0, 1.2, pi/2 - G.THETA - 0.002],
    4: [pi/2 - G.THETA + 0.002, 1.4, pi/2 - G.PHI - 0.002],
    5: [pi/2 - G.PHI + 0.002, 1.55, 1.57],
}
eps = 1e-6
print("fd-vs-analytic x'(t) per phase, interior samples")
for j, ts in samples.items():
    for t in ts:
        fd = (G.x(t + eps) - G.x(t - eps)) / (2*eps)
        an = G.x(t, der=1)
        gap = np.max(np.abs(fd - an))
        print(f"    phase {j}  t={t:.6f}   |gap| = {gap:.3e}   x' = ({an[0]:+.9f}, {an[1]:+.9f})")

print()
print("left/right analytic derivative at the four edges (eps=1e-7)")
e2 = 1e-7
for label, t in [("phi", G.PHI), ("theta", G.THETA),
                 ("pi/2-theta", pi/2 - G.THETA), ("pi/2-phi", pi/2 - G.PHI)]:
    dl = G.x(t - e2, der=1)
    dr = G.x(t + e2, der=1)
    fd = (G.x(t + eps) - G.x(t - 0.5*eps)) / eps   # forward diff from left
    print(f"    {label:<10s} left=({dl[0]:+.7f},{dl[1]:+.7f})  right=({dr[0]:+.7f},{dr[1]:+.7f})  "
          f"|jump|={np.max(np.abs(dr-dl)):.3e}")

# direct ODE verification: y(t) = R_{-t} x'(t) must satisfy y' = T y + v
print()
print("direct ODE check: y' - (T y + v), fd with eps=1e-7")


def Y(t):
    xx = G.x(t, der=1)
    c, s = cos(t), sin(t)
    return np.array([c*xx[0] + s*xx[1], -s*xx[0] + c*xx[1]])


odes = {
    1: (lambda y: np.array([0.0, 0.0]) @ y, None),  # placeholder; fill below
}
# ODE1 (paper): y' = T y + v with T = [[0, ?],...]; use printed T, v for ODE2/ODE3
# and verify phases 2,3 directly; print y itself for phases 1,4,5 for inspection.
T2 = np.array([[0.0, 0.0], [0.5, 0.0]]); v2 = np.array([-1.0, -0.5])
T3 = np.array([[0.0, 0.0], [0.0, 0.0]]); v3 = np.array([-1.0, -1.0])
for t in (0.2, 0.5):
    yy, dyy = Y(t), (Y(t + e2) - Y(t - e2)) / (2*e2)
    print(f"    phase2 t={t}:  y'-(Ty+v) = {np.max(np.abs(dyy - (T2 @ yy + v2))):.3e}")
for t in (0.8, 1.2):
    yy, dyy = Y(t), (Y(t + e2) - Y(t - e2)) / (2*e2)
    print(f"    phase3 t={t}:  y'-(Ty+v) = {np.max(np.abs(dyy - (T3 @ yy + v3))):.3e}")
for t in (0.01, 0.02, 0.3, 0.5):
    yy, dyy = Y(t), (Y(t + e2) - Y(t - e2)) / (2*e2)
    print(f"    phase1 t={t}:  y = ({yy[0]:+.9f}, {yy[1]:+.9f})   y' = ({dyy[0]:+.6f}, {dyy[1]:+.6f})")
for t in (1.45, 1.5, 1.57):
    yy, dyy = Y(t), (Y(t + e2) - Y(t - e2)) / (2*e2)
    print(f"    phase5 t={t}:  y = ({yy[0]:+.9f}, {yy[1]:+.9f})   y' = ({dyy[0]:+.6f}, {dyy[1]:+.6f})")

# symmetry relation (26): x'(pi/2 - t) =? diag(1,-1) x'(t)
print()
print("symmetry (26) residual at interior t")
for t in (0.02, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3):
    lhs = G.x(pi/2 - t, der=1)
    rhs = G.x(t, der=1) * np.array([1.0, -1.0])
    print(f"    t={t:<5.3f}  |lhs-rhs| = {np.max(np.abs(lhs - rhs)):.3e}")

# glue equations in velocity: (36) x1'(phi)=x2'(phi); (38) x2'(theta)=x3'(theta)
# (40) x3'(pi/2-theta)=x4'(pi/2-theta); (42) x4'(pi/2-phi)=x5'(pi/2-phi)
print()
print("velocity glue residuals (analytic, one-sided)")
pairs = [("36", G.PHI, G.x(G.PHI - e2, der=1), G.x(G.PHI, der=1)),
         ("38", G.THETA, G.x(G.THETA - e2, der=1), G.x(G.THETA, der=1)),
         ("40", pi/2 - G.THETA, G.x(pi/2 - G.THETA - e2, der=1), G.x(pi/2 - G.THETA, der=1)),
         ("42", pi/2 - G.PHI, G.x(pi/2 - G.PHI - e2, der=1), G.x(pi/2 - G.PHI, der=1))]
for tag, t, a, b in pairs:
    print(f"    ({tag})  gap = ({a[0]-b[0]:+.3e}, {a[1]-b[1]:+.3e})")
