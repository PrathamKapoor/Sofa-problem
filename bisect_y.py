"""Bisect the source of the 8e-8 area error: dense-grid y1/y2 vs closed form."""
import numpy as np
from math import pi, cos, sin
from scipy.integrate import quad
import gerver_sofa as g

(A, B, ph, th), _ = g.gerver_constants()


# ---- closed-form y1, y2 in double precision --------------------------------
def _r_coeffs(k):
    if k == 1:   # [0, phi)
        return (0.5, 0.0, 0.0)
    if k == 2:   # [phi, theta)
        return ((1 + A - ph) / 2, 0.5, 0.0)
    if k == 3:   # [theta, pi/2-theta)
        return (A - ph, 1.0, 0.0)
    if k == 4:   # [pi/2-theta, pi/2-phi]
        m = pi / 2 - ph
        return (B - 0.5 * m * (1 + A) - 0.25 * m * m,
                0.5 * (1 + A) + 0.5 * m,
                -0.25)


def _s_coeffs(k):
    c = _r_coeffs(k)
    return (1 - c[0], -c[1], -c[2])


def _Fk(c, t):
    c0, c1, c2 = c
    return (-c0 * cos(t)
            + c1 * (-t * cos(t) + sin(t))
            + c2 * (-t * t * cos(t) + 2 * t * sin(t) + 2 * cos(t)))


def _running(cfun, x):
    pieces = [(0.0, ph, 1), (ph, th, 2), (th, pi / 2 - th, 3), (pi / 2 - th, pi / 2 - ph, 4)]
    tot = 0.0
    for lo, hi, k in pieces:
        if x <= lo:
            break
        top = min(x, hi)
        tot += _Fk(cfun(k), top) - _Fk(cfun(k), lo)
        if top >= x:
            break
    return tot


def y1_exact(a):
    return 1.0 - _running(_r_coeffs, a)


def y2_exact(a):
    return 1.0 - _running(_s_coeffs, a)


# ---- compare against the dense-grid running integrals ----------------------
n = 400001
grid = np.linspace(0, pi / 2, n)
rr = np.array([g._r(x_, A, B, ph, th) for x_ in grid])
ss = 1 - rr
st = np.diff(grid)
y1g = 1 - np.concatenate([[0], np.cumsum(0.5 * st * (rr[:-1] * np.sin(grid[:-1]) + rr[1:] * np.sin(grid[1:])))])
y2g = 1 - np.concatenate([[0], np.cumsum(0.5 * st * (ss[:-1] * np.sin(grid[:-1]) + ss[1:] * np.sin(grid[1:])))])
y1f = lambda a: np.interp(a, grid, y1g)

print("y1: closed form vs dense grid")
for x0 in (0.2, 0.5, th, 1.0, 1.4, pi / 2 - ph):
    d = y1_exact(x0) - y1f(x0)
    print(f"    x={x0:.6f}  y1_exact={y1_exact(x0):.16f}  grid={y1f(x0):.16f}  diff={d:+.3e}")

# ---- recomputed integrals with exact y1/y2 ---------------------------------
r = lambda a: g._r(a, A, B, ph, th)
s = lambda a: 1 - r(a)
u_du = lambda a: g._u_du(a, A, B, ph, th)

f1 = lambda a: y1_exact(a) * r(a) * cos(a)
f2 = lambda a: y2_exact(a) * s(a) * cos(a)
f3 = lambda a: (lambda u, du:
                (y2_exact(a) - u * sin(a)) *
                (u * sin(a) - du * cos(a) - s(a) * cos(a)))(*u_du(a))

I1 = quad(f1, 0, pi / 2 - ph, points=[ph, th, pi / 2 - th], limit=400,
          epsabs=1e-13, epsrel=1e-11)[0]
I2 = quad(f2, 0, th, points=[ph], limit=400, epsabs=1e-13, epsrel=1e-11)[0]
I3 = quad(f3, ph, pi / 4, points=[th], limit=400, epsabs=1e-13, epsrel=1e-11)[0]
print("closed-form y1/y2:  I1=%.16f I2=%.16f I3=%.16f S=%.16f" % (I1, I2, I3, 2 * (I1 + I2 + I3)))

# same but with grid-interpolated y1/y2 (current gerver_sofa gerver_area)
gA = g.gerver_area(A, B, ph, th)
print("module gerver_area: S=%.16f" % gA)
