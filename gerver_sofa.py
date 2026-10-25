"""
Gerver's sofa -- exact construction & multi-way verification.

The moving sofa problem (Moser, 1966): find the planar shape of maximal area
that can be moved continuously (rigid motions only) around an L-shaped hallway
formed by two unit-width corridors.  Gerver (1992) constructed the 18-piece
shape below with area 2.21953166887196... (OEIS A128463), and Baek
(arXiv:2411.19826, 2024) proved that it is optimal.

This module reconstructs Gerver's sofa exactly from the ODE/rotation-path
theory of Romik ("Differential equations and exact solutions in the moving
sofa problem", Exp. Math. 27 (2018) 316-330; arXiv:1606.08111):

  * hallway conventions (Romik sec. 2):
        L_horiz = {(x,y): x <= 1, 0 <= y <= 1}
        L_vert  = {(x,y): 0 <= x <= 1, y <= 1},      L = L_horiz ∪ L_vert
    R_t = [[cos t, -sin t], [sin t, cos t]] (counter-clockwise).
  * a rotation path x: [0, pi/2] -> R^2, x(0) = (0,0), tracking the hallway's
    reflex ("inner") corner (0,0) in the sofa's frame; the hallway copy at
    "time" t is x(t) + R_t(L).  The shape pinned down by the motion is
    (Romik eq. (8)):
        S_x = L_horiz ∩ ⋂_{0≤t≤π/2} (x(t) + R_t L) ∩ (x(π/2) + R_{π/2} L_vert)
  * five phases (Romik eqs. (24)-(25)) with breaks φ, θ, π/2-θ, π/2-φ, on
    which x is the solution SOL_j of the j-th contact ODE (Romik Theorem 3):
        x_j(t) = R_t @ p_j(t) + κ_j ,
    with the Table-1 numerical constants reproduced below, plus the symmetry
    relations of Romik eqs. (27)-(34).
  * contact paths (Romik Theorem 1, eqs. (9)-(12)), with μ_t=(cos t, sin t),
    ν_t=(-sin t, cos t):
        A(t) = x + <x',μ>_t ν_t + μ_t      (outer wall "front-right")
        B(t) = x + <x',μ>_t ν_t            (inner wall, μ-wall side)
        C(t) = x - <x',ν>_t μ_t + ν_t      (outer wall "back-top")
        D(t) = x - <x',ν>_t μ_t            (inner wall, ν-wall side)

Verification (see main()): C^1/C^2 structure of x, Table-1 identities, contact
point endpoints, and THREE independent area computations that must agree:
  1. Gerver's analytic area integral (MathWorld "Gerver Sofa" eqs. (1)-(17),
     re-solving the 4x4 system for A, B, φ, θ from scratch);
  2. the numeric intersection of ~N rotated hallway copies (shapely), which
     probes eq. (8) directly and uses no contact-curve formulas, with a
     first-order (in 1/n) Richardson refinement;
  3. the analytic boundary polygon built from the contact curves A, B, C, D
     and the niche ceiling (shoelace), with a 2nd-order Richardson
     refinement in the sampling density.

Conventions for all functions here:
    x(t, der)  returns an ndarray of shape np.shape(t) + (2,).
    contact(name, t) likewise.
    Scalar inputs give shape (2,) outputs.
"""
from __future__ import annotations

import json
import numpy as np
from numpy import cos, sin, pi

# ----------------------------------------------------------------------------
# Constants of Gerver's sofa: Table 1 of Romik (2018), 18 significant digits,
# obtained by solving the 22 scalar equations (27)-(44) of his paper.
# ----------------------------------------------------------------------------
PHI   = 0.039177364790083641          # first transition angle  (radians)
THETA = 0.681301509382724894          # second transition angle (radians)

A1, A2 = 1.210322422072688751, -0.25
B1, B2 = -0.527624598026784624, 0.920258385160637622
C1, C2 = 0.626045522848465867, -0.944750803946430751
D1, D2 = 1.313022761424232933, -0.525382670414554437
E1, E2 = 1.210322422072688751, 0.25

KP = np.array([                        # translation vectors kappa_j, j = 1..5
    [-0.210322422072688751,  0.25],
    [-0.919179292771593322,  0.472406619750805465],
    [-0.613763229430251668,  0.889626479003221860],
    [-0.308347166088910014,  0.472406619750805465],
    [-1.017204036787814585,  0.25],
])

GERVER_AREA = 2.21953166887196         # OEIS A128463 (14 printed digits)
PHASE_EDGES = (0.0, PHI, THETA, pi/2 - THETA, pi/2 - PHI, pi/2)
_INTERIOR   = np.array([PHI, THETA, pi/2 - THETA, pi/2 - PHI])

T0, T1, T2, T3, T4, T5 = PHASE_EDGES   # Baek's t_0..t_5 notation


# ----------------------------------------------------------------------------
# Phase polynomial/trig pieces p_j(t) with p'_j and p''_j (SOL1..SOL5).
# x = R_t p + kappa  on phase j,  x' = R_t (p' + J p),
# x'' = R_t (2 J p' + p'' - p),  with J = [[0,-1],[1,0]].
# ----------------------------------------------------------------------------
def _stack(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.stack(np.broadcast_arrays(a, b), axis=-1)


def _P(t, j):
    t = np.asarray(t, dtype=float)
    if j == 1:
        return _stack(A1*cos(t) + A2*sin(t) - 1.0,
                      -A2*cos(t) + A1*sin(t) - 0.5)
    if j == 2:
        return _stack(-t*t/4.0 + B1*t + B2, t/2.0 - B1 - 1.0)
    if j == 3:
        return _stack(C1 - t, C2 + t)
    if j == 4:
        return _stack(-t/2.0 + D1 - 1.0, -t*t/4.0 + D1*t + D2)
    if j == 5:
        return _stack(E1*cos(t) + E2*sin(t) - 0.5,
                      -E2*cos(t) + E1*sin(t) - 1.0)
    raise ValueError(j)


def _dP(t, j):
    t = np.asarray(t, dtype=float)
    if j == 1:
        return _stack(-A1*sin(t) + A2*cos(t), A2*sin(t) + A1*cos(t))
    if j == 2:
        return _stack(-t/2.0 + B1, 0.5 + 0*t)
    if j == 3:
        return _stack(-1.0 + 0*t, 1.0 + 0*t)
    if j == 4:
        return _stack(-0.5 + 0*t, -t/2.0 + D1)
    if j == 5:
        return _stack(-E1*sin(t) + E2*cos(t), E2*sin(t) + E1*cos(t))
    raise ValueError(j)


def _ddP(t, j):
    t = np.asarray(t, dtype=float)
    if j in (1, 5):                    # harmonic: p'' = -(p - v), v = ODE forcing
        p = _P(t, j)
        v = _stack(-1.0 + 0*t, -0.5 + 0*t) if j == 1 else _stack(-0.5 + 0*t, -1.0 + 0*t)
        return -(p - v)
    if j == 2:                         # p = (-t²/4 + b1 t + b2, t/2 - b1 - 1)
        return _stack(-0.5 + 0*t, 0*t)
    if j == 3:
        return _stack(0*t, 0*t)
    if j == 4:                         # p = (-t/2 + d1 - 1, -t²/4 + d1 t + d2)
        return _stack(0*t, -0.5 + 0*t)
    raise ValueError(j)


def _phase_index(t):
    """0..4 = phase j-1 using the left-closed intervals of Romik eq. (25)."""
    return np.searchsorted(_INTERIOR, np.asarray(t, dtype=float), side="right")


def x(t, der=0):
    """Rotation path x(t) (Romik eq. (25)).

    Parameters
    ----------
    t   : scalar or array of t in [0, pi/2].
    der : 0 -> x(t), 1 -> x'(t), 2 -> x''(t).

    Returns ndarray of shape np.shape(t) + (2,).
    """
    tin = np.asarray(t, dtype=float)
    scalar = (tin.ndim == 0)
    t = np.atleast_1d(tin)
    if (t < -1e-12).any() or (t > pi/2 + 1e-12).any():
        raise ValueError("t must lie in [0, pi/2]")
    t = np.clip(t, 0.0, pi/2)
    idx = _phase_index(t)
    out = np.empty(t.shape + (2,))
    for j in (1, 2, 3, 4, 5):
        m = (idx == j-1)
        if not m.any():
            continue
        tt = t[m]
        p = _P(tt, j)
        if der == 0:
            q = p
        elif der == 1:
            dp = _dP(tt, j)
            q = _stack(dp[..., 0] - p[..., 1],       # (p' + Jp)_1
                       dp[..., 1] + p[..., 0])       # (p' + Jp)_2
        elif der == 2:
            dp = _dP(tt, j)
            ddp = _ddP(tt, j)
            q = _stack(ddp[..., 0] - 2.0*dp[..., 1] - p[..., 0],
                       ddp[..., 1] + 2.0*dp[..., 0] - p[..., 1])
        else:
            raise ValueError("der must be 0, 1 or 2")
        c, s = cos(tt), sin(tt)
        rx = c*q[..., 0] - s*q[..., 1]
        ry = s*q[..., 0] + c*q[..., 1]
        if der == 0:
            rx = rx + KP[j-1, 0]
            ry = ry + KP[j-1, 1]
        out[m, 0] = rx
        out[m, 1] = ry
    return out[0] if scalar else out


# ----------------------------------------------------------------------------
# Contact paths A, B, C, D (Romik Theorem 1, eqs. (9)-(12)).
# Valid domains (per phase table (24) and Baek, Def. 8.4.3):
#   A: [0, pi/2],  B: [t3, t5] = [pi/2-theta, pi/2],
#   C: [0, pi/2],  D: [t0, t2] = [0, theta].
# ----------------------------------------------------------------------------
def contact(name, t):
    """Contact path 'name' (one of A, B, C, D); ndarray shape np.shape(t)+(2,)."""
    tin = np.asarray(t, dtype=float)
    scalar = (tin.ndim == 0)
    t = np.atleast_1d(tin)
    X = x(t)
    V = x(t, der=1)
    mu = _stack(cos(t), sin(t))
    nu = _stack(-sin(t), cos(t))
    Xdmu = (V*mu).sum(-1)
    Xdnu = (V*nu).sum(-1)
    if name == "A":
        Q = X + Xdmu[..., None]*nu + mu
    elif name == "B":
        Q = X + Xdmu[..., None]*nu
    elif name == "C":
        Q = X - Xdnu[..., None]*mu + nu
    elif name == "D":
        Q = X - Xdnu[..., None]*mu
    else:
        raise ValueError("name must be one of A, B, C, D")
    return Q[0] if scalar else Q


# ----------------------------------------------------------------------------
# Sofa pose: where a sofa-frame point sits in the hallway frame at time t.
# (Romik sec. 2: hallway copies are x(t) + R_t L, hence g_t(p) = R_{-t}(p - x(t)).)
# ----------------------------------------------------------------------------
def pose(pts, t):
    """Map sofa-frame points (..., 2) to hallway-frame points at time t."""
    pts = np.asarray(pts, dtype=float)
    xx = x(t)
    d = pts - xx
    c, s = cos(t), sin(t)
    return _stack(c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1])


# ----------------------------------------------------------------------------
# Verification 1: numeric sofa = intersection of rotating hallways (eq. (8)).
# ----------------------------------------------------------------------------
def hallway_copy(t, big=14.0, buf=1e-12, which="both"):
    """Shapely polygon of the hallway copy x(t) + R_t L (or L_horiz/L_vert part)."""
    from shapely.geometry import Polygon, box
    c, s = cos(t), sin(t)
    R = np.array([[c, -s], [s, c]])
    P0 = np.array([x(t)[0], x(t)[1]])
    polys = []
    if which in ("both", "horiz"):
        C = np.array([[-big, 0.0], [1.0, 0.0], [1.0, 1.0], [-big, 1.0]])
        polys.append(Polygon(C @ R.T + P0))
    if which in ("both", "vert"):
        C = np.array([[0.0, -big], [1.0, -big], [1.0, 1.0], [0.0, 1.0]])
        polys.append(Polygon(C @ R.T + P0))
    if len(polys) == 2:
        from shapely.ops import unary_union
        g = unary_union(polys)
    else:
        g = polys[0]
    if buf:
        g = g.buffer(buf, join_style=2)
    return g


def sofa_by_intersection(n_angles=1200, big=14.0, buf=1e-12):
    """S_x = L_horiz ∩ ⋂_t (x(t)+R_t L) ∩ (x(pi/2) + R_{pi/2} L_vert).

    Returns (shapely polygon, angle grid used)."""
    from shapely.geometry import box
    S = box(-big, 0.0, 1.0, 1.0)                       # L_horiz (bounded copy)
    ts = np.linspace(0.0, pi/2, n_angles)
    for t in ts:
        S = S.intersection(hallway_copy(t, big, buf, "both"))
        if S.is_empty:
            raise RuntimeError("intersection collapsed at t=%.6f" % t)
        if S.geom_type == "MultiPolygon":
            S = max(S.geoms, key=lambda g: g.area)
    S = S.intersection(hallway_copy(pi/2, big, buf, "vert"))
    if S.geom_type == "MultiPolygon":
        S = max(S.geoms, key=lambda g: g.area)
    return S, ts


# ----------------------------------------------------------------------------
# Verification 3: analytic boundary polygon (contact curves + straight edges).
# ----------------------------------------------------------------------------
def poly_area(P):
    """Signed shoelace area of the point loop P (N,2); >0 when CCW."""
    P = np.asarray(P, dtype=float)
    px, py = P[:, 0], P[:, 1]
    return 0.5*float(np.sum(px*np.roll(py, -1) - np.roll(px, -1)*py))


def sofa_polygon(n_per_curve=20000):
    """Gerver's sofa boundary as one closed CCW point loop, ndarray (N,2).

    Assembly (pieces listed in CCW traversal order; A is frozen on [0, phi]
    and C on [pi/2-phi, pi/2], so those spans degenerate to points and are
    skipped):
        0. A on [phi, pi/2]          right outer curve (1,0) -> A(pi/2)
        1. segment A(pi/2) -> C(0)   flat top edge y = 1
        2. C on [0, pi/2-phi]        outer curve C(0) -> C(pi/2)
        3. segment C(pi/2) -> D(0)   flat edge y = 0
        4. D on [0, theta]           inner curve D(0) -> D(theta) = x(pi/2-phi)
        5. x on [pi/2-phi, phi]      niche ceiling x(pi/2-phi) -> x(phi)
        6. B on [pi/2-theta, pi/2]   inner curve B(pi/2-theta)=x(phi) -> B(pi/2)
        7. segment B(pi/2) -> A(0)   flat edge y = 0, closes loop at (1,0)

    The returned loop is closed (first row == last row, both exactly the
    glue corner A(0) = (1, 0)).  Flat-edge junction constants are snapped
    to exact ordinates (y in {0, 1}): the glue endpoints evaluate with
    ~1e-17 float noise, which otherwise pokes ring vertices below the
    bottom flats and makes GEOS flag a self-intersection near (1, 0).
    """
    def curve(name, t0, t1, n, p0=None, p1=None):
        t = np.linspace(t0, t1, n)
        pts = contact(name, t) if name else x(t)
        # glue endpoints evaluate to within ~4e-16 of the exact junction
        # constants; overwrite with the snapped constants so consecutive
        # pieces join bit-for-bit on their shared boundary flats
        for j, pe in ((0, p0), (-1, p1)):
            if pe is not None and np.linalg.norm(np.asarray(pts[j]) - pe) < 1e-12:
                pts[j] = pe
        return pts

    def seg(p, q):
        return np.array([p, q], dtype=float)

    def flat(p, y):
        """Snap a flat-edge endpoint to its exact ordinate (kills float noise)."""
        q = np.array(p, dtype=float)
        if abs(q[1] - y) < 1e-12:
            q[1] = float(y)
        return q

    A0 = flat(contact("A", 0.0), 0.0)       # exact glue corner (1, 0)
    A5 = flat(contact("A", pi/2), 1.0)      # top flat corner
    C0 = flat(contact("C", 0.0), 1.0)       # top flat corner
    C5 = flat(contact("C", pi/2), 0.0)      # reflex floor corner
    D0 = flat(contact("D", 0.0), 0.0)       # funnel mouth on the floor
    B5 = flat(contact("B", pi/2), 0.0)      # funnel mouth on the floor
    pieces = [
        curve("A", PHI, pi/2, n_per_curve, p0=A0),          # right outer curve
        seg(A5, C0),                                        # flat top edge
        curve("C", 0.0, pi/2 - PHI, n_per_curve, p0=C0, p1=C5),   # outer left
        seg(C5, D0),                                        # flat bottom-left
        curve("D", 0.0, THETA, n_per_curve, p0=D0),         # inner left curve
        curve(None, pi/2 - PHI, PHI, n_per_curve),          # niche ceiling
        curve("B", pi/2 - THETA, pi/2, n_per_curve, p1=B5), # inner right curve
        seg(B5, A0),                                        # flat bottom-right
    ]
    P = np.vstack([pieces[0]] + [q[1:] for q in pieces[1:]])
    # The glue hand-offs (Romik (43)/(44)) hold to ~1e-16, so consecutive
    # vertices at the junctions are near-duplicates; drop zero-length edges
    # (they would make GEOS flag the ring as invalid).
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    P = P[~np.concatenate([[False], d < 1e-9])]
    if np.linalg.norm(np.asarray(P[-1]) - P[0]) > 1e-12:
        P = np.vstack([P, P[:1]])   # close explicitly at A(0) = (1, 0);
                                    # skipped when the last piece already
                                    # ends there (a repeated vertex would
                                    # itself read as a self-intersection)
    return P


# ----------------------------------------------------------------------------
# Verification 2: Gerver's area integral (MathWorld "Gerver Sofa",
# eqs. (1)-(17); Gerver 1992 / Finch 2003).  Constants A, B are re-solved
# from the 4x4 nonlinear system (eqs. (1)-(4)).
# ----------------------------------------------------------------------------
def gerver_constants(xtol=1e-14):
    from scipy.optimize import root

    def eqs(v):
        A, B, phi, theta = v
        return [
            A*(cos(theta) - cos(phi)) - 2*B*sin(phi)
            + (theta - phi - 1)*cos(theta) - sin(theta) + cos(phi) + sin(phi),
            A*(3*sin(theta) + sin(phi)) - 2*B*cos(phi)
            + 3*(theta - phi - 1)*sin(theta) + 3*cos(theta) - sin(phi) + cos(phi),
            A*cos(phi) - (sin(phi) + 0.5 - 0.5*cos(phi) + B*sin(phi)),
            (A + pi/2 - phi - theta)
            - (B - 0.5*(theta - phi)*(1 + A) - 0.25*(theta - phi)**2),
        ]

    sol = root(eqs, [0.094, 1.399, 0.0392, 0.6813], method="hybr",
               options={"xtol": xtol})
    res = np.max(np.abs(eqs(sol.x)))
    return sol.x, res


def _r(a, A, B, phi, theta):
    """Gerver's radius function r(alpha), MathWorld eq. (9), on [0, pi/2-phi]."""
    if a < phi:
        return 0.5
    if a < theta:
        return 0.5*(1.0 + A + a - phi)
    if a < pi/2 - theta:
        return A + a - phi
    u = pi/2 - a - phi
    return B - 0.5*u*(1.0 + A) - 0.25*u*u        # a <= pi/2 - phi


def _u_du(a, A, B, phi, theta):
    """u(alpha) (MathWorld eq. (11)) and u'(alpha) (eq. (13)) on [phi, pi/4]."""
    if a < theta:
        u = B - 0.5*(a - phi)*(1.0 + A) - 0.25*(a - phi)**2
        du = -0.5*(1.0 + A) - 0.5*(a - phi)
    else:
        u = A + pi/2 - phi - a
        du = -1.0
    return u, du


def _r_coeffs(k, A, B, phi):
    """Polynomial coefficients c0 + c1*a + c2*a^2 of Gerver's r(alpha) per piece
    (MathWorld eq. (9)); needed only for the exact running integrals y1, y2."""
    if k == 1:   # [0, phi): r = 1/2
        return (0.5, 0.0, 0.0)
    if k == 2:   # [phi, theta): r = (1 + A + alpha - phi)/2
        return ((1.0 + A - phi) / 2.0, 0.5, 0.0)
    if k == 3:   # [theta, pi/2-theta): r = A + alpha - phi
        return (A - phi, 1.0, 0.0)
    if k == 4:   # [pi/2-theta, pi/2-phi]: quadratic in u = (pi/2-phi) - alpha
        m = pi/2 - phi
        return (B - 0.5*m*(1.0 + A) - 0.25*m*m,
                0.5*(1.0 + A) + 0.5*m,
                -0.25)
    raise ValueError(k)


def _Fprime(c, t):
    """F(t) = integral (c0 + c1 t + c2 t^2) sin t dt, vanishing-derivative form."""
    c0, c1, c2 = c
    return (-c0*cos(t)
            + c1*(-t*cos(t) + sin(t))
            + c2*(-t*t*cos(t) + 2.0*t*sin(t) + 2.0*cos(t)))


def _running_int(cfun, x, A, B, phi, theta):
    """Exact accumulation of int_0^x poly(t) sin t dt across the four pieces."""
    pieces = [(0.0, phi, 1), (phi, theta, 2), (theta, pi/2 - theta, 3),
              (pi/2 - theta, pi/2 - phi, 4)]
    tot = 0.0
    for lo, hi, k in pieces:
        if x <= lo:
            break
        top = min(x, hi)
        tot += _Fprime(cfun(k, A, B, phi), top) - _Fprime(cfun(k, A, B, phi), lo)
        if top >= x:
            break
    return tot


def gerver_area(A, B, phi, theta):
    """Area from MathWorld eq. (17):  S = 2(I1 + I2 + I3).

    The running integrals y1 = 1 - int r sin, y2 = 1 - int s sin (eqs. (14)-(16))
    are accumulated in closed form (r, s are piecewise polynomials with known
    antiderivatives); the outer integrals use adaptive quadrature with split
    points at the phase breaks.  (A dense-grid trapezoid approximation is ~1.6e-7
    off due to the jump of r at alpha = pi/2-theta inside a grid cell.)
    Sanity: y1(pi/2-phi) = 0 exactly, y1' = -r sin."""
    from scipy.integrate import quad

    y1f = lambda a: 1.0 - _running_int(_r_coeffs, a, A, B, phi, theta)
    s_coeffs = lambda k, A, B, phi: tuple(1.0 - c if i == 0 else -c
                                          for i, c in enumerate(_r_coeffs(k, A, B, phi)))
    y2f = lambda a: 1.0 - _running_int(s_coeffs, a, A, B, phi, theta)
    if abs(y1f(pi/2 - phi)) > 1e-12:
        raise RuntimeError("closed-form running integral failed its trivial check")

    def f1(a):
        return y1f(a)*_r(a, A, B, phi, theta)*cos(a)

    def f2(a):
        return y2f(a)*(1.0 - _r(a, A, B, phi, theta))*cos(a)

    def f3(a):
        u, du = _u_du(a, A, B, phi, theta)
        y3 = y2f(a) - u*sin(a)
        return y3*(u*sin(a) - du*cos(a) - (1.0 - _r(a, A, B, phi, theta))*cos(a))

    pts1 = [v for v in (phi, theta, pi/2 - theta) if 0.0 < v < pi/2 - phi]
    pts3 = [v for v in (theta,) if phi < v < pi/4]
    I1 = quad(f1, 0.0, pi/2 - phi, points=pts1, limit=400,
              epsabs=1e-13, epsrel=1e-11)[0]
    I2 = quad(f2, 0.0, theta, points=[phi] if 0.0 < phi < theta else None,
              limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    I3 = quad(f3, phi, pi/4, points=pts3, limit=400,
              epsabs=1e-13, epsrel=1e-11)[0]
    return 2.0*(I1 + I2 + I3)


# ----------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("GERVER'S SOFA -- construction & cross-checks")
    print("=" * 76)

    # ---- Table 1 identity checks (Romik eqs. (27)-(34)) ----------------------
    print("\n[0] Table-1 / symmetry identities (Romik eqs. (27)-(34))")
    idents = {
        "(27) e1 = a1":         E1 - A1,
        "(28) e2 = -a2":        E2 - (-A2),
        "(29) d1 = pi/4 - b1":  D1 - (pi/4 - B1),
        "(30) d2 = b2 + (pi/4)(2b1 - pi/4)":
            D2 - (B2 + (pi/4)*(2*B1 - pi/4)),
        "(31) c2 = c1 - pi/2":  C2 - (C1 - pi/2),
        "(32) k11 = 1 - a1":    KP[0, 0] - (1.0 - A1),
        "(33) k12 = 1/4":       KP[0, 1] - 0.25,
        "(34) a2 = -1/4":       A2 - (-0.25),
        "(44c) k42 = k22":      KP[3, 1] - KP[1, 1],
    }
    for k, v in idents.items():
        print(f"    {k:<42s}  residual = {v:.2e}")

    # ---- x(0), x'(0) endpoints ----------------------------------------------
    print("\n[1] rotation path endpoints")

    def fmt2(p):
        return f"({p[0]: .15f}, {p[1]: .15f})"

    x0 = x(0.0)
    xp0 = x(0.0, der=1)
    xf = x(pi/2)
    xpf = x(pi/2, der=1)
    print(f"    x(0)     = {fmt2(x0)}          [expect (0, 0)]")
    print(f"    x'(0)    = {fmt2(xp0)}   [expect (0, 1.420644844145378)]")
    print(f"    x(pi/2)  = {fmt2(xf)}")
    print(f"    x'(pi/2) = {fmt2(xpf)}          [expect (0, -1.42064..) by (26)]")

    # analytic-derivative self-check against central differences
    # (samples strictly inside phases; edges are C^1 but not C^2)
    eps = 1e-6
    for t in (0.1, PHI - 1e-3, 0.2, THETA - 1e-3, 1.0, 1.2, pi/2 - 4*PHI, 1.55):
        fd = (x(t + eps) - x(t - eps)) / (2*eps)
        an = x(t, der=1)
        err = np.max(np.abs(fd - an))
        fd2 = (x(t + eps, der=1) - x(t - eps, der=1)) / (2*eps)
        an2 = x(t, der=2)
        err2 = np.max(np.abs(fd2 - an2))
        print(f"    t={t:<11.7f} |x'-fd gap|={err:.2e}  |x''-fd gap|={err2:.2e}")

    # ---- C^1 gluing across the four phase edges ------------------------------
    print("\n[2] C^1 gluing across phase transitions (eps = 1e-7)")
    eps = 1e-7
    for label, t in [("phi", PHI), ("theta", THETA),
                     ("pi/2-theta", pi/2 - THETA), ("pi/2-phi", pi/2 - PHI)]:
        dl = x(t - eps, der=1)
        dr = x(t + eps, der=1)
        gap = np.max(np.abs(dl - dr))                # should be ~ eps*|x''|
        pp = 0.5*(x(t - eps) + x(t + eps))
        jmp = np.max(np.abs(x(t) - pp))              # position continuity
        print(f"    t={label:<10s} |dx'| = {gap:.2e}   |x-position edge gap| = {jmp:.2e}")

    # ---- contact path endpoints ----------------------------------------------
    print("\n[3] contact-path endpoints (Romik Theorem 1)")
    a0 = contact("A", 0.0)
    c0 = contact("C", 0.0)
    d0 = contact("D", 0.0)
    b3 = contact("B", T3)
    d2 = contact("D", T2)
    print(f"    A(0)   = {fmt2(a0)}          [expect (1, 0)]  (Romik (23))")
    print(f"    C(0)   = {fmt2(c0)}          (= x(pi/2) inner-wall offset)")
    print(f"    D(0)   = {fmt2(d0)}          [expect ({c0[0]:+.15f}, 0)]")
    print(f"    B(t3)  = {fmt2(b3)}")
    print(f"    D(t2)  = {fmt2(d2)}")
    back = np.max(np.abs(c0 - d0 - np.array([0.0, 1.0])))
    print(f"    |C(0) - D(0) - (0,1)| = {back:.2e}   (vertical alignment of back corners; not a boundary edge)")

    # A frozen on phase 1, C frozen on phase 5 (Romik eq. (21)/(22) <=> ODE1)
    aP = contact("A", PHI)
    cP = contact("C", pi/2 - 1e-7)
    cL = contact("C", T4)
    print(f"    |A(phi) - A(0)|            = {np.max(np.abs(aP - a0)):.2e}  [expect ~0]")
    print(f"    |C(pi/2) - C(pi/2-phi)|    = {np.max(np.abs(cP - cL)):.2e}  [expect ~0]")

    # glue equations (43)-(44)
    e43 = np.max(np.abs(x(PHI) - contact("B", T3)))
    e44 = np.max(np.abs(x(T4) - contact("D", T2)))
    print(f"    (43) |x(phi) - B(pi/2-theta)|   = {e43:.2e}")
    print(f"    (44) |x(pi/2-phi) - D(theta)|   = {e44:.2e}")

    # ---- re-solve Gerver's 4x4 system for A, B, phi, theta -------------------
    print("\n[4] re-solving Gerver's 4x4 system (MathWorld eqs. (1)-(4))")
    (A, B, ph, th), resid = gerver_constants()
    print(f"    A       = {A:.15f}   [MathWorld: 0.094426560843653]")
    print(f"    B       = {B:.15f}   [MathWorld: 1.399203727333547]")
    print(f"    phi     = {ph:.15f}   [Table 1 : {PHI:.15f}]")
    print(f"    theta   = {th:.15f}   [Table 1 : {THETA:.15f}]")
    print(f"    max residual of the 4 eqs = {resid:.2e}")
    print(f"    |phi - PHI| = {abs(ph - PHI):.2e},   |theta - THETA| = {abs(th - THETA):.2e}")

    # ---- area via Gerver's integral ------------------------------------------
    S_int = gerver_area(A, B, ph, th)
    print("\n[5] area, independent computation 1: Gerver integral (eq. 17)")
    print(f"    S = {S_int:.15f}")
    print(f"    reference (OEIS A128463) = {GERVER_AREA:.15f}   "
          f"diff = {S_int - GERVER_AREA:+.3e}")

    # ---- area via intersection of rotating hallways --------------------------
    print("\n[6] area, independent computation 2: hallway intersection (eq. 8)")
    areas = {}
    S3600, ts = None, None
    for n in (900, 1800, 3600):
        S, ts = sofa_by_intersection(n_angles=n)
        areas[n] = S.area
        print(f"    n={n:5d}:  area = {S.area:.15f}   "
              f"diff = {S.area - GERVER_AREA:+.2e}")
        S3600 = S
    rich1 = 2*areas[1800] - areas[900]      # first order in 1/n
    rich2 = 2*areas[3600] - areas[1800]
    print(f"    1/n-Richardson (900,1800)  = {rich1:.15f}   "
          f"diff = {rich1 - GERVER_AREA:+.2e}")
    print(f"    1/n-Richardson (1800,3600) = {rich2:.15f}   "
          f"diff = {rich2 - GERVER_AREA:+.2e}")
    bb = S3600.bounds
    print(f"    bbox: x in [{bb[0]:+.6f}, {bb[2]:+.6f}],  y in [{bb[1]:+.6f}, {bb[3]:+.6f}]")

    # ---- area, independent computation 3: analytic boundary polygon ---------
    print("\n[7] area, independent computation 3: boundary polygon (shoelace)")
    Pan = sofa_polygon(n_per_curve=20000)
    Pan40 = sofa_polygon(n_per_curve=40000)
    s1_, s2_ = poly_area(Pan), poly_area(Pan40)
    ar1, ar2 = abs(s1_), abs(s2_)
    richp = (4*ar2 - ar1)/3.0       # 2nd order in 1/n_per_curve (chord error)
    print(f"    signed area, n=20000 = {s1_:+.15f}   (expect > 0: CCW loop)")
    print(f"    area n=20000 = {ar1:.15f}   diff = {ar1 - GERVER_AREA:+.2e}")
    print(f"    area n=40000 = {ar2:.15f}   diff = {ar2 - GERVER_AREA:+.2e}")
    print(f"    2nd-order Richardson     = {richp:.15f}   "
          f"diff = {richp - GERVER_AREA:+.2e}")
    # boundary agreement against the (independent) shapely intersection shape
    import shapely as _sh
    dmax = float(np.max(_sh.distance(_sh.points(Pan[::37]), S3600.boundary)))
    print(f"    max |analytic boundary -> S_3600 boundary| = {dmax:.2e}")

    # ---- save artifacts -------------------------------------------------------
    poly = np.array(S3600.exterior.coords)
    np.save("sofa_polygon_xy.npy", poly)
    np.save("sofa_polygon_analytic_xy.npy", Pan)
    np.save("hallway_angles.npy", ts)
    np.savetxt("coordinates.csv", Pan[::10], delimiter=",", header="x,y",
               comments="# analytic boundary of Gerver's sofa (n_per_curve=20000,\n"
                        "# every 10th vertex; closed CCW loop)\n")
    out = {
        "A": A, "B": B, "phi": ph, "theta": th,
        "area_integral": S_int,
        "area_intersection_900": areas[900],
        "area_intersection_1800": areas[1800],
        "area_intersection_3600": areas[3600],
        "area_intersection_rich1_900_1800": rich1,
        "area_intersection_rich2_1800_3600": rich2,
        "area_boundary_20000": ar1,
        "area_boundary_40000": ar2,
        "area_boundary_richardson": richp,
        "boundary_gap_max": dmax,
        "reference": GERVER_AREA,
        "x0": list(x0), "xprime0": list(xp0), "xpi2": list(xf),
        "A0": list(a0), "C0": list(c0), "D0": list(d0),
        "bbox": list(bb),
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n    wrote sofa_polygon_xy.npy, sofa_polygon_analytic_xy.npy, "
          "hallway_angles.npy, coordinates.csv, results.json")
    print("\nAll checks printed above; acceptance gate: area agreement <= 1e-9.")


if __name__ == "__main__":
    main()
