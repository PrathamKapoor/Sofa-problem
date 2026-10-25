"""Area of Gerver's sofa at high precision (mpmath; closed-form running integrals).

Independent of scipy: solves Gerver's 4x4 system at 40 dps with mp.findroot and
computes MathWorld eq. (17) using exact piecewise antiderivatives for the
running integrals y1, y2 (MathWorld eqs. (14)-(16)).
"""
import mpmath as mp
from mpmath import mpf, cos, sin, pi

mp.mp.dps = 40
half = mpf(1)/2
quarter = mpf(1)/4


def solve_constants():
    def eqs(A, B, phi, theta):
        return [
            A*(cos(theta) - cos(phi)) - 2*B*sin(phi)
            + (theta - phi - 1)*cos(theta) - sin(theta) + cos(phi) + sin(phi),
            A*(3*sin(theta) + sin(phi)) - 2*B*cos(phi)
            + 3*(theta - phi - 1)*sin(theta) + 3*cos(theta) - sin(phi) + cos(phi),
            A*cos(phi) - (sin(phi) + half - half*cos(phi) + B*sin(phi)),
            (A + pi/2 - phi - theta)
            - (B - half*(theta - phi)*(1 + A) - quarter*(theta - phi)**2),
        ]
    x0 = [mpf("0.09442656084365283"), mpf("1.3992037273335471"),
          mpf("0.03917736479008362"), mpf("0.6813015093827246")]
    return mp.findroot(eqs, x0, tol=mp.mpf("1e-30"))


def main():
    A, B, phi, theta = solve_constants()
    print("A      =", mp.nstr(A, 20))
    print("B      =", mp.nstr(B, 20))
    print("phi    =", mp.nstr(phi, 20))
    print("theta  =", mp.nstr(theta, 20))
    m = pi/2 - phi

    # ---- coefficient tables c = (c0, c1, c2) of the piece polynomial in t ----
    def r_coeffs(k):
        if k == 1:   # [0, phi): r = 1/2
            return (half, mpf(0), mpf(0))
        if k == 2:   # [phi, theta): r = (1+A+a-phi)/2
            return ((1 + A - phi)/2, half, mpf(0))
        if k == 3:   # [theta, pi/2-theta): r = A + a - phi
            return (A - phi, mpf(1), mpf(0))
        if k == 4:   # [pi/2-theta, pi/2-phi]: quadratic in u = m - a
            return (B - half*m*(1 + A) - quarter*m*m,
                    half*(1 + A) + half*m,
                    -quarter)
        raise ValueError(k)

    def s_coeffs(k):
        c0, c1, c2 = r_coeffs(k)
        return (1 - c0, -c1, -c2)

    # F_k(t): antiderivative of (c0 + c1 t + c2 t^2) sin t
    def Fk(c, t):
        c0, c1, c2 = c
        return (-c0*cos(t)
                + c1*(-t*cos(t) + sin(t))
                + c2*(-t*t*cos(t) + 2*t*sin(t) + 2*cos(t)))

    # piece boundaries per target outer variable
    pieces_r = [(mpf(0), phi, 1), (phi, theta, 2),
                (theta, pi/2 - theta, 3), (pi/2 - theta, pi/2 - phi, 4)]

    def running(cfun, x):
        """exact accumulation of int_0^x (piecewise poly) sin t dt"""
        tot = mpf(0)
        for lo, hi, k in pieces_r:
            if x <= lo:
                break
            top = min(x, hi)
            tot += Fk(cfun(k), top) - Fk(cfun(k), lo)
            if top >= x:
                break
        return tot

    def y1(a):  # a <= pi/2 - phi only
        return 1 - running(r_coeffs, a)

    def y2(a):  # a <= pi/2 - theta < pi/4 uses pieces 1..3
        return 1 - running(s_coeffs, a)

    def r(a):
        k = 1 if a < phi else (2 if a < theta else
                               (3 if a < pi/2 - theta else 4))
        c0, c1, c2 = r_coeffs(k)
        return c0 + c1*a + c2*a*a

    def s(a):
        return 1 - r(a)

    def u_du(a):
        if a < theta:
            u = B - half*(a - phi)*(1 + A) - quarter*(a - phi)**2
            du = -half*(1 + A) - half*(a - phi)
        else:
            u = A + pi/2 - phi - a
            du = -1
        return u, du

    I1 = mp.quad(lambda a: y1(a)*r(a)*cos(a),
                 [0, phi, theta, pi/2 - theta, pi/2 - phi])
    I2 = mp.quad(lambda a: y2(a)*s(a)*cos(a), [0, phi, theta])
    I3 = mp.quad(
        lambda a: (y2(a) - u_du(a)[0]*sin(a))
                  * (u_du(a)[0]*sin(a) - u_du(a)[1]*cos(a) - s(a)*cos(a)),
        [phi, theta, pi/4])
    S = 2*(I1 + I2 + I3)
    print("I1 =", mp.nstr(I1, 24))
    print("I2 =", mp.nstr(I2, 24))
    print("I3 =", mp.nstr(I3, 24))
    print("S  =", mp.nstr(S, 24))
    print("?= 2.21953166887196   diff =", mp.nstr(S - mpf("2.21953166887196"), 12))


if __name__ == "__main__":
    main()
