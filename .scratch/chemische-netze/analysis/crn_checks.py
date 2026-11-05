#!/usr/bin/env python3
"""
Our own checks for `research.md`, sections 3.7 and 8.2. Not peer reviewed.

  1. The ReLU module read at a fixed time, against a closed formula
  2. The error at the kink times the read-out time
  3. A finite reverse rate, against the smoothed ReLU of source 9, Eq. 4
  4. The largest error against the free-energy drop per reaction
  5. The error at the kink against the overlap of the two rails
  6. A chain of reversible modules without weights
  7. A step rule, on a star of 50 reactions
  8. The kink error and the smallest concentration after 1000 RK4 steps
  9. The share of subnormal values in those steps, with and without reverse rates

Run: python3 crn_checks.py
Needs: numpy, scipy
"""

import numpy as np
from scipy.integrate import solve_ivp


def module_rhs(t, s, k1f, k1r, k2f, k2r):
    """X+ <-> M + Y+ and M + X- <-> Y-."""
    xp, xm, m, yp, ym = s
    j1 = k1f * xp - k1r * m * yp
    j2 = k2f * m * xm - k2r * ym
    return [-j1, -j2, j1 - j2, j1, j2]


def run_module(xp0, xm0, t_end, k1f = 1.0, k1r = 0.0, k2f = 1.0, k2r = 0.0):
    """Returns the output y+ - y- and y- at t_end."""
    sol = solve_ivp(
        module_rhs,
        (0.0, t_end),
        [xp0, xm0, 0.0, 0.0, 0.0],
        args = (k1f, k1r, k2f, k2r),
        method = "Radau",
        rtol = 1e-10,
        atol = 1e-13,
    )
    xp, xm, m, yp, ym = sol.y[:, -1]
    return yp - ym, ym


def chain_rhs(t, s, depth, kf, kr):
    """The outputs of one module are the inputs of the next."""
    out = np.zeros_like(s)

    for layer in range(depth):
        if layer == 0:
            ip, im = 0, 1
        else:
            previous = 2 + 3 * (layer - 1)
            ip, im = previous + 1, previous + 2

        base = 2 + 3 * layer
        m, yp, ym = base, base + 1, base + 2
        j1 = kf * s[ip] - kr * s[m] * s[yp]
        j2 = kf * s[m] * s[im] - kr * s[ym]
        out[ip] -= j1
        out[m] += j1 - j2
        out[yp] += j1
        out[im] -= j2
        out[ym] += j2

    return out


def chain_output(x, depth, d_g):
    k = np.exp(d_g)
    s0 = np.zeros(2 + 3 * depth)
    s0[0] = 1.0 + x
    s0[1] = 1.0
    sol = solve_ivp(
        chain_rhs,
        (0.0, 1e9),
        s0,
        args = (depth, 1.0, 1.0 / k),
        method = "Radau",
        rtol = 1e-9,
        atol = 1e-12,
    )
    last = 2 + 3 * (depth - 1)
    return sol.y[last + 1, -1] - sol.y[last + 2, -1]


def rk4(rhs, s, dt, steps):
    """Returns the final state, None if it diverges, and the smallest positive value seen."""
    smallest = np.inf

    for _ in range(steps):
        a = rhs(s)
        b = rhs(s + 0.5 * dt * a)
        c = rhs(s + 0.5 * dt * b)
        d = rhs(s + dt * c)
        s = s + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)

        if not np.all(np.isfinite(s)) or np.max(np.abs(s)) > 1e12:
            return None, smallest

        positive = s[s > 0.0]

        if positive.size:
            smallest = min(smallest, float(positive.min()))

    return s, smallest


def check_fixed_time():
    print("1. Fixed time, k1 = 1e4, k2 = 1, x- = 1: y = x / (1 - C exp(-k T x)), C = 1 - x / x+")

    for k_t in (10.0, 100.0):
        for x in (-0.5, -0.05, 0.0, 0.05, 0.5):
            xp0 = 1.0 + x
            y, _ = run_module(xp0, 1.0, k_t, k1f = 1e4)

            if abs(x) < 1e-12:
                formula = 1.0 / (1.0 / xp0 + k_t)
            else:
                formula = x / (1.0 - (1.0 - x / xp0) * np.exp(-k_t * x))

            print(f"   kT={k_t:6.1f} x={x:+.2f} ode={y:.6f} formula={formula:.6f} relu={max(x, 0):.2f}")


def check_kink_tail():
    print("2. Kink x = 0, x+ = x- = 1, k1 = k2 = 1: y times T tends to 1")

    for t_end in (10.0, 100.0, 1000.0, 10000.0):
        y, _ = run_module(1.0, 1.0, t_end)
        print(f"   T={t_end:8.0f} y={y:.6e} y*T={y * t_end:.4f}")


def check_smoothed():
    print("3. Equilibrium, only the second reaction reversible, K = exp(dG), h = y- / K")

    for d_g in (5.0, 9.2, 15.0):
        k = np.exp(d_g)

        for x in (-0.5, 0.0, 0.5):
            y, ym = run_module(1.0 + x, 1.0, 1e7, k1f = 1e3, k2r = 1.0 / k)
            smooth = (x + np.sqrt(x * x + 4.0 * ym / k)) / 2.0
            print(f"   dG={d_g:5.1f} x={x:+.1f} y={y:.6f} smoothed={smooth:.6f} relu={max(x, 0):.2f}")


def check_error_against_energy():
    print("4. Both reactions reversible with the same dG, x- = 1, x in [-1, 1]")
    grid = np.linspace(-1.0, 1.0, 41)

    for d_g in (5.0, 7.0, 9.2, 12.0, 15.0, 20.0):
        k = np.exp(d_g)
        errors = [
            abs(run_module(1.0 + x, 1.0, 1e9, k1r = 1.0 / k, k2r = 1.0 / k)[0] - max(x, 0.0))
            for x in grid
        ]
        worst = int(np.argmax(errors))
        print(f"   dG={d_g:5.1f} RT error={errors[worst]:.5f} at x={grid[worst]:+.2f} sqrt(1/K)={np.sqrt(1.0 / k):.5f}")


def check_overlap():
    print("5. Kink x = 0 with x+ = x- = c, both reactions reversible, dG = 9.2 RT")
    k = np.exp(9.2)

    for c in (0.01, 0.1, 1.0, 10.0):
        y, _ = run_module(c, c, 1e9, k1r = 1.0 / k, k2r = 1.0 / k)
        print(f"   c={c:6.2f} error={y:.6f} sqrt(c/K)={np.sqrt(c / k):.6f}")


def check_chain():
    print("6. Chain without weights at equilibrium, the same dG in all reactions")
    grid = np.linspace(-1.0, 1.0, 21)

    for d_g in (5.0, 9.2, 15.0):
        for depth in (1, 2, 3, 4):
            errors = [abs(chain_output(x, depth, d_g) - max(x, 0.0)) for x in grid]
            print(f"   dG={d_g:5.1f} depth={depth} error={max(errors):.5f}")


def check_step_rule():
    print("7. Star A -> B_i, 50 reactions with rate constant 1, RK4, bound about 2.785")
    n = 50

    def star(s):
        out = np.zeros_like(s)
        out[0] = -n * s[0]
        out[1:] = s[0]
        return out

    for dt in (0.05, 0.0557, 0.06, 0.1, 0.5, 2.0):
        s0 = np.zeros(n + 1)
        s0[0] = 1.0
        result, _ = rk4(star, s0, dt, 1000)
        state = "diverged" if result is None else f"A={result[0]:.3e}"
        print(f"   dt={dt:<7} single rate={dt:6.3f} loss sum={dt * n:7.3f} -> {state}")


def check_budget():
    print("8. ReLU module, 1000 RK4 steps, x+ = x- = 1, k1 = k2 = 1")

    def module(s):
        xp, xm, m, yp, ym = s
        j1 = xp
        j2 = m * xm
        return np.array([-j1, -j2, j1 - j2, j1, j2])

    for dt in (0.1, 0.5, 1.0, 1.5, 2.0):
        result, smallest = rk4(module, np.array([1.0, 1.0, 0.0, 0.0, 0.0]), dt, 1000)

        if result is None:
            print(f"   dt={dt}: diverged")
        else:
            print(f"   dt={dt}: T={dt * 1000:.0f} y={result[3] - result[4]:.3e} smallest positive value={smallest:.3e}")


def check_subnormal_share():
    print("9. Share of subnormal values among states and rates, module, 1000 RK4 steps, x+ = x- = 1")
    tiny = np.finfo(float).tiny

    for name, k_r in (("irreversible", 0.0), ("dG = 20 RT", np.exp(-20.0)), ("dG = 100 RT", np.exp(-100.0))):
        for dt in (1.0, 1.5):
            counts = [0, 0]
            smallest = [np.inf]

            def tally(values):
                values = np.asarray(values)
                counts[0] += int(np.sum((values != 0.0) & (np.abs(values) < tiny)))
                counts[1] += values.size
                positive = values[values > 0.0]

                if positive.size:
                    smallest[0] = min(smallest[0], float(positive.min()))

            def rhs(s):
                xp, xm, m, yp, ym = s
                rates = (xp, k_r * m * yp, m * xm, k_r * ym)
                tally(s)
                tally(rates)
                j1 = rates[0] - rates[1]
                j2 = rates[2] - rates[3]
                return np.array([-j1, -j2, j1 - j2, j1, j2])

            s = np.array([1.0, 1.0, 0.0, 0.0, 0.0])

            for _ in range(1000):
                a = rhs(s)
                b = rhs(s + 0.5 * dt * a)
                c = rhs(s + 0.5 * dt * b)
                d = rhs(s + dt * c)
                s = s + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)

            print(f"   {name:13} dt={dt}: subnormal share={100.0 * counts[0] / counts[1]:.3f} % smallest positive value={smallest[0]:.3e}")


def main():
    check_fixed_time()
    check_kink_tail()
    check_smoothed()
    check_error_against_energy()
    check_overlap()
    check_chain()
    check_step_rule()
    check_budget()
    check_subnormal_share()


if __name__ == "__main__":
    main()
