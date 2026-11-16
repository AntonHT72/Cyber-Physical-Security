"""Block 3 --- model-consistent false-data injection for AC state estimation."""
import numpy as np
from scipy.stats import chi2

from common import (SIGMA, NZ_TOL, branch_flows, bus_injections, load_case,
                    measurement_matrix)


net = load_case()
n = net["n"]
keep = np.asarray(net["keep"]) - 1
slack = net["slack"] - 1


def voltage_from_state(x):
    angles = np.zeros(n)
    angles[keep] = x[: n - 1]
    return x[n - 1:] * np.exp(1j * angles)


def state_from_voltage(voltage):
    return np.r_[np.angle(voltage[keep]), np.abs(voltage)]


def measurement_labels():
    labels = [f"P{i}" for i in range(1, n + 1)]
    labels += [f"Q{i}" for i in range(1, n + 1)]
    for i, j in zip(net["frm"], net["to"]):
        labels += [f"P{i},{j}", f"P{j},{i}"]
    for i, j in zip(net["frm"], net["to"]):
        labels += [f"Q{i},{j}", f"Q{j},{i}"]
    return labels + [f"V{i}" for i in range(1, n + 1)]


def measurement_model(x):
    voltage = voltage_from_state(x)
    injections = bus_injections(net, voltage)
    flows = branch_flows(net, voltage)
    return np.r_[
        injections.real, injections.imag,
        flows[:, 0].real, flows[:, 1].real,
        flows[:, 0].imag, flows[:, 1].imag,
        np.abs(voltage),
    ]


def numerical_jacobian(x, step=1e-6):
    H = np.empty((measurement_model(x).size, x.size))
    for column in range(x.size):
        dx = np.zeros_like(x)
        dx[column] = step
        H[:, column] = (measurement_model(x + dx) -
                        measurement_model(x - dx)) / (2 * step)
    return H


def estimate(z, x0):
    """Iterated AC WLS with equal measurement variances."""
    x = x0.copy()
    for iteration in range(30):
        residual = z - measurement_model(x)
        H = numerical_jacobian(x)
        delta = np.linalg.solve(H.T @ H, H.T @ residual)
        x += delta
        if np.linalg.norm(delta, np.inf) < 1e-9:
            return x, iteration + 1
    raise RuntimeError("AC WLS did not converge within 30 iterations")


labels = measurement_labels()
x_true = state_from_voltage(net["V"])
x_flat = np.r_[np.zeros(n - 1), np.ones(n)]
m = measurement_model(x_true).size
states = x_true.size
rng = np.random.default_rng(1)
z = measurement_model(x_true) + SIGMA * rng.standard_normal(m)


def Jstat(zz, x0=x_flat):
    x_hat, _ = estimate(zz, x0)
    residual = zz - measurement_model(x_hat)
    return residual @ residual / SIGMA**2, x_hat


H = numerical_jacobian(x_true)
rank = np.linalg.matrix_rank(H)
K = m - states
tau = chi2.ppf(0.95, K)
J, x_hat = Jstat(z)
print(f"measurement count: {m}, state count: {states}, Jacobian rank: {rank}")
print(f"J = {J:.2f}   K = {K}   tau = {tau:.2f}")

# --- Task 3.1 ---------------------------------------------------------------
# For a nonlinear AC model, the model-consistent attack is the difference
# between measurements at the shifted and original states.
c = np.zeros(states)
angle8_col = 8 - 2
c[angle8_col] = 0.05
x_shifted = x_true + c
a = measurement_model(x_shifted) - measurement_model(x_true)
Jplus, x_plus = Jstat(z + a, x0=x_shifted)
print(f"AC attack changes {np.count_nonzero(np.abs(a) > NZ_TOL)} of {m} meters")
print(f"estimated bus-8 angle shift: {x_plus[angle8_col] - x_hat[angle8_col]:.6f} rad")
print(f"clean J = {J:.2f}, attacked J = {Jplus:.2f}")

# --- Task 3.2 ---------------------------------------------------------------
# A random measurement corruption with the same norm is not model-consistent.
arand = np.linalg.norm(a) * rng.standard_normal(m)
arand *= 1.0 / np.linalg.norm(arand)
Jrandom, _ = Jstat(z + arand)
print(f"random attack J = {Jrandom:.2f}; detected = {Jrandom > tau}")

# --- Task 3.3 ---------------------------------------------------------------
# The exact graph-cut minimum from the DC model does not apply to nonlinear
# AC measurements. Report the support and its measurement labels instead.
support = np.flatnonzero(np.abs(a) > NZ_TOL)
print("changed measurements:", ", ".join(labels[i] for i in support))

# --- Task 3.4 ---------------------------------------------------------------
# Removing the changed rows from the local AC Jacobian shows the remaining
# local observability of this particular attack.
H_remaining = np.delete(H, support, axis=0)
remaining_rank = np.linalg.matrix_rank(H_remaining)
print(f"local Jacobian rank after removing attack rows: "
      f"{remaining_rank} of {states}")

# --- Task 3.5 ---------------------------------------------------------------
# Apply the corresponding DC attack to the AC estimator. The DC vector only
# contains active-power injections and active-power branch flows, so the
# reactive-power and voltage-magnitude entries remain unchanged.
H_dc = measurement_matrix(net)
c_dc = np.zeros(n - 1)
c_dc[8 - 2] = 0.05
a_dc = H_dc @ c_dc
a_dc_ac = np.zeros(m)
a_dc_ac[:n] = a_dc[:n]
a_dc_ac[2 * n:2 * n + 2 * net["nb"]] = a_dc[n:]
J_dc_attack, x_dc_attack = Jstat(z + a_dc_ac)
print(f"DC attack embedded in AC measurements changes "
      f"{np.count_nonzero(np.abs(a_dc_ac) > NZ_TOL)} of {m} meters")
print(f"DC attack against AC estimator: J = {J_dc_attack:.2f}, "
      f"tau = {tau:.2f}, detected = {J_dc_attack > tau}")
print(f"AC-estimated bus-8 angle shift from DC attack: "
      f"{x_dc_attack[8 - 2] - x_hat[8 - 2]:.6f} rad")
