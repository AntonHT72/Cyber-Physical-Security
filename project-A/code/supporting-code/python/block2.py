"""Block 2 --- AC measurements and the estimator (L2, L3).

The measurement vector contains active and reactive bus injections, active and
reactive branch-end flows, and voltage magnitudes for the IEEE 57-bus case.
The AC measurement model is nonlinear, so the estimator uses Gauss-Newton
weighted least squares rather than the linear DC normal equations.
"""
import numpy as np
from scipy.stats import chi2

from common import SIGMA, branch_flows, bus_injections, load_case


net = load_case()
n = net["n"]
slack = net["slack"] - 1
keep = np.asarray(net["keep"]) - 1


def voltage_from_state(x):
    """Convert [non-slack angles, voltage magnitudes] to complex bus voltages."""
    angles = np.zeros(n)
    angles[keep] = x[: n - 1]
    magnitudes = x[n - 1:]
    return magnitudes * np.exp(1j * angles)


def state_from_voltage(voltage):
    """Convert complex bus voltages to the estimator state ordering."""
    return np.r_[np.angle(voltage[keep]), np.abs(voltage)]


def measurement_labels():
    labels = [f"P{i}" for i in range(1, n + 1)]
    labels += [f"Q{i}" for i in range(1, n + 1)]
    for i, j in zip(net["frm"], net["to"]):
        labels += [f"P{i},{j}", f"P{j},{i}"]
    for i, j in zip(net["frm"], net["to"]):
        labels += [f"Q{i},{j}", f"Q{j},{i}"]
    labels += [f"V{i}" for i in range(1, n + 1)]
    return labels


def measurement_model(x):
    """Return h(x) for the full AC measurement set, in per unit."""
    voltage = voltage_from_state(x)
    injections = bus_injections(net, voltage)
    flows = branch_flows(net, voltage)
    return np.r_[
        injections.real,
        injections.imag,
        flows[:, 0].real,
        flows[:, 1].real,
        flows[:, 0].imag,
        flows[:, 1].imag,
        np.abs(voltage),
    ]


def numerical_jacobian(x, step=1e-6):
    """Central-difference Jacobian of the AC measurement model."""
    h0 = measurement_model(x)
    H = np.empty((h0.size, x.size))
    for column in range(x.size):
        dx = np.zeros_like(x)
        dx[column] = step
        H[:, column] = (measurement_model(x + dx) - measurement_model(x - dx)) / (2 * step)
    return H


def estimate(z, x0):
    """Estimate the AC state with iterated Gauss-Newton WLS."""
    x = x0.copy()
    for iteration in range(30):
        h = measurement_model(x)
        H = numerical_jacobian(x)
        delta = np.linalg.solve(H.T @ H, H.T @ (z - h))
        x += delta
        if np.linalg.norm(delta, np.inf) < 1e-9:
            return x, iteration + 1
    raise RuntimeError("AC WLS did not converge within 30 iterations")


labels = measurement_labels()
x_true = state_from_voltage(net["V"])
m = measurement_model(x_true).size
rng = np.random.default_rng(1)
noise = SIGMA * rng.standard_normal(m)
z = measurement_model(x_true) + noise

print(f"measurement count: {m}")
print(f"state count: {x_true.size} ({n - 1} angles + {n} magnitudes)")
print(f"measurement groups: {n} P, {n} Q, {2 * net['nb']} P-flow, "
      f"{2 * net['nb']} Q-flow, {n} V")

# --- Task 2.1 ---------------------------------------------------------------
# Unlike the DC model, H is the Jacobian of the nonlinear AC measurement map.
x_flat = np.r_[np.zeros(n - 1), np.ones(n)]
H = numerical_jacobian(x_true)
print(f"Jacobian shape: {H.shape}, rank: {np.linalg.matrix_rank(H)}")
if np.linalg.matrix_rank(H) < x_true.size:
    raise RuntimeError("the AC measurement set is not observable")

# --- Task 2.2 ---------------------------------------------------------------
# z = h(x) + e, e ~ N(0, SIGMA^2 I). Estimate around a flat-voltage start.
x_hat, iterations = estimate(z, x_flat)
H_hat = numerical_jacobian(x_hat)
r = z - measurement_model(x_hat)
Rinv = np.eye(m) / SIGMA**2
G = H_hat.T @ Rinv @ H_hat
covariance = np.linalg.inv(G)
print(f"Gauss-Newton iterations: {iterations}")
print(f"max angle error: {np.degrees(np.abs(x_hat[:n - 1] - x_true[:n - 1])).max():.4f} deg")
print(f"max voltage-magnitude error: {np.abs(x_hat[n - 1:] - x_true[n - 1:]).max():.5f} p.u.")

# --- Task 2.3 ---------------------------------------------------------------
# The local linearized residual projector gives the usual AC BDD statistic.
projection = H_hat @ np.linalg.solve(G, H_hat.T @ Rinv)
J = r @ Rinv @ r
K = m - (x_true.size)
tau = chi2.ppf(0.95, K)
print(f"J = {J:.2f}   K = {K}   tau = {tau:.2f}")
print(f"residual norm: {np.linalg.norm(r):.5f}")

# --- Task 2.4 ---------------------------------------------------------------
# The normalized residual identifies the most suspicious measurement.
Omega = SIGMA**2 * (np.eye(m) - projection)
normalized_residual = np.abs(r) / np.sqrt(np.maximum(np.diag(Omega), 1e-14))
bad = int(np.argmax(normalized_residual))
print(f"largest normalized residual = {normalized_residual[bad]:.2f} "
      f"at z_{bad + 1} ({labels[bad]})")

# --- Task 2.5 (stretch) -----------------------------------------------------
# Monte Carlo check of the local chi-squared residual statistic.
residual_projector = np.eye(m) - projection
Jmc = np.empty(2000)
for sample in range(Jmc.size):
    e = SIGMA * rng.standard_normal(m)
    residual = residual_projector @ e
    Jmc[sample] = residual @ Rinv @ residual
print(f"Task 2.5  mean J = {Jmc.mean():.2f} (K = {K}), "
      f"var J = {Jmc.var():.1f} (2K = {2 * K}), "
      f"fraction J > tau = {np.mean(Jmc > tau):.3f}")
