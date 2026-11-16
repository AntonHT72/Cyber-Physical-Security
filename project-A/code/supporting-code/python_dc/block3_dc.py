"""Block 3 --- stealthy false-data injection for the 57-bus DC estimator."""
import networkx as nx
import numpy as np
from scipy.stats import chi2

from common import (NZ_TOL, SIGMA, load_case, load_noise, measurement_matrix,
                    meter_labels, min_attack_cut, state_col)


net = load_case()
n = net["n"]
m = n + 2 * net["nb"]
labels = meter_labels(net)
H = measurement_matrix(net)

# Recreate the DC operating point used by block2_dc.py.
P = net["Pg"] - net["Pd"]
keep = np.asarray(net["keep"]) - 1
B = np.zeros((n, n))
for i, j, x in zip(net["frm"], net["to"], net["x"]):
    i, j = i - 1, j - 1
    y = 1.0 / x
    B[i, i] += y
    B[j, j] += y
    B[i, j] -= y
    B[j, i] -= y
theta = np.linalg.solve(B[np.ix_(keep, keep)], P[keep])

e = load_noise()
z = H @ theta + e
Rinv = np.eye(m) / SIGMA**2
G = H.T @ Rinv @ H


def estimate(measurements):
    """Linear WLS estimate for the DC model."""
    return np.linalg.solve(G, H.T @ Rinv @ measurements)


def Jstat(measurements):
    residual = measurements - H @ estimate(measurements)
    return residual @ Rinv @ residual


K = m - (n - 1)
tau = chi2.ppf(0.95, K)
theta_hat = estimate(z)
J = Jstat(z)
print(f"measurement count: {m}, state count: {n - 1}, rank: {np.linalg.matrix_rank(H)}")
print(f"J = {J:.2f}   K = {K}   tau = {tau:.2f}")

# --- Task 3.1 ---------------------------------------------------------------
# For the linear DC model, a model-consistent attack is a = H c.
c = np.zeros(n - 1)
c[state_col(8)] = 0.05
a = H @ c
theta_plus = estimate(z + a)
Jplus = Jstat(z + a)
print(f"estimated bus-8 angle shift: "
      f"{theta_plus[state_col(8)] - theta_hat[state_col(8)]:.6f} rad")
print(f"attack changes {np.count_nonzero(np.abs(a) > NZ_TOL)} of {m} meters")
print(f"clean J = {J:.2f}, attacked J = {Jplus:.2f}, "
      f"residual unchanged = {abs(Jplus - J) < 1e-8}")

# --- Task 3.2 ---------------------------------------------------------------
# A random measurement corruption with the same norm is not model-consistent.
arand = np.linalg.norm(a) * e / np.linalg.norm(e)
Jrandom = Jstat(z + arand)
print(f"random attack J = {Jrandom:.2f}; detected = {Jrandom > tau}")

# --- Task 3.3 ---------------------------------------------------------------
# The exact minimum-support cut remains applicable to this DC measurement set.
support = np.flatnonzero(np.abs(a) > NZ_TOL)
print("changed measurements:", ", ".join(labels[i] for i in support))
kstar, astar, sstar, gstar = min_attack_cut(net)
print(f"minimum cut: {astar}, separates buses {sstar}, k* = {kstar}")
print(f"cut measurements: {', '.join(labels[i] for i in gstar)}")

graph = nx.Graph(zip(net["frm"], net["to"]))
bridges = list(nx.bridges(graph))
single_bus_sizes = [np.count_nonzero(np.abs(H[:, j]) > NZ_TOL)
                    for j in range(n - 1)]
print(f"grid bridges: {bridges}")
print(f"smallest single-bus attack: {min(single_bus_sizes)} meters")

# --- Task 3.4 ---------------------------------------------------------------
Hcut = np.delete(H, support, axis=0)
rank_cut = np.linalg.matrix_rank(Hcut)
print(f"rank after deleting the attack measurements: {rank_cut} of {n - 1}")
if Hcut.shape[0]:
    _, _, Vt = np.linalg.svd(Hcut)
    print("lost direction:", np.round(Vt[-1], 3))

# --- Task 3.5 ---------------------------------------------------------------
print(f"K + 1 = {K + 1} meters are always enough; "
      f"the constructed attack changes {len(support)}")
