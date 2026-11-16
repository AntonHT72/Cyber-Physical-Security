"""Block 2 --- DC measurements and the estimator for the IEEE 57-bus case."""
import numpy as np
from scipy.stats import chi2

from common import (NZ_TOL, SIGMA, flow_meter, load_case, load_noise,
                    load_perms, measurement_matrix, meter_labels, state_col)


net = load_case()
n = net["n"]
m = n + 2 * net["nb"]
labels = meter_labels(net)

# The true DC state is obtained exactly as in Block 1.
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

# --- Task 2.1 ---------------------------------------------------------------
# The DC measurement matrix contains one injection meter per bus and two flow
# meters per branch. It has n-1 angle columns after removing the slack column.
H = measurement_matrix(net)
rank = np.linalg.matrix_rank(H)
print(f"measurement matrix shape: {H.shape}, rank: {rank} = {n - 1}")
if rank != n - 1:
    raise RuntimeError("the 57-bus DC measurement set is not observable")

# --- Task 2.2 ---------------------------------------------------------------
# z = H theta + e, with independent Gaussian meter noise.
e = load_noise()
z = H @ theta + e
Rinv = np.eye(m) / SIGMA**2
G = H.T @ Rinv @ H
theta_hat = np.linalg.solve(G, H.T @ Rinv @ z)
print(f"max |theta-hat - theta| = {np.abs(theta_hat - theta).max():.4f} rad")

# --- Task 2.3 ---------------------------------------------------------------
# Linear WLS residual and its projection matrix.
Pm = H @ np.linalg.solve(G, H.T @ Rinv)
r = z - H @ theta_hat
print(f"projection error ||P^2-P|| = {np.linalg.norm(Pm @ Pm - Pm):.2e}")
print(f"weighted normal-equation residual = {np.linalg.norm(H.T @ Rinv @ r):.2e}")

# --- Task 2.4 ---------------------------------------------------------------
J = r @ Rinv @ r
K = m - (n - 1)
tau = chi2.ppf(0.95, K)
print(f"J = {J:.2f}   K = {K}   tau = {tau:.2f}")

# --- Task 2.5 ---------------------------------------------------------------
# Flip one flow measurement, identify it with the largest normalized residual,
# then remove it and repeat the WLS estimate.
kbad = flow_meter(net, 1, 2)
zb = z.copy()
zb[kbad] = -zb[kbad]
theta_bad = np.linalg.solve(G, H.T @ Rinv @ zb)
rb = zb - H @ theta_bad
Omega = SIGMA**2 * (np.eye(m) - Pm)
rN = np.abs(rb) / np.sqrt(np.maximum(np.diag(Omega), 1e-14))
flagged = int(np.argmax(rN))
print(f"largest normalized residual = {rN[flagged]:.2f} "
      f"at z_{flagged + 1} ({labels[flagged]})")

kept = [i for i in range(m) if i != flagged]
Hk, zk = H[kept], zb[kept]
Gk = Hk.T @ Hk / SIGMA**2
theta_kept = np.linalg.solve(Gk, Hk.T @ zk / SIGMA**2)
rk = zk - Hk @ theta_kept
Jk = rk @ rk / SIGMA**2
Kk = len(kept) - np.linalg.matrix_rank(Hk)
print(f"after removing z_{flagged + 1}: J = {Jk:.2f}, "
      f"tau = {chi2.ppf(0.95, Kk):.2f}")

# --- Task 2.6 ---------------------------------------------------------------
# Random meter subsets, now generated with 217 rows for the 57-bus case.
perms = load_perms()
N = perms.shape[0]


def rank_subset(rows):
    return np.linalg.matrix_rank(H[rows])


def full_rank_fraction(size):
    return np.mean([rank_subset(perms[i, :size]) == n - 1 for i in range(N)])


print(f"fraction of {N} random subsets with full rank:")
for size in (n - 1, n + 5, n + 20, 2 * n, 3 * n):
    if size <= m:
        print(f"    m = {size:3d}   f = {full_rank_fraction(size):.3f}")

m1 = next(size for size in range(1, m + 1)
           if rank_subset(perms[0, :size]) == n - 1)
full_rows = list(perms[0, :m1])
touch8 = [i for i in range(m) if abs(H[i, state_col(8)]) > NZ_TOL]
deficient_rows = [i for i in full_rows if i not in touch8]
print(f"first full-rank subset: {m1} meters")
print(f"rank after removing meters seeing angle 8: {rank_subset(deficient_rows)}")
