"""
Block 3 --- Stealthy false-data injection                 (L4)
    python block3.py
Python counts from 0: meter z_k is row k-1; theta_b is entry state_col(b).
"""

# AI declaration: generated with generative-AI assistants (Claude, Codex, OpenCode;
# 4-8 October 2026), then reviewed, corrected and verified by the instructor; the
# numbers this file prints are asserted by verify.jl. Full record: AI.log.
import networkx as nx
import numpy as np
from scipy.stats import chi2

from common import (NZ_TOL, SIGMA, HINT, check, load_case, load_checkpoint,
                    load_noise, measurement_matrix, meter_labels, min_attack_cut,
                    nnz_meters, state_col)

net = load_case()
lab = meter_labels(net)
n, m = net["n"], 54
H = measurement_matrix(net)
z = load_checkpoint("z.csv")
e = load_noise()
Rinv = np.eye(m) / SIGMA**2
G = H.T @ Rinv @ H


def estimate(zz):                               # WLS, as in Block 2
    return np.linalg.solve(G, H.T @ Rinv @ zz)


def Jstat(zz):
    r = zz - H @ estimate(zz)
    return r @ Rinv @ r


K = m - (n - 1)
tau = chi2.ppf(0.95, K)
theta_hat, J = estimate(z), Jstat(z)

# --- Task 3.1 ---------------------------------------------------------------
# Where this comes from: L4 'A model-consistent attack leaves no residual
# trace', 'The estimate moves even when the residual does not':
#   z_+ = z + a,  a = H c  =>  theta-hat_+ = theta-hat + c,  r_+ = r.
c = np.zeros(n - 1)
c[state_col(8)] = 0.05                          # move theta_8 by 0.05 rad
a = H @ c
#HINT("3.1", "the stealthy attack vector a = H c for the state shift c")
Jplus = Jstat(z + a)
check(np.allclose(estimate(z + a), theta_hat + c), "theta-hat_+ = theta-hat + c",
      "L4 'The estimate moves even when the residual does not'")
check(abs(Jplus - J) < 1e-8, f"J_+ = J = {Jplus:.2f}: the detector sees nothing",
      "L4 'A model-consistent attack leaves no residual trace'")

# --- Task 3.2 ---------------------------------------------------------------
arand = np.linalg.norm(a) * e / np.linalg.norm(e)
Jr = Jstat(z + arand)
#HINT("3.2", "the chi-squared statistic J of the attacked measurements z + arand (same Jstat as Block 2)")
check(Jr > tau, f"J = {Jr:.0f} > tau = {tau:.1f}: detected", "L4 'The detector's blind spot'")

# --- Task 3.3 ---------------------------------------------------------------
# Where this comes from: L4 'The minimum attack set as a graph cut',
# 'A pendant cut needs four meters':  k* = min over cuts A, h(A) >= 2, of |g(A)| - h(A) + 2.
supp = np.flatnonzero(np.abs(a) > NZ_TOL)
#HINT("3.3a", "the meter indices the attack changes: the non-zero entries of a")
print("the attack changes", len(supp), "meters:", ", ".join(lab[i] for i in supp))

# (i) Exact: the given black box searches every cut.
kstar, Astar, Sstar, gstar = min_attack_cut(net)    # GIVEN (common.py)
print(f"cheapest cut A* = {Astar} (cuts off buses {Sstar}):  k* = {kstar},  "
      f"g(A*) = {', '.join(lab[i] for i in gstar)}")
check(kstar == len(supp) and gstar == list(supp),
      f"k* = {kstar} = the meters your attack changes: no stealthy attack is cheaper",
      "L4 'The minimum attack set as a graph cut'")
# By hand: for A*, list g(A*), count h(A*), evaluate |g(A)| - h(A) + 2.

# (ii) Bounds without the box. Every cut you try is a stealthy attack, so its size is an
# upper bound on k*: single-line cuts (bridges) and single buses (c = e_j).
gr = nx.Graph(list(zip(net["frm"], net["to"])))
br = list(nx.bridges(gr))
print("bridges of the grid:", br, " (a single-line cut costs 2 + 2 - 2 + 2 = 4 meters)")
nnz_bus = [nnz_meters(H[:, j]) for j in range(n - 1)]
print("meters touched by c = e_j, buses 2..14:", nnz_bus)
kbus = min(nnz_bus)
#HINT("3.3b", "the best single-bus attack size: min over buses j of ||H e_j||_0")
kup = min(kbus, 4) if br else kbus          # best upper bound
klow = 4                                    # L4: |g(A)| - h(A) + 2 >= |A| + 3 >= 4 (show it by hand)
check(klow <= kstar <= kup, f"lower {klow} <= k* = {kstar} <= upper {kup}: distance {kup - klow}",
      "L4 exercise 'Certify the global minimum'")

# --- Task 3.4 ---------------------------------------------------------------
# Where this comes from: L4 'Stealthy attacks and observability'.
Hcut = np.delete(H, supp, axis=0)
rcut = np.linalg.matrix_rank(Hcut)
#HINT("3.4", "the rank of H after deleting the rows of the attack's meters")
check(rcut == n - 2, f"rank drops to {rcut}", "L4 'Stealthy attacks and observability'")
_, _, Vt = np.linalg.svd(Hcut)
print("lost direction (null space of Hcut):", np.round(Vt[-1], 3))

# --- Task 3.5 ---------------------------------------------------------------
# Where this comes from: L4 'A sufficient access bound' (k > K always suffices).
print(f"K + 1 = {K + 1} meters are always enough; the cut needs {len(supp)}")
