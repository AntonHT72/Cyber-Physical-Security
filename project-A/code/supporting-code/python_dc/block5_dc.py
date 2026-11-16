"""
Block 5 --- Defence and wrap-up (OPTIONAL)           (L1, L4, L6)
    python block5.py
Run only if time permits: turns the L4/L6 indices into a defence decision.
Python counts from 0: meter z_k is row k-1.
"""

# AI declaration: generated with generative-AI assistants (Claude, Codex, OpenCode;
# 4-8 October 2026), then reviewed, corrected and verified by the instructor; the
# numbers this file prints are asserted by verify.jl. Full record: AI.log.
import numpy as np

from common import (NZ_TOL, HINT, beta_index, load_case, load_checkpoint,
                    measurement_matrix, meter_labels, nnz_meters)

net = load_case()
lab = meter_labels(net)
n, m = net["n"], 54
H = measurement_matrix(net)
beta = load_checkpoint("beta.csv")
abar = load_checkpoint("abar.csv").astype(int)

# --- Task 5.1 --- the redundancy ladder, with alpha --------------------------
# First the two small placements: 13 injections (buses 2..14) and 14 injections (all
# buses). Their rank, redundancy K = |S| - rank(H_S) and Omega_ii = 0 (critical meter)
# give you alpha_k BY HAND using a row-basis argument for the targeted bound
# alpha_k <= K + 1 (L2, beyond L4's untargeted access bound) and L6's
# alpha_k = 1 iff z_k is critical. Omega / sigma^2 = I - P for R = sigma^2 I.
S13, S14 = list(range(1, 14)), list(range(14))     # rows of P_2..P_14 / P_1..P_14
print("13 injections:", ", ".join(lab[i] for i in S13), " (one injection per non-slack bus)")
print("14 injections:", ", ".join(lab[i] for i in S14), " (one injection per bus; sum P_i = 0)")


def minOmega(S):
    return HINT("5.1", "min_i Omega_ii/sigma^2 for the meter subset S: the smallest diagonal entry of I - P_S")


for name, S in [("13 injections", S13), ("14 injections", S14), ("all 54 meters", list(range(m)))]:
    rk = np.linalg.matrix_rank(H[S])
    print(f"{name:<14s} m = {len(S):2d}  rank = {rk:2d}  K = {len(S) - rk:2d}  "
          f"min Omega_ii/sigma^2 = {max(minOmega(S), 0):.3f}")
# Now derive alpha_k BY HAND (sheet) for m = 13 and m = 14, then confirm with the L6
# bound abar_k computed on the smaller placements.


def abar_on(S):
    HS = H[S]
    out = []
    for k in range(len(S)):
        a2 = nnz_meters(beta_index(HS, k)[1])
        a1 = min(nnz_meters(HS[:, i]) for i in range(n - 1) if abs(HS[k, i]) > NZ_TOL)
        out.append(min(a1, a2))
    return out


print("13 injections: abar =", abar_on(S13))
print("14 injections: abar =", abar_on(S14))
print("54 meters:     min abar =", abar.min())

# --- Task 5.2 --- defence budget ---------------------------------------------
# Where this comes from: L6 'Defence Is Resource Allocation', L1 'The risk
# concept'. Choose THREE meters (Python rows = lecture number - 1).
protect = HINT("5.2", "your three meters to protect, given as row indices (lecture number - 1)")
betap = np.full(m, np.inf); abarp = np.full(m, 10**9)
for k in [i for i in range(m) if i not in protect]:
    betap[k], ak = beta_index(H, k, protect)
    if np.isfinite(betap[k]):
        abarp[k] = nnz_meters(ak)
reach = np.isfinite(betap)
print(f"protecting {', '.join(lab[i] for i in protect)}: min abar {abar.min()} -> {abarp[reach].min()};"
      f"  min beta {beta.min():.2f} -> {betap[reach].min():.2f}")
