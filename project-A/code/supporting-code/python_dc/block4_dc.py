"""
Block 4 --- Imperfect knowledge and security indices       (L6)
    python block4.py
Python counts from 0: meter z_k is row k-1 (the L4 cut z_7, z_8, z_41, z_42
is rows 6, 7, 40, 41).
"""

# AI declaration: generated with generative-AI assistants (Claude, Codex, OpenCode;
# 4-8 October 2026), then reviewed, corrected and verified by the instructor; the
# numbers this file prints are asserted by verify.jl. Full record: AI.log.
import numpy as np
from scipy.stats import chi2

from common import (NZ_TOL, SIGMA, HINT, beta_index, check, load_case,
                    load_checkpoint, measurement_matrix, meter_labels,
                    nnz_meters, state_col)

net = load_case()
lab = meter_labels(net)
n, m = net["n"], 54
H = measurement_matrix(net)
z = load_checkpoint("z.csv")
K = m - (n - 1)
tau = chi2.ppf(0.95, K)
# With R = sigma^2 I the weighted P of L3 is the orthogonal projector onto R(H),
# so the L6 formulas (written for R = I) apply unchanged.
PNH = np.eye(m) - H @ np.linalg.pinv(H)        # P_{N(H')}


def Jof(zz):
    return np.sum((PNH @ zz) ** 2) / SIGMA**2


# --- Task 4.1 ---------------------------------------------------------------
# Where this comes from: L6 'False data injection: DC state estimation,
# mitigating Critique 2'. a = H_+ c with H_+ = H + Delta H.
# Proposition: ||Delta r|| <= cos(gamma) ||a||_2;  Lemma: cos(gamma) = ||P_N(H') P_R(H_+)||_2.
l15 = [l for l, ft in enumerate(zip(net["frm"], net["to"])) if ft == (1, 5)][0]


def Hplus(delta):
    netp = dict(net); netp["x"] = net["x"].copy(); netp["x"][l15] *= 1 + delta
    return measurement_matrix(netp)


c5 = np.zeros(n - 1); c5[state_col(5)] = 0.1
print(" delta   cos(gamma)  ||Delta r||  cos(gamma)||a||     J    alarm")
for delta in (0.05, 0.10, 0.20, 0.50):
    Hp = Hplus(delta)
    ap = Hp @ c5
    cosg = HINT("4.1a", "cos(gamma) = ||P_N(H') P_R(H_+)||_2 = norm(P_N(H') H_+ pinv(H_+), 2)")
    dr = HINT("4.1b", "the leaked residual norm ||P_N(H') a||_2 for the attack a = H_+ c")
    Jp = Jof(z + ap)
    print(f"{delta:5.2f}    {cosg:.4f}      {dr:.4f}        {cosg*np.linalg.norm(ap):.4f}       "
          f"{Jp:6.1f}   {'yes' if Jp > tau else 'no'}")
# Now: (i) replace Hp by 1.2 * H; (ii) keep Hp = Hplus(0.5) but use c6 = 0.1 e_6.
# Hint. Branch (1,5) touches only bus 5: where is Delta H = Hplus(delta) - H nonzero?
#   (i) Does Hp = 1.2 H change Range(H)?
#   (ii) Which column does c6 excite -- is Delta H nonzero there?
#   cos(gamma) is the worst case over all c, not a function of the c you pick.
# (i) a pure scaling error Hp = 1.2 * H: same range as H, so nothing leaks.
Hp = 1.2 * H
cosg1 = HINT("4.1c", "cos(gamma) for H_+ = 1.2 H: the same principal-angle formula")
dr1   = HINT("4.1d", "the leaked residual norm for H_+ = 1.2 H with c = c5")
print(f"H_+ = 1.2 H        cos(gamma) = {cosg1:.1e}   ||Delta r|| = {dr1:.1e}")
# (ii) delta = 0.50 but c = 0.1 e_6: Delta H c = 0, so nothing leaks.
Hp = Hplus(0.50)
c6 = np.zeros(n - 1); c6[state_col(6)] = 0.1
cosg2 = HINT("4.1e", "cos(gamma) for H_+ = Hplus(0.50): the same principal-angle formula")
dr2   = HINT("4.1f", "the leaked residual norm for c = c6 (0.1 e_6)")
print(f"c = 0.1 e_6, 50%   cos(gamma) = {cosg2:.4f}   ||Delta r|| = {dr2:.1e}")

# --- Task 4.2 ---------------------------------------------------------------
# Where this comes from: L6 'Effort as Sparsity: the Index alpha_k',
# 'Bounding alpha_k Cheaply', 'Effort as Magnitude: the Index beta_k'.
beta = np.zeros(m); abar2 = np.zeros(m, dtype=int)
for k in range(m):
    beta[k], ak = beta_index(H, k)
    abar2[k] = nnz_meters(ak)
abar1 = HINT("4.2a", "abar^1_k = min over columns i with H_ki != 0 of ||H_{.,i}||_0, one per meter k")
abar = HINT("4.2b", "abar_k = min(abar^1_k, abar^2_k)")
weak = np.flatnonzero(abar == abar.min())
print("smallest abar_k =", abar.min(), "at", ", ".join(lab[i] for i in weak))
check(list(weak) == [6, 7, 40, 41], "the weakest meters are the L4 cut of Task 3.3",
      "L6 'Effort as Sparsity' meets L4 'A pendant cut needs four meters'")

# --- Task 4.3 ---------------------------------------------------------------
# Where this comes from: L6 'sensor's vulnerability, heuristics', 'Redundancy
# Heuristics Do Not Measure Security', 'Predict: Does Redundancy Buy Security?',
# 'Redundancy != Sparsity-Security (IEEE 14-Bus)'.
# PREDICT FIRST (write it on the sheet).
Pu = H @ np.linalg.solve(H.T @ H, H.T)
r1 = HINT("4.3a", "r^1_k = number of i with |P_ik / P_kk| >= 0.33, one per meter k")
r2 = HINT("4.3b", "r^2_k = sum_i |P_ik / P_kk|, one per meter k")
print(f"rho(abar, r^1) = {np.corrcoef(abar, r1)[0, 1]:.3f}    rho(beta, r^2) = {np.corrcoef(beta, r2)[0, 1]:.3f}")
print(f"meter z_8: (abar, beta, r^1, r^2) = ({abar[7]}, {beta[7]:.2f}, {r1[7]}, {r2[7]:.2f})")
for v in sorted(set(r1)):
    idx = np.flatnonzero(r1 == v)
    print(f"r^1 = {v:2d}  ->  abar in {sorted(set(abar[idx].tolist()))}  ({len(idx)} meters)")
# Plot (optional): matplotlib, abar and r1 against k = 1..54; beta and r2 likewise.
