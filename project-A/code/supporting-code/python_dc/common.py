"""Shared DC state-estimation helpers for the IEEE 57-bus case.

Notation follows the lecture decks (L3, L4, L6):
  theta in R^{n-1}  bus angles without the slack bus r = 1 (theta_r = 0)
  H in R^{m x (n-1)}, z = H theta + e, e ~ N(0, R), R = sigma^2 I
  K = m - (n-1),  tau = Quantile_0.95(chi^2_K)

Python uses 0-based array indices while bus and meter labels remain 1-based.
"""
import os
import re

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)              # supporting-code

BASE_MVA = 100.0
SIGMA = 0.01        # standard deviation of every meter, p.u. (= 1 MW)
NZ_TOL = 1e-6       # |a_i| > NZ_TOL counts as a changed meter


def load_case():
    """Load the IEEE 57-bus MATPOWER case for the DC model."""
    path = os.path.join(DATA, "data", "case57.m")
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def table(name):
        match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
        if match is None:
            raise ValueError(f"case file does not contain mpc.{name}")
        rows = []
        for line in match.group(1).splitlines():
            line = line.split("%", 1)[0].strip()
            if line:
                rows.append([float(value) for value in re.split(r"[\s,]+", line.rstrip(";"))])
        return np.asarray(rows, dtype=float)

    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text)
    if base_match is None:
        raise ValueError("case file does not define mpc.baseMVA")
    base_mva = float(base_match.group(1))
    bus, gen, branch = table("bus"), table("gen"), table("branch")
    n = bus.shape[0]
    slack_rows = np.flatnonzero(bus[:, 1] == 3)
    if len(slack_rows) != 1:
        raise ValueError("case must contain exactly one reference bus")
    slack = int(bus[slack_rows[0], 0])
    pg = np.zeros(n)
    for row in gen[gen[:, 7] > 0]:
        pg[int(row[0]) - 1] += row[1] / base_mva
    return {
        "n": n, "nb": branch.shape[0], "slack": slack,
        "keep": [b for b in range(1, n + 1) if b != slack],
        "frm": branch[:, 0].astype(int).tolist(),
        "to": branch[:, 1].astype(int).tolist(),
        "x": branch[:, 3],
        "Pg": pg, "Pd": bus[:, 2] / base_mva,
        "Va": np.radians(bus[:, 8]),
    }


def meter_labels(net):
    """Lecture names of the 54 meters: P1..P14 (injections), then P_{i,j},
    P_{j,i} for every branch (i,j) in branch order."""
    lab = [f"P{i}" for i in range(1, net["n"] + 1)]
    for i, j in zip(net["frm"], net["to"]):
        lab += [f"P{i},{j}", f"P{j},{i}"]
    return lab


def meter_no(i):
    """Lecture meter number k of Python row i."""
    return i + 1


def flow_meter(net, i, j):
    """Python row of the flow P_{i,j} measured at bus i of branch (i,j)."""
    n = net["n"]
    for l, (f, t) in enumerate(zip(net["frm"], net["to"])):
        if (f, t) == (i, j):
            return n + 2 * l
        if (f, t) == (j, i):
            return n + 2 * l + 1
    raise ValueError(f"no branch between buses {i} and {j}")


def state_col(b):
    """Python column of H (entry of theta) holding the angle of bus b = 2..14."""
    return b - 2


def measurement_matrix(net):
    """The L3 measurement matrix H (54 x 13), slack column removed:
      flow row      P_{ij} = (theta_i - theta_j) / x_ij
      injection row P_i    = sum_j (theta_i - theta_j) / x_ij"""
    n, nb = net["n"], net["nb"]
    Hfull = np.zeros((n + 2 * nb, n))
    for l, (i, j) in enumerate(zip(net["frm"], net["to"])):
        y = 1.0 / net["x"][l]
        i0, j0 = i - 1, j - 1
        Hfull[n + 2 * l, i0], Hfull[n + 2 * l, j0] = y, -y            # P_{i,j}
        Hfull[n + 2 * l + 1, j0], Hfull[n + 2 * l + 1, i0] = y, -y    # P_{j,i}
        Hfull[i0, i0] += y; Hfull[i0, j0] -= y                       # P_i
        Hfull[j0, j0] += y; Hfull[j0, i0] -= y                       # P_j
    return Hfull[:, [b - 1 for b in net["keep"]]]


def load_noise():
    """Return a deterministic noise vector with the current meter count."""
    m = load_case()["n"] + 2 * load_case()["nb"]
    path = os.path.join(DATA, "noise.csv")
    if os.path.exists(path):
        noise = np.loadtxt(path)
        if noise.size == m:
            return noise
    return np.random.default_rng(1).standard_normal(m) * SIGMA


def load_perms():
    """Return deterministic random meter permutations for the current case."""
    m = load_case()["n"] + 2 * load_case()["nb"]
    path = os.path.join(DATA, "perms.csv")
    if os.path.exists(path):
        perms = np.loadtxt(path, delimiter=",", dtype=int) - 1
        if perms.shape[1] == m:
            return perms
    rng = np.random.default_rng(1)
    return np.array([rng.permutation(m) for _ in range(1000)])


def load_checkpoint(name):
    """Read a checkpoint CSV from checkpoints/ (vector or matrix)."""
    return np.loadtxt(os.path.join(DATA, "checkpoints", name), delimiter=",")


def check(ok, what, link=""):
    """Print a self-check line and the lecture claim the number confirms."""
    print("  " + ("[ok]  " if ok else "[FAIL]") + " " + what
          + (f"\n           lecture link: {link}" if link else ""))
    return bool(ok)


def beta_index(H, k, protect=()):
    """GIVEN LP for the L6 index (k is a Python row index)
        beta_k = minimise_c ||H c||_1  subject to  (H c)_k = 1,
    written with slacks t: minimise sum(t) s.t. -t <= H c <= t, (H c)_k = 1.
    Optional `protect`: rows the attacker cannot touch, (H c)_i = 0.
    Returns (beta_k, a) with a = H c the optimal attack."""
    m, p = H.shape
    cost = np.r_[np.zeros(p), np.ones(m)]
    A_ub = np.block([[H, -np.eye(m)], [-H, -np.eye(m)]])
    rows = [k] + list(protect)
    A_eq = np.c_[H[rows, :], np.zeros((len(rows), m))]
    b_eq = np.r_[1.0, np.zeros(len(protect))]
    res = linprog(cost, A_ub=A_ub, b_ub=np.zeros(2 * m), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(None, None)] * p + [(0, None)] * m, method="highs")
    if res.status != 0:
        return np.inf, np.full(m, np.nan)
    return res.fun, H @ res.x[:p]


def cut_lines(net, S):
    """Branches with exactly one end bus in the bus set S: the cut delta(S), as Python
    branch indices (0-based). Buses are lecture bus numbers 1..14."""
    return [l for l, (i, j) in enumerate(zip(net["frm"], net["to"])) if (i in S) != (j in S)]


def cut_meters(net, S):
    """|g(A)| for the cut A = delta(S) under the standard placement: two flow meters on
    every cut line plus one injection meter at every end bus of a cut line."""
    A = cut_lines(net, S)
    return 2 * len(A) + len({b for l in A for b in (net["frm"][l], net["to"][l])})


def min_attack_cut(net):
    """BLACK BOX (given). Solves the L4 graph-cut problem of Kosut et al. (2010), Theorem 2,
        k* = min over line sets A with h(A) >= 2 of  |g(A)| - h(A) + 2,
    for the standard placement (one injection meter per bus, two flow meters per line:
    all 54 meters). Returns (k, A, S, g):
      k   the fewest meters a stealthy attack must change,
      A   an optimal cut, as bus pairs (i, j) in branch order,
      S   the buses the cut separates from the slack bus: the attack is c = 1 on S, 0 elsewhere,
      g   the meters g(A), as sorted Python ROWS (lecture number = row + 1): exactly the
          meters that attack changes.

    How it works (not needed for the session): see min_attack_cut in common.jl.
    Queyranne's algorithm (1998) minimises F(S) = |g(delta(S))| = cut_meters(net, S) over
    bus sets S; identical steps and tie-breaking, so both languages return the same cut."""
    def F(X):
        return cut_meters(net, X)
    groups = [[v] for v in range(1, net["n"] + 1)]   # each element: the buses merged so far
    k, best = np.inf, []
    while len(groups) > 1:
        # maximum-adjacency ordering: next = argmin_y F(W u y) - F(y), ties to the lowest index
        W, rest, order = list(groups[0]), list(range(1, len(groups))), [0]
        while rest:
            y = min(rest, key=lambda i: F(W + groups[i]) - F(groups[i]))
            order.append(y); W += groups[y]; rest.remove(y)
        t, u = order[-2], order[-1]              # pendent pair: groups[u] is a best set separating t and u
        if F(groups[u]) < k:
            k, best = F(groups[u]), groups[u]
        groups[t] = groups[t] + groups[u]        # merge u into t
        del groups[u]
    n = net["n"]
    S = sorted(set(range(1, n + 1)) - set(best) if net["slack"] in best else best)
    A = cut_lines(net, S)
    g = sorted({b - 1 for l in A for b in (net["frm"][l], net["to"][l])}
               | {n + 2 * l for l in A} | {n + 2 * l + 1 for l in A})
    return k, [(net["frm"][l], net["to"][l]) for l in A], S, g


def TODO(hint):
    """Placeholder in the starter files: running a block stops at the first
    TODO you have not replaced yet and prints the line to write."""
    raise NotImplementedError(f"\n\n  TODO --- replace this line with:  {hint}\n")


def HINT(id, hint):
    """Placeholder in the starter files: names the quantity and the lecture formula, but
    not the code. Write the line yourself; the check lines below it tell you if it is right."""
    raise NotImplementedError(f"\n\n  HINT [{id}] --- write this line yourself:  {hint}\n")


def nnz_meters(a):
    """Number of meters an attack vector changes, ||a||_0 with tolerance NZ_TOL."""
    return int(np.sum(np.abs(a) > NZ_TOL))
