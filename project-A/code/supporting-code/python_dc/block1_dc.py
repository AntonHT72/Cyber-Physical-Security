"""Block 1 --- the IEEE 57-bus grid and its DC operating point."""
import numpy as np

from common import BASE_MVA, load_case


net = load_case()
n = net["n"]

print("generation at buses",
      [int(bus) for bus in np.flatnonzero(net["Pg"] > 0) + 1],
      ":", BASE_MVA * net["Pg"][net["Pg"] > 0], "MW")
print("total load", BASE_MVA * net["Pd"].sum(), "MW")

P = net["Pg"] - net["Pd"]
print("net injection at buses", list(range(1, n + 1)), ":", BASE_MVA * P, "MW")

# DC susceptance matrix: line resistance, charging, and transformer ratios are
# ignored, as in the original 14-bus exercise.
B = np.zeros((n, n))
for i, j, x in zip(net["frm"], net["to"], net["x"]):
    i, j = i - 1, j - 1
    y = 1.0 / x
    B[i, i] += y
    B[j, j] += y
    B[i, j] -= y
    B[j, i] -= y

keep = np.asarray(net["keep"]) - 1
theta = np.linalg.solve(B[np.ix_(keep, keep)], P[keep])
theta_full = np.zeros(n)
theta_full[keep] = theta

print(f"buses: {n}, branches: {net['nb']}, state dimension: {theta.size}")
print(f"slack injection from DC model: {BASE_MVA * (B @ theta_full)[0]:.1f} MW")
print(f"case-file slack injection: {BASE_MVA * P[0]:.1f} MW")
print(f"DC angle range: {np.degrees(theta_full).min():.2f} .. "
      f"{np.degrees(theta_full).max():.2f} degrees")
