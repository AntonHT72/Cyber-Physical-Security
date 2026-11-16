"""Block 1 --- The grid and its AC operating point."""
import numpy as np

from common import BASE_MVA, branch_flows, bus_injections, load_case, state_vector, ybus

net = load_case()
n = net["n"]

# --- Task 1.1 ---------------------------------------------------------------
# The case file fixes the demand and the generator dispatch in per unit.
gen = np.flatnonzero(net["Pg"] > 0)
print("generation at buses", [int(bus) for bus in gen + 1], ":", BASE_MVA * net["Pg"][gen], "MW")
print("total load", BASE_MVA * net["Pd"].sum(), "MW")
P = np.array(net["Pg"] - net["Pd"])
Q = np.array(net["Qg"] - net["Qd"])
print("net active injection at buses", list(range(1, n + 1)), ":", BASE_MVA * P, "MW")
print("net reactive injection at buses", list(range(1, n + 1)), ":", BASE_MVA * Q, "MVAr")

# --- Task 1.2 ---------------------------------------------------------------
# The AC analogue of the DC susceptance matrix is the complex bus admittance
# matrix Ybus. It includes line resistance, charging, and transformer taps.
Y = ybus(net)
print(f"Ybus shape: {Y.shape}")
print(f"number of branches: {net['nb']}")

# --- Task 1.3 ---------------------------------------------------------------
# The reference angle is fixed at bus 1. The remaining state contains all
# non-reference angles and every bus voltage magnitude.
x_true = state_vector(net)
V = net["V"]
S = bus_injections(net, V)
flows = branch_flows(net, V)
print(f"state dimension: {x_true.size} ({n - 1} angles + {n} magnitudes)")
print(f"reference angle: {net['Va'][net['slack'] - 1]:.6f} rad")
print(f"voltage magnitude range: {net['Vm'].min():.4f} .. {net['Vm'].max():.4f} p.u.")
print(f"voltage angle range: {np.degrees(net['Va'].min()):.2f} .. "
      f"{np.degrees(net['Va'].max()):.2f} degrees")
print(f"network active loss: {S.real.sum() * BASE_MVA:.5f} MW")
print(f"network reactive loss: {S.imag.sum() * BASE_MVA:.5f} MVAr")
print(f"branch active loss: {sum((f + t).real for f, t in flows) * BASE_MVA:.5f} MW")
