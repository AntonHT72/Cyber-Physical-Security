"""Q4/Q5 simulation: constrained AC and DC state estimation on IEEE 57.

Run from this directory:

    python ac_dc_simulation.py

The default study uses 2,000 noise draws (takes a loon time) at each requested standard
deviation. Use ``--draws N`` for a shorter development run.
"""
import argparse
import csv
import os

import numpy as np
from scipy.optimize import root
from scipy.linalg import null_space
from scipy.stats import chi2

from common import branch_flows, bus_injections, load_case, ybus


NOISE_LEVELS = (0.01, 0.02, 0.05, 0.1)
ATTACK_LEVELS = np.logspace(-4, 0, 9)
TOL = 1e-8


def solve_power_flow(net):
    """Solve the MATPOWER P/V/Q specifications with bus 1 as reference."""
    Y = ybus(net)
    n = net["n"]
    slack = net["slack"] - 1
    pv = np.flatnonzero(net["bus_type"] == 2)
    pq = np.flatnonzero(net["bus_type"] == 1)
    angle_buses = np.asarray([i for i in range(n) if i != slack])
    p_spec = net["Pg"] - net["Pd"]
    q_spec = net["Qg"] - net["Qd"]

    def unpack(u):
        angles = np.zeros(n)
        angles[slack] = net["Va"][slack]
        angles[angle_buses] = u[: n - 1]
        magnitudes = net["Vm"].copy()
        magnitudes[pq] = u[n - 1:]
        return magnitudes * np.exp(1j * angles)

    def mismatch(u):
        voltage = unpack(u)
        calculated = voltage * np.conj(Y @ voltage)
        p_error = (calculated - p_spec)[angle_buses]
        q_error = (calculated.imag - q_spec)[pq]
        return np.r_[p_error.real, q_error]

    initial = np.r_[net["Va"][angle_buses], net["Vm"][pq]]
    result = root(mismatch, initial, method="hybr", tol=1e-10)
    if not result.success or np.linalg.norm(mismatch(result.x), np.inf) > 1e-7:
        raise RuntimeError(f"AC power flow failed: {result.message}")
    return unpack(result.x), pv, pq


def state_from_voltage(voltage, keep):
    return np.r_[np.angle(voltage[keep]), np.abs(voltage)]


def voltage_from_state(x, n, keep):
    angles = np.zeros(n)
    angles[keep] = x[: n - 1]
    return x[n - 1:] * np.exp(1j * angles)


def ac_measurements(net, voltage, injection_buses):
    injections = bus_injections(net, voltage)
    flows = branch_flows(net, voltage)
    return np.r_[
        injections[injection_buses].real,
        injections[injection_buses].imag,
        flows[:, 0].real,
        flows[:, 1].real,
        flows[:, 0].imag,
        flows[:, 1].imag,
        np.abs(voltage),
        np.angle(voltage),
    ]


def ac_labels(net, injection_buses):
    labels = [f"P{i + 1}" for i in injection_buses]
    labels += [f"Q{i + 1}" for i in injection_buses]
    labels += [f"P{f},{t}" for f, t in zip(net["frm"], net["to"])]
    labels += [f"P{t},{f}" for f, t in zip(net["frm"], net["to"])]
    labels += [f"Q{f},{t}" for f, t in zip(net["frm"], net["to"])]
    labels += [f"Q{t},{f}" for f, t in zip(net["frm"], net["to"])]
    labels += [f"V{i + 1}" for i in range(net["n"])]
    labels += [f"theta{i + 1}" for i in range(net["n"])]
    return labels


def dc_measurements(net, theta, injection_buses):
    """Return P injections, P flows, and all angle measurements."""
    p = np.zeros(net["n"])
    flows_from = []
    flows_to = []
    for f, t, x in zip(net["frm"], net["to"], net["x"]):
        value = (theta[f - 1] - theta[t - 1]) / x
        p[f - 1] += value
        p[t - 1] -= value
        flows_from.append(value)
        flows_to.append(-value)
    return np.r_[p[injection_buses], flows_from, flows_to, theta]


def dc_matrix(net, injection_buses):
    """Build the linear DC matrix in the same order as dc_measurements."""
    n = net["n"]
    keep = np.asarray(net["keep"]) - 1
    rows = []
    for bus in injection_buses:
        row = np.zeros(n - 1)
        for f, t, x in zip(net["frm"], net["to"], net["x"]):
            y = 1.0 / x
            if bus == f - 1:
                if t != 1:
                    row[t - 2] -= y
                if f != 1:
                    row[f - 2] += y
            elif bus == t - 1:
                if f != 1:
                    row[f - 2] -= y
                if t != 1:
                    row[t - 2] += y
        rows.append(row)
    from_rows = []
    to_rows = []
    for f, t, x in zip(net["frm"], net["to"], net["x"]):
        row = np.zeros(n - 1)
        y = 1.0 / x
        if f != 1:
            row[f - 2] += y
        if t != 1:
            row[t - 2] -= y
        from_rows.append(row)
        to_rows.append(-row)
    rows.extend(from_rows)
    rows.extend(to_rows)
    angle_rows = np.zeros((n, n - 1))
    angle_rows[1:, :] = np.eye(n - 1)
    rows.extend(angle_rows)
    return np.asarray(rows)


def numerical_jacobian(model, x):
    values = model(x)
    jacobian = np.empty((values.size, x.size))
    for column in range(x.size):
        step = 1e-6
        dx = np.zeros_like(x)
        dx[column] = step
        jacobian[:, column] = (model(x + dx) - model(x - dx)) / (2 * step)
    return jacobian


def constrained_ac_estimate(net, z, x0, injection_buses, zero_buses, sigma):
    n = net["n"]
    keep = np.asarray(net["keep"]) - 1

    def model(x):
        return ac_measurements(net, voltage_from_state(x, n, keep), injection_buses)

    def constraints(x):
        s = bus_injections(net, voltage_from_state(x, n, keep))
        return np.r_[s[zero_buses].real, s[zero_buses].imag]

    x = x0.copy()
    for _ in range(40):
        residual = z - model(x)
        jacobian = numerical_jacobian(model, x)
        constraint = constraints(x)
        constraint_jacobian = numerical_jacobian(constraints, x)
        weight = np.eye(z.size) / sigma**2
        lhs = jacobian.T @ weight @ jacobian
        rhs = jacobian.T @ weight @ residual
        if constraint_jacobian.size:
            kkt = np.block([
                [lhs, constraint_jacobian.T],
                [constraint_jacobian, np.zeros((constraint.size, constraint.size))],
            ])
            update = np.linalg.solve(
                kkt, np.r_[rhs, -constraint])
            delta = update[:x.size]
        else:
            delta = np.linalg.solve(lhs, rhs)
        x += delta
        if np.linalg.norm(delta, np.inf) < 1e-9:
            return x
    raise RuntimeError("constrained AC WLS did not converge")


def constrained_dc_estimate(H, z, constraints, sigma):
    """Linear DC WLS with zero-injection equality constraints."""
    weight = np.eye(z.size) / sigma**2
    lhs = H.T @ weight @ H
    rhs = H.T @ weight @ z
    if constraints.size:
        kkt = np.block([
            [lhs, constraints.T],
            [constraints, np.zeros((constraints.shape[0], constraints.shape[0]))],
        ])
        return np.linalg.solve(
            kkt, np.r_[rhs, np.zeros(constraints.shape[0])])[:H.shape[1]]
    return np.linalg.solve(lhs, rhs)


def residual_statistics(measurements, predicted, jacobian, sigma,
                        constraint_jacobian=None):
    residual = measurements - predicted
    if constraint_jacobian is not None and constraint_jacobian.size:
        free_states = null_space(constraint_jacobian)
        tangent = jacobian @ free_states
    else:
        tangent = jacobian
    projection = tangent @ np.linalg.solve(
        tangent.T @ tangent, tangent.T)
    covariance = sigma**2 * (np.eye(measurements.size) - projection)
    normalized = np.abs(residual) / np.sqrt(np.maximum(np.diag(covariance), 1e-14))
    return residual @ residual / sigma**2, normalized.max()


def main(draws):
    net = load_case()
    n = net["n"]
    keep = np.asarray(net["keep"]) - 1
    voltage, _, pq = solve_power_flow(net)
    x_true = state_from_voltage(voltage, keep)
    p_spec = net["Pg"] - net["Pd"]
    q_spec = net["Qg"] - net["Qd"]
    injection_buses = np.flatnonzero(
        (np.abs(p_spec) > TOL) | (np.abs(q_spec) > TOL))
    zero_buses = np.asarray([
        i for i in range(n) if i not in set(injection_buses)])
    labels = ac_labels(net, injection_buses)
    ac_true = ac_measurements(net, voltage, injection_buses)
    Hdc = dc_matrix(net, injection_buses)
    theta_true = x_true[: n - 1]
    dc_true = dc_measurements(net, np.r_[0.0, theta_true], injection_buses)
    dc_zero_constraints = dc_matrix(net, zero_buses)[:len(zero_buses)]
    rng = np.random.default_rng(20261120)

    def ac_model(x):
        return ac_measurements(net, voltage_from_state(x, n, keep), injection_buses)

    def ac_constraints(x):
        s = bus_injections(net, voltage_from_state(x, n, keep))
        return np.r_[s[zero_buses].real, s[zero_buses].imag]

    ac_constraint_rank = np.linalg.matrix_rank(
        numerical_jacobian(ac_constraints, x_true))
    ac_df = ac_true.size - (x_true.size - ac_constraint_rank)
    dc_constraint_rank = np.linalg.matrix_rank(dc_zero_constraints)
    dc_df = dc_true.size - (theta_true.size - dc_constraint_rank)
    ac_threshold = chi2.ppf(.95, ac_df)
    dc_threshold = chi2.ppf(.95, dc_df)

    print(f"AC solution: {n} buses, {net['nb']} branches")
    print(f"AC measurements: {ac_true.size}; DC measurements: {dc_true.size}")
    print(f"AC states: {x_true.size}; DC states: {theta_true.size}")
    print(f"non-zero-injection buses: {len(injection_buses)}; "
          f"zero-injection buses: {len(zero_buses)}")

    # Use a common state direction: bus 2 angle only, projected into the
    # respective zero-injection constraint null spaces.
    direction_dc = np.zeros(n - 1)
    direction_dc[0] = 1.0
    if dc_zero_constraints.size:
        cmat = dc_zero_constraints
        direction_dc -= cmat.T @ np.linalg.lstsq(
            cmat @ cmat.T, cmat @ direction_dc, rcond=None)[0]
    direction_dc /= np.linalg.norm(direction_dc)
    direction_ac = np.r_[direction_dc, np.zeros(n)]
    print(f"bus-2 direction: DC {direction_dc[0]:.6f}, "
          f"AC {direction_ac[0]:.6f}")

    results = []
    for sigma in NOISE_LEVELS:
        noise_ac = sigma * rng.standard_normal(ac_true.size)
        z_ac = ac_true + noise_ac
        # Select the same P and angle noise components for the DC estimator.
        p_count = len(injection_buses)
        flow_count = 2 * net["nb"]
        dc_noise = np.r_[
            noise_ac[:p_count],
            noise_ac[2 * p_count:2 * p_count + flow_count],
            noise_ac[-n:],
        ]
        z_dc = dc_true + dc_noise
        x0 = np.r_[np.zeros(n - 1), np.ones(n)]
        xhat_ac = constrained_ac_estimate(
            net, z_ac, x0, injection_buses, zero_buses, sigma)
        j_ac = numerical_jacobian(ac_model, xhat_ac)
        ac_constraint_jacobian = numerical_jacobian(ac_constraints, xhat_ac)
        ac_stats = residual_statistics(
            z_ac, ac_model(xhat_ac), j_ac, sigma, ac_constraint_jacobian)
        theta_hat_dc = constrained_dc_estimate(
            Hdc, z_dc, dc_zero_constraints, sigma)
        dc_stats = residual_statistics(
            z_dc, Hdc @ theta_hat_dc, Hdc, sigma, dc_zero_constraints)
        print(f"sigma={sigma:.2f}: AC J={ac_stats[0]:.2f}, "
              f"DC J={dc_stats[0]:.2f}, "
              f"AC threshold={ac_threshold:.2f}, "
              f"DC threshold={dc_threshold:.2f}")
        results.append((sigma, z_ac, z_dc, xhat_ac, theta_hat_dc))

    # Attack comparison at every magnitude, with and without measurement noise.
    for sigma, z_ac, z_dc, xhat_ac, theta_hat_dc in results[:1]:
        attack_center = xhat_ac
        jacobian = numerical_jacobian(ac_model, attack_center)
        dc_attack_base = Hdc @ direction_dc
        dc_attack = np.zeros(ac_true.size)
        p_count = len(injection_buses)
        flow_count = 2 * net["nb"]
        dc_attack[:p_count] = dc_attack_base[:p_count]
        dc_attack[2 * p_count:2 * p_count + flow_count] = \
            dc_attack_base[p_count:p_count + flow_count]
        dc_attack[-n:] = [0.0] + direction_dc.tolist()
        noisy_base = z_ac - ac_true
        for noise_name, base in (("noise", z_ac), ("no_noise", ac_true)):
            if noise_name == "no_noise":
                attack_center = x_true
                jacobian = numerical_jacobian(ac_model, attack_center)
            print(f"\nattack sweep ({noise_name}, sigma={sigma:.2f})")
            for magnitude in ATTACK_LEVELS:
                c = magnitude * direction_ac
                jacobian_attack = magnitude * (jacobian @ direction_ac)
                nonlinear_attack = ac_model(attack_center + c) - ac_model(attack_center)
                attacks = {
                    "dc": magnitude * dc_attack,
                    "jacobian": jacobian_attack,
                    "nonlinear": nonlinear_attack,
                }
                for name, attack in attacks.items():
                    attacked = base + attack
                    try:
                        estimate = constrained_ac_estimate(
                            net, attacked, attack_center, injection_buses,
                            zero_buses, sigma)
                    except RuntimeError:
                        print(f"{name},{magnitude:.6g},non-convergent")
                        continue
                    local_jacobian = numerical_jacobian(ac_model, estimate)
                    statistic, largest = residual_statistics(
                        attacked, ac_model(estimate), local_jacobian, sigma,
                        numerical_jacobian(ac_constraints, estimate))
                    print(f"{name},{magnitude:.6g},{statistic:.6f},"
                          f"{largest:.6f},{estimate[0] - xhat_ac[0]:.6f}")

    # False-alarm study and Q-Q data for every requested noise level.
    samples = {}
    for sigma in NOISE_LEVELS:
        j_ac_samples = []
        j_dc_samples = []
        for _ in range(draws):
            noise_ac = sigma * rng.standard_normal(ac_true.size)
            z_ac = ac_true + noise_ac
            p_count = len(injection_buses)
            flow_count = 2 * net["nb"]
            z_dc = dc_true + np.r_[
                noise_ac[:p_count],
                noise_ac[2 * p_count:2 * p_count + flow_count],
                noise_ac[-n:],
            ]
            try:
                xhat = constrained_ac_estimate(
                    net, z_ac, x0, injection_buses, zero_buses, sigma)
                jacobian = numerical_jacobian(ac_model, xhat)
                j_ac_samples.append(residual_statistics(
                    z_ac, ac_model(xhat), jacobian, sigma,
                    numerical_jacobian(ac_constraints, xhat))[0])
                theta_hat = constrained_dc_estimate(
                    Hdc, z_dc, dc_zero_constraints, sigma)
                j_dc_samples.append(residual_statistics(
                    z_dc, Hdc @ theta_hat, Hdc, sigma,
                    dc_zero_constraints)[0])
            except RuntimeError:
                continue
        samples[sigma] = (j_ac_samples, j_dc_samples)
        ac_rate = np.mean(np.asarray(j_ac_samples) > ac_threshold)
        dc_rate = np.mean(np.asarray(j_dc_samples) > dc_threshold)
        print(f"\nfalse-alarm study sigma={sigma:.2f}: "
              f"{len(j_ac_samples)} successful draws, "
              f"{draws - len(j_ac_samples)} non-convergent")
        print(f"AC mean={np.mean(j_ac_samples):.2f}, variance={np.var(j_ac_samples):.2f}, "
              f"df={ac_df}, false alarms={ac_rate:.4f}")
        print(f"DC mean={np.mean(j_dc_samples):.2f}, variance={np.var(j_dc_samples):.2f}, "
              f"df={dc_df}, false alarms={dc_rate:.4f}")
    j_ac_samples, j_dc_samples = samples[NOISE_LEVELS[0]]
    with open("qq_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ac_j", "dc_j"])
        for ac_j, dc_j in zip(j_ac_samples, j_dc_samples):
            writer.writerow([ac_j, dc_j])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import probplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        probplot(j_ac_samples, dist=chi2(df=ac_df), plot=axes[0])
        probplot(j_dc_samples, dist=chi2(df=dc_df), plot=axes[1])
        axes[0].set_title(f"AC test statistic Q-Q (df={ac_df})")
        axes[1].set_title(f"DC test statistic Q-Q (df={dc_df})")
        fig.tight_layout()
        fig.savefig("qq-test-statistics.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; Q-Q data remains in qq_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    main(parser.parse_args().draws)
