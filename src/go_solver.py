"""
go_solver.py
------------
Global Optimization (GO) via Differential Evolution (DE) for isolated pure-compound and mixture
critical points using Thermopack's original PC-SAFT EOS.

Notes
-----
- PC-SAFT EOS is provided by Thermopack; this script implements the optimization workflow.
- We search over (T, V) for fixed overall composition n (normalized to mole fractions).
- Objective: F(T,V) = f1^2 + (∇_n f1 · n)^2, where f1 = λ_min(dμ/dn) at fixed (T,V).
- All binary interaction parameters (k_ij) are set to zero (predictive baseline).

Usage
-----
> pip install -r requirements.txt
> python src/go_solver.py
"""

import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from scipy.optimize import differential_evolution
import thermopack as tp



COMP_STRING = "CO2,N2,C1,C2,C3,IC4,NC4,IC5,NC5,NC6,NC7"

# 11-component example composition (will be normalized)
n_raw = jnp.array(
    [0.0109, 0.0884, 0.8286, 0.0401, 0.0174, 0.0030, 0.0055, 0.0019, 0.0012, 0.0014, 0.0006],
    dtype=jnp.float64
)
n = n_raw / jnp.sum(n_raw)

# Search bounds in (T, V)
BOUNDS = [(150.0, 300.0), (1.00e-4, 1.30e-4)]  # K, m^3/mol

DE_CFG = dict(
    strategy="best1bin",
    popsize=20,
    maxiter=600,
    mutation=(0.5, 1.0),
    recombination=0.7,
    tol=1e-6,
    polish=True,
    updating="deferred",
    workers=1,
    seed=None,
)

# ------------------- EOS + math helpers -------------------

def build_eos():
    """Construct PC-SAFT EOS from Thermopack and set kij=0 (predictive baseline)."""
    eos = tp.pcsaft.pcsaft(COMP_STRING)
    nc = len(COMP_STRING.split(","))
    for i in range(1, nc + 1):
        for j in range(i + 1, nc + 1):
            eos.set_kij(i, j, 0.0)
    return eos

def min_eig_dmudn(eos, n_vec, T, V):
    """Return λ_min of dμ/dn at fixed (T,V,n)."""
    mu, dmudn = eos.chemical_potential_tv(float(T), float(V), np.array(n_vec),
                                          None, None, 1, property_flag='IR')
    dmudn = 0.5 * (dmudn + dmudn.T)  # symmetrize
    evals = jnp.linalg.eigh(jnp.array(dmudn, dtype=jnp.float64))[0]
    return float(jnp.min(evals))

def grad_n_min_eig(eos, n_vec, T, V, h=1e-8):
    """Central finite-difference gradient of λ_min(dμ/dn) w.r.t. n."""
    n_vec = jnp.array(n_vec, dtype=jnp.float64)
    g = jnp.zeros_like(n_vec)
    for k in range(len(n_vec)):
        n_plus  = n_vec.at[k].add(h)
        n_minus = n_vec.at[k].add(-h)
        f_plus  = min_eig_dmudn(eos, n_plus,  T, V)
        f_minus = min_eig_dmudn(eos, n_minus, T, V)
        g = g.at[k].set((f_plus - f_minus) / (2.0 * h))
    return g

def objective_F(eos, n_vec, T, V):
    """F(T,V) = f1^2 + (∇_n f1 · n)^2; return large value if state is invalid."""
    BIG = 1e6
    try:
        f1 = min_eig_dmudn(eos, n_vec, T, V)
        if not np.isfinite(f1):
            return BIG
        g = grad_n_min_eig(eos, n_vec, T, V)
        if not np.all(np.isfinite(g)):
            return BIG
        return float(f1**2 + (np.dot(np.array(g, float), np.array(n_vec, float)))**2)
    except Exception:
        return BIG

def run_de_search(eos, n_vec, bounds):
    """Run Differential Evolution over (T, V) to minimize F(T,V)."""
    def wrapper(x):
        T_, V_ = x
        return objective_F(eos, n_vec, T_, V_)
    return differential_evolution(wrapper, bounds=bounds, **DE_CFG)


def main():
    print("[INFO] Building PC-SAFT EOS (Thermopack)…")
    eos = build_eos()

    print("[INFO] Running Differential Evolution search over (T, V)…")
    res = run_de_search(eos, n, BOUNDS)

    T_opt, V_opt = float(res.x[0]), float(res.x[1])
    F_opt = float(res.fun)

    print("\n[RESULT] GO/DE optimum (k_ij=0 baseline):")
    print(f"  T*  = {T_opt:.6g} K")
    print(f"  V*  = {V_opt:.6g} m^3/mol")
    print(f"  F*  = {F_opt:.6g}")

    # Diagnostics
    try:
        p_out = eos.pressure_tv(T_opt, V_opt, np.array(n))
        P_opt = p_out[0] if isinstance(p_out, (tuple, list)) else p_out
        print(f"  P(T*,V*,n) = {P_opt:.6g} Pa  ({P_opt/1e6:.6g} MPa)")
    except Exception as e:
        print("  [WARN] pressure_tv failed near optimum (near spinodal):", e)

    try:
        lam_min = min_eig_dmudn(eos, n, T_opt, V_opt)
        print(f"  λ_min(dμ/dn) = {lam_min:.6g}")
    except Exception as e:
        print("  [WARN] eigenvalue diagnostic failed:", e)

if __name__ == "__main__":
    main()
