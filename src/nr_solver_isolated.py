"""
Newton–Raphson (NR) solver via Heidemann-Khalil criticality conditions for isolated pure-compound and mixture
critical points using Thermopack's original PC-SAFT EOS.

Notes:
- Uses the component list in (T,V) bounds.
- Fixes overall composition (x = z).
- Solves HK residuals: [g_i = (H d)_i, g = directional cubic, h = ||d||^2-1].
- Renormalizes Δn every update.
- All binary interaction parameters (k_ij) are set to zero (predictive baseline).

Usage:
  pip install -r requirements.txt
  python src/nr_solver.py
"""

import numpy as np
import thermopack as tp

COMP_STRING = "CO2,N2,C1,C2,C3,IC4,NC4,IC5,NC5,NC6,NC7"
NC = 11

# 11-component example composition (will be normalized)
z_raw = np.array(
    [0.0109, 0.0884, 0.8286, 0.0401, 0.0174, 0.0030, 0.0055, 0.0019, 0.0012, 0.0014, 0.0006],
    dtype=float
)
z = z_raw / z_raw.sum()

# (T,V) search window
T_BOUNDS = (150.0, 300.0)        # K
V_BOUNDS = (1.00e-4, 1.30e-4)    # m^3/mol

BIG = 1e8


# EOS handler
def build_eos():
    """PC-SAFT from Thermopack; set all k_ij = 0 (predictive baseline)."""
    eos = tp.pcsaft.pcsaft(COMP_STRING)
    for i in range(1, NC + 1):
        for j in range(i + 1, NC + 1):
            eos.set_kij(i, j, 0.0)
    return eos


def get_mu_H(eos, T, V, n_vec):
    """Return (mu, H=∂μ/∂n)|_{T,V}. Symmetrize H for numerical stability."""
    try:
        mu, dmu_dn = eos.chemical_potential_tv(float(T), float(V), np.array(n_vec), None, None, 1, property_flag='IR')
        H = 0.5 * (dmu_dn + dmu_dn.T)
        return np.asarray(mu, float), np.asarray(H, float)
    except Exception:
        return None, None


def pure_critical_table(eos):
    """Build (Tci, Vci) per component via EOS; NaN on failure."""
    Tci, Vci = np.full(NC, np.nan), np.full(NC, np.nan)
    for k in range(NC):
        n_pure = np.zeros(NC); n_pure[k] = 1.0
        try:
            Tc_k, Vc_k, _ = eos.critical(n_pure)
            Tci[k], Vci[k] = float(Tc_k), float(Vc_k)
        except Exception:
            pass
    return Tci, Vci


def kay_seed_TV(eos, z_):
    """Kay-type seeds: Tc0 = Σ z_i Tci; Vc0 = Σ z_i Vci; fallback to mid-box."""
    Tci, Vci = pure_critical_table(eos)
    T0 = float(np.dot(z_, Tci)) if np.all(np.isfinite(Tci)) else 0.5 * (T_BOUNDS[0] + T_BOUNDS[1])
    V0 = float(np.dot(z_, Vci)) if np.all(np.isfinite(Vci)) else 0.5 * (V_BOUNDS[0] + V_BOUNDS[1])
    # clamp to bounds
    T0 = min(max(T0, T_BOUNDS[0]), T_BOUNDS[1])
    V0 = min(max(V0, V_BOUNDS[0]), V_BOUNDS[1])
    return T0, V0


def seed_direction(eos, T, V, n_vec):
    """Use the smallest-eigenvector of H(T,V;n) as Δn; fallback to z^(2/3)."""
    _, H = get_mu_H(eos, T, V, n_vec)
    if H is not None and np.all(np.isfinite(H)):
        w, U = np.linalg.eigh(H)
        d0 = U[:, np.argmin(w)]
        s = np.linalg.norm(d0)
        if np.isfinite(s) and s > 0:
            return d0 / s
    d0 = np.power(n_vec, 2.0 / 3.0)
    s = np.linalg.norm(d0)
    return d0 / s if s > 0 else np.ones_like(n_vec) / np.sqrt(n_vec.size)


def phi_dir(eos, T, V, n_vec, d):
    """φ(0) = d^T H(T,V;n) d — used for the cubic directional derivative."""
    _, H = get_mu_H(eos, T, V, n_vec)
    if H is None:
        return np.inf
    return float(d @ H @ d)


# HK residuals and Jacobian
def hk_residuals(eos, T, V, d, n_vec, h2=1e-5):
    """
    HK residual vector F = [g_1..g_nc, g, h]:
      g_i = (H d)_i
      g   = d/dα φ(α) |_{α=0}  with φ(α) = d^T H(T,V;n+α d) d  (centered FD)
      h   = ||d||^2 - 1
    """
    _, H = get_mu_H(eos, T, V, n_vec)
    if H is None or not np.all(np.isfinite(H)):
        g_vec  = np.full(NC, np.sqrt(BIG))
        g_cub  = np.sqrt(BIG)
        h_norm = float(d @ d - 1.0)
        return np.concatenate([g_vec, [g_cub, h_norm]])

    g_vec = H @ d

    phi_p = phi_dir(eos, T, V, n_vec + h2 * d, d)
    phi_m = phi_dir(eos, T, V, n_vec - h2 * d, d)
    g_cub = np.sqrt(BIG) if (not np.isfinite(phi_p) or not np.isfinite(phi_m)) else (phi_p - phi_m) / (2.0 * h2)

    h_norm = float(d @ d - 1.0)
    return np.concatenate([g_vec, [g_cub, h_norm]])


def numerical_jacobian(eos, T, V, d, n_vec, hT=None, hV=None, hd=2e-6):
    """
    Central differences for T and V; forward difference for each component of d.
    Returns J ((NC+2) x (2+NC)) and F0.
    """
    F0 = hk_residuals(eos, T, V, d, n_vec)
    m = F0.size
    J = np.zeros((m, 2 + NC), float)

    # scale-aware steps
    hT = hT or max(1e-2, 1e-4 * max(200.0, abs(T)))
    hV = hV or max(5e-8, 1e-3 * max(1e-4, abs(V)))

    # ∂F/∂T (central)
    FTp = hk_residuals(eos, T + hT, V, d, n_vec)
    FTm = hk_residuals(eos, T - hT, V, d, n_vec)
    J[:, 0] = (FTp - FTm) / (2.0 * hT)

    # ∂F/∂V (central)
    FVp = hk_residuals(eos, T, V + hV, d, n_vec)
    FVm = hk_residuals(eos, T, V - hV, d, n_vec)
    J[:, 1] = (FVp - FVm) / (2.0 * hV)

    # ∂F/∂d_i (forward)
    for i in range(NC):
        dd = np.zeros_like(d); dd[i] = hd
        Fd = hk_residuals(eos, T, V, d + dd, n_vec)
        J[:, 2 + i] = (Fd - F0) / hd

    return J, F0


# Newton Solver
def newton_hk_solver(eos, z_,
                     T0=None, V0=None,
                     max_iter=150,
                     tol_F=1e-8, tol_step=1e-9,
                     h2=1e-5, lam0=1e-2,
                     verbose=True):
    """
    Levenberg–Marquardt–style damping:
      (J^T J + λI) Δx = -J^T F,  x = (T, V, d_1..d_NC)
    Always renormalizes d after a step.
    Accepts “near-zero HK geometry” when residuals are tiny.
    """
    
    # Initial (T,V) and Δn
    if T0 is None or V0 is None:
        T0, V0 = kay_seed_TV(eos, z_)
    T, V = float(T0), float(V0)
    d = seed_direction(eos, T, V, z_)

    lam = lam0
    F = hk_residuals(eos, T, V, d, z_, h2=h2)
    best = {"T": T, "V": V, "d": d.copy(), "F": F.copy(), "norm": float(np.linalg.norm(F))}

    if verbose:
        print(f"[init] T0={T:.4f} K, V0={V:.6e} m^3/mol, ‖F‖={best['norm']:.3e}")

    for it in range(1, max_iter + 1):
        J, F = numerical_jacobian(eos, T, V, d, z_)
        res0 = float(np.linalg.norm(F))

        accept = False
        lam_try = lam
        for _ in range(12):
            JTJ = J.T @ J
            rhs = - J.T @ F
            dx = np.linalg.solve(JTJ + lam_try * np.eye(J.shape[1]), rhs)

            dT, dV, dd = dx[0], dx[1], dx[2:]
            T_try = min(max(T + dT, T_BOUNDS[0]), T_BOUNDS[1])
            V_try = min(max(V + dV, V_BOUNDS[0]), V_BOUNDS[1])
            d_try = d + dd
            s = np.linalg.norm(d_try)
            if not np.isfinite(s) or s == 0:
                lam_try *= 5.0
                continue
            d_try = d_try / s  # keep ||d|| = 1

            F_try = hk_residuals(eos, T_try, V_try, d_try, z_, h2=h2)
            res1 = float(np.linalg.norm(F_try))
            if np.isfinite(res1) and res1 < res0:
                T, V, d = T_try, V_try, d_try
                lam = max(lam_try / 3.0, 1e-8)
                if res1 < best["norm"]:
                    best = {"T": T, "V": V, "d": d.copy(), "F": F_try.copy(), "norm": res1}
                accept = True
                break
            lam_try *= 3.0

        if verbose:
            _, Hc = get_mu_H(eos, T, V, z_)
            lmin = (np.min(np.linalg.eigh(Hc)[0]) if Hc is not None else np.nan)
            print(f"[it {it:02d}] ‖F‖={best['norm']:.3e}, T={T:.3f}, V={V:.7e}, λ_min={lmin:.3e}, λ={lam:.1e}")

        # primary convergence
        if best["norm"] < tol_F:
            return {"status": "converged", "iter": it, **best}

        # stalled step → “geometry acceptance” check
        if not accept:
            Ffin = hk_residuals(eos, T, V, d, z_, h2=h2)
            g_lin = Ffin[:NC]      # H d
            g_cub = Ffin[NC]       # cubic directional
            h_nrm = Ffin[NC + 1]   # ||d||^2 - 1
            _, Hfin = get_mu_H(eos, T, V, z_)
            lam_min = (np.min(np.linalg.eigh(Hfin)[0]) if Hfin is not None else np.nan)

            if (np.isfinite(lam_min) and abs(lam_min) < 1e-8 and
                abs(h_nrm) < 1e-10 and
                np.linalg.norm(g_lin) < 1e-6 and
                abs(g_cub) < 1e-6):
                return {"status": "converged", "iter": it, "T": T, "V": V, "d": d.copy(),
                        "F": Ffin.copy(), "norm": float(np.linalg.norm(Ffin))}
            return {"status": "stalled", "iter": it, **best}

        # tiny step + small residuals
        if np.linalg.norm(dx) < tol_step and best["norm"] < 10 * tol_F:
            return {"status": "converged", "iter": it, **best}

    # after max_iter, try geometry acceptance again
    Ffin = hk_residuals(eos, T, V, d, z_, h2=h2)
    g_lin = Ffin[:NC]; g_cub = Ffin[NC]; h_nrm = Ffin[NC + 1]
    _, Hfin = get_mu_H(eos, T, V, z_)
    lam_min = (np.min(np.linalg.eigh(Hfin)[0]) if Hfin is not None else np.nan)
    if (np.isfinite(lam_min) and abs(lam_min) < 1e-8 and
        abs(h_nrm) < 1e-10 and
        np.linalg.norm(g_lin) < 1e-6 and
        abs(g_cub) < 1e-6):
        return {"status": "converged", "iter": max_iter, "T": T, "V": V, "d": d.copy(),
                "F": Ffin.copy(), "norm": float(np.linalg.norm(Ffin))}

    return {"status": "max_iter", "iter": max_iter, **best}


# CLI
def main():
    eos = build_eos()
    sol = newton_hk_solver(
        eos, z,
        T0=None, V0=None,          # Kay seeding
        max_iter=150,
        tol_F=1e-8, tol_step=1e-9,
        h2=1e-5, lam0=1e-2,
        verbose=True
    )

    print("\n=== NR result (HK) ===")
    print(f"status  : {sol['status']} (iters={sol['iter']})")
    print(f"T_c*    : {sol['T']:.6f} K")
    print(f"v_c*    : {sol['V']:.9f} m^3/mol")
    print(f"||F||   : {sol['norm']:.3e}")

    
    try:
        p_out = eos.pressure_tv(sol["T"], sol["V"], z)
        P_opt = p_out[0] if isinstance(p_out, (tuple, list)) else p_out
        print(f"P(T*,V*,z): {P_opt:.6g} Pa ({P_opt/1e6:.6g} MPa)")
    except Exception as e:
        print("pressure_tv failed near optimum:", e)


if __name__ == "__main__":
    main()
