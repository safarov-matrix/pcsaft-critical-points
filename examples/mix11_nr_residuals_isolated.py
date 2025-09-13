"""
Full Newton–Raphson (NR) example for an 11-component mixture using Thermopack's PC-SAFT EOS, 
including 2D/3D plotting of log ||F|| (HK residual norm).

- Uses Thermopack's PC-SAFT EOS.
- All binary interaction parameters k_ij are set to 0 (predictive baseline).
- Solves Heidemann–Khalil (HK) criticality conditions directly with a damped NR method.
- Objective: drive HK residuals (g_i, g, h) → 0 at fixed (T,V,n).

Run:
  pip install -r requirements.txt
  python examples/mix11_nr_surface.py
"""


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.ticker as mticker
from scipy.interpolate import RegularGridInterpolator
import thermopack as tp

# Mixture and EOS
comp_string = "CO2,N2,C1,C2,C3,IC4,NC4,IC5,NC5,NC6,NC7"
eos = tp.pcsaft.pcsaft(comp_string)
nc = 11

# kij = 0 predictive baseline
for i in range(1, nc+1):
    for j in range(i+1, nc+1):
        eos.set_kij(i, j, 0.0)

# Overall composition
n_raw = np.array([0.0109, 0.0884, 0.8286, 0.0401, 0.0174, 0.0030, 0.0055, 0.0019, 0.0012, 0.0014, 0.0006], float)
z = n_raw / n_raw.sum()

# (T, V) search window
T_bounds = (150.0, 300.0)        # K
V_bounds = (0.60e-4, 1.40e-4)    # m^3/mol

BIG = 1e8


def get_mu_H(T, V, n_vec):
    """Return (mu, H) where H = dmu/dn|_{T,V}. Symmetrize for stability."""
    try:
        mu, dmu_dn = eos.chemical_potential_tv(float(T), float(V), np.array(n_vec), None, None, 1, property_flag='IR')
        H = 0.5*(dmu_dn + dmu_dn.T)
        return np.asarray(mu, float), np.asarray(H, float)
    except Exception:
        return None, None

def safe_pressure(T, V, n_vec):
    try:
        p_out = eos.pressure_tv(float(T), float(V), np.array(n_vec))
        return p_out[0] if isinstance(p_out, (tuple, list)) else float(p_out)
    except Exception:
        return np.nan

def phi_dir(T, V, n_vec, d):
    """φ(0) = d^T H(T,V;n) d used for the HK cubic directional derivative."""
    _, H = get_mu_H(T, V, n_vec)
    if H is None: return np.inf
    return float(d @ H @ d)

def pure_critical_table():
    """Build per-component (Tci, Vci) via EOS (fallbacks to None if not available)."""
    Tci, Vci = np.full(nc, np.nan), np.full(nc, np.nan)
    for k in range(nc):
        n_pure = np.zeros(nc); n_pure[k] = 1.0
        try:
            Tc_k, Vc_k, Pc_k = eos.critical(n_pure)
            Tci[k], Vci[k] = float(Tc_k), float(Vc_k)
        except Exception:
            pass
    return Tci, Vci

def kay_seed_TV(z):
    """Kay-type seeds: Tc0 = sum z_i Tci; vc0 = sum z_i vci; fallback to mid-box."""
    Tci, Vci = pure_critical_table()
    if np.all(np.isfinite(Tci)):
        T0 = float(np.dot(z, Tci))
    else:
        T0 = 0.5*(T_bounds[0] + T_bounds[1])
    if np.all(np.isfinite(Vci)):
        V0 = float(np.dot(z, Vci))
    else:
        V0 = 0.5*(V_bounds[0] + V_bounds[1])
    # clamp
    T0 = min(max(T0, T_bounds[0]), T_bounds[1])
    V0 = min(max(V0, V_bounds[0]), V_bounds[1])
    return T0, V0


def hk_residuals(T, V, d, n_vec, h2=1e-5):
    """
    g_i: (H d)_i = 0, i=1..nc
    g  : centered directional derivative of φ(α)=d^T H(n+α d) d at α=0
    h  : ||d||^2 - 1 = 0
    """
    _, H = get_mu_H(T, V, n_vec)
    if H is None or not np.all(np.isfinite(H)):
        g_vec  = np.full(nc, np.sqrt(BIG))
        g_cub  = np.sqrt(BIG)
        h_norm = d @ d - 1.0
        return np.concatenate([g_vec, [g_cub, h_norm]])

    g_vec = H @ d

    phi_p = phi_dir(T, V, n_vec + h2*d, d)
    phi_m = phi_dir(T, V, n_vec - h2*d, d)
    g_cub = np.sqrt(BIG) if (not np.isfinite(phi_p) or not np.isfinite(phi_m)) else (phi_p - phi_m)/(2*h2)

    h_norm = d @ d - 1.0
    return np.concatenate([g_vec, [g_cub, h_norm]])


def numerical_jacobian(T, V, d, n_vec, hT=None, hV=None, hd=2e-6):
    """
    Central differences for T and V; forward for d-components.
    Returns J ((nc+2) x (2+nc)) and F0.
    """
    F0 = hk_residuals(T, V, d, n_vec)
    m = F0.size
    J = np.zeros((m, 2+nc), float)

    # scale-aware steps
    hT = hT or max(1e-2, 1e-4*max(200.0, abs(T)))
    hV = hV or max(5e-8, 1e-3*max(1e-4, abs(V)))

    # d/dT (central)
    FTp = hk_residuals(T + hT, V, d, n_vec)
    FTm = hk_residuals(T - hT, V, d, n_vec)
    J[:, 0] = (FTp - FTm)/(2*hT)

    # d/dV (central)
    FVp = hk_residuals(T, V + hV, d, n_vec)
    FVm = hk_residuals(T, V - hV, d, n_vec)
    J[:, 1] = (FVp - FVm)/(2*hV)

    # d/dd_i (forward)
    for i in range(nc):
        dd = np.zeros_like(d); dd[i] = hd
        Fd = hk_residuals(T, V, d + dd, n_vec)
        J[:, 2+i] = (Fd - F0)/hd

    return J, F0

def lm_step(J, F, lam):
    """Levenberg–Marquardt: (J^T J + lam I) dx = -J^T F."""
    JTJ = J.T @ J
    A = JTJ + lam*np.eye(J.shape[1])
    b = -J.T @ F
    dx = np.linalg.solve(A, b)
    return dx


def seed_direction(T, V, n_vec):
    """Use smallest-eigenvector of H(T,V;n) as Δn seed; fallback to z^(2/3)."""
    _, H = get_mu_H(T, V, n_vec)
    if H is not None and np.all(np.isfinite(H)):
        w, U = np.linalg.eigh(H)
        d0 = U[:, np.argmin(w)]
        s = np.linalg.norm(d0)
        if np.isfinite(s) and s > 0:
            return d0/s
    d0 = np.power(n_vec, 2.0/3.0)
    s  = np.linalg.norm(d0)
    return d0/s if s > 0 else np.ones_like(n_vec)/np.sqrt(n_vec.size)

def clamp_TV(T, V):
    T = min(max(T, T_bounds[0]), T_bounds[1])
    V = min(max(V, V_bounds[0]), V_bounds[1])
    return T, V

def newton_hk_solver(n_vec,
                     T0=None, V0=None,
                     max_iter=120,
                     tol_F=1e-8, tol_step=1e-9,
                     h2=1e-5,
                     lam0=1e-2,
                     verbose=True):
    """
    Solve F(T,V,d)=0 with NR.
    Your run reports CONVERGED.
    """

    if T0 is None or V0 is None:
        T0, V0 = kay_seed_TV(n_vec)
    T, V = clamp_TV(T0, V0)
    d    = seed_direction(T, V, n_vec)

    lam = lam0
    F = hk_residuals(T, V, d, n_vec, h2=h2)
    best = {"T":T, "V":V, "d":d.copy(), "F":F.copy(), "norm":np.linalg.norm(F)}

    if verbose:
        print(f"[init] T0={T:.4f} K, V0={V:.6e} m^3/mol, ‖F‖={best['norm']:.3e}")

    for it in range(1, max_iter+1):
        J, F = numerical_jacobian(T, V, d, n_vec)
        res0 = np.linalg.norm(F)

        accept = False
        lam_try = lam
        for _ in range(12):
            # (J^T J + λI) dx = -J^T F
            JTJ = J.T @ J
            dx = np.linalg.solve(JTJ + lam_try*np.eye(J.shape[1]), -J.T @ F)

            dT, dV, dd = dx[0], dx[1], dx[2:]
            T_try, V_try = clamp_TV(T + dT, V + dV)
            d_try = d + dd
            s = np.linalg.norm(d_try)
            if not np.isfinite(s) or s == 0:
                lam_try *= 5.0
                continue
            d_try = d_try / s  # renormalize after each update

            F_try = hk_residuals(T_try, V_try, d_try, n_vec, h2=h2)
            res1 = np.linalg.norm(F_try)
            if np.isfinite(res1) and res1 < res0:
                T, V, d = T_try, V_try, d_try
                lam = max(lam_try/3.0, 1e-8)
                if res1 < best["norm"]:
                    best = {"T":T, "V":V, "d":d.copy(), "F":F_try.copy(), "norm":res1}
                accept = True
                break
            lam_try *= 3.0

        if verbose:
            _, Hc = get_mu_H(T, V, n_vec)
            lmin = np.min(np.linalg.eigh(Hc)[0]) if Hc is not None else np.nan
            print(f"[it {it:02d}] ‖F‖={best['norm']:.3e}, T={T:.3f}, V={V:.7e}, λ_min={lmin:.3e}, λ={lam:.1e}")

        # primary convergence checks
        if best["norm"] < tol_F:
            return {"status":"converged", "iter":it, **best}
        if not accept:
            # If HK geometry is satisfied, accept as converged
            Ffin = hk_residuals(T, V, d, n_vec, h2=h2)
            g_lin = Ffin[:nc]         # H d
            g_cub = Ffin[nc]          # cubic directional
            h_nrm = Ffin[nc+1]        # normalization
            _, Hfin = get_mu_H(T, V, n_vec)
            lam_min = (np.min(np.linalg.eigh(Hfin)[0]) if Hfin is not None else np.nan)

            if (np.isfinite(lam_min) and abs(lam_min) < 1e-8 and
                abs(h_nrm) < 1e-10 and
                np.linalg.norm(g_lin) < 1e-6 and
                abs(g_cub) < 1e-6):
                return {"status":"converged", "iter":it, "T":T, "V":V, "d":d.copy(),
                        "F":Ffin.copy(), "norm":np.linalg.norm(Ffin)}
            return {"status":"stalled", "iter":it, **best}

        if np.linalg.norm(dx) < tol_step and best["norm"] < 10*tol_F:
            return {"status":"converged", "iter":it, **best}

    Ffin = hk_residuals(T, V, d, n_vec, h2=h2)
    g_lin = Ffin[:nc]; g_cub = Ffin[nc]; h_nrm = Ffin[nc+1]
    _, Hfin = get_mu_H(T, V, n_vec)
    lam_min = (np.min(np.linalg.eigh(Hfin)[0]) if Hfin is not None else np.nan)
    if (np.isfinite(lam_min) and abs(lam_min) < 1e-8 and
        abs(h_nrm) < 1e-10 and
        np.linalg.norm(g_lin) < 1e-6 and
        abs(g_cub) < 1e-6):
        return {"status":"converged", "iter":max_iter, "T":T, "V":V, "d":d.copy(),
                "F":Ffin.copy(), "norm":np.linalg.norm(Ffin)}

    return {"status":"max_iter", "iter":max_iter, **best}


sol = newton_hk_solver(z,
                       T0=None, V0=None,
                       max_iter=150,
                       tol_F=1e-8, tol_step=1e-9,
                       h2=1e-5, lam0=1e-2,
                       verbose=True)

print("\n=== NR result (HK, NR-only) ===")
print(f"status = {sol['status']} (iters={sol['iter']})")
print(f"T_c*   = {sol['T']:.6f} K")
print(f"v_c*   = {sol['V']:.9f} m^3/mol")

P_star = safe_pressure(sol["T"], sol["V"], z)
if np.isfinite(P_star):
    print(f"P(T*,V*,n) = {P_star:.6g} Pa  ({P_star/1e6:.6g} MPa)")
else:
    print("P(T*,V*,n) could not be evaluated robustly near spinodal.")

# Spinodal/cubic checks
_, H_star = get_mu_H(sol["T"], sol["V"], z)
if H_star is not None:
    w, _ = np.linalg.eigh(H_star)
    print(f"λ_min(H)     = {np.min(w):.6e}")
print(f"‖F‖ at soln  = {sol['norm']:.3e}")


# Grid for visualization
NT, NV = 180, 180
Tvec = np.linspace(*T_bounds, NT)
Vvec = np.linspace(*V_bounds, NV)
TT, VV = np.meshgrid(Tvec, Vvec)
d_plot = sol["d"]
Z = np.full_like(TT, np.nan, float)
for j in range(NV):
    for i in range(NT):
        Fij = hk_residuals(TT[j,i], VV[j,i], d_plot, z)
        val = norm(Fij)
        Z[j,i] = np.nan if not np.isfinite(val) else val

eps = 1e-16
Zplot = np.log(Z + eps); Zplot[~np.isfinite(Zplot)] = np.nan

normc = colors.Normalize(vmin=np.nanmin(Zplot), vmax=np.nanpercentile(Zplot, 97))
S = 1e4
VVp = VV * S
Vopt_p = sol["V"] * S

plt.close("all")
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
fig.subplots_adjust(left=0.20, right=0.84, bottom=0.18, top=0.95)
ax.set_axisbelow(True)

pc = ax.pcolormesh(TT, VVp, Zplot, shading="auto", cmap=cm.viridis, norm=normc, rasterized=True)

try:
    levels = np.linspace(np.nanmin(Zplot), np.nanpercentile(Zplot, 97), 5)
    ax.contour(TT, VVp, Zplot, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
except Exception:
    pass


ax.plot([sol["T"]], [Vopt_p], marker="*", ms=11, mfc="white", mec="#d62728",
        mew=1.5, linestyle="none", zorder=6)

ax.set_xlabel("Temperature (K)", labelpad=30, fontsize=30)
ax.set_ylabel(r"Molar volume ($\times 10^{-4}$ m$^3$/mol)", labelpad=30, fontsize=30)
ax.set_xlim(150, 300)
ax.set_ylim(0.598, 1.40)

ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.tick_params(axis="both", which="major", labelsize=22, direction="in", length=7, width=1.1, pad=8)
ax.tick_params(axis="both", which="minor", direction="in", length=4, width=1.0, labelsize=0)
ax.grid(True, which="major", color="0.92", linewidth=1.0)

cax = fig.add_axes([0.86, 0.18, 0.03, 0.77])
cb = fig.colorbar(pc, cax=cax)
cb.set_label(r"$\log\|\mathbf{F}\|$", fontsize=30, labelpad=8)
cb.ax.tick_params(labelsize=16, direction="in")
cb.locator = mticker.MaxNLocator(5); cb.update_ticks()

plt.savefig("NR_heatmap_logF.png", dpi=900, bbox_inches="tight")
plt.savefig("NR_heatmap_logF.pdf",  bbox_inches="tight")
plt.show()

Zlog = np.log(np.maximum(Z, eps)); Zlog[~np.isfinite(Zlog)] = np.nan
normc = colors.Normalize(vmin=np.nanmin(Zlog), vmax=np.nanpercentile(Zlog, 97))

Vvec_plot = VV[:, 0]
Tvec_plot = TT[0, :]
interp = RegularGridInterpolator((Vvec_plot, Tvec_plot), Zlog, bounds_error=False, fill_value=np.nan)
try:
    z_star = float(interp([[float(sol["V"]), float(sol["T"])]]))
except Exception:
    z_star = np.nan

plt.close('all')
fig = plt.figure(figsize=(18, 16), dpi=300)
fig.subplots_adjust(left=0.10, right=0.88, bottom=0.08, top=0.97)

ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(TT, VVp, Zlog, cmap=cm.viridis, norm=normc, linewidth=0, antialiased=False)

ax.view_init(elev=25, azim=-135)
ax.set_xlabel("Temperature (K)", fontsize=25, labelpad=14)
ax.set_ylabel(r"Molar volume ($\times 10^{-4}$ m$^3$/mol)", fontsize=25, labelpad=16)
ax.set_zlabel(r"$\log\|\mathbf{F}\|$", fontsize=20, labelpad=16)


ax.set_xlim(150, 300)
ax.set_ylim(0.55, 1.40)


ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.minorticks_on()
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

ax.tick_params(axis='x', which='major', labelsize=20, pad=6, direction='in')
ax.tick_params(axis='y', which='major', labelsize=20, pad=6, direction='in')
ax.tick_params(axis='z', which='major', labelsize=20, pad=6, direction='in')

if np.isfinite(z_star):
    ax.scatter([sol["T"]], [sol["V"]*S], [z_star + 0.02], s=100, c='white',
               edgecolors='#d62728', linewidths=1.6, marker='*',
               depthshade=False, zorder=10)

CB = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
CB.set_label(r"$\log\|\mathbf{F}\|$", fontsize=25)
CB.ax.tick_params(labelsize=20, direction='in')

plt.savefig("NR_surface_3D_logF.png", dpi=900, bbox_inches="tight")
plt.savefig("NR_surface_3D_logF.pdf",  bbox_inches="tight")
plt.show()
