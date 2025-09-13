"""
Full GO/DE example for an 11-component mixture using Thermopack's PC-SAFT EOS, 
including 2D/3D plotting of log F(T,V) (Objective Function).

- Uses Thermopack's PC-SAFT EOS (we do NOT implement PC-SAFT ourselves).
- All binary interaction parameters k_ij are set to 0 (predictive baseline).
- Objective: F(T,V) = f1^2 + (∇_n f1 · n)^2, where f1 = λ_min(dμ/dn) at fixed (T,V).

Run:
  pip install -r requirements.txt
  python examples/mix11_go_surface.py

"""

import os
import numpy as np
import pandas as pd

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm, colors

from scipy.optimize import differential_evolution
from scipy.interpolate import RegularGridInterpolator

import thermopack as tp


# Mixture + EOS setup

comp_string = "CO2,N2,C1,C2,C3,IC4,NC4,IC5,NC5,NC6,NC7"
eos = tp.pcsaft.pcsaft(comp_string)
nc = 11

# Set all k_ij = 0 for i<j  (predictive baseline)
for i in range(1, nc + 1):
    for j in range(i + 1, nc + 1):
        eos.set_kij(i, j, 0.0)

# overall mole numbers (will be normalized to fractions)
n_raw = jnp.array(
    [0.0109, 0.0884, 0.8286, 0.0401, 0.0174, 0.0030, 0.0055, 0.0019, 0.0012, 0.0014, 0.0006],
    dtype=jnp.float64
)
n = n_raw / jnp.sum(n_raw)

# search bounds in (T, V)
bounds = [(150.0, 300.0), (1.0e-4, 1.30e-4)]

# Objective function pieces
def min_eig_dmudn(eos, n_vec, T, V):
    """Smallest eigenvalue of dμ/dn at (T,V,n)."""
    mu, dmudn = eos.chemical_potential_tv(float(T), float(V), np.array(n_vec),
                                          None, None, 1, property_flag='IR')
    dmudn = 0.5*(dmudn + dmudn.T)  # symmetrize for numerical safety
    evals = jnp.linalg.eigh(jnp.array(dmudn, dtype=jnp.float64))[0]
    return float(jnp.min(evals))


def grad_n_min_eig(eos, n_vec, T, V, h=1e-8):
    """Central finite-difference gradient of λ_min(dμ/dn) wrt n."""
    n_vec = jnp.array(n_vec, dtype=jnp.float64)
    g = jnp.zeros_like(n_vec)
    for k in range(len(n_vec)):
        n_plus  = n_vec.at[k].add(h)
        n_minus = n_vec.at[k].add(-h)
        f_plus  = min_eig_dmudn(eos, n_plus,  T, V)
        f_minus = min_eig_dmudn(eos, n_minus, T, V)
        g = g.at[k].set((f_plus - f_minus) / (2.0*h))
    return g


def ObjF(eos, n_vec, T, V):
    """
    F(T,V) = f1^2 + (∇_n f1 · n)^2, with f1 = λ_min(dμ/dn).
    Return BIG if state is invalid.
    """
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


def minimize_ObjF(eos, n_vec, bounds, pop_size=20, max_iter=600, mutation=(0.5, 1.0), recombination=0.7):
    """Global search over (T, V) for fixed n using Differential Evolution."""
    def objective_wrapper(state):
        T_, V_ = state
        return ObjF(eos, n_vec, T_, V_)

    res = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        strategy='best1bin',
        popsize=pop_size,
        maxiter=max_iter,
        mutation=mutation,
        recombination=recombination,
        tol=1e-6,
        seed=None,
        polish=True,
        updating='deferred',
        workers=1
    )
    return (float(res.x[0]), float(res.x[1])), float(res.fun)



# Run optimization
(T_opt, V_opt), F_opt = minimize_ObjF(eos, n, bounds)
print(f"Optimal T, V = {T_opt:.6g} K, {V_opt:.6g} m^3/mol")
print(f"Optimal F     = {F_opt:.6g}")

# Pressure & diagnostics
try:
    p_out = eos.pressure_tv(T_opt, V_opt, np.array(n))
    P_opt = p_out[0] if isinstance(p_out, (tuple, list)) else p_out
    print(f"P(T*,V*,n)    = {P_opt:.6g} Pa  ({P_opt/1e6:.6g} MPa)")
except Exception as e:
    print("pressure_tv failed at optimum (often fine near critical/spinodal):", str(e))

try:
    lam_min = min_eig_dmudn(eos, n, T_opt, V_opt)
    print(f"λ_min(dμ/dn)  = {lam_min:.6g}")
except Exception as e:
    print("λ_min diagnostic failed:", str(e))



# Derived densities
comp_order = ["CO2","N2","C1","C2","C3","IC4","NC4","IC5","NC5","NC6","NC7"]
M_i = np.array([
    44.0095e-3, 28.0134e-3, 16.043e-3, 30.069e-3, 44.097e-3,
    58.123e-3, 58.123e-3, 72.151e-3, 72.151e-3, 86.178e-3, 100.205e-3
], dtype=float)

x = np.array(n, dtype=float)
x = x / x.sum()
M_mix = float((x * M_i).sum())
rho_molar = 1.0 / float(V_opt)        # mol/m^3
rho_mass  = rho_molar * M_mix         # kg/m^3

print(f"M_mix        = {M_mix:.6g} kg/mol")
print(f"ρ_molar      = {rho_molar:.6g} mol/m^3")
print(f"ρ_mass       = {rho_mass:.6g}  kg/m^3")



# Build grid and evaluate F

NT, NV = 200, 200
Tvec = np.linspace(bounds[0][0], bounds[0][1], NT)      # K
Vvec = np.linspace(bounds[1][0], bounds[1][1], NV)      # m^3/mol
TT, VV = np.meshgrid(Tvec, Vvec)  # shapes (NV, NT)

Z = np.full_like(TT, np.nan, dtype=float)
for j in range(NV):
    for i in range(NT):
        try:
            Z[j, i] = ObjF(eos, n, TT[j, i], VV[j, i])
        except Exception:
            Z[j, i] = np.nan

# log plot array
eps = 1e-16
Zplot = np.log(Z + eps)
Zplot[~np.isfinite(Zplot)] = np.nan



# 2D heatmap
os.makedirs("results", exist_ok=True)

norm = colors.Normalize(vmin=np.nanmin(Zplot), vmax=np.nanmax(Zplot))
S = 1e4
VVp = VV * S
Vopt_p = V_opt * S

plt.close("all")
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
fig.subplots_adjust(left=0.20, right=0.84, bottom=0.18, top=0.95)
ax.set_axisbelow(True)

pc = ax.pcolormesh(TT, VVp, Zplot, shading="auto", cmap=cm.viridis, norm=norm, rasterized=True)

# light contours
try:
    vmin = np.nanmin(Zplot); vmax = np.nanpercentile(Zplot, 97)
    levels = np.linspace(vmin, vmax, 5)
    ax.contour(TT, VVp, Zplot, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
except Exception:
    pass

# mark optimum
ax.plot([T_opt], [Vopt_p], marker="*", ms=11, mfc="white", mec="#d62728",
        mew=1.5, linestyle="none", zorder=6)

ax.set_xlabel("Temperature (K)", labelpad=30, fontsize=30)
ax.set_ylabel(r"Molar volume ($\times 10^{-4}$ m$^3$/mol)", labelpad=30, fontsize=30)
ax.set_xlim(149.9, 300)
ax.set_ylim(0.999, 1.30)

ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

for s in ("left","bottom","right","top"):
    ax.spines[s].set_position(("outward", 0))
ax.grid(True, which="major", color="0.92", linewidth=1.0)

cax = fig.add_axes([0.86, 0.18, 0.03, 0.77])
cb = fig.colorbar(pc, cax=cax)
cb.set_label("log(F)", fontsize=30, labelpad=8)
cb.ax.tick_params(labelsize=16, direction="in")
cb.locator = mticker.MaxNLocator(5); cb.update_ticks()

fig.savefig("results/GO_surface_heatmap_pub_clean.png", dpi=900, bbox_inches="tight")
fig.savefig("results/GO_surface_heatmap_pub_clean.pdf",  bbox_inches="tight")
plt.show()


# 3D surface
Zlog = np.log(np.maximum(Z, eps))
Zlog[~np.isfinite(Zlog)] = np.nan

# color normalization reuse
if 'norm' not in globals():
    norm = colors.Normalize(vmin=np.nanmin(Zlog), vmax=np.nanmax(Zlog))

Vvec_col = VV[:, 0]
Tvec_row = TT[0, :]
interp = RegularGridInterpolator((Vvec_col, Tvec_row), Zlog, bounds_error=False, fill_value=np.nan)
try:
    z_star = float(interp([[float(V_opt), float(T_opt)]]))
except Exception:
    z_star = np.nan

plt.close('all')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans']

fig = plt.figure(figsize=(18, 16), dpi=300)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(TT, VVp, Zlog, cmap=cm.viridis, norm=norm, linewidth=0, antialiased=False)

ax.view_init(elev=25, azim=-135)
ax.set_xlabel("Temperature (K)", fontsize=25, labelpad=14)
ax.set_ylabel(r"Molar volume ($\times 10^{-4}$ m$^3$/mol)", fontsize=25, labelpad=16)
ax.set_zlabel("log(F)", fontsize=20, labelpad=16)
ax.set_xlim(150, 300)
ax.set_ylim(0.95, 1.30)

ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
ax.minorticks_on()
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())

if np.isfinite(z_star):
    ax.scatter([T_opt], [Vopt_p], [z_star + 0.02], s=100, c='white',
               edgecolors='#d62728', linewidths=1.6, marker='*',
               depthshade=False, zorder=10)

CB = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.08)
CB.set_label("log(F)", fontsize=25)
CB.ax.tick_params(labelsize=20, direction="in")

fig.tight_layout()
fig.savefig("results/GO_surface_3D_pub_units_matched.png", dpi=900, bbox_inches="tight")
fig.savefig("results/GO_surface_3D_pub_units_matched.pdf", bbox_inches="tight")
plt.show()
