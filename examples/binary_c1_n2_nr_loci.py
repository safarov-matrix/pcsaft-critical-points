"""
examples/binary_c1_n2_nr_locus.py
Full NR example for a binary mixture (C1–N2) using Thermopack PC-SAFT,
including CSV export and an overlay plot against experimental data.

- Uses Thermopack's PC-SAFT EOS (we do NOT implement PC-SAFT ourselves).
- All binary interaction parameters k_ij are set to 0 (predictive baseline).
- Locus is computed by calling eos.critical(n) on a composition grid.

Run:
  pip install -r requirements.txt
  python examples/binary_c1_n2_nr_locus.py

Input:
  data/Methane_Nitrogen_Critical_Data.csv  (experimental points)

Output:
  C1-N2_NR_critical_locus_PT.csv
  C1-N2_NR_overlay_clean.png / .pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator

import jax
jax.config.update("jax_enable_x64", True)

import thermopack as tp

# ------------------- Plot style -------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 28,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ------------------- Mixture / EOS -------------------
species   = ("C1", "N2")   # Thermopack IDs
mix_label = "C1-N2"

eos = tp.pcsaft.pcsaft(",".join(species))

# Set all k_ij = 0 (predictive baseline)
nc = len(species)
for i in range(1, nc+1):
    for j in range(i+1, nc+1):
        eos.set_kij(i, j, 0.0)

# ------------------- Composition grid -------------------
N_GRID = 61
EPS_Z  = 1e-6
z1_grid = np.linspace(EPS_Z, 1.0 - EPS_Z, N_GRID)

# ------------------- NR locus sweep -------------------
rows, skipped = [], 0
for z1 in z1_grid:
    n = np.array([z1, 1.0 - z1], dtype=float)
    try:
        Tc, Vc, pc = eos.critical(n)          # T [K], V [m^3/mol], p [Pa]
        rho_c  = 1.0 / float(Vc)              # mol/m^3 (since sum(n)=1)
        Pc_MPa = float(pc) / 1e6              # MPa
        rows.append([float(z1), 1.0 - float(z1), float(Tc), Pc_MPa, rho_c, "NR"])
    except Exception:
        skipped += 1
        continue

df_nr = pd.DataFrame(rows, columns=["z1","z2","Tc[K]","Pc[MPa]","rho_c[mol/m^3]","solver"])
df_nr = df_nr[np.isfinite(df_nr["Tc[K]"]) & np.isfinite(df_nr["Pc[MPa]"])]
df_nr = df_nr.drop_duplicates(subset=["z1"]).sort_values("z1").reset_index(drop=True)

# Save + quick stats
csv_path = f"{mix_label}_NR_critical_locus_PT.csv"
df_nr.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}  |  points: {len(df_nr)}  |  skipped: {skipped}")
print(df_nr[["Tc[K]","Pc[MPa]"]].describe())

# ------------------- Load experimental CSV -------------------
# Expecting: data/Methane_Nitrogen_Critical_Data.csv
exp_path = "data/Methane_Nitrogen_Critical_Data.csv"
exp = pd.read_csv(exp_path).copy()

# Normalize headers
exp.rename(columns={c: c.strip() for c in exp.columns}, inplace=True)
lower = {c.lower(): c for c in exp.columns}
if "tc_k"   in lower: exp.rename(columns={lower["tc_k"]  :"Tc_K"},   inplace=True)
if "pc_mpa" in lower: exp.rename(columns={lower["pc_mpa"]:"Pc_MPa"}, inplace=True)

# Ensure we have methane mole fraction in 'z1'
if "z1" not in exp.columns:
    # Try common aliases for methane composition
    for k in ["z1_C1","x1","x_C1","z1(C1)","x(C1)","x_methane","z_C1"]:
        if k in exp.columns:
            exp.rename(columns={k: "z1"}, inplace=True)
            break

# If only nitrogen composition provided, infer z1 = 1 - z_N2
if "z1" not in exp.columns:
    for k in ["z1_N2","x_N2","x(N2)","z_N2","y_N2","X_N2"]:
        if k in exp.columns:
            exp["z1"] = 1.0 - exp[k]
            break

if "z1" not in exp.columns:
    raise ValueError(
        "Experimental CSV must include methane composition (e.g., 'z1' or 'x1'), "
        "or a nitrogen composition column so z1 can be inferred as 1 - z_N2."
    )

# ------------------- RMSE vs experiment -------------------
z_mod = df_nr["z1"].to_numpy()
T_mod = df_nr["Tc[K]"].to_numpy()
P_mod = df_nr["Pc[MPa]"].to_numpy()

exp_keep = exp[(exp["z1"] >= z_mod.min()) & (exp["z1"] <= z_mod.max())].reset_index(drop=True)
Tc_hat = np.interp(exp_keep["z1"], z_mod, T_mod)
Pc_hat = np.interp(exp_keep["z1"], z_mod, P_mod)
rmse_T = float(np.sqrt(np.mean((Tc_hat - exp_keep["Tc_K"])**2)))
rmse_P = float(np.sqrt(np.mean((Pc_hat - exp_keep["Pc_MPa"])**2)))
print(f"NR RMSE ({mix_label}): ΔT = {rmse_T:.2f} K, ΔP = {rmse_P:.3f} MPa (N={len(exp_keep)})")

# ------------------- Smooth locus with PCHIP -------------------
z1 = df_nr["z1"].to_numpy()
Tc = df_nr["Tc[K]"].to_numpy()
Pc = df_nr["Pc[MPa]"].to_numpy()
z_dense = np.linspace(z1.min(), z1.max(), 1500)
T_line  = PchipInterpolator(z1, Tc)(z_dense)
P_line  = PchipInterpolator(z1, Pc)(z_dense)

# ------------------- Plot: NR + Experimental -------------------
plt.close('all')
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
fig.subplots_adjust(left=0.20, right=0.86, bottom=0.20, top=0.95)
ax.set_axisbelow(True)

nr_color = "#d62728"   # NR red
underlay = dict(color="white", lw=6.5,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)

# NR curve + square markers
ax.plot(T_line, P_line, **underlay)
ax.plot(T_line, P_line, color=nr_color, lw=4.5, zorder=3,
        solid_capstyle="round", solid_joinstyle="round", label="Newton–Raphson")
ax.plot(T_line, P_line, linestyle="none", marker="s", markevery=40,
        ms=7, mfc="white", mec=nr_color, mew=1.6, zorder=5)

# Experimental: solid black triangles
ax.scatter(exp_keep["Tc_K"], exp_keep["Pc_MPa"],
           marker="^", facecolors="black", edgecolors="black",
           s=110, linewidths=1.2, zorder=6, label="Experimental")

# Labels
ax.set_xlabel("Critical Temperature, K", labelpad=28)
ax.set_ylabel("Critical Pressure, MPa", labelpad=28)

# Axes & grid
ax.grid(True, which="major", color="0.92", linewidth=1.0)
ax.set_xlim(120, 200)
ax.set_ylim(2.5, 6.0)
ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
               length=6, width=1.2, pad=12)
ax.tick_params(axis="x", which="major", pad=16)
ax.tick_params(axis="both", which="minor", length=3, width=1.0)
for s in ("left","bottom","right","top"):
    ax.spines[s].set_position(("outward", 0))

# Legend
ax.legend(loc="upper left", frameon=False, fontsize=22,
          handlelength=2.8, bbox_to_anchor=(0.03, 0.97), borderaxespad=0.3)

# Save & show
plt.savefig(f"{mix_label}_NR_overlay_clean.png", dpi=900, bbox_inches="tight")
plt.savefig(f"{mix_label}_NR_overlay_clean.pdf",  bbox_inches="tight")
plt.show()
