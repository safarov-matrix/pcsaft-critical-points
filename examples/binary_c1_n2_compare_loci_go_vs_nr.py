import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from google.colab import files
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

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

print("⬆️ Upload 3 CSV files: Newton-Raphson (NR) loci, Global Optimization (GO) loci, and Experimental data (in this example, C1–N2).")
uploaded = files.upload()
assert uploaded, "No files uploaded."

dfs = {}
for name, content in uploaded.items():
    df = pd.read_csv(name)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    dfs[name] = df  

nr_name = None
go_name = None
exp_name = None

for name, df in dfs.items():
    cols = {c.lower() for c in df.columns}
    if {"tc_k", "pc_mpa"}.issubset(cols) or ("tc_k" in cols) or ("pc_mpa" in cols):
        exp_name = name
    elif "go" in name.lower():
        go_name = name
    elif "nr" in name.lower():
        nr_name = name

candidates = [n for n in dfs if n not in {exp_name}]
if go_name is None or nr_name is None:
    model_like = [n for n in candidates if {"Tc[K]", "Pc[MPa]"}.issubset(set(dfs[n].columns))]
    if len(model_like) == 2:
        if nr_name is None: nr_name = model_like[0]
        if go_name is None: go_name = model_like[1]

assert exp_name is not None,  "Could not detect the Experimental CSV (needs Tc_K, Pc_MPa)."
assert nr_name  is not None,  "Could not detect the NR CSV (try including 'NR' in filename)."
assert go_name  is not None,  "Could not detect the GO CSV (try including 'GO' in filename)."

print("Detected files:")
print(" • Experimental:", exp_name)
print(" • NR:", nr_name)
print(" • GO:", go_name)

exp = dfs[exp_name].copy()
lower = {c.lower(): c for c in exp.columns}
if "tc_k"   in lower: exp.rename(columns={lower["tc_k"]  :"Tc_K"},   inplace=True)
if "pc_mpa" in lower: exp.rename(columns={lower["pc_mpa"]:"Pc_MPa"}, inplace=True)
if "z1" not in exp.columns:
    for cand in ["z1_C1","x1","x_C1","z1(C1)","z1(C1)","x(C1)","x(C1)"]:
        if cand in exp.columns:
            exp.rename(columns={cand:"z1"}, inplace=True)
            break
if "z1" not in exp.columns:
    raise ValueError("Experimental CSV needs a CH4 composition column (e.g., z1 or z1_CH4/x1).")

# NR loci
df_nr = dfs[nr_name].copy()
if "z1" not in df_nr.columns:
    if "z2" in df_nr.columns:
        df_nr["z1"] = 1.0 - df_nr["z2"]
    else:
        raise ValueError("NR CSV must include 'z1' (or 'z2').")

# GO loci
df_go = dfs[go_name].copy()
if "z1" not in df_go.columns:
    if "z2" in df_go.columns:
        df_go["z1"] = 1.0 - df_go["z2"]
    else:
        raise ValueError("GO CSV must include 'z1' (or 'z2').")

def _clean(df):
    df = df[np.isfinite(df.get("Tc[K]", np.nan)) & np.isfinite(df.get("Pc[MPa]", np.nan))]
    df = df.drop_duplicates(subset=["z1"]).sort_values("z1").reset_index(drop=True)
    return df

df_nr = _clean(df_nr)
df_go = _clean(df_go)

zmin = max(df_nr["z1"].min(), df_go["z1"].min())
zmax = min(df_nr["z1"].max(), df_go["z1"].max())
exp_keep = exp[(exp["z1"] >= zmin) & (exp["z1"] <= zmax)].reset_index(drop=True)

print(f"Counts → NR: {len(df_nr)}  GO: {len(df_go)}  EXP used: {len(exp_keep)}")

# building lines
def line_from_nr(df, npts=1500):
    z = df["z1"].to_numpy()
    T = df["Tc[K]"].to_numpy()
    P = df["Pc[MPa]"].to_numpy()
    z_dense = np.linspace(z.min(), z.max(), npts)
    T_line  = PchipInterpolator(z, T)(z_dense)
    P_line  = PchipInterpolator(z, P)(z_dense)
    return T, P, T_line, P_line

def line_from_go(df, npts=1500, s_factor=0.002):
    T = df["Tc[K]"].to_numpy()
    P = df["Pc[MPa]"].to_numpy()
    idx = np.argsort(T)
    T_ord, P_ord = T[idx], P[idx]
    s = s_factor * len(T_ord) * (np.ptp(P_ord) ** 2)
    spl = UnivariateSpline(T_ord, P_ord, k=3, s=s)
    T_line = np.linspace(T_ord.min(), T_ord.max(), npts)
    P_line = spl(T_line)
    return T, P, T_line, P_line

T_nr, P_nr, Tline_nr, Pline_nr = line_from_nr(df_nr)
T_go, P_go, Tline_go, Pline_go = line_from_go(df_go, s_factor=0.002)

# RMSE vs experiment (C1–C3)
def rmse_vs_exp(df_model, exp_df):
    z = df_model["z1"].to_numpy()
    T = df_model["Tc[K]"].to_numpy()
    P = df_model["Pc[MPa]"].to_numpy()
    Tc_mod = np.interp(exp_df["z1"], z, T)
    Pc_mod = np.interp(exp_df["z1"], z, P)
    rmse_T = float(np.sqrt(np.mean((Tc_mod - exp_df["Tc_K"])**2)))
    rmse_P = float(np.sqrt(np.mean((Pc_mod - exp_df["Pc_MPa"])**2)))
    return rmse_T, rmse_P

rmseT_nr, rmseP_nr = rmse_vs_exp(df_nr, exp_keep)
rmseT_go, rmseP_go = rmse_vs_exp(df_go, exp_keep)
print(f"NR RMSE (C1–N2): ΔT={rmseT_nr:.2f} K, ΔP={rmseP_nr:.3f} MPa")
print(f"GO RMSE (C1–N2): ΔT={rmseT_go:.2f} K, ΔP={rmseP_go:.3f} MPa")


plt.close('all')
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
fig.subplots_adjust(left=0.20, right=0.86, bottom=0.20, top=0.95)
ax.set_axisbelow(True)

nr_color = "#d62728"
go_color = "#2a9d8f"

underlay = dict(color="white", lw=6.5,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)


ax.plot(Tline_nr, Pline_nr, **underlay)
ax.plot(Tline_nr, Pline_nr, color=nr_color, lw=4.5, zorder=3)
ax.plot(Tline_nr, Pline_nr, linestyle="none", marker="s", markevery=40,
        ms=7, mfc="white", mec=nr_color, mew=1.6, zorder=5)

ax.plot(Tline_go, Pline_go, **underlay)
ax.plot(Tline_go, Pline_go, color=go_color, lw=4.5, zorder=4)
ax.plot(Tline_go, Pline_go, linestyle="none", marker="o", markevery=40,
        ms=7, mfc="white", mec=go_color, mew=1.6, zorder=6)

ax.scatter(exp_keep["Tc_K"], exp_keep["Pc_MPa"],
           marker="^", facecolors="none", edgecolors="k",
           s=110, linewidths=1.8, zorder=7)

ax.set_xlabel("Temperature (K)", labelpad=28, fontsize=30)
ax.set_ylabel("Pressure (MPa)", labelpad=28, fontsize=30)
ax.tick_params(axis="both", which="major", labelsize=30)

ax.grid(True, which="major", color="0.92", linewidth=1.0)

# Set the limits
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

legend_elements = [
    Line2D([0],[0], color=nr_color, lw=4.5, marker="s", mfc="white", mec=nr_color, mew=1.6, label="Newton–Raphson"),
    Line2D([0],[0], color=go_color, lw=4.5, marker="o", mfc="white", mec=go_color, mew=1.6, label="Global Optimization"),
    Line2D([0],[0], color="k", lw=0.0, marker="^", mfc="none", mec="k", mew=1.6, label="Experimental"),
]

ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=25,
          handlelength=2.8, bbox_to_anchor=(0.03, 0.97), borderaxespad=0.3)

z = df_nr["z1"].to_numpy()
T = df_nr["Tc[K]"].to_numpy()
P = df_nr["Pc[MPa]"].to_numpy()

plt.savefig("C1–N2_NR_GO_EXP_overlay_clean.png", dpi=900, bbox_inches="tight")
plt.savefig("C1–N2_NR_GO_EXP_overlay_clean.pdf",  bbox_inches="tight")
plt.show()
