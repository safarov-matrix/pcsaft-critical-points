#!pip install thermopack

import thermopack as tp
import jax.numpy as jnp
import jax
import pandas as pd
import scipy.constants as spc
import os
import re
import importlib
from google.colab import files
jax.config.update("jax_enable_x64", True)


import numpy as np
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
from numpy import sqrt
from numpy import isfinite
from numpy.random import rand
from numpy.random import choice


from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import dual_annealing


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot
from matplotlib import cm


from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter




uploaded = files.upload()

for _name in ['min','max','sum','argmin','argmax','around','clip','asarray','rand','choice','median']:
    if _name in globals():
        del globals()[_name]

import numpy as np
N = importlib.import_module("numpy")
print("np.min from:", getattr(np.min, '__module__', '?'))



exp = pd.read_csv("Methane_Nitrogen_Critical_Data.csv")

# Normalize headers (robust to small variations)
exp.rename(columns={c: c.strip() for c in exp.columns}, inplace=True)
lower = {c.lower(): c for c in exp.columns}

if "pc_mpa" in lower:
    exp.rename(columns={lower["pc_mpa"]:"Pc_MPa"}, inplace=True)
if "tc_k" in lower:
    exp.rename(columns={lower["tc_k"] :"Tc_K"}, inplace=True)
if "source" in lower:
    exp.rename(columns={lower["source"]:"Source"}, inplace=True)

# If your exp file uses z1_CH4, map it to z1
if "z1" not in exp.columns:
    for cand in ["z1_CH4","x1","x_CH4","z1(CH4)","z1(C1)"]:
        if cand in exp.columns:
            exp.rename(columns={cand:"z1"}, inplace=True)
            break

# convenience for merges
if "x1" not in exp.columns and "z1" in exp.columns:
    exp["x1"] = exp["z1"]

print("EXP columns:", list(exp.columns))


def _mu_C_TV(eos, T, V, n):
    mu, C = eos.chemical_potential_tv(T, V, n, None, None, 1, property_flag='IR')
    return np.asarray(mu, float), np.asarray(C, float)

def _pressure_TV(eos, T, V, n):
    try:
        out = eos.pressure_tv(T, V, n)
    except TypeError:
        out = eos.pressure_tv(T, V, n, None, None, 1)
    p = out[0] if isinstance(out,(tuple,list)) else out
    return float(np.asarray(p).reshape(()))/1e6

def hk_residuals_Trho(eos, T, rho_mol, z, eps=3e-6):
    n = np.asarray(z,float); V = float(np.sum(n))/float(rho_mol)
    mu0, C = _mu_C_TV(eos,T,V,n)
    Csym = 0.5*(C+C.T)
    w,U = np.linalg.eigh(Csym)
    i_min = int(np.argmin(w)); lam = float(w[i_min])
    u = U[:,i_min]; u = u/np.linalg.norm(u)
    s = eps*max(1.0,float(np.max(n)))
    if np.any(n - s*np.abs(u) <= 0.0):
        s = 0.5*float(np.min(n/(np.abs(u)+1e-16)))
    mu_p,_ = _mu_C_TV(eos,T,V,n+s*u)
    mu_m,_ = _mu_C_TV(eos,T,V,n-s*u)
    proj = float(u @ (mu_p - 2.0*mu0 + mu_m)/(s**2))
    Pc = _pressure_TV(eos,T,V,n)
    return lam, proj, Pc, {"V":V,"rho":rho_mol,"u":u}


def hk_objective(x, eos, z, pc_window=None, alpha=5.0, eps=3e-6):
    T = float(x[0]); rho = float(np.exp(x[1]))
    if T<=1.0 or rho<=1e-12: return 1e12
    try:
        r1,r2,Pc,_ = hk_residuals_Trho(eos,T,rho,z,eps=eps)
    except Exception:
        return 1e9
    pen = 0.0
    if pc_window is not None:
        lo,hi = pc_window; dlo=max(0.0,lo-Pc); dhi=max(0.0,Pc-hi)
        pen = 10.0*(max(dlo,dhi)**2)
    return r1*r1 + alpha*(r2*r2) + pen

def _clamp_additive_window(center, broad, halfwidth):
    lo = max(broad[0], center-halfwidth); hi = min(broad[1], center+halfwidth)
    return (lo,hi) if (hi>lo) else broad

def _clamp_multiplicative_window(center, broad, flo=0.75, fhi=1.33, min_span_ratio=1.12):
    lo = max(broad[0], flo*center); hi = min(broad[1], fhi*center)
    if not (hi>lo): return broad
    if (hi/lo) < min_span_ratio:
        mid = np.sqrt(lo*hi)
        lo = max(broad[0], mid/np.sqrt(min_span_ratio))
        hi = min(broad[1], mid*np.sqrt(min_span_ratio))
    if not (hi>lo): return broad
    return (lo,hi)


def minimize_ObjF_DE(eos, z, T_bounds, rho_bounds,
                     alpha=5.0, eps=3e-6,
                     popsize=24, maxiter=450, seed=17, pc_window=None):
    rlo,rhi = float(rho_bounds[0]), float(rho_bounds[1])
    if not (rhi>rlo) or (rlo<=0.0): raise ValueError("Invalid rho bounds")
    bnds = [(float(T_bounds[0]), float(T_bounds[1])), (np.log(rlo), np.log(rhi))]

    def fun(x): return hk_objective(x, eos, np.asarray(z,float),
                                    pc_window=pc_window, alpha=alpha, eps=eps)

    res = differential_evolution(fun, bounds=bnds, strategy="best1bin",
                                 popsize=int(popsize), maxiter=int(maxiter),
                                 mutation=(0.5,1.0), recombination=0.7,
                                 tol=1e-6, polish=True, seed=int(seed))

    loc = minimize(fun, res.x, method="Nelder-Mead",
                   options={"xatol":1e-8,"fatol":1e-10,"maxiter":2000})

    xbest = loc.x if loc.fun < res.fun else res.x
    T = float(xbest[0]); rho = float(np.exp(xbest[1]))
    r1,r2,Pc,info = hk_residuals_Trho(eos,T,rho,z,eps=eps)
    return {"T":T,"rho":rho,"Pc":Pc,"r1":r1,"r2":r2,"info":info,
            "res":(res,loc),"method":"DE+NM"}


def crit_at_z_binary_GO_warm(eos, z1, prev=None, alpha=5.0, eps=3e-6):
    z2 = 1.0 - z1
    if (z1<=0.0) or (z2<=0.0): raise ValueError("z must be strictly inside (0,1)")
    z = [z1,z2]
    T_broad = (115.0, 205.0)
    rho_broad = (200.0, 20000.0)
    Pc_broad = (2.5, 5.5)
    if prev is None:
        pcw = Pc_broad
        return minimize_ObjF_DE(eos, z, T_broad, rho_broad,
                                alpha=alpha, eps=eps, popsize=28,
                                maxiter=500, seed=17, pc_window=pcw)
    T0, rho0, Pc0 = prev["T"], prev["rho"], prev["Pc"]
    T_win = _clamp_additive_window(T0, T_broad, halfwidth=10.0)
    r_win = _clamp_multiplicative_window(rho0, rho_broad,
                                         flo=0.80, fhi=1.25, min_span_ratio=1.10)
    Pc_win= _clamp_additive_window(Pc0, Pc_broad, halfwidth=0.25)
    return minimize_ObjF_DE(eos, z, T_win, r_win,
                            alpha=alpha, eps=eps, popsize=18,
                            maxiter=350, seed=17, pc_window=Pc_win)


N_GRID = 81
EPS_Z = 1e-6
eos = tp.pcsaft.pcsaft("C1,N2")
z1_grid = np.linspace(EPS_Z, 1.0 - EPS_Z, N_GRID)

T_BROAD = (115.0, 205.0)
RHO_BROAD = (200.0, 20000.0)
PC_BROAD = (2.5, 5.5)

def solve_DE(z1, prev=None):
    return minimize_ObjF_DE(eos, [z1, 1.0 - z1],
                            T_bounds=T_BROAD, rho_bounds=RHO_BROAD,
                            alpha=5.0, eps=3e-6,
                            popsize=16, maxiter=220, seed=17,
                            pc_window=PC_BROAD)

def solve_local(z1, prev):
    T0, rho0, Pc0 = prev["T"], prev["rho"], prev["Pc"]
    T_win = _clamp_additive_window(T0, T_BROAD, halfwidth=10.0)
    r_win = _clamp_multiplicative_window(rho0, RHO_BROAD,
                                         flo=0.80, fhi=1.25, min_span_ratio=1.10)
    Pc_win = _clamp_additive_window(Pc0, PC_BROAD, halfwidth=0.25)
    x0 = np.array([T0, np.log(rho0)], float)
    bnds = [T_win, (np.log(r_win[0]), np.log(r_win[1]))]

    def fun(x): return hk_objective(x, eos, np.array([z1, 1.0 - z1]),
                                    pc_window=Pc_win, alpha=5.0, eps=3e-6)

    res = minimize(fun, x0, method="Powell", bounds=bnds,
                   options={"xtol":1e-3,"ftol":1e-9,"maxiter":200,"maxfev":4000,"disp":False})

    xb = res.x
    T = float(xb[0]); rho = float(np.exp(xb[1]))
    r1, r2, Pc, info = hk_residuals_Trho(eos, T, rho, [z1, 1.0 - z1], eps=3e-6)
    return {"T": T, "rho": rho, "Pc": Pc, "r1": r1, "r2": r2,
            "info": info, "res": res, "method": "Powell"}

def append_row(rows, z1, out):
    rows.append([z1, 1.0 - z1, out["T"], out["Pc"], out["rho"], out["r1"], out["r2"], out["method"]])


z_anchor = 0.75
out_anchor = solve_DE(z_anchor)
rows = []
append_row(rows, z_anchor, out_anchor)
print(f"Anchor z1={z_anchor:.3f} -> T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa [{out_anchor['method']}]")

for z1 in z1_grid[z1_grid > z_anchor]:
    try:
        out_anchor = solve_local(z1, out_anchor)
        append_row(rows, z1, out_anchor)
        if len(rows) % 10 == 0:
            print(f" .. z1={z1:.3f} T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa")
    except Exception as e:
        try:
            out_anchor = solve_DE(z1)
            append_row(rows, z1, out_anchor)
            print(f" !! fallback-DE z1={z1:.3f} T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa")
        except Exception:
            print(f" xx skipped z1={z1:.3f} ({e})")

out_left = rows[0]
out_left = {"T": out_left[2], "Pc": out_left[3], "rho": out_left[4],
            "r1": out_left[5], "r2": out_left[6], "method": "Powell"}

for z1 in z1_grid[(z1_grid < z_anchor)][::-1]:
    try:
        out_left = solve_local(z1, out_left)
        append_row(rows, z1, out_left)
        if len(rows) % 10 == 0:
            print(f" .. z1={z1:.3f} T={out_left['T']:.2f} K, Pc={out_left['Pc']:.3f} MPa")
    except Exception as e:
        try:
            out_left = solve_DE(z1)
            append_row(rows, z1, out_left)
            print(f" !! fallback-DE z1={z1:.3f} T={out_left['T']:.2f} K, Pc={out_left['Pc']:.3f} MPa")
        except Exception:
            print(f" xx skipped z1={z1:.3f} ({e})")

df_go = pd.DataFrame(rows, columns=["z1","z2","Tc[K]","Pc[MPa]","rho_c[mol/m^3]","r1","r2","solver"])
df_go = df_go[np.isfinite(df_go["Tc[K]"]) & np.isfinite(df_go["Pc[MPa]"])]
df_go = df_go.drop_duplicates(subset=["z1"]).sort_values("z1").reset_index(drop=True)
df_go.to_csv("C1-N2_GO_DE_critical_locus_PT.csv", index=False)
print("Saved: C1-N2_GO_DE_critical_locus_PT.csv")
print(df_go[["Tc[K]","Pc[MPa]"]].describe())

z_anchor = 0.75
out_anchor = solve_DE(z_anchor)
rows = []
append_row(rows, z_anchor, out_anchor)
print(f"Anchor z1={z_anchor:.3f} -> T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa [{out_anchor['method']}]")

for z1 in z1_grid[z1_grid > z_anchor]:
    try:
        out_anchor = solve_local(z1, out_anchor)
        append_row(rows, z1, out_anchor)
        if len(rows) % 10 == 0:
            print(f" .. z1={z1:.3f} T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa")
    except Exception as e:
        try:
            out_anchor = solve_DE(z1)
            append_row(rows, z1, out_anchor)
            print(f" !! fallback-DE z1={z1:.3f} T={out_anchor['T']:.2f} K, Pc={out_anchor['Pc']:.3f} MPa")
        except Exception:
            print(f" xx skipped z1={z1:.3f} ({e})")

out_left = rows[0]
out_left = {"T": out_left[2], "Pc": out_left[3], "rho": out_left[4],
            "r1": out_left[5], "r2": out_left[6], "method": "Powell"}

for z1 in z1_grid[(z1_grid < z_anchor)][::-1]:
    try:
        out_left = solve_local(z1, out_left)
        append_row(rows, z1, out_left)
        if len(rows) % 10 == 0:
            print(f" .. z1={z1:.3f} T={out_left['T']:.2f} K, Pc={out_left['Pc']:.3f} MPa")
    except Exception as e:
        try:
            out_left = solve_DE(z1)
            append_row(rows, z1, out_left)
            print(f" !! fallback-DE z1={z1:.3f} T={out_left['T']:.2f} K, Pc={out_left['Pc']:.3f} MPa")
        except Exception:
            print(f" xx skipped z1={z1:.3f} ({e})")

df_go = pd.DataFrame(rows, columns=["z1","z2","Tc[K]","Pc[MPa]","rho_c[mol/m^3]","r1","r2","solver"])
df_go = df_go[np.isfinite(df_go["Tc[K]"]) & np.isfinite(df_go["Pc[MPa]"])]
df_go = df_go.drop_duplicates(subset=["z1"]).sort_values("z1").reset_index(drop=True)
df_go.to_csv("C1-N2_GO_DE_critical_locus_PT.csv", index=False)
print("Saved: C1-N2_GO_DE_critical_locus_PT.csv")
print(df_go[["Tc[K]","Pc[MPa]"]].describe())

df = df_go.copy()
df = df[np.isfinite(df["Tc[K]"]) & np.isfinite(df["Pc[MPa]"])]
df = df.sort_values("z1").drop_duplicates(subset=["z1"]).reset_index(drop=True)

z = df["z1"].to_numpy()
T = df["Tc[K]"].to_numpy()
P = df["Pc[MPa]"].to_numpy()

exp_keep = exp[(exp["z1"] >= z.min()) & (exp["z1"] <= z.max())].reset_index(drop=True)
Tc_mod = np.interp(exp_keep["z1"], z, T)
Pc_mod = np.interp(exp_keep["z1"], z, P)
rmse_T = float(np.sqrt(np.mean((Tc_mod - exp_keep["Tc_K"])**2)))
rmse_P = float(np.sqrt(np.mean((Pc_mod - exp_keep["Pc_MPa"])**2)))
print(f"RMSE (C1–N2): ΔT = {rmse_T:.2f} K, ΔP = {rmse_P:.3f} MPa")

def choose_sg_window(n, polyorder=3, cap=11):
    maxw = min(cap, n)
    if maxw <= polyorder + 2: return None
    w = maxw if (maxw % 2 == 1) else (maxw - 1)
    if w <= polyorder: w = polyorder + 3
    if w % 2 == 0: w += 1
    if w > maxw: return None
    return w

w = choose_sg_window(len(z), polyorder=3, cap=11)
if w is not None:
    T_s = savgol_filter(T, window_length=w, polyorder=3)
    P_s = savgol_filter(P, window_length=w, polyorder=3)
else:
    T_s, P_s = T, P

idxT = np.argsort(T)
T_ord = T[idxT]; P_ord = P[idxT]
s = 0.002 * len(T_ord) * (np.ptp(P_ord) ** 2)
spl = UnivariateSpline(T_ord, P_ord, k=3, s=s)
T_line = np.linspace(T_ord.min(), T_ord.max(), 1500)
P_line = spl(T_line)

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

plt.close('all')
fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
fig.subplots_adjust(left=0.20, right=0.86, bottom=0.20, top=0.95)
ax.set_axisbelow(True)

ax.plot(T_line, P_line, color="white", lw=6.5, zorder=3,
        solid_capstyle="round", solid_joinstyle="round", clip_on=True, label="_nolegend_")
ax.plot(T_line, P_line, color="#b3b3b3", lw=4.5, zorder=4,
        solid_capstyle="round", solid_joinstyle="round", clip_on=True, label="Global Optimization")

sc = ax.scatter(T, P, c=z, cmap="viridis", vmin=0.0, vmax=1.0, s=55,
                edgecolors="k", linewidths=0.4, zorder=5)

ax.scatter(exp_keep["Tc_K"], exp_keep["Pc_MPa"],
           marker="^", facecolors="none", edgecolors="k", s=110,
           linewidths=2.0, zorder=6, label="Experimental")

ax.set_xlabel("Critical Temperature, K", labelpad=28)
ax.set_ylabel("Critical Pressure, MPa", labelpad=28)

cbar = plt.colorbar(sc, ax=ax, pad=0.03, fraction=0.06)
cbar.set_label(r"CH$_4$ mole fraction", labelpad=28)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
cbar.ax.tick_params(labelsize=24)

ax.grid(True, which="major", color="0.92", linewidth=1.0)
ax.set_xlim(120, 200)
ax.set_ylim(2.5, 6.5)
ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

ax.tick_params(axis="both", which="both", direction="in",
               top=True, right=True, length=6, width=1.2, pad=12)
ax.tick_params(axis="x", which="major", pad=16)
ax.tick_params(axis="both", which="minor", length=3, width=1.0)

for s_ in ("left","bottom","right","top"):
    ax.spines[s_].set_position(("outward", 0))

ax.legend(loc="upper left", frameon=False, fontsize=22,
          handlelength=2.8, bbox_to_anchor=(0.05, 0.95), borderaxespad=0.3)

plt.savefig("C1-N2_GO_overlay_clean.png", dpi=900, bbox_inches="tight")
plt.savefig("C1-N2_GO_overlay_clean.pdf", bbox_inches="tight")
plt.show()

