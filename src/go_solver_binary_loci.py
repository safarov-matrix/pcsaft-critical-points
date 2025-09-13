"""
Global Optimization (GO) via Differential Evolution (DE) for binary mixture critical loci using Thermopack's PC-SAFT EOS.

Notes:
- PC-SAFT EOS is provided by Thermopack; this script implements the optimization workflow.
- All binary interaction parameters (k_ij) are set to zero (predictive baseline).
- Traces (T_c, P_c) along composition z1 ∈ (0,1) using DE + local refinement.

Usage:
> pip install -r requirements.txt
> python src/go_solver_binary_loci.py
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import thermopack as tp

# EOS helpers

def _mu_C_TV(eos, T, V, n):
    mu, C = eos.chemical_potential_tv(T, V, n, None, None, 1, property_flag='IR')
    return np.asarray(mu, float), np.asarray(C, float)

def _pressure_TV(eos, T, V, n):
    try:
        out = eos.pressure_tv(T, V, n)
    except TypeError:
        out = eos.pressure_tv(T, V, n, None, None, 1)
    p = out[0] if isinstance(out,(tuple,list)) else out
    return float(np.asarray(p).reshape(()))/1e6  # MPa

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

# Objective + windowing
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

# DE + local polish

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

# Minimal CLI sanity check
def main():
    eos = tp.pcsaft.pcsaft("C1,N2")
    # (optional) set kij = 0 explicitly (Thermopack default may already be 0)
    eos.set_kij(1, 2, 0.0)

    out = minimize_ObjF_DE(
        eos, [0.5, 0.5],
        T_bounds=(115.0, 205.0),
        rho_bounds=(200.0, 20000.0),
        alpha=5.0, eps=3e-6, popsize=16, maxiter=220, seed=17,
        pc_window=(2.5, 5.5),
    )
    print("[Binary GO] example at z1=0.5:", {k: out[k] for k in ["T","Pc","rho","r1","r2","method"]})

if __name__ == "__main__":
    main()