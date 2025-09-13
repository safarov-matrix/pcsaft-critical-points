"""
nr_solver_binary_loci_simple.py
--------------------------------
Binary critical locus via Thermopack's built-in NR critical finder.

What it does
------------
- For a binary mixture at a grid of overall compositions z = (z1, 1 - z1),
  call: Tc, Vc, pc = eos.critical(n) with n = [z1, 1-z1].
- Convert to outputs:
    Tc [K], Pc [MPa], rho_c [mol/m^3] with rho_c = (sum n) / Vc (sum n = 1).

Notes
-----
- PC-SAFT EOS is provided by Thermopack; we do NOT implement the EOS here.
- This module contains NO plotting and NO file I/O; it's meant to be imported.
- If you want kij = 0 (predictive baseline), set it on your eos instance
  BEFORE calling nr_locus_via_eos_critical (see example script).

API
---
nr_locus_via_eos_critical(eos, z1_grid)
    -> returns a dict with a pandas DataFrame 'df' and counters.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def nr_locus_via_eos_critical(eos, z1_grid):
    """
    Compute a binary critical locus by looping over compositions and calling
    Thermopack's eos.critical(n).

    Parameters
    ----------
    eos : Thermopack EOS object
        e.g., eos = tp.pcsaft.pcsaft("C1,N2")
    z1_grid : array-like
        Compositions z1 in (0,1). z2 = 1 - z1 will be used.

    Returns
    -------
    out : dict
        {
          "df": DataFrame with columns
                ["z1","z2","Tc[K]","Pc[MPa]","rho_c[mol/m^3]","solver"],
          "points": int,   # number of successful points
          "skipped": int   # number of exceptions/failed states
        }
    """
    rows = []
    skipped = 0

    for z1 in np.asarray(z1_grid, float):
        n = np.array([z1, 1.0 - z1], dtype=float)
        try:
            Tc, Vc, pc = eos.critical(n)            # Tc [K], Vc [m^3/mol], pc [Pa]
            rho_c  = float(np.sum(n)) / float(Vc)    # mol/m^3 (sum(n) = 1)
            Pc_MPa = float(pc) / 1e6                 # MPa
            rows.append([float(z1), 1.0 - float(z1), float(Tc), Pc_MPa, rho_c, "NR"])
        except Exception:
            skipped += 1
            continue

    df = pd.DataFrame(rows, columns=["z1","z2","Tc[K]","Pc[MPa]","rho_c[mol/m^3]","solver"])
    df = df[np.isfinite(df["Tc[K]"]) & np.isfinite(df["Pc[MPa]"])].copy()
    df = df.drop_duplicates(subset=["z1"]).sort_values("z1").reset_index(drop=True)

    return {"df": df, "points": len(df), "skipped": skipped}