This repository accompanies our manuscript on mixture critical points with the Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT) Equation of State (EOS).  
It provides reproducible code to calculate:
- Pure compound and multicomponent critical points
- Binary critical loci

Two computational strategies are implemented:
- Newton–Raphson (NR) formulation of the Heidemann-Khalil (HK) criticality conditions based on PC-SAFT EOS 
- Global Optimization (GO) formulation solved with Differential Evolution (DE) based on PC-SAFT EOS

Note: This project uses the readily available PC-SAFT EOS implementation from [Thermopack](https://github.com/thermotools/thermopack).  
This repository focuses on the computational methods (NR and GO/DE) and their performance comparison.


\# Binary critical loci (Hicks \& Young, 1975)


Columns:

\- `Mixture`: label (e.g., C1–C2)

\- `Comp_1`, `Comp_2`: component identifiers (Thermopack IDs)

\- `z1`: mole fraction of `Comp_1`

\- `Tc_K`: critical temperature \[K]

\- `Pc_MPa`: critical pressure \[MPa]


Source: J. S. Hicks and S. T. Young, 1975
