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



File: `critloci\_hicks1975.csv`



Columns:

\- `mixture`: label (e.g., C1–C2)

\- `comp1`, `comp2`: component identifiers (Thermopack IDs)

\- `z1`: mole fraction of `comp1`

\- `Tc\_K`: critical temperature \[K]

\- `Pc\_MPa`: critical pressure \[MPa]

\- `source`: citation key (Hicks1975)



Source: J. S. Hicks and S. T. Young, 1975.
