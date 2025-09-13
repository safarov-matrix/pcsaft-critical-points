This repository accompanies our manuscript on mixture critical points with the Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT) Equation of State (EOS).  It focuses on the computational methods and their performance comparison.

The repository provides reproducible code to calculate:

* Pure compound and multicomponent critical points
* Binary critical loci

Two computational strategies are implemented:

* Newton–Raphson (NR) formulation of the Heidemann-Khalil (HK) criticality conditions based on PC-SAFT EOS
* Global Optimization (GO) formulation solved with Differential Evolution (DE) based on PC-SAFT EOS

Note: This project uses the readily available PC-SAFT EOS implementation from [Thermopack](https://github.com/thermotools/thermopack).  

Multicomponent mixtures (Dimitrakopoulos et al., 2014)

File: `data/mixtures\_dim-wen-2014.csv`  

Columns:

- `Mixture_No`: index (1–44, as in Table S1 of your manuscript)  

- `CO2`, `N2`, `C1`, `C2`, `C3`, `iC4`, `nC4`, `iC5`, `nC5`, `nC6`, `nC7`: mole fractions of each component (dimensionless).  

- Columns not used in a given mixture are left blank (or zero).  

Source: P. Dimitrakopoulos, W. Jia, C. Li, An improved computational method for the calculation of mixture liquid–vapor critical points, Int. J. Thermophys. 35 (2014) 865–889. https://doi.org/10.1007/s10765-014-1680-7.

Binary critical loci (Hicks & Young, 1975):

Columns:

- `Mixture`: label (e.g., C1–C2)

- `Comp_1`, `Comp_2`: component identifiers (Thermopack IDs)

- `z1`: mole fraction of `Comp\_1`

- `Tc_K`: critical temperature \\\[K]

- `Pc_MPa`: critical pressure \\\[MPa]

Source: C.P. Hicks, C.L. Young, Gas–liquid critical properties of binary mixtures, Chem. Rev. 75 (1975) 119–175. https://doi.org/10.1021/cr60294a001.
