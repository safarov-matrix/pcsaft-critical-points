This repository accompanies our manuscript on mixture critical points calculation using the Perturbed-Chain Statistical Associating Fluid Theory (PC-SAFT) Equation of State (EOS) via two distinct computational methods, namely the root-finding and optimization formulations. It specifically focuses on the computational methods and their performance comparison when combined with PC-SAFT EOS.

The repository provides reproducible code to calculate:

1. Pure-compound and multicomponent critical points
2. Binary critical loci

Two computational strategies are implemented:

1. Newton–Raphson (NR) formulation of the Heidemann-Khalil (HK) criticality conditions based on PC-SAFT EOS
2. Global Optimization (GO) formulation solved with Differential Evolution (DE) based on PC-SAFT EOS

In this study, 11 pure compounds (hydrocarbon and hon-hydrocarbon) and 44 mixtures (from 2 components to 11 components) are tested. Additionally, 6 binary mixtures from a different compilations are used to build composition-dependent critical loci using both computational methods.

Note: This project implements the readily available PC-SAFT EOS from [Thermopack](https://github.com/thermotools/thermopack).  

Sources:
[1] SINTEF Energy Research, NTNU, ThermoPack [software], GitHub repository.https://github.com/thermotools/thermopack, accessed 8 December 2024.
[2] Ø. Wilhelmsen, A. Aasen, G. Skaugen, P. Aursand, A. Austegard, E. Aursand, M.Aa. Gjennestad, H. Lund, G. Linga, M. Hammer, Thermodynamic modeling with equations of state: Present challenges with established methods, Ind. Eng. Chem. Res. 56 (2017) 3503–3515. https://doi.org/10.1021/acs.iecr.7b00317.
[3] P. Aursand, M.Aa. Gjennestad, E. Aursand, M. Hammer, Ø. Wilhelmsen, The spinodal of single- and multi-component fluids and its role in the development of modern equations of state, Fluid Phase Equilib. 436 (2017) 98–112. https://doi.org/10.1016/j.fluid.2016.12.018.
[4] A. Aasen, M. Hammer, Å. Ervik, E.A. Müller, Ø. Wilhelmsen, Equation of state and force fields for Feynman–Hibbs-corrected Mie fluids. I. Application to pure helium, neon, hydrogen, and deuterium, J. Chem. Phys. 151 (2019) 035103. https://doi.org/10.1063/1.5111364.
[5] T. van Westen, M. Hammer, B. Hafskjold, A. Aasen, J. Gross, Ø. Wilhelmsen, Perturbation theories for fluids with short-ranged attractive forces: A case study of the Lennard-Jones spline fluid, J. Chem. Phys. 156 (2022) 104504. https://doi.org/10.1063/5.0082690.

Pure component PC-SAFT parameters and critical data:

File: `data/pure_components_pcsaft_params.csv`

Columns:

- `Component`: identifier (Thermopack ID, e.g., C1, C2, nC4, CO2, N2)
- `m`: segment number (dimensionless)
- `sigma_A`: segment diameter [Å]
- `eps_k`: segment energy parameter [K]
- `omega`: acentric factor (dimensionless)
- `Tc_exp`: experimental critical temperature [K]
- `Pc_exp`: experimental critical pressure [MPa]
- `Source`: reference for the EOS parameters and critical data

Sources:  
- J. Gross, G. Sadowski, Ind. Eng. Chem. Res. 40 (2001) 1244–1260.  
- D. Bücker, W. Wagner, J. Phys. Chem. Ref. Data 35 (2006) 205–266, 929–1019.  
- E.W. Lemmon, M.O. McLinden, W. Wagner, J. Chem. Eng. Data 54 (2009) 3141–3180.  
- D. Ambrose, J. Walton, Pure Appl. Chem. 61 (1989) 1395–1403.  
- R. Span, W. Wagner, J. Phys. Chem. Ref. Data 25 (1996) 1509–1596.  
- R. Span et al., J. Phys. Chem. Ref. Data 29 (2000) 1361–1433.  
- U. Setzmann, W. Wagner, J. Phys. Chem. Ref. Data 20 (1991) 1061–1155.  
- J. Shi et al., AIChE J. 70 (2024) e18466.  

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
