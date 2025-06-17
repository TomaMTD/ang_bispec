# 🌀 `ang_bispec`: Python code for galaxy number count angular bispectrum

Based on T. Montandon, et al., “Full angular bispectrum of galaxy number counts: Newtonian and GR terms without Limber,” arXiv:2501.05422 (2025).

**Description**  
`ang_bispec` is a high-performance Python implementation for computing the angular bispectrum of galaxy number counts. It significantly extends the existing **Byspectrum** code by incorporating **finite redshift bins**, **redshift-space distortions (RSD)**, and leading **general relativistic (GR) projection** and **radiation effects**, all without relying on the Limber approximation.

---

## Features

- **Complete theoretical model**  
  Based on first- and second-order number count expressions from the literature, covering:
  - Newtonian terms (density, RSD, quadratic terms)
  - Non‑integrated projection effects  
  - Radiative and dynamical GR effects  

- **Efficient hypergeometric evaluation**  
  Translated the Mathematica expressions from Assassi et al. (2017) into numba‑accelerated Python for fast computation of the ₂F₁ hypergeometric function.

- **Pipeline overview**  
  1. **Linear cosmology module**  
     - Solves growth functions **D(z), f(z), v(z), w(z)**  
     - Interfaces with **CLASS** to extract potential transfer functions and power spectra  
  2. **FFTLog module**  
     - Computes Hankel transforms of the potential power spectrum and transfer function  
  3. **Generalized power spectra \(C_ℓ(χ)\)**  
     - Precomputes 7 spectra across ℓ and χ values—this is the main bottleneck  
  4. **Precomputation of radial integrals**  
     - Integrals over \(r_1\) for all \(f_{n,m}\) coefficients  
  5. **Main bispectrum loop**  
     - Computes contributions for all (ℓ₁, ℓ₂, ℓ₃) triplets, including:
       - Density, RSD, projection, quadratic, and cross terms  

---

### Prerequisites

- Python ≥3.8  
- [numba](https://numba.pydata.org/) for JIT acceleration  
- [CLASS](https://github.com/lesgourg/class_public) installed and accessible  

### Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/TomaMTD/ang_bispec.git
cd ang_bispec
