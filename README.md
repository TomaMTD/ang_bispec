# `ang_bispec`: Python code for the galaxy number count angular bispectrum

Based on T. Montandon et al., *"Angular bispectrum of matter number counts in cosmic structures"*, [arXiv:2501.05422](https://arxiv.org/abs/2501.05422) (2025).

**Description**
`ang_bispec` computes the angular bispectrum of galaxy number counts $B_{\ell_1\ell_2\ell_3}$ without the Limber approximation. It extends the **Byspectrum** code by including **finite redshift bins**, **redshift-space distortions**, leading **general relativistic projection effects**, **early radiation** corrections, and **primordial non-Gaussianity** (local / equilateral / orthogonal shapes and scale-dependent $f_{\rm NL}$).

---

## Features

- **Complete theoretical model** — first- and second-order number count expressions:
  - Newtonian terms (density, RSD, quadratic terms)
  - Non-integrated projection effects (Doppler, potentials, lensing)
  - Radiative and dynamical GR effects
  - Primordial shapes and scale-dependent bias from $f_{\rm NL}^{\rm local}$

- **Efficient hypergeometric evaluation** — the Mathematica expressions of Assassi et al. (2017)
  translated into numba-accelerated Python for the ₂F₁ function.

- **Restartable** — every expensive stage is cached to disk and skipped on re-run unless `--force`.

---

## Installation

**Prerequisites**

- Python ≥ 3.8, `numpy`, `scipy`, `h5py`, `numba`, `art`
- [CLASS](https://github.com/lesgourg/class_public) with its Python wrapper `classy`

```bash
git clone https://github.com/TomaMTD/ang_bispec.git
cd ang_bispec
```

Then edit `path.py` so that `path` is the absolute path of the directory *containing* the clone
(the code imports from `path.path + '/ang_bispec/source'`):

```python
path = "/home/<you>/software/"
```

---

## Quick start

The full pipeline for one Euclid bin (this is `run.sh`):

```bash
# 1. angular power spectra C_l(chi) -- the expensive stage
python byspectrum.py -m cl -w all -l all       -r 1 -N 1 -f 0

# 2. bispectra for every configuration
python byspectrum.py -m bl -w all -l all       -r 1 -N 1 -f 0
python byspectrum.py -m bl -w all -l noproj    -r 1 -N 1 -f 0
python byspectrum.py -m bl -w all -l all       -r 0 -N 0 -fnl_local 1 -f 0

# 3. primordial shapes (local, equilateral, orthogonal)
python byspectrum.py -m cl -w primordial -l all -r 0 -N 0 -f 0
python byspectrum.py -m bl -w primordial -l all -r 0 -N 0 -f 0
```

Step 1 is optional in principle — `-m bl` computes any missing $C_\ell$ on the fly — but running
it first keeps the two stages separate and makes restarts cheaper.

`-r 1 -N 1` is **not** a physical configuration; it is shorthand for *"run all three cases"*:
GR `(rad=0, Newton=0)`, radiation `(1, 0)` and Newtonian `(0, 1)`, in that order, in a single
invocation. It works in both `cl` and `bl` modes.

---

## Command-line options

| Flag | Default (`param.py`) | Meaning |
|---|---|---|
| `-m`, `--mode` | `bl` | `cl` → angular power spectra; `merge` → fold `Cls_fallback/*.npy` back into `Cls.h5`; anything else → bispectra |
| `-w`, `--which` | `all` | Which term(s) to compute — see below |
| `-l`, `--lterm` | `all` | Which linear terms to sum — see below |
| `-N`, `--Newton` | `0` | Newtonian dynamics only |
| `-r`, `--rad` | `0` | Early-radiation contribution |
| `-f`, `--force` | — | `1` = recompute and overwrite, `0` = skip anything already in the output |
| `-config` | `esf` | Triangle configuration(s): `equi`, `squ`, `squ2`, `folded`, `es`, `esf`, `all` |
| `-fnl_local` | `0` | Amplitude of local primordial non-Gaussianity |
| `-ell`, `-ellmax`, `-Nell` | `4`, `1024`, `32` | Multipole grid (log-spaced; rounded to even $\ell$ for `-config equi/squ/squ2/folded/esf`) |
| `-Nchi` | `501` | Points on the radial grid (must stay uniform) |
| `-o`, `--output_dir` | `output_euclid_bin4/` | Output directory |
| `-window_type` | `euclid` | `euclid`, `ska` or `nbody` |
| `-euclid_bin_idx` | `4` | Euclid photometric bin, 0–9 |
| `-z0`, `-dz`, `-sigma_z` | — | Redshift bin centre / half-width / smoothing (for `ska` and `nbody`) |
| `-h100`, `-omega_b`, `-omega_cdm`, `-A_s`, `-n_s`, … | Planck 2018 | Cosmological parameters |

### `-w` (which)

| Mode | Value | Expands to |
|---|---|---|
| `cl` | `all` | `FG2, d2v, d1v, d3v, d1d, F2, G2, dv2` (only `F2, G2, dv2` when `-r 1`) |
| `cl` | `cl` | the **linear** angular power spectrum $C_\ell$ → `Cl_<lterm>.h5` |
| `bl` | `all` | the 15 second-order terms `F2, G2, d2vd2v, d1vd3v, d1vd1d, d0dd0d, dv2, d2vd0d, d1vd2v, d1vd0d, d1vdod, davd1v, d0pd3v, d0pd1d, d1vd2p` |
| both | `primordial` | the three primordial shapes (`local`, `equi`, `ortho` blocks → `local`, `equilateral`, `orthogonal`) |
| both | any single name | just that term |

### `-l` (lterm)

`density`, `rsd`, `doppler`, `pot`, `dpot`, `pot_gr`, `pot_fnl`, `lensing`, plus:

- `all` — every term above (`pot_gr` is GR-only, so it is dropped when `-N 1`)
- `noproj` — `density, rsd, pot_gr, pot_fnl, lensing` (no projection terms; `pot_gr` dropped when `-N 1`)
- `a+b+c` — an explicit sum, e.g. `-l density+rsd`

The lterm string also names the output file: `-l all` → `bl_all.h5`, `-l noproj` → `bl_noproj.h5`.

---

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| 1 | `source/lincosmo.py` | Growth functions $D, f, v, w$; calls CLASS for transfer functions and $P(k)$ (cached in `class_transfer.npy`, `time_dict.npy`) |
| 2 | `source/fctr.py` | Radial kernels $f_{n,m}(r)$: window derivatives, bias, growth factors (cached in `window_derivs_cache.npz`) |
| 3 | `source/fftlog.py` | FFTLog decomposition of $P(k)$ / transfer functions into power laws → the $c_p$ coefficients |
| 4 | `source/mathematica.py` | numba ₂F₁ evaluation and the per-$\ell$ `t` grid |
| 5 | `source/general_ps.py` | The generalised spectra $C_\ell(\chi)$ — **the bottleneck** — written to `Cls.h5` |
| 6 | `source/bispectrum.py` | Assembles $C_\ell(\chi)$ into $B_{\ell_1\ell_2\ell_3}$ over all valid triplets (numba, parallel) |
| 7 | `source/binning.py` | Post-processing: bin the bispectrum in $\ell$ |

`byspectrum.py` is the entry point and holds the CLI, the cosmology setup and the mode dispatch.

---

## Output files

Everything lands in `output_dir` (default `output_euclid_bin4/`).

**`Cls.h5`** — the generalised power spectra, one dataset per $\ell$:

```
n_<n>_m_<m>/            # e.g. n_0_m_0, n_-1_m_1  (FG2, d1v, d2v, d3v, d1d)
    <lterm>/ell_<L>     # (n_chi,)
    chi_list
primordial_n_<n>_m_0/   # the alpha/beta legs of the primordial shapes (n = 0, 1/3, 2/3, 1)
    <lterm>/ell_<L>
F2/                     # second-order multipoles
    fm2, fm4            # Am terms  (GR)      -- F2 has no Newtonian counterpart
    fm2_rad, fm4_rad    # Il terms  (radiation)
G2/  dv2/
    f0, fm2             # Am terms  (GR)
    f0_newton           # Am terms  (Newtonian)
    fm2_rad, fm4_rad    # Il terms  (radiation)
```

**`bl_<lterm>[_rad|_newton][_fnl<X>].h5`** — the bispectra, grouped by configuration:

```
equilateral/            # squeezed_ell4/, squeezed2_ell1024/, folded_ell1024/, all/
    bl_<which>          # (n_triplets,)  reduced bispectrum
    ell | ell1, ell23   # the varying multipole(s)
    wigner              # the 3j symbol for each triplet
    variance            # only for -config all, and only if Cl_<lterm>.h5 exists
```

The Wigner symbol is stored but **not** applied: to plot $B_{\ell_1\ell_2\ell_3}$ multiply by
$\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}\,\mathrm{wigner}^2$.

Also written: `Cl_<lterm>.h5` (linear spectra from `-w cl`), `fctr_of_r_<which>.npy`,
`cp_<which>.npy`, `time_dict.npy`, `param_used.py`, and the `triplets_cache_*.npz` /
`window_derivs_cache.npz` caches.

---

## Restarting, caching and `--force`

- `-f 0` skips any $(n,m)$ / lterm / multipole already present in `Cls.h5` **for every requested
  $\ell$**; adding a new $\ell$ re-triggers that block. `-f 1` recomputes and overwrites.
- Triangle triplets, Wigner symbols and window derivatives are cached and reused across runs;
  the caches are validated against the current grid before being trusted.
- HDF5 writes retry on lock contention, so several jobs may share one output directory. If a
  write still fails it falls back to `Cls_fallback/*.npy`; run `-m merge` afterwards to fold
  those back into `Cls.h5`.

> **Gotcha:** `param_used.py` is written into the output directory on the *first* run only, and
> every module imports it (`from param_used import *`). Editing `param.py` afterwards will **not**
> affect an existing output directory — delete `param_used.py` or use a fresh `-o` directory.

---

## Configuration

`source/param.py` holds the defaults for every CLI flag plus the settings that have no flag:
`rmin_global` (inner edge of the radial grid), `ell_spacing`, `bins`, the window type and its
redshift bin, and the Planck 2018 cosmology. Anything passed on the command line overrides it.
