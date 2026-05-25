# Subcortical Defense and Collapse: Nonlinear Atrophy Dynamics along a Dementia Continuum

## Overview

This repository contains the full analysis pipeline for the paper:

> **Kakuda N.** *Subcortical Defense and Collapse: Nonlinear Atrophy Dynamics along a Dementia Continuum.* (2025)

Cross-sectional T1-weighted MRI (n = 483) and neuropsychological scores (MMSE, FAB, RCPM) are used to reconstruct a continuous disease progression axis (pseudotime) and model how atrophy co-evolves across 15 brain networks from cognitive normality to advanced dementia. A key finding is a temporally ordered subcortical sequence in which VTA atrophy shifts from suppression to acceleration of network-wide atrophy propagation (t* = 0.460), followed by the thalamus (t* = 0.483), with both transitions converging near MMSE ≈ 20.

---

## Repository Structure

```
project_config.py                              # Central path configuration (edit before running)

── Project 1: Preprocessing ──────────────────────────────────────────────
batch_masking.sh                               # Batch brain masking in MNI space (FSL)
fsl_anat_parallel.sh                           # Parallel fsl_anat execution (GNU parallel)
fsl_anat_parallel_seasons.sh                   # Runs fsl_anat_parallel for each season
project_1_preprocessing_1.py                   # DICOM → NIfTI conversion (dcm2niix)
project_1_preprocessing_2.py                   # NIfTI → .npy + Z-score standardization
project_1_preprocessing_3.py                   # Concatenate MRI arrays with psychological scores → .npz

── Project 2: Supervised VAE (sVAE) ──────────────────────────────────────
project_2_sVAE_1_encode_.py                    # 3D-CNN-sVAE training & grid search; saves best z-matrix
project_2_sVAE_2_PHATE_.py                     # PHATE embedding; defines pseudotime axis
project_2_sVAE_3_decode_.py                    # Generative reconstruction along PHATE1 axis; Jacobian computation

── Project 3: Atrophy Dynamics ───────────────────────────────────────────
project_3_atrophy_dynamics_1_parcellation_.py  # Voxel-wise atrophy energy → 15-network parcellation (AAL3)
project_3_atrophy_dynamics_2_SNR_.py           # tSNR analysis; network selection (Extended Data Fig. 1)

── Project 4: Dynamical Model ────────────────────────────────────────────
project_4_dynamical_model_1.py                 # Time-varying linear dynamical model; PEB/BMR/BMA inference
```

---

## Pipeline Summary

```
MRI (DICOM)
    │
    ▼  project_1_preprocessing_1.py + fsl_anat_parallel.sh + batch_masking.sh
MNI-registered, brain-masked NIfTI (.nii.gz)
    │
    ▼  project_1_preprocessing_2.py
Z-score normalized .npy files (91×109×91)
    │
    ▼  project_1_preprocessing_3.py
all_mri_psyc_dataset_final.npz  ← MRI + MMSE/FAB/RCPM + metadata
    │
    ▼  project_2_sVAE_1_encode_.py
64-dimensional latent variables z  (grid_results_npz/z_matrix_fold1_epoch*.npz)
    │
    ▼  project_2_sVAE_2_PHATE_.py
Pseudotime axis (normalized PHATE1 rank, 0.0–1.0)
    │
    ▼  project_2_sVAE_3_decode_.py
Reconstructed MRI series along pseudotime → Jacobian (atrophy energy) arrays
    │
    ▼  project_3_atrophy_dynamics_1_parcellation_.py
Network-level atrophy energy (15 networks × 483 pseudotime points)
    │
    ▼  project_3_atrophy_dynamics_2_SNR_.py
tSNR-validated 15-network set (5 low-SNR networks excluded)
    │
    ▼  project_4_dynamical_model_1.py
A / B1 / B2 matrices + PEB/BMR/BMA → Figures 4, 7
```

---

## Environment

| Item | Specification |
|---|---|
| OS | Ubuntu 24 (WSL2) |
| Python | 3.10+ |
| GPU | CUDA-compatible (TensorFlow GPU) |
| MRI tool | FSL (fsl_anat, applywarp, fslmaths) |
| DICOM conversion | dcm2niix |
| Parallelization | GNU parallel |

### Key Python packages

```
tensorflow >= 2.12
nibabel
numpy / pandas / scipy / scikit-learn
matplotlib / seaborn
phate
statsmodels
pydicom
```

---

## Configuration

All file paths are centralized in **`project_config.py`**. Edit this file before running any script:

```python
BASE_DIR      = "/mnt/e/T1_2024"          # Root directory for processed data
JACOBIAN_DIR  = "/mnt/e/T1_2024/jacobian" # Jacobian (atrophy energy) arrays
FIGURE_DIR    = "/mnt/e/T1_2024"          # Output directory for figures

DATASET_NPZ   = f"{BASE_DIR}/all_mri_psyc_dataset_final.npz"
GRID_RESULTS  = f"{BASE_DIR}/grid_results_npz"

AAL_LABEL_TXT = "/mnt/c/Users/.../AAL3/AAL3v1.nii.txt"
AAL_LABEL_NII = "/mnt/c/Users/.../AAL3/AAL3v1.nii"
```

---

## Data

- **Participants:** 483 subjects (after exclusion of 69); memory clinic patients, April 2024 – March 2025
- **MRI:** 1.5T T1-weighted MPRAGE (Siemens Magnetom Symphony Tim); voxel size 0.9×0.9×1.25 mm; registered to MNI152 2mm (91×109×91)
- **Scores:** MMSE (mean 20.5 ± 4.8), FAB (mean 9.9 ± 3.1), RCPM (mean 22.8 ± 5.8)
- **Atlas:** AAL3 (Rolls et al. 2020), organized into 15 functional networks based on Yeo et al. (2011)

> Raw MRI data are not distributed due to patient privacy. The preprocessed `.npz` dataset may be shared upon reasonable request.

---

## Execution Order

```bash
# 1. Convert DICOM to NIfTI
python project_1_preprocessing_1.py

# 2. Run FSL anatomical processing (per season, in parallel)
bash fsl_anat_parallel_seasons.sh

# 3. Brain masking in MNI space
bash batch_masking.sh

# 4. NIfTI → Z-score .npy
python project_1_preprocessing_2.py

# 5. Concatenate MRI + psychological scores
python project_1_preprocessing_3.py

# 6. Train sVAE, extract latent z
python project_2_sVAE_1_encode_.py

# 7. PHATE embedding → pseudotime
python project_2_sVAE_2_PHATE_.py

# 8. Generative reconstruction → Jacobian
python project_2_sVAE_3_decode_.py

# 9. Parcellate atrophy energy into networks
python project_3_atrophy_dynamics_1_parcellation_.py

# 10. SNR analysis / network selection
python project_3_atrophy_dynamics_2_SNR_.py

# 11. Time-varying dynamical model + PEB/BMA inference
python project_4_dynamical_model_1.py
```

---

## Output Files

| File | Description |
|---|---|
| `all_mri_psyc_dataset_final.npz` | Preprocessed MRI + clinical scores (N×91×109×91 + metadata) |
| `grid_results_npz/z_matrix_fold1_epoch*.npz` | Latent z-matrix from sVAE best fold |
| `jacobian/jacobian0.npy` | Raw reconstructed MRI along pseudotime (N_t × voxels) |
| `jacobian/jacobian1.npy` | First derivative (Savitzky-Golay, window=21, poly=2) |
| `jacobian/jacobian2.npy` | Squared first derivative (atrophy energy) |
| `jacobian/network_mask_1d.npy` | Voxel-to-network assignment (AAL3 → 15/20 networks) |
| `Figure2.pdf / .tif` | PHATE embedding & pseudotime projection |
| `Figure3.pdf / .tif` | Pseudotime vs MMSE/FAB/RCPM |
| `Figure4.pdf / .tif` | B1 / B2 coupling matrices (BMA) |
| `Figure7.pdf / .tif` | A / B1 / B2 matrices combined |
| `FigureS1.pdf / .tif` | tSNR across 20 candidate networks |

---

## Methods Summary

### 3D Supervised VAE (sVAE)
A 3D-CNN encoder–decoder with a regression head predicts MMSE/FAB/RCPM from latent space. Training loss:

```
L_total = L_Rec + β·L_KL + λ1·L_Sup + λ2·L_Corr + λ3·L_Cov
```

- `L_Rec`: voxel-wise MSE reconstruction loss  
- `L_KL`: KL divergence with beta-annealing  
- `L_Sup`: Elastic Net regression loss on cognitive scores  
- `L_Corr`: linear correlation maximization  
- `L_Cov`: latent redundancy penalty  
- Latent dimension: **z = 64**

### PHATE Pseudotime
Parameters: `n_components=2, knn=30, t=20`. PHATE1 rank-normalized to [0, 1] as the pseudotime index *t*.

### Atrophy Energy
Voxel-wise first derivative along pseudotime computed with a Savitzky-Golay filter (window=21, poly=2). Squared derivative = "atrophy energy."

### Time-Varying Linear Dynamical Model
```
dz/dt = (A + B1·t + B2·t²) · z
```
Transition point: `t* = −B1 / (2·B2)`

### Statistical Inference
Parametric Empirical Bayes (PEB) + Bayesian Model Reduction (BMR) + Bayesian Model Averaging (BMA), following Friston et al. (2016). Edges reported at Pp > 0.75 (*) and Pp > 0.90 (**).

---

## Key Results

| Structure | Transition point t* | Role shift |
|---|---|---|
| VTA | 0.460 | Suppression → Acceleration (earlier) |
| Thalamus | 0.483 | Suppression → Acceleration |

Both transitions converge near **MMSE ≈ 20**, the clinical threshold for moderate dementia and the eligibility boundary of recently approved disease-modifying therapies (lecanemab MMSE ≥ 22; donanemab MMSE ≥ 20).

---

## License

For academic and research use. Raw patient data are not included. Please contact the author for data sharing inquiries.

## Contact

**Naoyuki Kakuda, MD**  
Department of Neurology, Higashiyamato Hospital, Tokyo, Japan  
n-kakuda7@kxf.biglobe.ne.jp
