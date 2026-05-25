#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth set.")
    except RuntimeError as e:
        print(e)


# In[2]:


import os
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import swish
from tensorflow.image import ssim
from datetime import datetime

print("finished preparing")


# In[ ]:


# Check the number of voxels and the Signal-to-Noise Ratio (SNR) in each network


# In[33]:


import numpy as np
import os
from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR

# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR
jacobian_files = ["jacobian0.npy", "jacobian1.npy", "jacobian2.npy"]

for file in jacobian_files:
    file_path = os.path.join(save_dir, file)

    data = np.load(file_path)
    print(f"{file}: {data.shape}")


# In[34]:


import numpy as np
import os

# 設定
# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR
eps = 1e-10

# 1. load
print("Loading files...")
raw = np.load(os.path.join(save_dir, "jacobian0.npy"))
first_deriv = np.load(os.path.join(save_dir, "jacobian1.npy"))

# 2. Signal and Noise
print("Calculating tSNR...")
signal = np.mean(raw, axis=0)
noise_rms = np.sqrt(np.mean(np.square(first_deriv), axis=0))

# 3. tSNR
tsnr_raw = signal / (noise_rms + eps)
noise_sq = np.mean(np.square(first_deriv), axis=0)
tsnr_sq = signal / (noise_sq + eps)

# 4. save
print("Saving results...")
path_raw = os.path.join(save_dir, "jacobian0_tSNR.npy")
path_sq = os.path.join(save_dir, "jacobian1sq_tSNR.npy")

np.save(path_raw, tsnr_raw)
np.save(path_sq, tsnr_sq)

# 5. shape of files
print("-" * 30)
print(f"DONE!")
print(f"jacobian0_tSNR.npy shape: {tsnr_raw.shape}")
print(f"jacobian1sq_tSNR.npy shape: {tsnr_sq.shape}")
print(f"Saved directory: {save_dir}")


# In[35]:


import numpy as np
import os
import gc

# --- 1. setting ---
# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR
raw_path = os.path.join(save_dir, "jacobian0.npy")      # Raw data
deriv_path = os.path.join(save_dir, "jacobian1.npy")    # 1st derivative
mask_path = os.path.join(save_dir, "network_mask_1d.npy")
eps = 1e-10

# --- 2. mask ---
#network_mask_1d = np.load(mask_path)
#net_ids = range(1, 21) # Network 1-20

def calculate_tsnr_with_stats(file_path, is_derivative=False):
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")
    data = np.load(file_path, mmap_mode='r')

    if is_derivative:
        print("Step 1: Squaring 1st derivative values...")
        working_data = np.square(data)
    else:
        working_data = data

    print("Step 2: Calculating voxel-wise temporal Mean and Std...")
    v_mean = np.mean(working_data, axis=0)
    v_std = np.std(working_data, axis=0)

    v_tsnr = v_mean / (v_std + eps)

    roi_stats = []
    print("Step 3: Aggregating ROI statistics...")
    for net_id in net_ids:
        mask = (network_mask_1d == net_id)
        n_voxels = np.sum(mask)

        if n_voxels > 0:
            tsnrs_in_roi = v_tsnr[mask]

            mean_val = np.mean(tsnrs_in_roi)      
            std_val = np.std(tsnrs_in_roi)
            se_val = std_val / np.sqrt(n_voxels) 

            roi_stats.append([mean_val, se_val, n_voxels])
        else:
            roi_stats.append([0.0, 0.0, 0.0])

    return np.array(roi_stats), v_tsnr

# --- 3. 実行 ---

# left axis: Raw tSNR
raw_results, raw_tsnr_map = calculate_tsnr_with_stats(raw_path, is_derivative=False)

# rigth axis: 1st Deriv Sq tSNR
sq_results, sq_tsnr_map = calculate_tsnr_with_stats(deriv_path, is_derivative=True)

# --- 4. results ---
np.save(os.path.join(save_dir, "results_raw_stats.npy"), raw_results)
np.save(os.path.join(save_dir, "results_sq_stats.npy"), sq_results)

np.save(os.path.join(save_dir, "jacobian0_tSNR.npy"), raw_tsnr_map)
np.save(os.path.join(save_dir, "jacobian1sq_tSNR.npy"), sq_tsnr_map)

print("\n--- Final Summary (Network 20) ---")
lc_raw = raw_results[18]
lc_sq = sq_results[18]
print(f"LC (voxels: {int(lc_raw[2])})")
print(f"  Raw tSNR: {lc_raw[0]:.4f} ± {lc_raw[1]:.4f} (SE)")
print(f"  Sq tSNR : {lc_sq[0]:.4f} ± {lc_sq[1]:.4f} (SE)")

print("\nAll processing completed successfully.")


# In[36]:


import numpy as np
import os
import gc

# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR
jacobian_files = ["jacobian0_tSNR.npy", "jacobian1sq_tSNR.npy"]

net_ids = range(1, 21) 
n_timepoints = 483
n_rows = 21

if 'network_mask_1d' not in locals():
    raise NameError("network_mask_1d is not found!")

voxel_counts = np.zeros(n_rows, dtype=np.float32)
for net_idx, net_id in enumerate(net_ids):
    voxel_counts[net_idx] = np.sum(network_mask_1d == net_id)

voxel_counts[20] = np.sum(voxel_counts[0:20])
counts_reshaped = voxel_counts[:, np.newaxis] # ブロードキャスト用 (20, 1)

print(f"Voxel counts check: Total voxels in networks = {int(voxel_counts[19])}")

for filename in jacobian_files:
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        print(f"Skip: {filename} not found.")
        continue

    print(f"\n--- Processing {filename} (mmap mode) ---")
    jac = np.load(file_path, mmap_mode='r')

    net_means = np.zeros(n_rows, dtype=np.float32)

    for net_idx, net_id in enumerate(net_ids):
        mask_indices = (network_mask_1d == net_id)
        if np.any(mask_indices):
            net_means[net_idx] = np.mean(jac[mask_indices])
        else:
            net_means[net_idx] = 0.0

        print(f"  Network {net_id:02d}: {net_means[net_idx]:.6f} (voxels: {int(voxel_counts[net_idx])})")

    all_net_mask = (network_mask_1d >= 1) & (network_mask_1d <= 20)
    if np.any(all_net_mask):
        net_means[20] = np.mean(jac[all_net_mask])

    print(f"  --- Total Network Mean (Index 20): {net_means[20]:.6f} ---")

    save_name = filename.replace(".npy", "_ROI_means.npy")
    np.save(os.path.join(save_dir, save_name), net_means)
    print(f"Saved results to: {save_name}")

    del jac
    gc.collect()

print("\nAll processing completed.")


# In[37]:


import matplotlib.pyplot as plt

raw_ses = raw_results[:, 1]
sq_means = sq_results[:, 0]
sq_ses = sq_results[:, 1]

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(1, 21)
width = 0.35

# left axis: Raw tSNR
bars1 = ax1.bar(x - width/2, raw_means, width, yerr=raw_ses, 
                label='Raw tSNR', color='skyblue', capsize=3)
ax1.set_ylabel('Raw tSNR Scale')
ax1.set_xlabel('Network ID')

# right axis: 1st Deriv Sq tSNR
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, sq_means, width, yerr=sq_ses, 
                label='1stDeriv Sq tSNR', color='salmon', capsize=3)
ax2.set_ylabel('1stDeriv Sq Scale')

sq_min = np.min(sq_means[sq_means > 0])
sq_max = np.max(sq_means)
ax2.set_ylim(sq_min * 0.95, sq_max * 1.05) 

plt.title('Comparison of tSNR across Networks (with SE bars)')
ax1.set_xticks(x)
fig.tight_layout()
plt.show()


# In[38]:


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

network_names = [yeo7_plus_network_labels[i] for i in range(1, 21)]

fig, ax1 = plt.subplots(figsize=(14, 7)) 
x = np.arange(1, 21)
width = 0.35

# left axis: Raw tSNR
bars1 = ax1.bar(x - width/2, raw_means, width, yerr=raw_ses, 
                label='Raw tSNR', color='skyblue', capsize=3)
ax1.set_ylabel('Raw tSNR Scale', fontsize=12)
ax1.set_xlabel('Network Name') 

# right axis: 1st Deriv Sq tSNR
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, sq_means, width, yerr=sq_ses, 
                label='1stDeriv Sq tSNR', color='salmon', capsize=3)
ax2.set_ylabel('1stDeriv Sq Scale', fontsize=12)

sq_min = np.min(sq_means[sq_means > 0])
sq_max = np.max(sq_means)
ax2.set_ylim(sq_min * 0.95, sq_max * 1.05) 

plt.title('Comparison of tSNR across Networks (with SE bars)\n', fontsize=16)

ax1.set_xticks(x)
ax1.set_xticklabels(network_names, rotation=45, ha="right") 

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()

# save_path = '/mnt/e/T1_2024/FigureS1'
save_path = f'{FIGURE_DIR}/FigureS1'
# 1. PDF
plt.savefig(f'{save_path}.pdf', 
            format='pdf', 
            bbox_inches='tight', 
            pad_inches=0.05)
# 2. TIFF
plt.savefig(f'{save_path}.tif', 
            dpi=300, 
            format='tiff', 
            bbox_inches='tight', 
            pad_inches=0.05, 
            pil_kwargs={"compression": "tiff_lzw"})

plt.show()

plt.close()


# In[39]:


# Extract 15 networks (remove 5 networks)

full_target_data = combined_jacobian_data[1, 5, 0:20, :] # square of first derivative

# 15 networks(15, 483)
keep_indices = list(range(14)) + [19]
target_data = full_target_data[keep_indices, :]

print(f"original networks: {full_target_data.shape}")
print(f"restricted networks: {target_data.shape}")


# In[40]:


# yeo7_plus_network_labels = {
#     1: "Visual",
#     2: "Somatomotor",
#     3: "Dorsal Attn",
#     4: "Ventral Attn",
#     5: "Limbic-cortical",
#     6: "ParaHippocampal",    
#     #6: "Subcortical",
#     7: "Frontoparietal",
#     8: "DMN",
#     9: "hippocampus",
#     10: "amygdala",
#     11: "thalamus",
#     12: "basal ganglia",
#     13: "cerebellum",
#     14: "VTA",
#     15: "SN_pc",
#     16: "SN_pr",
#     17: "Red",
#     18: "LC",
#     19: "Raphe_D", 
#     20: "Precuneus",
# }


# In[ ]:


# Finished!: Parcellation of atrophy per voxel into 15 networks


# In[ ]:




