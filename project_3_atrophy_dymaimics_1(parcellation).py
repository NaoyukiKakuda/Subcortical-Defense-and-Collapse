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


# In[13]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR, AAL_LABEL_TXT, AAL_LABEL_NII

# --- AALラベル読み込み ---
# label_path = "/mnt/c/Users/n-kak/Downloads/AAL3/AAL3v1.nii.txt"
label_path = AAL_LABEL_TXT
roi_name_dict = {}
with open(label_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            idx = int(parts[0])
            name = " ".join(parts[1:])
            roi_name_dict[idx] = name
print(roi_name_dict)


# In[19]:


# ============================================
# AAL3 → Yeo7 network, plus
# ============================================

aal_to_yeo7_plus = {
    # ------------------------
    # Visual (1)
    # ------------------------
    "Calcarine_L":1, "Calcarine_R":1,
    "Cuneus_L":1, "Cuneus_R":1,
    "Lingual_L":1, "Lingual_R":1,
    "Occipital_Sup_L":1, "Occipital_Sup_R":1,
    "Occipital_Mid_L":1, "Occipital_Mid_R":1,
    "Occipital_Inf_L":1, "Occipital_Inf_R":1,
    "Fusiform_L":1, "Fusiform_R":1,

    # ------------------------
    # Somatomotor (2)
    # ------------------------
    "Precentral_L":2, "Precentral_R":2,
    "Postcentral_L":2, "Postcentral_R":2,
    "Supp_Motor_Area_L":2, "Supp_Motor_Area_R":2,
    "Paracentral_Lobule_L":2, "Paracentral_Lobule_R":2,
    "Rolandic_Oper_L":2, "Rolandic_Oper_R":2,
    "Heschl_L":2, "Heschl_R":2,
    "Temporal_Sup_L":2, "Temporal_Sup_R":2,
    "Temporal_Pole_Sup_L":2, "Temporal_Pole_Sup_R":2,

    # ------------------------
    # Dorsal Attention (3)
    # ------------------------
    "Parietal_Sup_L":3, "Parietal_Sup_R":3,
    "Parietal_Inf_L":3, "Parietal_Inf_R":3,
    "SupraMarginal_L":3, "SupraMarginal_R":3,

    # ------------------------
    # Ventral Attention (4)
    # ------------------------
    "Insula_L":4, "Insula_R":4,
    "Cingulate_Mid_L":4, "Cingulate_Mid_R":4,
    "Cingulate_Ant_L":4, "Cingulate_Ant_R":4,

    # ------------------------
    # Limbic-cortical (5)
    # ------------------------
    "Olfactory_L":5, "Olfactory_R":5,
    "Rectus_L":5, "Rectus_R":5,
    "OFCmed_L":5, "OFCmed_R":5,
    "OFClat_L":5, "OFClat_R":5,
    "OFCant_L":5, "OFCant_R":5,
    "OFCpost_L":5, "OFCpost_R":5,

    # ------------------------
    # Limbic-subcortical (6)
    # ------------------------
    #"Amygdala_L":6, "Amygdala_R":6,
    #"Hippocampus_L":6, "Hippocampus_R":6,
    "ParaHippocampal_L":6, "ParaHippocampal_R":6,

    # ------------------------
    # Frontoparietal / Control (7)
    # ------------------------
    "Frontal_Sup_L":7, "Frontal_Sup_R":7,
    "Frontal_Sup_Medial_L":7, "Frontal_Sup_Medial_R":7,
    "Frontal_Mid_L":7, "Frontal_Mid_R":7,
    "Frontal_Inf_Oper_L":7, "Frontal_Inf_Oper_R":7,
    "Frontal_Inf_Tri_L":7, "Frontal_Inf_Tri_R":7,
    "Frontal_Inf_Orb_L":7, "Frontal_Inf_Orb_R":7,

    # ------------------------
    # Default Mode (8)
    # ------------------------
    "Precuneus_L":20, "Precuneus_R":20,
    "Cingulate_Post_L":8, "Cingulate_Post_R":8,
    "Angular_L":8, "Angular_R":8,
    "Temporal_Mid_L":8, "Temporal_Mid_R":8,
    "Temporal_Pole_Mid_L":8, "Temporal_Pole_Mid_R":8,
    "Temporal_Inf_L":8, "Temporal_Inf_R":8,

    # ------------------------
    #  Hippocampus (9)
    # ------------------------
    "Hippocampus_L":9, "Hippocampus_R":9, 

    # ------------------------
    #  Amygdala (10)
    # ------------------------
    "Amygdala_L":10, "Amygdala_R":10,

    # ------------------------
    # thalamus (11)
    # ------------------------
    "Thal_AV_L":11, "Thal_AV_R":11, 
    "Thal_LP_L":11, "Thal_LP_R":11,  
    "Thal_VA_L":11, "Thal_VA_R":11, 
    "Thal_VL_L":11, "Thal_VL_R":11, 
    "Thal_VPL_L":11, "Thal_VPL_R":11, 
    "Thal_IL_L":11, "Thal_IL_R":11, 
    "Thal_Re_L":11, "Thal_Re_R":11, 
    "Thal_MDm_L":11, "Thal_MDm_R":11, 
    "Thal_MDl_L":11, "Thal_MDl_R":11, 
    "Thal_MGN_L":11, "Thal_MGN_R":11, 
    "Thal_PuI_L":11, "Thal_PuI_R":11, 
    "Thal_PuM_L":11, "Thal_PuM_R":11, 
    "Thal_PuL_L":11, "Thal_PuL_R":11,
    "Thalamus_L":11, "Thalamus_R":11,
    "Thal_LGN_L":11, "Thal_LGN_R":11,
    "Thal_PuA_L":11, "Thal_PuA_R":11,

    # ------------------------
    # basal ganglia (12)
    # ------------------------
    "Caudate_L":12, "Caudate_R":12,
    "Putamen_L":12, "Putamen_R":12,
    "Pallidum_L":12, "Pallidum_R":12,
    "N_Acc_L":12, "N_Acc_R":12, 

    # cerebellm (13)
    # ------------------------
    "Cerebellum_Crus1_L":13, "Cerebellum_Crus1_R":13,
    "Cerebellum_Crus2_L":13, "Cerebellum_Crus2_R":13,
    "Cerebellum_3_L":13, "Cerebellum_3_R":13,
    "Cerebellum_4_5_L":13, "Cerebellum_4_5_R":13,
    "Cerebellum_6_L":13, "Cerebellum_6_R":14,
    "Cerebellum_7b_L":13, "Cerebellum_7b_R":13,
    "Cerebellum_8_L":13, "Cerebellum_8_R":13,
    "Cerebellum_9_L":13, "Cerebellum_9_R":13,
    "Cerebellum_10_L":13, "Cerebellum_10_R":13,
    "Vermis_1_2":13, "Vermis_3":13,
    "Vermis_4_5":13, "Vermis_6":13,
    "Vermis_7":13, "Vermis_8":13, 
    "Vermis_9":13, "Vermis_10":13, 

    # ------------------------
    # brainstem (14)
    # ------------------------
    "VTA_L":14, "VTA_R":14,
    "SN_pc_L":15, "SN_pc_R":15,
    "SN_pr_L":16, "SN_pr_R":16,
    "Red_N_L":17, "Red_N_R":17, 
    "LC_L":18, "LC_R":18,
    "Raphe_D":19, "Raphe_M":19,
}

yeo7_plus_network_labels = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attn",
    4: "Ventral Attn",
    5: "Limbic-cortical",
    6: "ParaHippocampal",    
    #6: "Subcortical",
    7: "Frontoparietal",
    8: "DMN",
    9: "hippocampus",
    10: "amygdala",
    11: "thalamus",
    12: "basal ganglia",
    13: "cerebellum",
    14: "VTA",
    15: "SN_pc",
    16: "SN_pr",
    17: "Red",
    18: "LC",
    19: "Raphe_D", 
    20: "Precuneus",
}

def normalize_aal(name):
    name = name.split()[0]
    name = name.replace("_2_L", "_L").replace("_2_R", "_R")
    return name


# In[20]:


#"/mnt/c/Users/n-kak/Downloads/AAL3/AAL3v1.nii"
import nibabel as nib
import numpy as np
import pandas as pd

label_path_nii = AAL_LABEL_NII

# roi_nii = nib.load("/mnt/c/Users/n-kak/Downloads/AAL3/AAL3v1.nii") 
roi_nii = nib.load(label_path_nii) 
roi_mask = roi_nii.get_fdata().astype(int)

print(roi_mask.shape)#, delta_img.shape)


# In[21]:


import numpy as np

def create_final_network_mask(roi_mask, aal_id_to_name, aal_to_yeo7_plus, normalize_aal_func):

    id_mapping = {}
    for aal_id, name in aal_id_to_name.items():
        norm_name = normalize_aal_func(name)
        if norm_name in aal_to_yeo7_plus:
            id_mapping[aal_id] = aal_to_yeo7_plus[norm_name]
        else:
            id_mapping[aal_id] = 0 # 定義外の領域は背景(0)とする

    clsnp_mask_3d = np.zeros_like(roi_mask, dtype=int)
    for aal_id, clsnp_id in id_mapping.items():
        if clsnp_id != 0:
            clsnp_mask_3d[roi_mask == aal_id] = clsnp_id

    network_mask_1d = clsnp_mask_3d.reshape(-1)

    return clsnp_mask_3d, network_mask_1d

# aal_label_file = "/mnt/c/Users/n-kak/Downloads/AAL3/AAL3v1.nii.txt"
aal_label_file = AAL_LABEL_TXT
labels_df = pd.read_csv(aal_label_file, sep='\s+', header=None, names=['id', 'name', 'index'])
aal3_id_to_name = dict(zip(labels_df['index'], labels_df['name']))

print(f"Loaded {len(aal3_id_to_name)} regions from AAL3 label file.")

clsnp_mask_3d, network_mask_1d = create_final_network_mask(
    roi_mask, aal3_id_to_name, aal_to_yeo7_plus, normalize_aal
)

print(f"Mask created. Shape: {network_mask_1d.shape}")
print(f"Found networks: {np.unique(network_mask_1d)}")


# In[28]:


import numpy as np
import os
import gc
from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR

# --- 設定 ---
# save path
save_dir = JACOBIAN_DIR
jacobian_files = ["jacobian0.npy", "jacobian1.npy", "jacobian2.npy"]

# networks
net_ids = range(1, 21) 
n_timepoints = 483
n_rows = 21  # 20 netwros and a whole brain

# --- check mask ---
if 'network_mask_1d' not in locals():
    raise NameError("network_mask_1d is not found!")

# calculate the number of voxels in each netwprk
voxel_counts = np.zeros(n_rows, dtype=np.float32)
for net_idx, net_id in enumerate(net_ids):
    voxel_counts[net_idx] = np.sum(network_mask_1d == net_id)

# the number of voxels in 20 networks
voxel_counts[20] = np.sum(voxel_counts[0:20])
counts_reshaped = voxel_counts[:, np.newaxis] 

print(f"Voxel counts check: Total voxels in networks = {int(voxel_counts[19])}")

# --- main loop ---
for i, filename in enumerate(jacobian_files):
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        print(f"Skip: {filename} not found.")
        continue

    print(f"\n--- Processing {filename} (mmap mode) ---")
    # mmap_mode='r': save memory
    jac = np.load(file_path, mmap_mode='r')

    # matrix for saving(21*483)
    mean_results = np.zeros((n_rows, n_timepoints), dtype=np.float32)
    square_results = np.zeros((n_rows, n_timepoints), dtype=np.float32)

    # 1. ネットワークごとの基礎計算
    for net_idx, net_id in enumerate(net_ids):
        indices = np.where(network_mask_1d == net_id)[0]
        num_voxels = voxel_counts[net_idx]

        if num_voxels > 0:
            temp_data = jac[:, indices].astype(np.float32)

            # index1: Mean
            mean_results[net_idx, :] = temp_data.mean(axis=1)
            # index2: Square
            square_results[net_idx, :] = np.sum(temp_data**2, axis=1)

            del temp_data
        else:
            print(f"  Network {net_id} is empty (0 voxels).")

    # 2. whole brain
    square_results[20, :] = np.sum(square_results[0:20, :], axis=0)

    # 3. index3: Share of Square (%)
    total_square_all = square_results[20, :] + 1e-10
    share_results = (square_results / total_square_all) * 100
    share_results[20, :] = 100.0

    # 4. index4: Share per Voxel (% of each voxel)
    share_per_voxel = np.divide(share_results, counts_reshaped, 
                                out=np.zeros_like(share_results), 
                                where=counts_reshaped != 0)

    # 5. index5: Square per Voxel 
    square_per_voxel = np.divide(square_results, counts_reshaped, 
                                 out=np.zeros_like(square_results), 
                                 where=counts_reshaped != 0)

    # 6. index6: Z-score of Square per Voxel 
    spv_mean_time = square_per_voxel.mean(axis=1, keepdims=True)
    spv_std_time = square_per_voxel.std(axis=1, keepdims=True)
    z_square_per_voxel = np.divide(square_per_voxel - spv_mean_time, spv_std_time,
                                   out=np.zeros_like(square_per_voxel),
                                   where=spv_std_time != 0)

    # --- save ---
    prefix = f"jac{i}"
    output_dict = {
        "mean": mean_results,
        "square": square_results,
        "share": share_results,
        "share_per_voxel": share_per_voxel,
        "square_per_voxel": square_per_voxel,
        "z_square_per_voxel": z_square_per_voxel
    }

    for key, data in output_dict.items():
        np.save(os.path.join(save_dir, f"{prefix}_{key}.npy"), data)

    print(f"Successfully processed and saved all metrics for {filename}")

    del jac, mean_results, square_results, share_results, share_per_voxel, square_per_voxel, z_square_per_voxel
    gc.collect()

print("\n----- Finished! -----")


# In[29]:


import numpy as np

# 1. List
# axis 0 (Jacobian): jac0, jac1, jac2
# axis 1 (Mode): mean, square, share

# save_dir = "/mnt/e/T1_2024/jacobian"
save_dir = JACOBIAN_DIR

filename = "jac0_mean.npy"
file_path = os.path.join(save_dir, filename)
jac0_mean = np.load(file_path, mmap_mode='r')
filename = "jac0_square.npy"
file_path = os.path.join(save_dir, filename)
jac0_square = np.load(file_path, mmap_mode='r')
filename = "jac0_share.npy"
file_path = os.path.join(save_dir, filename)
jac0_share = np.load(file_path, mmap_mode='r')
filename = "jac0_share_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac0_share_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac0_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac0_square_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac0_z_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac0_z_square_per_voxel = np.load(file_path, mmap_mode='r')

filename = "jac1_mean.npy"
file_path = os.path.join(save_dir, filename)
jac1_mean = np.load(file_path, mmap_mode='r')
filename = "jac1_square.npy"
file_path = os.path.join(save_dir, filename)
jac1_square = np.load(file_path, mmap_mode='r')
filename = "jac1_share.npy"
file_path = os.path.join(save_dir, filename)
jac1_share = np.load(file_path, mmap_mode='r')
filename = "jac1_share_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac1_share_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac1_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac1_square_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac1_z_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac1_z_square_per_voxel = np.load(file_path, mmap_mode='r')

filename = "jac2_mean.npy"
file_path = os.path.join(save_dir, filename)
jac2_mean = np.load(file_path, mmap_mode='r')
filename = "jac2_square.npy"
file_path = os.path.join(save_dir, filename)
jac2_square = np.load(file_path, mmap_mode='r')
filename = "jac2_share.npy"
file_path = os.path.join(save_dir, filename)
jac2_share = np.load(file_path, mmap_mode='r')
filename = "jac2_share_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac2_share_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac2_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac2_square_per_voxel = np.load(file_path, mmap_mode='r')
filename = "jac2_z_square_per_voxel.npy"
file_path = os.path.join(save_dir, filename)
jac2_z_square_per_voxel = np.load(file_path, mmap_mode='r')

print(jac0_mean.shape, jac0_square.shape, jac0_share.shape, jac0_share_per_voxel.shape, jac0_square_per_voxel.shape, jac0_z_square_per_voxel.shape)
print(jac1_mean.shape, jac1_square.shape, jac1_share.shape, jac1_share_per_voxel.shape, jac1_square_per_voxel.shape, jac1_z_square_per_voxel.shape)
print(jac2_mean.shape, jac2_square.shape, jac2_share.shape, jac2_share_per_voxel.shape, jac2_square_per_voxel.shape, jac2_z_square_per_voxel.shape)

jac0_stack = np.stack([jac0_mean, jac0_square, jac0_share, jac0_share_per_voxel, jac0_square_per_voxel, jac0_z_square_per_voxel], axis=0) 
jac1_stack = np.stack([jac1_mean, jac1_square, jac1_share, jac1_share_per_voxel, jac1_square_per_voxel, jac1_z_square_per_voxel], axis=0) 
jac2_stack = np.stack([jac2_mean, jac2_square, jac2_share, jac2_share_per_voxel, jac2_square_per_voxel, jac2_z_square_per_voxel], axis=0) 

combined_jacobian_data = np.stack([jac0_stack, jac1_stack, jac2_stack], axis=0)
print(f"combined_jacobian_data.shape: {combined_jacobian_data.shape}")

phate_path = os.path.join(save_dir, "sorted_phate1.npy")
sorted_phate1 = np.load(phate_path)
print(f"sorted_phate1.shape: {sorted_phate1.shape}")

save_path = os.path.join(save_dir, "combined_jacobian_metrics.npy")
np.save(save_path, combined_jacobian_data)


# In[30]:


def get_network_name(labels, idx):

    net_id = idx + 1

    if isinstance(labels, dict):
        return labels.get(net_id, labels.get(str(net_id), f"Network {net_id}"))
    elif isinstance(labels, (list, np.ndarray)):
        if idx < len(labels):
            return labels[idx]
        else:
            return f"Network {net_id}"
    else:
        return f"Network {net_id}"


# In[33]:


import numpy as np
import os

save_dir = JACOBIAN_DIR
jacobian_files = ["jacobian0.npy", "jacobian1.npy", "jacobian2.npy"]

for file in jacobian_files:
    file_path = os.path.join(save_dir, file)

    data = np.load(file_path)
    print(f"{file}: {data.shape}")


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




