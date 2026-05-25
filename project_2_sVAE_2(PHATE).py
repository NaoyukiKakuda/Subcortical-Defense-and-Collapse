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


# In[22]:


# PHATE

import numpy as np
import phate
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt

from project_config import GRID_RESULTS

# z_npz = np.load("/mnt/e/T1_2024/grid_results_npz/z_matrix_fold1_epoch66.npz")
save_path = f"{GRID_RESULTS}/z_matrix_fold1_epoch66.npz"
z_npz = np.load(save_path)

z_matrix = z_npz['z_matrix']
scaler_z = StandardScaler()
z_std = scaler_z.fit_transform(z_matrix)

phate_op = phate.PHATE(n_components=2, knn=30, t=20)
z_phate = phate_op.fit_transform(z_std)  # (483, 2)

z_phate[:, 0] *= -1

phate1 = z_phate[:, 0] 
phate2 = z_phate[:, 1]

phate_df = pd.DataFrame(z_phate, columns=["PHATE1", "PHATE2"])

plt.figure(figsize=(6,5))
plt.scatter(z_phate[:, 0], z_phate[:, 1], c=phate1)
plt.colorbar(label="Pseudotime (PHATE1)")
plt.title("PHATE embedding colored by pseudotime")
plt.xlabel("PHATE1")
plt.ylabel("PHATE2")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. dataframe
df_subjects = pd.DataFrame({
    'PHATE1': z_phate[:, 0]
})

# 2. histogram
plt.figure(figsize=(10, 5))
sns.histplot(df_subjects['PHATE1'], bins=50, kde=True, color='teal')
plt.title("Sample Density along PHATE1 Axis (Verification of Linearity)", fontsize=14)
plt.xlabel("Raw PHATE1 Value", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

# 3. rank order
df_subjects['Pseudo_Time_Rank'] = df_subjects['PHATE1'].rank(pct=True)

plt.figure(figsize=(6, 6))
plt.scatter(df_subjects['PHATE1'], df_subjects['Pseudo_Time_Rank'], s=10, alpha=0.5)
plt.title("Raw PHATE1 vs. Rank-based Pseudo-time")
plt.xlabel("Raw PHATE1")
plt.ylabel("Normalized Rank (0.0 - 1.0)")
plt.grid(True, alpha=0.2)
plt.show()


# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable # 追加

import matplotlib.pyplot as plt

# 1. set fonto
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

phate1_raw = z_phate[:, 0]
phate2_raw = z_phate[:, 1]

phate1_normalized = (phate1_raw - phate1_raw.min()) / (phate1_raw.max() - phate1_raw.min())
phate2_normalized = (phate2_raw - phate2_raw.min()) / (phate2_raw.max() - phate2_raw.min())

pseudotime = pd.Series(phate1_raw).rank(pct=True).values

# graph (2*1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.suptitle("Dimensionality reduction and definition of pseudotime using PHATE\n", fontsize=16)

# --- left: PHATE Embedding ---
sc = ax1.scatter(phate1_normalized, phate2_normalized, c=pseudotime, cmap='viridis', s=30, alpha=0.6, edgecolors='none')
ax1.set_title("Normalized PHATE coordinates")
ax1.set_xlabel("Normalized PHATE1 (0.0 - 1.0)")
ax1.set_ylabel("Normalized PHATE2 (0.0 - 1.0)")
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle=':', alpha=0.3)

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1) # 5%の幅で横に追加
cbar = fig.colorbar(sc, cax=cax1)
cbar.set_label("Pseudotime (Normalized Rank)")

# --- right: Mapping ---
ax2.scatter(phate1_normalized, pseudotime, alpha=0.6, edgecolors='none')
ax2.set_title("Pseudotime Projection")
ax2.set_xlabel("Normalized PHATE1 (0.0 - 1.0)")
ax2.set_ylabel("Pseudotime (Normalized Rank 0.0 - 1.0)")
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle='--', alpha=0.4)

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
cax2.axis('off') 

plt.tight_layout()

save_path = f'{FIGURE_DIR}/Figure2' # for example
plt.savefig(f'{save_path}.pdf', 
            format='pdf', 
            bbox_inches='tight', 
            pad_inches=0.05)
plt.savefig(f'{save_path}.tif', 
            dpi=300, 
            format='tiff', 
            bbox_inches='tight', 
            pad_inches=0.05, 
            pil_kwargs={"compression": "tiff_lzw"})

plt.show()
plt.close()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

psyc_labels = ["MMSE", "FAB", "RCPM"] 

plt.figure(figsize=(15, 6))
plt.suptitle("Correlation between PHATE-derived pseudotime and clinical scores\n", fontsize=16)

for i in range(3):
    plt.subplot(1, 3, i+1)

    x = z_phate[:, 0] 
    y = psyc_original[:, i]

    r, p = pearsonr(x, y)
    sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5, 's':10}, line_kws={'color':'red'})
    plt.title(f"PHATE1 vs {psyc_labels[i]}\n$r = {r:.3f}, p = {p:.3e}$")
    plt.xlabel("PHATE 1")
    plt.ylabel(psyc_labels[i])
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/Figure6.tif', 
            dpi=300,
            format='tiff',
            bbox_inches='tight',
            pad_inches=0.05,
            pil_kwargs={"compression": "tiff_lzw"})
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np # 追加

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

psyc_labels = ["MMSE", "FAB", "RCPM"] 
critical_point = 0.470

plt.figure(figsize=(15, 6))
plt.suptitle("Correlation between PHATE-derived pseudotime and clinical scores\n", fontsize=16)

x_raw = z_phate[:, 0]

x_min, x_max = x_raw.min(), x_raw.max()
x_norm = (x_raw - x_min) / (x_max - x_min)

for i in range(3):
    ax = plt.subplot(1, 3, i+1)

    y_data = psyc_original[:, i]
    r, p = pearsonr(x_norm, y_data)
    sns.regplot(x=x_norm, y=y_data, 
                scatter_kws={'alpha':0.4, 's':15, 'edgecolors':'none'}, 
                line_kws={'color':'#d62728', 'lw':2}, 
                ax=ax)
    ax.axvline(x=critical_point, color='#4d4d4d', linestyle='--', linewidth=1.2, 
                label=f'Threshold ({critical_point:.3f})')
    ax.set_xlim(-0.05, 1.05) 

    ax.set_title(f"PHATE1 vs {psyc_labels[i]}\n$r = {r:.3f}, p = {p:.2e}$", fontsize=12)
    ax.set_xlabel("Normalized PHATE1", fontsize=10)
    ax.set_ylabel(psyc_labels[i], fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

save_path = f'{FIGURE_DIR}/Figure3' #for example
plt.savefig(f'{save_path}.pdf', 
            format='pdf', 
            bbox_inches='tight', 
            pad_inches=0.05)
plt.savefig(f'{save_path}.tif', 
            dpi=300, 
            format='tiff', 
            bbox_inches='tight', 
            pad_inches=0.05, 
            pil_kwargs={"compression": "tiff_lzw"})
plt.show()
plt.close()


# In[ ]:




