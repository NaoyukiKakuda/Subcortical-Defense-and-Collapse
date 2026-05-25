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


# In[39]:


# original data (20 ROI x 483 samples)
# full_target_data shape: (20, 483)
full_target_data = combined_jacobian_data[1, 5, 0:20, :] # square of first derivative

# saved index: 0-13 and 19
# 15 networks(15, 483)
keep_indices = list(range(14)) + [19]
target_data = full_target_data[keep_indices, :]

print(f"original networks: {full_target_data.shape}")
print(f"restricted networks: {target_data.shape}")


# In[ ]:





# In[42]:


# time-varying dynamical model: B1 and B2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR

# --- 1. concanating the data ---
roi_names = [
    "Visual", "Somatomotor", "Dorsal Attn", "Ventral Attn",
    "Limbic-cortical", "ParaHippocampal", "Frontoparietal", "DMN",
    "hippocampus", "amygdala", "thalamus", "basal ganglia",
    "cerebellum", "VTA", "Precuneus"
]

df_jacobian = pd.DataFrame(target_data[0:15, :].T, columns=roi_names).reset_index(drop=True)
df_combined = pd.concat([df.reset_index(drop=True), df_jacobian], axis=1, join='inner')

def run_continuous_dcm_analysis(df, roi_names, dt=1.0):
    n_roi = len(roi_names)
    n_samples = len(df)
    data = df[roi_names].values
    X_data = data[:-1]
    Y_data = (data[1:] - data[:-1]) / dt
    time_axis = np.linspace(0, 1, n_samples - 1)

    results = {
        'b_lin': np.zeros((n_roi, n_roi)), 'p_lin': np.ones((n_roi, n_roi)),
        'b_quad': np.zeros((n_roi, n_roi)), 'p_quad': np.ones((n_roi, n_roi))
    }

    print("Running OLS for each edge...")
    for i in range(n_roi):
        for j in range(n_roi):
            if i == j: continue
            coupling = Y_data[:, i] / (X_data[:, j] + 1e-6)
            coupling = np.clip(coupling, -5, 5)
            X_reg = sm.add_constant(np.column_stack([time_axis, time_axis**2]))
            model = sm.OLS(coupling, X_reg).fit()

            results['b_lin'][i, j] = model.params[1]
            results['p_lin'][i, j] = model.pvalues[1]
            results['b_quad'][i, j] = model.params[2]
            results['p_quad'][i, j] = model.pvalues[2]

    out = {k: pd.DataFrame(v, index=roi_names, columns=roi_names) for k, v in results.items()}
    return out

# --- 2. calculate ---
dcm_results = run_continuous_dcm_analysis(df_combined, roi_names)

# --- 3. left: B1(Linear), right: B2(Quadratic) ---
fig, axes = plt.subplots(1, 2, figsize=(26, 10))
plt.suptitle("Estimating Modulatory Effects (B-matrix) over the sorted trajectory\n", fontsize=24)#, fontweight='bold')


# --- left:Linear Trajectory ---
b_lin = dcm_results['b_lin']
p_lin = dcm_results['p_lin']
annot_lin = p_lin.applymap(lambda x: '★' if x < 0.05 else ('+' if x < 0.1 else ''))
vlim_lin = np.nanmax(np.abs(b_lin.values)) * 0.7

sns.heatmap(b_lin, ax=axes[0], cmap='RdBu_r', center=0, vmax=vlim_lin, vmin=-vlim_lin,
            annot=annot_lin, fmt='', annot_kws={'size': 15, 'color': 'yellow'},
            square=True, cbar_kws={'label': 'Linear Coefficient'})
axes[0].set_title('Linear Trajectory: Overall Progression Trend\n(+: p < 0.10)', fontsize=16)#, fontweight='bold')

# --- right:Quadratic Wave Dynamics ---
b_quad = dcm_results['b_quad']
p_quad = dcm_results['p_quad']
annot_quad = p_quad.applymap(lambda x: '★' if x < 0.05 else ('+' if x < 0.1 else ''))
vlim_quad = np.nanmax(np.abs(b_quad.values)) * 0.7

sns.heatmap(b_quad, ax=axes[1], cmap='PRGn', center=0, vmax=vlim_quad, vmin=-vlim_quad,
            annot=annot_quad, fmt='', annot_kws={'size': 30, 'color': 'yellow'},
            square=True, cbar_kws={'label': 'Quadratic Coefficient'})
axes[1].set_title('Quadratic Wave: Mid-stage Specific Dynamics\n(+: p < 0.10)', fontsize=16)#, fontweight='bold')

for ax in axes:
    ax.set_xlabel('Source (From)')
    ax.set_ylabel('Target (To)')

plt.tight_layout()
# plt.savefig('/mnt/e/T1_2024/Figure4.tif', 
plt.savefig(f'{FIGURE_DIR}/Figure4.tif', 
            dpi=300,
            format='tiff',
            bbox_inches='tight',
            pad_inches=0.05,
            # compression ではなく pil_kwargs を使います
            pil_kwargs={"compression": "tiff_lzw"})
plt.show()


# In[43]:


# calculate A matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_a_matrix_from_ols(df, roi_names, dt=1.0):
    n_roi = len(roi_names)
    data = df[roi_names].values
    X_data = data[:-1]
    Y_data = (data[1:] - data[:-1]) / dt
    time_axis = np.linspace(0, 1, len(df) - 1)

    a_mat = np.zeros((n_roi, n_roi))
    p_mat = np.ones((n_roi, n_roi))

    for i in range(n_roi):
        for j in range(n_roi):
            if i == j: continue
            coupling = Y_data[:, i] / (X_data[:, j] + 1e-6)
            coupling = np.clip(coupling, -5, 5)
            # Aは const、B_linは time、B_quadは time^2
            X_reg = sm.add_constant(np.column_stack([time_axis, time_axis**2]))
            model = sm.OLS(coupling, X_reg).fit()

            a_mat[i, j] = model.params[0] # 切片をA-matrixとする
            p_mat[i, j] = model.pvalues[0]

    return pd.DataFrame(a_mat, index=roi_names, columns=roi_names), \
           pd.DataFrame(p_mat, index=roi_names, columns=roi_names)

a_matrix, a_pvalues = get_a_matrix_from_ols(df_combined, roi_names)

# --- graphs：A, B(Linear), B(Quadratic) ---
fig, axes = plt.subplots(1, 3, figsize=(36, 10))
fig.suptitle("Global Connectivity Landscape: Baseline (A) and Modulations (B)\n", fontsize=32)

label_fontsize = 18 
cbar_label_fontsize = 16
cbar_tick_fontsize = 16   
annot_kws = {'size': 14, 'color': 'yellow'}

matrices = [a_matrix, dcm_results['b_lin'], dcm_results['b_quad']]
p_values = [a_pvalues, dcm_results['p_lin'], dcm_results['p_quad']]
cbar_labels = ['Intrinsic Strength (A)', 'Linear Modulation (B1)', 'Quadratic Modulation (B2)']
cmaps = ['RdBu_r', 'RdBu_r', 'PRGn']

for i, ax in enumerate(axes):
    vlimit = np.nanmax(np.abs(matrices[i].values)) * 0.7
    annot_text = p_values[i].applymap(lambda x: '★' if x < 0.05 else ('+' if x < 0.1 else ''))

    sns.heatmap(matrices[i], ax=ax, cmap=cmaps[i], center=0, vmax=vlimit, vmin=-vlimit,
                annot=annot_text, fmt='', annot_kws=annot_kws, square=True,
                yticklabels=roi_names if i == 0 else False,
                cbar_kws={'label': ''}) # ここでラベルを空に

    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_labels[i], fontsize=cbar_label_fontsize, labelpad=20) 
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

    ax.set_xlabel('Source (From)', fontsize=24)
    ax.set_xticklabels(roi_names, fontsize=label_fontsize, rotation=45, ha="right")
    if i == 0:
        ax.set_ylabel('Target (To)', fontsize=24)
        ax.set_yticklabels(roi_names, fontsize=label_fontsize, rotation=0)
    else:
        ax.set_ylabel('')

axes[0].set_title('A-matrix: Average Intrinsic Connectivity', fontsize=22)
axes[1].set_title('B-matrix (Linear): Progression Trend', fontsize=22)
axes[2].set_title('B-matrix (Quadratic): Mid-stage Dynamics', fontsize=22)

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/Figure7.tif', 
            dpi=300, 
            format='tiff', 
            bbox_inches='tight', 
            pad_inches=0.05, 
            pil_kwargs={"compression": "tiff_lzw"})
plt.show()


# In[44]:


# close approximation of full logic of PEB/BMR/BMA

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def run_full_peb_analysis(dcm_results, a_matrix, a_pvalues, roi_names):
    results_peb = {}
    matrix_keys = ['a_matrix', 'b_lin', 'b_quad']

    # prior variance (close to SPM default)
    prior_variance = 1/16 

    for key in matrix_keys:
        if key == 'a_matrix':
            curr_b = a_matrix.values
            curr_p = a_pvalues.values
        else:
            curr_b = dcm_results[key].values
            curr_p = dcm_results[key.replace('b_', 'p_')].values

        # --- 1. PEB (Parametric Empirical Bayes) ---
        # Precision in each edge
        # SE calculated from p-value: z = beta / SE => SE = beta / z
        z_scores = np.abs(stats.norm.ppf(curr_p / 2.0))
        z_scores = np.clip(z_scores, 1e-5, 8.0) 
        se_estimates = np.abs(curr_b) / z_scores
        precisions = 1.0 / (se_estimates**2 + 1e-6)

        # --- 2. BMR (Bayesian Model Reduction) ---
        # approximation of change of free energy(dF) by Savage-Dickey ratio
        # for simplicity, difference between Log-model evidences
        # dF = -0.5 * ( (beta^2 * prec) - log(prec_prior / prec_post) )

        post_precisions = (1/prior_variance) + precisions
        dF = 0.5 * ( (curr_b**2 * precisions) - np.log(post_precisions * prior_variance) )

        # Pp = exp(dF) / (1 + exp(dF))
        post_probs = 1.0 / (1.0 + np.exp(-dF))

        # --- 3. BMA (Bayesian Model Averaging) ---
        b_bma = post_probs * curr_b
        b_bma[post_probs < 0.75] = 0 

        results_peb[key] = pd.DataFrame(b_bma, index=roi_names, columns=roi_names)
        results_peb[key + '_prob'] = pd.DataFrame(post_probs, index=roi_names, columns=roi_names)

    return results_peb

peb_results = run_full_peb_analysis(dcm_results, a_matrix, a_pvalues, roi_names)

fig, axes = plt.subplots(1, 3, figsize=(30, 8))
fig.suptitle("Full PEB/BMR/BMA Results: Bayesian Model Averaging\n", fontsize=24)

titles = ['Intrinsic (A)', 'Linear Mod. (B1)', 'Quadratic Mod. (B2)']
for i, key in enumerate(['a_matrix', 'b_lin', 'b_quad']):
    data = peb_results[key]
    probs = peb_results[key + '_prob']

    annot_labels = data.copy().astype(str)
    for r in range(len(roi_names)):
        for c in range(len(roi_names)):
            val = data.iloc[r, c]
            p = probs.iloc[r, c]
            if p > 0.90:
                annot_labels.iloc[r, c] = f"{val:.2f}**"
            elif p > 0.75:
                annot_labels.iloc[r, c] = f"{val:.2f}*"
            else:
                annot_labels.iloc[r, c] = ""

    sns.heatmap(data, ax=axes[i], cmap='RdBu_r', center=0, 
                annot=annot_labels, fmt='', annot_kws={'size': 9}, square=True)
    axes[i].set_title(f"{titles[i]}\n(Annot: ** Pp>.90, * Pp>.75)", fontsize=16)

plt.tight_layout()
plt.show()


# In[56]:


fig, axes = plt.subplots(1, 2, figsize=(28, 12))
fig.suptitle("Full PEB/BMR/BMA Results: Bayesian Model Averaging\n\n", fontsize=24)

titles = ['Linear Mod. (B1)', 'Quadratic Mod. (B2)']
for i, key in enumerate(['b_lin', 'b_quad']):
    data = peb_results[key]
    probs = peb_results[key + '_prob']

    annot_labels = data.copy().astype(str)
    for r in range(len(roi_names)):
        for c in range(len(roi_names)):
            val = data.iloc[r, c]
            p = probs.iloc[r, c]
            if p > 0.90:
                annot_labels.iloc[r, c] = f"{val:.2f}**"
            elif p > 0.75:
                annot_labels.iloc[r, c] = f"{val:.2f}*"
            else:
                annot_labels.iloc[r, c] = ""

    sns.heatmap(data, ax=axes[i], cmap='RdBu_r', center=0, 
            annot=annot_labels, fmt='', annot_kws={'size': 16, 'fontweight': 'bold'}, square=True)
    axes[i].set_title(f"{titles[i]}\n(Annot: ** Pp>.90, * Pp>0.75)", fontsize=18)
    axes[i].set_title(f"{titles[i]}\n(Annot: ** Pp>.90, * Pp>0.75)", fontsize=18)

    axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=16, rotation=45, ha='right')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=16, rotation=0)
    axes[i].set_title(f"{titles[i]}\n(Annot: ** Pp>.90, * Pp>.75)\n", fontsize=22)

plt.tight_layout()
# save_path = '/mnt/e/T1_2024/Figure4'
save_path = f'{FIGURE_DIR}/Figure4'
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




