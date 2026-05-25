#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR, AAL_LABEL_TXT, AAL_LABEL_NII


# In[7]:


# Concanate MRI and psycologocal score

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

input_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked" 

# ===== 1. read CSV =====
# csv_path = "/mnt/e/T1_2024/_2024allseasons_brain_psyc.csv"  # for example
csv_path = f"{BASE_DIR}/_2024allseasons_brain_psyc.csv"
df = pd.read_csv(csv_path)

# ===== 2. remove nan data =====
df[["MMSE", "FAB", "RCPM"]] = df[["MMSE", "FAB", "RCPM"]].replace(0, np.nan)
df = df.dropna(subset=["MMSE","FAB","RCPM","seasons","files","Sex","Birthday"]).reset_index(drop=True)

# ===== 3. MRIs =====
npy_base_dirs = {
    "spring":  f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed",  # for example
    "summer":  f"{BASE_DIR}/_2024summer_mni152_brain_masked_npy_zscore_outlier_removed",  # for example
    "autumn":  f"{BASE_DIR}/_2024autumn_mni152_brain_masked_npy_zscore_outlier_removed",  # for example
    "winter":  f"{BASE_DIR}/_2024winter_mni152_brain_masked_npy_zscore_outlier_removed",  # for example
}


# ===== 4. MRI, psyc.scores, sex, birthday and the season of examination =====
all_mri, all_scores, all_sex, all_bday, all_seasons = [], [], [], [], []
skipped = []

for i, row in df.iterrows():
    season = row["seasons"].lower().strip()
    fname  = row["files"]

    if season not in npy_base_dirs:
        skipped.append((i, season, fname, "invalid season"))
        continue

    npy_path = os.path.join(npy_base_dirs[season], fname)
    if not os.path.exists(npy_path):
        skipped.append((i, season, fname, "file not found"))
        continue

    try:
        arr = np.load(npy_path)   # shape = (91,109,91)
    except Exception as e:
        skipped.append((i, season, fname, f"load error: {e}"))
        continue

    all_mri.append(arr)
    all_scores.append([row["MMSE"], row["FAB"], row["RCPM"]])
    all_sex.append(row["Sex"])
    all_bday.append(str(row["Birthday"]))
    all_seasons.append(season)

# ===== 5. tranform to numpy =====
all_mri     = np.array(all_mri)            # (N, 91, 109, 91)
all_scores  = np.array(all_scores)         # (N, 3)
all_sex     = np.array(all_sex)            # dtype='<U1'
all_bday    = np.array(all_bday)           # dtype='<U10'
all_seasons = np.array(all_seasons)        # dtype='<U10'

print("MMSE, FAB, RCPM")
print("mean:", all_scores[:,0].mean(), all_scores[:,1].mean(),all_scores[:,2].mean())
print("std :", all_scores[:,0].std(),  all_scores[:,1].std(), all_scores[:,2].std())
print("max :", all_scores[:,0].max(),  all_scores[:,1].max(), all_scores[:,2].max())
print("min :", all_scores[:,0].min(),  all_scores[:,1].min(), all_scores[:,2].min())

# ===== 5.1 standazation of psychological scores =====
scaler = StandardScaler()
all_scores = scaler.fit_transform(all_scores)

print("MRI shape:", all_mri.shape)
print("Scores shape:", all_scores.shape)
print("Sex shape:", all_sex.shape)
print("Birthday shape:", all_bday.shape)
print("Season shape:", all_seasons.shape)
print("Skipped:", len(skipped), "cases")

# ===== 6. save =====
SAVE_PATH = f"{BASE_DIR}/all_mri_psyc_dataset_final.npz"
np.savez(SAVE_PATH,
         mri=all_mri,
         scores=all_scores,
         sex=all_sex,
         birthday=all_bday,
         season=all_seasons)


# In[13]:


data = np.load(SAVE_PATH, allow_pickle=True)
seasons = data["season"]
import pandas as pd
df_season = pd.DataFrame({"season": seasons})
print(df_season['season'].value_counts())


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

data = np.load(SAVE_PATH, allow_pickle=True)
scores = data["scores"]  # shape = (531, 3)

score_names = ["MMSE", "FAB", "RCPM"]

plt.figure(figsize=(15,4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.hist(scores[:,i], bins=15, color='skyblue', edgecolor='black')
    plt.title(score_names[i])
    plt.xlabel("Score")
    plt.ylabel("Count")
plt.tight_layout()
plt.show()


# In[ ]:




