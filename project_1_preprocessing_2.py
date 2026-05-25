#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# After "project_1_preprocessing_1" and before "project_1_preprocessing_2",
# "fsl_anat_parallel.sh", "fsl_anat_parallel_seasons.sh" and "batch_masking.sh" were executed.


# In[ ]:


from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR, AAL_LABEL_TXT, AAL_LABEL_NII


# In[32]:


# confirm the shape of output files (masked.nii.gz) by batch_masking.sh

import os
import nibabel as nib

input_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked"  # for example

for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz") or filename.endswith(".nii"):
        filepath = os.path.join(input_dir, filename)
        try:
            img = nib.load(filepath)
            shape = img.shape
            print(f"{filename}: shape = {shape}")
        except Exception as e:
            print(f"❌ Failed to read {filename}: {e}")


# In[33]:


# save the output files (masked.nii.gz) by batch_masking.sh as .npy format
import os
import nibabel as nib
import numpy as np

input_dir  = f"{BASE_DIR}/_2024spring_mni152_brain_masked"      # for example
output_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy"  # for example

# make an output folder, if not existed
os.makedirs(output_dir, exist_ok=True)

saved_count = 0  # conter of saved files

for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz") or filename.endswith(".nii"):
        filepath = os.path.join(input_dir, filename)
        try:
            # read NIfTI files
            img = nib.load(filepath)
            data = img.get_fdata(dtype=np.float32)

            # output file names (extension is .npy）
            base = os.path.splitext(os.path.splitext(filename)[0])[0]
            out_path = os.path.join(output_dir, base + ".npy")

            # save
            np.save(out_path, data)
            saved_count += 1
#            print(f"✅ Saved {out_path} shape={data.shape}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

print(f"Finished! Total .npy files saved: {saved_count}")


# In[34]:


#　show an image and a histogram of each masked.npy file
#　filter 0>

import os
import numpy as np
import matplotlib.pyplot as plt

npy_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy"  # for example

files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
files.sort()  # sort by file nema

for i, filename in enumerate(files, start=1):
    filepath = os.path.join(npy_dir, filename)
    data = np.load(filepath)

    # show the slice of the center of z_axis
    mid_slice = data[:, :, data.shape[2] // 2]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(mid_slice.T, cmap='gray', origin='lower')
    plt.title(f"{filename}\nShape: {data.shape}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.hist(data[data > 0].flatten(), bins=100, color='blue', alpha=0.7)
    plt.title("Voxel Intensity Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


    print(f"[{i}/{len(files)}] Displayed {filename} with shape {data.shape}")


# In[35]:


#　summary of voxels of all masked.npy files
#　filter 0> and <1000

import os
import numpy as np

npy_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy"  # for example

total_voxels = 0
zero_voxels = 0

filtered_voxels = 0  # 0 < voxel < 1000
high_voxels = 0      # voxel >= 1000

for filename in os.listdir(npy_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_dir, filename)
        try:
            data = np.load(filepath)

            total = data.size
            zeros = np.sum(data == 0)
            filtered = np.sum((data > 0) & (data < 1000))
            high = np.sum(data >= 1000)

            total_voxels += total
            zero_voxels += zeros
            filtered_voxels += filtered
            high_voxels += high

        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

# outputs
print(f"zero voxels    : {zero_voxels:,}")
print(f"filtered voxels: {filtered_voxels:,}")
print(f"Total voxels   : {total_voxels:,}")
print(f"0 < voxel < 1000 (percent): {filtered_voxels / total_voxels * 100:.4f}%")
print(f"1000 < voxel     (percent): {high_voxels / total_voxels * 100:.4f}%")


# In[36]:


#　全masked.npy filesのpixel値のヒストグラム
#　filter 0> and <1000

import os
import numpy as np
import matplotlib.pyplot as plt

npy_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy"  # for example
filtered_voxels_all = []

for filename in os.listdir(npy_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(npy_dir, filename)
        try:
            data = np.load(filepath)
            filtered = data[(data > 0) & (data < 1000)]
            filtered_voxels_all.append(filtered.flatten())  # flattenで1次元化
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

# すべての画像のボクセル値を結合
all_filtered_voxels = np.concatenate(filtered_voxels_all)

# ===== 外れ値除去後の統計 =====
mean_val = all_filtered_voxels.mean()
std_val  = all_filtered_voxels.std()
min_val  = all_filtered_voxels.min()
max_val  = all_filtered_voxels.max()
voxel_count = all_filtered_voxels.size

print(f"Voxel count: {voxel_count:,}")
print(f"Mean       : {mean_val:.4f}")
print(f"Std        : {std_val:.4f}")
print(f"Max        : {max_val:.4f}")
print(f"Min        : {min_val:.4f}")

# ===== ヒストグラム =====
plt.figure(figsize=(8, 5))
plt.hist(all_filtered_voxels, bins=200, density=True, color="steelblue", alpha=0.7)
plt.title("Histogram of voxel intensities (outliers removed)")
plt.xlabel("Voxel intensity")
plt.ylabel("Density")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# In[37]:


import os
import nibabel as nib
import numpy as np

# input and output directories
# input_dir = "/mnt/d/T1_2024/_2024spring_mni152_brain_masked"                              # for example
# output_dir = "/mnt/d/T1_2024/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed"  # for example
input_dir =  f"{BASE_DIR}/_2024spring_mni152_brain_masked"
output_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed"

os.makedirs(output_dir, exist_ok=True)

saved_count = 0  # 保存したファイル数カウンタ

# remove teh extreme outliears (±5σ)
sigma_threshold = 5.0

for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz") or filename.endswith(".nii"):
        filepath = os.path.join(input_dir, filename)
        try:
            img = nib.load(filepath)
            data = img.get_fdata(dtype=np.float32)

            mask = (data > 0) & (data < 1000)

            if not np.any(mask):
                print(f"⚠️ Skipping {filename}: No valid voxels.")
                continue

            mean = data[mask].mean()
            std = data[mask].std()
            if std == 0:
                print(f"⚠️ Skipping {filename}: std == 0.")
                continue

            # Z-score satndarzation
            data[mask] = (data[mask] - mean) / std

            # remove outliers
            outlier_mask = (data > sigma_threshold) | (data < -sigma_threshold)
            data[outlier_mask] = 0

            # backgraound = 0
            data[~mask] = 0

            # output file names
            base = os.path.splitext(os.path.splitext(filename)[0])[0]
            out_path = os.path.join(output_dir, base + ".npy")
            np.save(out_path, data)
            saved_count += 1


            valid_after = data[data != 0]

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print(f"Finished! Total .npy files saved: {saved_count}")


# In[38]:


# mask = remove background and outliers, done
# Z-score standardization, done

import os
import numpy as np
import matplotlib.pyplot as plt

# input directory
# input_dir = "/mnt/d/T1_2024/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed" 
# Here, output_dir in the above session is now input!
input_dir = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed"

# list for summary
all_values = []

for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        filepath = os.path.join(input_dir, filename)
        try:
            data = np.load(filepath)
            # Effective voxels are only non zero.
            filtered = data[data != 0]
            all_values.append(filtered.flatten())
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")

# combine all values
all_values = np.concatenate(all_values)

# statsitics
n_voxels = all_values.size
mean_val = all_values.mean()
std_val = all_values.std()
max_val = all_values.max()
min_val = all_values.min()

print(f"Voxel count: {n_voxels:,}")
print(f"Mean       : {mean_val:.4f}")
print(f"Std        : {std_val:.4f}")
print(f"Max        : {max_val:.4f}")
print(f"Min        : {min_val:.4f}")

# histogram
plt.figure(figsize=(8, 5))
plt.hist(all_values, bins=100, color='skyblue', edgecolor='black')
plt.title("Histogram of voxel values (z-score)")
plt.xlabel("Voxel intensity (z-score)")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[39]:


import os
import numpy as np
import matplotlib.pyplot as plt

def load_z44_slices_with_hist(folder_path, z_index=44):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    data = []

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        vol = np.load(file_path) 

        if vol.ndim != 3 or vol.shape[2] <= z_index:
            print(f"❌ Skipping {file} due to invalid shape: {vol.shape}")
            continue

        slice_img = vol[:, :, z_index]  # extract the slice of z_index
        data.append(slice_img)

        valid_voxels = slice_img[slice_img != 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(slice_img.T, origin='lower', cmap='gray')
        axes[0].set_title(f"{i+1:03d}: {file}")
        axes[0].axis('off')

        axes[1].hist(valid_voxels.flatten(), bins=100, color='skyblue', edgecolor='black')
        axes[1].set_title("Pixel value histogram (z-score)")
        axes[1].set_xlabel("Voxel intensity")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    return np.array(data)

folder_path = f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed"  # for example

load_z44_slices_with_hist(folder_path, z_index=44)


# In[ ]:





# In[1]:


import os
import pandas as pd

season_dirs = {
    "spring": f"{BASE_DIR}/_2024spring_mni152_brain_masked_npy_zscore_outlier_removed",
    "summer": f"{BASE_DIR}/_2024summer_mni152_brain_masked_npy_zscore_outlier_removed",
    "autumn": f"{BASE_DIR}/_2024autumn_mni152_brain_masked_npy_zscore_outlier_removed",
    "winter": f"{BASE_DIR}/_2024winter_mni152_brain_masked_npy_zscore_outlier_removed",
}

file_records = []

for season, dir_path in season_dirs.items():
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".npy"):
            file_records.append([season, fname])

df = pd.DataFrame(file_records, columns=["seasons", "files"])

output_csv = f"{BASE_DIR}/mni152_brain_masked_npy_zscore_outlier_removed_list.csv"
df.to_csv(output_csv, index=False)

print(f"✅ finished saving: {output_csv}")


# In[ ]:




