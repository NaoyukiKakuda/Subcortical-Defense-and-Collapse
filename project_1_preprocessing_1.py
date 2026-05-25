#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pydicom


# In[7]:


# Covert DICOM to NII
# In this exprimental analysis, "/mnt/d" and "/mnt/e" are not fixed, but changeable.

import os
import pydicom
import shutil
import subprocess
from project_config import BASE_DIR, JACOBIAN_DIR, FIGURE_DIR, AAL_LABEL_TXT, AAL_LABEL_NII

# format accessible through WSL
dcm_root = f"{BASE_DIR}/_2024spring"      # for example
nii_root = f"{BASE_DIR}/_2024spring_nii"  # for example

# Windows → WSL path transformation
def windows_to_wsl_path(win_path):
    return "/mnt/" + win_path[0].lower() + win_path[2:].replace("\\", "/")

os.makedirs(nii_root, exist_ok=True)

for subject in os.listdir(dcm_root):
    subject_path = os.path.join(dcm_root, subject)
    if not os.path.isdir(subject_path):
        continue

    dcm_files = [f for f in os.listdir(subject_path) if f.lower().endswith(".dcm")]
    if not dcm_files:
        continue

    dcm_path = os.path.join(subject_path, dcm_files[0])
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    birth = dcm.get("PatientBirthDate", "unknown")
    sex = dcm.get("PatientSex", "X")
    patient_id = dcm.get("PatientID", subject)

    tmp_output = os.path.join(nii_root, "tmp")
    os.makedirs(tmp_output, exist_ok=True)

    # dcm2niix through WSL
    wsl_subject_path = subject_path
    wsl_tmp_output = tmp_output

    subprocess.run(["dcm2niix", "-z", "y", "-o", wsl_tmp_output, wsl_subject_path])

    # rename NIfTI files
    for file in os.listdir(tmp_output):
        if file.endswith(".nii.gz"):
            old_path = os.path.join(tmp_output, file)
            new_name = f"{patient_id}_{birth}_{sex}.nii.gz"
            new_path = os.path.join(nii_root, new_name)
            shutil.move(old_path, new_path)

    # clear tmp folder
    for f in os.listdir(tmp_output):
        os.remove(os.path.join(tmp_output, f))

# remove tmpfolder
os.rmdir(tmp_output)

print("conversion to nii finished")


# In[ ]:




