#!/bin/bash

# ====== 設定 ======
INPUT_DIR="/mnt/d/T1_2024/_2024spring_nii"
OUTPUT_DIR="/mnt/d/T1_2024/_2024spring_fsl_anat_outputs"
LOG_FILE="${OUTPUT_DIR}/fsl_anat_parallel.log"
PARALLEL_JOBS=16

mkdir -p "$OUTPUT_DIR"
echo "=== fsl_anat parallel processing started: $(date) ===" | tee -a "$LOG_FILE"

# EXPORTしてparallel内で参照可能にする
export OUTPUT_DIR LOG_FILE

# ====== 並列処理 ======
ls "${INPUT_DIR}"/*.nii.gz | parallel -j $PARALLEL_JOBS '
    base=$(basename {} .nii.gz)
    out_folder="${OUTPUT_DIR}/${base}"

    if [ -d "${out_folder}.anat" ]; then
        echo "▶ Skipping $base (already processed)" | tee -a "$LOG_FILE"
        continue
    fi

    echo "▶ Processing $base" | tee -a "$LOG_FILE"
    fsl_anat -i {} -o "$out_folder" | tee -a "$LOG_FILE"
'

echo "=== fsl_anat parallel processing finished: $(date) ===" | tee -a "$LOG_FILE"
