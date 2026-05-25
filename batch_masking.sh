#!/bin/bash

# ====== 設定 ======
ANAT_ROOT="/mnt/d/T1_2024/_2024winter_fsl_anat_outputs"
OUT_DIR="/mnt/d/T1_2024/_2024winter_mni152_brain_masked"
MNI_REF="/home/n-kak/fsl/data/standard/MNI152_T1_2mm.nii.gz"

mkdir -p "$OUT_DIR"

# ====== 全フォルダをループ処理 ======
for folder in "${ANAT_ROOT}"/*_*.anat; do
    subj=$(basename "$folder" .anat)
    echo "▶ Processing: $subj"

    # 入力ファイル（拡張子の違いに対応）
    T1WARP="${folder}/T1_to_MNI_nonlin.nii"
    [[ ! -f "$T1WARP" ]] && T1WARP="${folder}/T1_to_MNI_nonlin.nii.gz"

    MASK_NATIVE="${folder}/T1_biascorr_brain_mask.nii"
    [[ ! -f "$MASK_NATIVE" ]] && MASK_NATIVE="${folder}/T1_biascorr_brain_mask.nii.gz"

#    WARPCOEF="${folder}/T1_biascorr_warpcoef.nii"
#    [[ ! -f "$WARPCOEF" ]] && WARPCOEF="${folder}/T1_biascorr_warpcoef.nii.gz"

    WARPCOEF="${folder}/T1_to_MNI_nonlin_coeff.nii"
    [[ ! -f "$WARPCOEF" ]] && WARPCOEF="${folder}/T1_to_MNI_nonlin_coeff.nii.gz"

    # 中間・出力ファイル
    MASK_MNI="${folder}/${subj}_mask_in_MNI.nii.gz"
    MASKED="${OUT_DIR}/${subj}_masked.nii.gz"

    # 存在チェック
    if [[ ! -f "$T1WARP" || ! -f "$MASK_NATIVE" || ! -f "$WARPCOEF" ]]; then
        echo "  ⚠️ Missing file(s) for $subj, skipping."
        continue
    fi

    # Step 1: 脳マスクをMNI空間にwarp
    if [[ ! -f "$MASK_MNI" ]]; then
        echo "  ⮕ Warping mask to MNI space..."
        applywarp --in="$MASK_NATIVE" --ref="$MNI_REF" --warp="$WARPCOEF" --out="$MASK_MNI" --interp=nn
    fi

    # Step 2: T1（MNI空間画像）にマスクを適用
    if [[ ! -f "$MASKED" ]]; then
        echo "  ⮕ Applying brain mask..."
        fslmaths "$T1WARP" -mas "$MASK_MNI" "$MASKED"
    fi
done

echo "✅ 全処理完了。出力先: $OUT_DIR"
