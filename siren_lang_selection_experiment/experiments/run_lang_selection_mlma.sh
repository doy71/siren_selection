#!/usr/bin/env bash
set -euo pipefail

# Run this from the root of CSSLab/SIREN or doy71/real_siren-compatible repo.
# Edit MLMA_DATASET and column names to match the exact MLMA source you used before.

MODEL=${MODEL:-qwen3-4b}
MLMA_DATASET=${MLMA_DATASET:-"YOUR_MLMA_HF_DATASET_ID"}
OUT=${OUT:-"outputs/mlma_lang_selection_${MODEL}"}

python experiments/lang_selection_siren_mlma.py \
  --model "$MODEL" \
  --hf_dataset "$MLMA_DATASET" \
  --hf_split train \
  --text_column text \
  --label_column label \
  --language_column language \
  --languages en ko fr \
  --pooling_types residual_mean \
  --max_per_lang 1000 \
  --extract_batch_size 16 \
  --seeds 1 2 3 \
  --c_values 50 100 200 \
  --stability_tau 0.6 \
  --specific_tau_low 0.3 \
  --mlp_hidden_dims 512 256 \
  --mlp_epochs 100 \
  --mlp_patience 10 \
  --output_dir "$OUT" \
  --save_models
