#!/usr/bin/env bash
set -euo pipefail

# Run this from the root of CSSLab/SIREN or doy71/real_siren-compatible repo.
#
# Dataset: nedjmaou/MLMA_hate_speech (Arabic / French / English)
#
# Key differences from a generic MLMA-style invocation:
#   --languages en fr ar       MLMA has Arabic, French, English (NOT Korean)
#   --label_column sentiment   Actual column name in the HF dataset
#   --normal_label normal      MLMA binarization rule: normal→0, everything else→1
#                              (handles compound labels like offensive_disrespectful,
#                              hateful_normal, fearful, disrespectful, etc.)
#   --infer_language           nedjmaou/MLMA_hate_speech has NO language column;
#                              uses Arabic-script detection + langdetect fallback
#
# Prerequisite: pip install langdetect

MODEL=${MODEL:-qwen3-4b}
OUT=${OUT:-"outputs/mlma_lang_selection_${MODEL}"}

python experiments/lang_selection_siren_mlma.py \
  --model "$MODEL" \
  --hf_dataset "nedjmaou/MLMA_hate_speech" \
  --hf_split train \
  --text_column tweet \
  --label_column sentiment \
  --normal_label normal \
  --languages en fr ar \
  --infer_language \
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
