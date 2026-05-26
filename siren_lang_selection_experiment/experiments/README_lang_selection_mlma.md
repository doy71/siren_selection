# MLMA cross-lingual SIREN selection experiment

This experiment keeps the official SIREN pipeline but changes the **neuron selection source** and retrains a fresh SIREN classifier for each selection strategy.

## What it compares

For each strategy below, the script builds selected-neuron masks, aggregates selected internal dimensions across layers with validation-performance layer weights, then trains a new MLP SIREN classifier on the same pooled EN+KO+FR training data.

- `en_selected`: nonzero neurons from English L1 probes
- `ko_selected`: nonzero neurons from Korean L1 probes
- `fr_selected`: nonzero neurons from French L1 probes
- `pooled_selected`: nonzero neurons from EN+KO+FR pooled L1 probes
- `shared_only`: neurons stably nonzero in EN, KO, and FR probes
- `shared_plus_all_specific`: union of shared neurons and all language-specific neurons
- `routed_shared_specific`: shared block + language-routed specific blocks; requires language id
- `random_same_size_as_pooled`: random baseline with the same per-layer selected count as pooled selection

## Why this is the right comparison

The evaluation target is not the selected neuron set alone. The target is the full **SIREN classifier** trained under each selection strategy. This follows the official SIREN design: layer-wise L1 probes identify safety-relevant dimensions, validation performance gives layer weights, selected internal features are aggregated, and a lightweight MLP classifier is trained on top.

## Required placement

Copy the `experiments/` directory into the root of either:

- `CSSLab/SIREN`, or
- a compatible fork such as `doy71/real_siren` if it preserves the same `train/` and `utils/` import paths.

Expected imports:

```python
from train.probe_trainer import LinearProbe, extract_layer_features
from train.train_general_siren import AdaptiveMLPClassifier
from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor
```

If your fork renamed the extractor, pass for example:

```bash
--extractor_class utils.model_hooks.YourExtractorClass
```

## Run

Edit `MLMA_DATASET`, column names, and model name in `run_lang_selection_mlma.sh`, then:

```bash
bash experiments/run_lang_selection_mlma.sh
```

Or run directly:

```bash
python experiments/lang_selection_siren_mlma.py \
  --model qwen3-4b \
  --hf_dataset YOUR_MLMA_HF_DATASET_ID \
  --text_column text \
  --label_column label \
  --language_column language \
  --languages en ko fr \
  --pooling_types residual_mean \
  --max_per_lang 1000 \
  --seeds 1 2 3 \
  --output_dir outputs/mlma_lang_selection_qwen
```

For a local MLMA CSV/JSONL instead of Hugging Face:

```bash
python experiments/lang_selection_siren_mlma.py \
  --model qwen3-4b \
  --local_file /path/to/mlma.csv \
  --text_column text \
  --label_column label \
  --language_column language \
  --languages en ko fr \
  --output_dir outputs/mlma_lang_selection_qwen
```

## Outputs

Inside `output_dir`:

- `representations.pkl`: cached internal representations
- `probe_runs.pkl`: cached L1 probe results
- `<pooling_type>/selection_summary.csv`: selected neuron counts per strategy/layer
- `<pooling_type>/overlap_summary.csv`: observed vs expected-random overlap
- `<pooling_type>/classifier_results.csv`: main comparison table
- `<pooling_type>/classifier_results.json`: full metrics
- `<pooling_type>/models/*.pt`: saved classifier heads if `--save_models` is set

## Main metrics to report

Use `classifier_results.csv` and focus on:

- `macro_over_languages_f1`
- `worst_language_macro_f1`
- `language_gap_macro_f1`
- per-language macro-F1 and AUROC

Interpretation:

- `shared_only` strong → evidence for a compact language-neutral safety core.
- `pooled_selected` strong → multilingual selection helps even without strict shared neurons.
- `routed_shared_specific` or `shared_plus_all_specific` strongest → safety signal likely has both shared and language-specific components.
- `random_same_size_as_pooled` close to selected strategies → current selection criterion is weak or overly broad.

## Important notes

- `nonzero` selection is maintained by default as `abs(weight) > --nonzero_eps`.
- Stability comes from repeated seeds: a dimension is stable if its nonzero frequency is `>= --stability_tau`.
- Language-specific dimensions are selected when a dimension is stable in one language and below `--specific_tau_low` in the others.
- The script uses a fallback of at least one shared neuron per layer to keep the experiment runnable; check `selection_summary.csv` to ensure this fallback is not dominating.
