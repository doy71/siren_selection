#!/usr/bin/env python3
"""
Cross-lingual SIREN selection experiment on MLMA-style multilingual data.

Goal
----
Compare SIREN classifiers trained with different neuron-selection sources:
  - en_selected
  - ko_selected
  - fr_selected
  - pooled_selected
  - shared_only
  - shared_plus_all_specific
  - routed_shared_specific
  - random_same_size_as_pooled

This script is designed to be placed under the repository root of CSSLab/SIREN
or a compatible fork such as real_siren, then run as:

  python experiments/lang_selection_siren_mlma.py \
    --model qwen3-4b \
    --hf_dataset <YOUR_MLMA_HF_DATASET_ID> \
    --text_column text --label_column label --language_column language \
    --languages en ko fr \
    --pooling_type residual_mean \
    --output_dir outputs/mlma_lang_selection_qwen \
    --seeds 1 2 3

Notes
-----
1. Official SIREN identifies safety-relevant dimensions with layer-wise L1 probes,
   computes validation-performance-based layer weights, aggregates selected internal
   dimensions across layers, and trains an MLP classifier on top.
2. This script keeps that structure, but changes the selection source and retrains
   a fresh SIREN classifier for each selection strategy.
3. `nonzero` selection is kept by default: abs(weight) > --nonzero_eps.

Expected repo imports
---------------------
From CSSLab/SIREN:
  - utils.config.MODEL_CONFIGS
  - utils.model_hooks.Qwen3RepresentationExtractor
  - train.probe_trainer.LinearProbe, extract_layer_features
  - train.train_general_siren.AdaptiveMLPClassifier

If your fork renamed the extractor, pass --extractor_class or edit `build_extractor`.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import hypergeom
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make the repository root importable when this file lives in ./experiments.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.probe_trainer import LinearProbe, extract_layer_features  # noqa: E402
from train.train_general_siren import AdaptiveMLPClassifier  # noqa: E402
from utils.config import MODEL_CONFIGS  # noqa: E402


LANG_ALIASES = {
    "en": "en", "eng": "en", "english": "en",
    "ko": "ko", "kor": "ko", "korean": "ko", "kr": "ko",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "ar": "ar", "ara": "ar", "arabic": "ar",
}

DEFAULT_UNSAFE_LABELS = {
    "1", "true", "yes", "y", "unsafe", "harmful", "hate", "hateful",
    "offensive", "abusive", "toxic", "toxicity", "dangerous", "attack",
}

TEXT_COLUMN_CANDIDATES = ["text", "tweet", "comment", "sentence", "content", "body"]
LABEL_COLUMN_CANDIDATES = ["label", "labels", "class", "category", "annotation", "hate", "hateful", "sentiment"]
LANG_COLUMN_CANDIDATES = ["lang", "language", "locale"]


@dataclass
class SplitData:
    texts: List[str]
    labels: np.ndarray
    langs: np.ndarray
    dataset_ids: np.ndarray


@dataclass
class ProbeRun:
    source: str
    seed: int
    pooling_type: str
    layer_idx: int
    best_C: float
    train_f1: float
    val_f1: float
    test_f1: float
    weights_abs: np.ndarray
    nonzero_indices: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_lang(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return LANG_ALIASES.get(s, s)


def auto_column(columns: Sequence[str], candidates: Sequence[str], explicit: Optional[str], kind: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{kind} column '{explicit}' not found. Available columns: {columns}")
        return explicit
    lower_to_col = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    raise ValueError(f"Could not auto-detect {kind} column. Available columns: {columns}")


def normalize_label(value: Any, unsafe_labels: set[str], normal_label: Optional[str] = None) -> int:
    """Map MLMA-style labels to binary 0=safe/non-hate, 1=unsafe/hate.

    Two binarization modes:
    - ``normal_label`` mode (recommended for MLMA): any value that is NOT
      exactly equal to ``normal_label`` is treated as unsafe (1).  This
      correctly handles compound labels such as "offensive_disrespectful",
      "hateful_normal", "disrespectful", "fearful", etc., all of which the
      MLMA paper considers toxic.  Use ``--normal_label normal``.
    - ``unsafe_set`` mode (default): a value is unsafe iff it appears in
      ``unsafe_labels``.  Suitable for datasets with clean binary string labels.
    """
    if value is None:
        raise ValueError("Label is None")
    if isinstance(value, (int, np.integer)):
        return int(value > 0)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            raise ValueError("Label is NaN")
        return int(float(value) > 0.0)
    if isinstance(value, (list, tuple, np.ndarray)):
        # Multi-label case: any positive / unsafe-like element means unsafe.
        return int(any(normalize_label(v, unsafe_labels, normal_label) == 1 for v in value))
    s = str(value).strip().lower()
    if normal_label is not None:
        # MLMA rule: exactly the safe label → 0, everything else → 1.
        return 0 if s == normal_label else 1
    return int(s in unsafe_labels)


import re as _re
_AR_SCRIPT = _re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')


def infer_language(text: str, candidates: Sequence[str]) -> Optional[str]:
    """Heuristically detect the language of *text* and return a normalized code.

    Strategy (ordered by reliability):
    1. Arabic-script character detection — fast, near-perfect for Arabic.
    2. ``langdetect`` fallback — covers Latin-script language pairs like
       French vs English.  Requires ``pip install langdetect``.

    Returns the normalized language code (via :func:`normalize_lang`) if it
    is in *candidates*, otherwise ``None``.
    """
    if "ar" in candidates and _AR_SCRIPT.search(text):
        return "ar"
    try:
        from langdetect import detect, LangDetectException
        try:
            detected = normalize_lang(detect(text))
            return detected if detected in candidates else None
        except LangDetectException:
            return None
    except ImportError:
        raise ImportError(
            "langdetect is required for --infer_language.  "
            "Install it with: pip install langdetect"
        )


def load_mlma_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load MLMA-style data from HF or local CSV/JSON/JSONL."""
    if args.local_file:
        suffix = Path(args.local_file).suffix.lower()
        if suffix == ".csv":
            ds = load_dataset("csv", data_files=args.local_file, split="train")
        elif suffix in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=args.local_file, split="train")
        else:
            raise ValueError(f"Unsupported local_file suffix: {suffix}. Use CSV/JSON/JSONL.")
    else:
        if not args.hf_dataset:
            raise ValueError("Provide --hf_dataset or --local_file for MLMA data.")
        load_kwargs = {}
        if args.hf_config:
            ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, **load_kwargs)
        else:
            ds = load_dataset(args.hf_dataset, split=args.hf_split, **load_kwargs)

    columns = list(ds.column_names)
    text_col = auto_column(columns, TEXT_COLUMN_CANDIDATES, args.text_column, "text")
    label_col = auto_column(columns, LABEL_COLUMN_CANDIDATES, args.label_column, "label")

    # Language column: optional when --infer_language is set.
    has_lang_col = True
    try:
        lang_col = auto_column(columns, LANG_COLUMN_CANDIDATES, args.language_column, "language")
    except ValueError:
        if not args.infer_language:
            raise ValueError(
                "No language column found in the dataset and --infer_language is not set. "
                "For nedjmaou/MLMA_hate_speech (which has no language column), add "
                "--infer_language to the command.  Alternatively, pass --language_column "
                "with the correct column name if your dataset has one."
            )
        lang_col = None
        has_lang_col = False

    unsafe_labels = {x.strip().lower() for x in args.unsafe_labels.split(",") if x.strip()}
    normal_label = args.normal_label.strip().lower() if args.normal_label else None

    wanted_langs = [normalize_lang(x) for x in args.languages]
    wanted_langs = [x for x in wanted_langs if x is not None]

    skipped_lang = 0
    records = []
    for row in ds:
        # ── Language ──────────────────────────────────────────────────────────
        if has_lang_col:
            lang = normalize_lang(row.get(lang_col))
        else:
            text_raw = str(row.get(text_col, "")).strip()
            lang = infer_language(text_raw, wanted_langs)
        if lang not in wanted_langs:
            skipped_lang += 1
            continue

        # ── Text ──────────────────────────────────────────────────────────────
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue

        # ── Label ─────────────────────────────────────────────────────────────
        try:
            label = normalize_label(row.get(label_col), unsafe_labels, normal_label)
        except Exception:
            continue

        records.append({"text": text, "label": int(label), "lang": lang})

    if skipped_lang and not has_lang_col:
        print(
            f"[warn] infer_language: {skipped_lang} rows dropped "
            f"(detected language not in {wanted_langs} or detection failed)."
        )
    if not records:
        raise ValueError(
            "No records left after filtering.  "
            "Check --languages, --text_column, --label_column.  "
            "If using nedjmaou/MLMA_hate_speech, make sure you set "
            "--languages en fr ar --label_column sentiment --normal_label normal "
            "--infer_language."
        )
    return records


def balanced_subsample(records: List[Dict[str, Any]], max_per_lang: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if not max_per_lang or max_per_lang <= 0:
        return records
    rng = np.random.RandomState(seed)
    out = []
    for lang in sorted(set(r["lang"] for r in records)):
        lang_records = [r for r in records if r["lang"] == lang]
        # Try to preserve label balance within the chosen cap.
        # Track by list index to avoid id() aliasing issues.
        chosen_indices: set[int] = set()
        per_label_cap = max(1, max_per_lang // 2)
        for label in [0, 1]:
            cls_indices = [i for i, r in enumerate(lang_records) if r["label"] == label]
            if len(cls_indices) > per_label_cap:
                picked = rng.choice(len(cls_indices), size=per_label_cap, replace=False)
                chosen_indices.update(cls_indices[p] for p in picked)
            else:
                chosen_indices.update(cls_indices)
        # Fill up to max_per_lang from the remainder if still short.
        target = min(max_per_lang, len(lang_records))
        if len(chosen_indices) < target:
            rest_indices = [i for i in range(len(lang_records)) if i not in chosen_indices]
            need = target - len(chosen_indices)
            if rest_indices and need > 0:
                picked = rng.choice(len(rest_indices), size=min(need, len(rest_indices)), replace=False)
                chosen_indices.update(rest_indices[p] for p in picked)
        chosen = [lang_records[i] for i in sorted(chosen_indices)]
        rng.shuffle(chosen)
        out.extend(chosen[:max_per_lang])
    rng.shuffle(out)
    return out


def split_records_by_language(
    records: List[Dict[str, Any]],
    languages: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    val_seed: Optional[int] = None,
) -> Dict[str, SplitData]:
    """Create stratified train/validation/test splits inside each language.

    Args:
        val_seed: Random state for the val/test sub-split. Defaults to seed + 17
            if not provided. Kept as an explicit argument so experiments are
            fully reproducible without relying on an implicit arithmetic offset.
    """
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1
    if val_seed is None:
        val_seed = seed + 17
    split_records: Dict[str, list] = {"train": [], "validation": [], "test": []}
    rng = np.random.RandomState(seed)

    for lang in languages:
        lang = normalize_lang(lang)
        items = [r for r in records if r["lang"] == lang]
        if len(items) < 10:
            raise ValueError(f"Too few records for language={lang}: {len(items)}")
        labels = np.array([r["label"] for r in items])
        stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels)) >= 2 else None
        idx_all = np.arange(len(items))
        idx_train, idx_tmp = train_test_split(
            idx_all,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        if val_ratio == 0.0:
            # No validation split requested: val gets nothing, test gets idx_tmp.
            idx_val: np.ndarray = np.array([], dtype=np.int64)
            idx_test: np.ndarray = idx_tmp
        else:
            tmp_labels = labels[idx_tmp]
            test_ratio_within_tmp = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
            stratify_tmp = (
                tmp_labels
                if len(np.unique(tmp_labels)) == 2 and min(np.bincount(tmp_labels)) >= 2
                else None
            )
            idx_val, idx_test = train_test_split(
                idx_tmp,
                test_size=test_ratio_within_tmp,
                random_state=val_seed,
                shuffle=True,
                stratify=stratify_tmp,
            )
        for split_name, idxs in [("train", idx_train), ("validation", idx_val), ("test", idx_test)]:
            split_records[split_name].extend([items[int(i)] for i in idxs])

    out = {}
    lang_to_id = {normalize_lang(l): i for i, l in enumerate(languages)}
    for split_name, items in split_records.items():
        rng.shuffle(items)
        out[split_name] = SplitData(
            texts=[r["text"] for r in items],
            labels=np.array([r["label"] for r in items], dtype=np.int64),
            langs=np.array([r["lang"] for r in items]),
            dataset_ids=np.array([lang_to_id[r["lang"]] for r in items], dtype=np.int64),
        )
    return out


def build_extractor(model_name: str, device: str, batch_size: int, rep_types: Sequence[str], extractor_class: str):
    """Build the representation extractor.

    By default, this uses utils.model_hooks.Qwen3RepresentationExtractor, which is what
    the official training script imports. If your fork has a Llama-specific extractor,
    pass e.g. --extractor_class utils.model_hooks.LlamaRepresentationExtractor.
    """
    model_config = MODEL_CONFIGS[model_name]
    if "." in extractor_class:
        module_name, cls_name = extractor_class.rsplit(".", 1)
    else:
        module_name, cls_name = "utils.model_hooks", extractor_class
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(
        model_config["model_path"],
        device=device,
        batch_size=batch_size,
        rep_types=list(rep_types),
    )


def extract_or_load_representations(
    args: argparse.Namespace,
    splits: Dict[str, SplitData],
    output_dir: Path,
) -> Dict[str, Any]:
    cache_path = output_dir / "representations.pkl"
    if args.reuse_representations and cache_path.exists():
        print(f"[cache] Loading representations from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[1/4] Extracting internal representations...")
    extractor = build_extractor(
        args.model,
        args.device,
        args.extract_batch_size,
        args.pooling_types,
        args.extractor_class,
    )
    extractor.register_hooks()

    all_reps: Dict[str, Any] = {}
    for split_name, split in splits.items():
        reps = []
        print(f"  - {split_name}: {len(split.texts)} samples")
        for i in tqdm(range(0, len(split.texts), args.extract_batch_size), desc=f"extract {split_name}"):
            batch_texts = split.texts[i:i + args.extract_batch_size]
            with torch.no_grad():
                batch_reps = extractor.extract_batch(batch_texts)
            reps.extend(batch_reps)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_reps[split_name] = {
            "representations": reps,
            "labels": split.labels,
            "langs": split.langs,
            "dataset_ids": split.dataset_ids,
            "texts": split.texts if args.save_texts else None,
            "num_layers": MODEL_CONFIGS[args.model]["num_layers"],
        }

    extractor.remove_hooks()
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open(cache_path, "wb") as f:
        pickle.dump(all_reps, f)
    print(f"[cache] Saved representations to {cache_path}")
    return all_reps


def source_mask(langs: np.ndarray, source: str) -> np.ndarray:
    if source == "pooled":
        return np.ones(len(langs), dtype=bool)
    return langs == source


def safe_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_one_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_dataset_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dataset_ids: np.ndarray,
    c_values: Sequence[float],
    device: str,
    seed: int,
    batch_size: int,
) -> Tuple[LinearProbe, float, float, float, float]:
    best_C = None
    best_val = -1.0
    for C in c_values:
        probe = LinearProbe(C=C, penalty="l1", device=device, batch_size=batch_size)
        probe.train(X_train, y_train, X_val, y_val, val_dataset_ids, quick_eval=True, random_seed=seed)
        val_score = probe.evaluate(X_val, y_val, val_dataset_ids, metric="f1_macro")
        if val_score > best_val:
            best_val = float(val_score)
            best_C = float(C)
    if best_C is None:
        raise RuntimeError("Failed to choose C for probe")
    final_probe = LinearProbe(C=best_C, penalty="l1", device=device, batch_size=batch_size)
    final_probe.train(X_train, y_train, X_val, y_val, val_dataset_ids, quick_eval=False, random_seed=seed)
    train_f1 = final_probe.evaluate(X_train, y_train, metric="f1_macro")
    val_f1 = final_probe.evaluate(X_val, y_val, val_dataset_ids, metric="f1_macro")
    test_f1 = final_probe.evaluate(X_test, y_test, test_dataset_ids, metric="f1_macro")
    return final_probe, float(train_f1), float(val_f1), float(test_f1), float(best_C)


def train_selection_probes(args: argparse.Namespace, all_reps: Dict[str, Any], output_dir: Path) -> List[ProbeRun]:
    cache_path = output_dir / "probe_runs.pkl"
    if args.reuse_probes and cache_path.exists():
        print(f"[cache] Loading probe runs from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[2/4] Training layer-wise L1 probes for selection sources...")
    num_layers = all_reps["train"]["num_layers"]
    sources = [normalize_lang(s) for s in args.languages] + ["pooled"]
    probe_runs: List[ProbeRun] = []

    for pooling_type in args.pooling_types:
        rep_type, pooling = pooling_type.split("_")[0], "_".join(pooling_type.split("_")[1:])

        # --- FIX: hoist extract_layer_features outside the source loop so we
        #     compute each (layer, pooling_type) feature matrix only ONCE instead
        #     of once per source.  This reduces calls from O(n_layers × n_sources)
        #     to O(n_layers), which is ~4× faster for 3 languages + pooled.
        for layer_idx in range(num_layers):
            X_all: Dict[str, np.ndarray] = {}
            y_all: Dict[str, np.ndarray] = {}
            langs_all: Dict[str, np.ndarray] = {}
            dataset_ids_all: Dict[str, np.ndarray] = {}
            for split_name in ("train", "validation", "test"):
                X_all[split_name] = extract_layer_features(
                    all_reps[split_name]["representations"], layer_idx, rep_type, pooling
                )
                y_all[split_name] = all_reps[split_name]["labels"]
                langs_all[split_name] = all_reps[split_name]["langs"]
                dataset_ids_all[split_name] = all_reps[split_name]["dataset_ids"]

            for source in sources:
                print(f"\n  layer={layer_idx:02d} source={source}, pooling={pooling_type}")
                train_m = source_mask(langs_all["train"], source)
                val_m = source_mask(langs_all["validation"], source)
                test_m = source_mask(langs_all["test"], source)
                if train_m.sum() < 10 or val_m.sum() < 4:
                    print(f"    [skip] too few samples for source={source}")
                    continue

                X_train = X_all["train"][train_m]
                y_train = y_all["train"][train_m]
                X_val = X_all["validation"][val_m]
                y_val = y_all["validation"][val_m]
                val_dataset_ids = dataset_ids_all["validation"][val_m]
                X_test = X_all["test"][test_m]
                y_test = y_all["test"][test_m]
                test_dataset_ids = dataset_ids_all["test"][test_m]

                for seed in args.seeds:
                    set_seed(seed)
                    probe, train_f1, val_f1, test_f1, best_C = train_one_probe(
                        X_train, y_train, X_val, y_val, val_dataset_ids,
                        X_test, y_test, test_dataset_ids,
                        c_values=args.c_values,
                        device=args.device,
                        seed=seed,
                        batch_size=args.probe_batch_size,
                    )
                    weights_abs = probe.get_feature_importance()
                    nz = np.flatnonzero(weights_abs > args.nonzero_eps)
                    if len(nz) == 0:
                        nz = np.array([int(np.argmax(weights_abs))], dtype=np.int64)
                    probe_runs.append(ProbeRun(
                        source=str(source),
                        seed=int(seed),
                        pooling_type=pooling_type,
                        layer_idx=int(layer_idx),
                        best_C=float(best_C),
                        train_f1=float(train_f1),
                        val_f1=float(val_f1),
                        test_f1=float(test_f1),
                        weights_abs=weights_abs.astype(np.float32),
                        nonzero_indices=nz.astype(np.int64),
                    ))
                    print(
                        f"    seed={seed}: "
                        f"val_f1={val_f1:.4f} test_f1={test_f1:.4f} "
                        f"C={best_C} nonzero={len(nz)}/{weights_abs.shape[0]}"
                    )
                    # Explicit cleanup to avoid GPU memory accumulation across
                    # the many (layer, source, seed) iterations.
                    del probe
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    with open(cache_path, "wb") as f:
        pickle.dump(probe_runs, f)
    print(f"[cache] Saved probe runs to {cache_path}")
    return probe_runs


def group_probe_runs(probe_runs: List[ProbeRun], pooling_type: str) -> Dict[Tuple[str, int], List[ProbeRun]]:
    grouped: Dict[Tuple[str, int], List[ProbeRun]] = {}
    for r in probe_runs:
        if r.pooling_type != pooling_type:
            continue
        grouped.setdefault((r.source, r.layer_idx), []).append(r)
    return grouped


def frequency_mask(runs: List[ProbeRun], tau: float, hidden_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    if not runs:
        raise ValueError("No probe runs")
    if hidden_dim is None:
        hidden_dim = runs[0].weights_abs.shape[0]
    freq = np.zeros(hidden_dim, dtype=np.float32)
    for r in runs:
        freq[r.nonzero_indices] += 1.0
    freq /= float(len(runs))
    selected = np.flatnonzero(freq >= tau).astype(np.int64)
    if len(selected) == 0:
        # Fallback: choose the most stable/highest-frequency dimension.
        selected = np.array([int(np.argmax(freq))], dtype=np.int64)
    val_score = float(np.mean([r.val_f1 for r in runs]))
    return selected, freq, val_score


def minmax_layer_weights(layer_scores: Dict[int, float], floor: float = 0.1) -> Dict[int, float]:
    """Normalize per-layer validation scores to the range [floor, 1.0].

    When all layers have the same validation score (mn == mx), min-max
    normalization would produce 0 for every layer, so the floor would apply
    and all weights would be ``floor`` (e.g. 0.1). That is semantically wrong:
    if all layers perform identically there is no reason to down-weight any of
    them, so we return weight = 1.0 for all layers in that case.
    """
    if not layer_scores:
        return {}
    values = np.array(list(layer_scores.values()), dtype=np.float32)
    mn, mx = float(values.min()), float(values.max())
    if mx == mn:
        # All layers perform identically; give every layer full weight.
        return {int(k): 1.0 for k in layer_scores}
    denom = mx - mn
    return {int(k): float(max(floor, (v - mn) / denom)) for k, v in layer_scores.items()}


def build_selection_masks(
    args: argparse.Namespace,
    probe_runs: List[ProbeRun],
    pooling_type: str,
    num_layers: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[int, float]], Dict[str, Any]]:
    """Return standard masks and routed masks.

    standard_masks[strategy][key] = selected neuron indices for key='layer{idx}_{pooling_type}'
    layer_weights[strategy][layer_idx] = performance-derived layer weight.
    metadata contains frequencies and language-specific components.
    """
    grouped = group_probe_runs(probe_runs, pooling_type)
    langs = [normalize_lang(l) for l in args.languages]
    masks: Dict[str, Dict[str, np.ndarray]] = {}
    layer_weights: Dict[str, Dict[int, float]] = {}
    metadata: Dict[str, Any] = {"freq": {}, "specific": {}, "shared": {}}

    # Per-source stable nonzero masks.
    for source in langs + ["pooled"]:
        source_masks: Dict[str, np.ndarray] = {}
        scores: Dict[int, float] = {}
        for layer_idx in range(num_layers):
            runs = grouped.get((source, layer_idx), [])
            if not runs:
                continue
            selected, freq, val_score = frequency_mask(runs, args.stability_tau)
            key = f"layer{layer_idx}_{pooling_type}"
            source_masks[key] = selected
            scores[layer_idx] = val_score
            metadata["freq"][(source, layer_idx)] = freq
        masks[f"{source}_selected"] = source_masks
        layer_weights[f"{source}_selected"] = minmax_layer_weights(scores, floor=args.layer_weight_floor)

    # Shared-only: stable in all languages.
    shared_masks: Dict[str, np.ndarray] = {}
    shared_scores: Dict[int, float] = {}
    specifics: Dict[str, Dict[str, np.ndarray]] = {lang: {} for lang in langs}
    for layer_idx in range(num_layers):
        freqs = {}
        dim = None
        for lang in langs:
            freq = metadata["freq"].get((lang, layer_idx))
            if freq is None:
                continue
            freqs[lang] = freq
            dim = len(freq)
        if len(freqs) != len(langs) or dim is None:
            continue
        shared_bool = np.ones(dim, dtype=bool)
        for lang in langs:
            shared_bool &= freqs[lang] >= args.stability_tau
        shared = np.flatnonzero(shared_bool).astype(np.int64)
        if len(shared) == 0:
            # Keep the experiment runnable; pick dimensions with highest min frequency.
            min_freq = np.minimum.reduce([freqs[lang] for lang in langs])
            k = max(1, args.shared_fallback_topk)
            shared = np.argsort(min_freq)[::-1][:k].astype(np.int64)
        key = f"layer{layer_idx}_{pooling_type}"
        shared_masks[key] = shared
        metadata["shared"][layer_idx] = shared
        # Average language probe validation score for shared layer weighting.
        vals = []
        for lang in langs:
            vals.extend([r.val_f1 for r in grouped.get((lang, layer_idx), [])])
        shared_scores[layer_idx] = float(np.mean(vals)) if vals else 0.0

        for lang in langs:
            this = freqs[lang] >= args.stability_tau
            others_max = np.maximum.reduce([freqs[o] for o in langs if o != lang])
            spec = np.flatnonzero(this & (others_max < args.specific_tau_low)).astype(np.int64)
            specifics[lang][key] = spec

    masks["shared_only"] = shared_masks
    layer_weights["shared_only"] = minmax_layer_weights(shared_scores, floor=args.layer_weight_floor)
    metadata["specific"] = specifics

    # Single-classifier non-routed approximation: shared + all language-specific features.
    shared_all: Dict[str, np.ndarray] = {}
    for key, shared in shared_masks.items():
        pieces = [shared]
        for lang in langs:
            pieces.append(specifics[lang].get(key, np.array([], dtype=np.int64)))
        union = np.unique(np.concatenate([p for p in pieces if len(p) > 0])).astype(np.int64)
        if len(union) == 0:
            union = shared
        shared_all[key] = union
    masks["shared_plus_all_specific"] = shared_all
    layer_weights["shared_plus_all_specific"] = layer_weights["shared_only"]

    # Random same-size baseline against pooled selection.
    rng = np.random.RandomState(args.random_seed)
    random_masks: Dict[str, np.ndarray] = {}
    pooled = masks.get("pooled_selected", {})
    for key, idx in pooled.items():
        # Find hidden dim from corresponding frequency.
        layer_idx = int(key.split("_")[0].replace("layer", ""))
        freq = metadata["freq"].get(("pooled", layer_idx))
        if freq is None:
            continue
        k = len(idx)
        random_masks[key] = np.sort(rng.choice(len(freq), size=k, replace=False)).astype(np.int64)
    masks["random_same_size_as_pooled"] = random_masks
    layer_weights["random_same_size_as_pooled"] = layer_weights.get("pooled_selected", {})

    return masks, layer_weights, metadata


def aggregate_standard(
    representations: Sequence[Any],
    pooling_type: str,
    selected_neurons: Dict[str, np.ndarray],
    weights: Dict[int, float],
    selected_layers: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if selected_layers is None:
        selected_layers = sorted(weights.keys())
    features = []
    for rep in representations:
        sample = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            if key not in selected_neurons:
                continue
            idx = selected_neurons[key]
            if len(idx) == 0:
                continue
            sample.append(rep[layer_idx][pooling_type][idx] * float(weights.get(layer_idx, 1.0)))
        if not sample:
            raise ValueError("No features selected for at least one sample. Check masks/layer weights.")
        features.append(np.concatenate(sample))
    return np.asarray(features, dtype=np.float32)


def aggregate_routed_shared_specific(
    representations: Sequence[Any],
    langs: np.ndarray,
    pooling_type: str,
    shared_masks: Dict[str, np.ndarray],
    specific_masks: Dict[str, Dict[str, np.ndarray]],
    weights: Dict[int, float],
    language_order: Sequence[str],
) -> np.ndarray:
    selected_layers = sorted(weights.keys())
    out = []
    for rep, lang in zip(representations, langs):
        sample = []
        for layer_idx in selected_layers:
            key = f"layer{layer_idx}_{pooling_type}"
            w = float(weights.get(layer_idx, 1.0))
            shared_idx = shared_masks.get(key, np.array([], dtype=np.int64))
            if len(shared_idx) > 0:
                sample.append(rep[layer_idx][pooling_type][shared_idx] * w)
            # Fixed block layout: EN-specific, KO-specific, FR-specific; only own block filled.
            for block_lang in language_order:
                spec_idx = specific_masks.get(block_lang, {}).get(key, np.array([], dtype=np.int64))
                if len(spec_idx) == 0:
                    continue
                if lang == block_lang:
                    sample.append(rep[layer_idx][pooling_type][spec_idx] * w)
                else:
                    sample.append(np.zeros(len(spec_idx), dtype=np.float32))
        if not sample:
            raise ValueError("No routed features selected. Check shared/specific masks.")
        out.append(np.concatenate(sample))
    return np.asarray(out, dtype=np.float32)


def eval_classifier(model: nn.Module, X: np.ndarray, y: np.ndarray, langs: np.ndarray, device: torch.device, batch_size: int) -> Dict[str, Any]:
    model.eval()
    probs_1 = []
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(bx)
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs_1.extend(p.tolist())
            preds.extend((p >= 0.5).astype(int).tolist())
    probs = np.asarray(probs_1, dtype=np.float32)
    pred = np.asarray(preds, dtype=np.int64)
    metrics: Dict[str, Any] = {}
    metrics["overall"] = binary_metrics(y, pred, probs)
    for lang in sorted(set(langs.tolist())):
        m = langs == lang
        metrics[f"lang_{lang}"] = binary_metrics(y[m], pred[m], probs[m])
    f1s = [metrics[f"lang_{l}"]["macro_f1"] for l in sorted(set(langs.tolist()))]
    metrics["language_gap_macro_f1"] = float(np.nanmax(f1s) - np.nanmin(f1s)) if f1s else float("nan")
    metrics["worst_language_macro_f1"] = float(np.nanmin(f1s)) if f1s else float("nan")
    metrics["macro_over_languages_f1"] = float(np.nanmean(f1s)) if f1s else float("nan")
    return metrics


def binary_metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else float("nan"),
        "binary_precision": float(p),
        "binary_recall": float(r),
        "binary_f1": float(f1),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "pred_positive_rate": float(np.mean(pred)) if len(pred) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, probs))
        out["auprc"] = float(average_precision_score(y, probs))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def train_siren_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    langs_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    langs_val: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[nn.Module, Dict[str, Any]]:
    input_dim = X_train.shape[1]
    layer_dims = list(args.mlp_hidden_dims)
    dropout_rates = [args.dropout] * len(layer_dims)
    model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = -1.0
    patience = 0

    for epoch in range(args.mlp_epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        losses = []
        for start in range(0, len(X_train), args.mlp_batch_size):
            idx = perm[start:start + args.mlp_batch_size]
            bx = torch.as_tensor(X_train[idx], dtype=torch.float32, device=device)
            by = torch.as_tensor(y_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = eval_classifier(model, X_val, y_val, langs_val, device, args.eval_batch_size)
        val_key = val_metrics["macro_over_languages_f1"]
        if val_key > best_val:
            best_val = val_key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if args.verbose:
            print(f"      epoch={epoch+1:03d} loss={np.mean(losses):.4f} val_lang_macro_f1={val_key:.4f}")
        if patience >= args.mlp_patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    train_metrics = eval_classifier(model, X_train, y_train, langs_train, device, args.eval_batch_size)
    val_metrics = eval_classifier(model, X_val, y_val, langs_val, device, args.eval_batch_size)
    return model, {"train": train_metrics, "validation": val_metrics, "best_val_macro_over_languages_f1": best_val}


def expected_random_jaccard(a: int, b: int, d: int) -> Tuple[float, float, float, float]:
    """Compute the **exact** expected Jaccard index for two random subsets.

    Given two subsets A and B drawn uniformly at random *without replacement*
    from a universe of size *d*, with |A| = a and |B| = b, this returns the
    true expectation:

        E[J(A,B)]  =  E[ |A∩B| / |A∪B| ]

    The intersection size k = |A∩B| follows a Hypergeometric(N=d, K=a, n=b)
    distribution, so the exact expected Jaccard is:

        E[J] = Σ_{k=k_min}^{k_max}  k/(a+b−k)  ·  P(|A∩B|=k)

    where k_min = max(0, a+b−d) and k_max = min(a, b).

    This differs from the ratio-of-expectations approximation E[|A∩B|]/E[|A∪B|]
    = (ab/d)/(a+b−ab/d), which overestimates the true E[J] whenever the
    denominator (a+b−k) has variance > 0 (Jensen's inequality, convexity of
    1/x).

    Returns:
        exp_inter  : E[|A∩B|] = a*b/d  (exact, analytical)
        exp_jaccard: E[|A∩B|/|A∪B|]   (exact, hypergeometric sum)
        exp_union  : E[|A∪B|] = a+b−a*b/d  (exact, analytical)
        d          : universe size (passed through for convenience)

    Edge cases:
        - d <= 0                 → all NaN
        - a == 0 and b == 0     → J is 0/0, exp_jaccard = NaN
        - a == 0 or b == 0      → intersection is always 0, E[J] = 0
        - a == b == d            → A = B = universe always, E[J] = 1
    """
    if d <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    exp_inter = float(a * b) / d
    exp_union = a + b - exp_inter

    # Degenerate cases where the hypergeometric sum is trivial.
    if a == 0 and b == 0:
        # J = 0/0 is undefined.
        return 0.0, float("nan"), 0.0, float(d)
    if a == 0 or b == 0:
        # One set is always empty → intersection = 0 → J = 0 always.
        return 0.0, 0.0, float(max(a, b)), float(d)
    if a == d and b == d:
        # Both sets must equal the full universe → J = 1 always.
        return float(d), 1.0, float(d), float(d)

    # General case: sum over the support of Hypergeometric(N=d, K=a, n=b).
    k_min = max(0, a + b - d)
    k_max = min(a, b)

    # scipy hypergeom parameterisation: hypergeom(M, n, N) where
    #   M = population size = d
    #   n = number of "success states" in population = a
    #   N = number of draws = b
    # pmf(k) = C(n,k)*C(M-n,N-k) / C(M,N)
    rv = hypergeom(M=d, n=a, N=b)
    ks = np.arange(k_min, k_max + 1)
    probs = rv.pmf(ks)                      # shape (k_max - k_min + 1,)
    union_sizes = a + b - ks               # |A∪B| for each k; always ≥ 1 here
    jaccards = ks / union_sizes            # J(A,B) for each k

    exp_jaccard = float(np.dot(jaccards, probs))

    return exp_inter, exp_jaccard, exp_union, float(d)


def _get_hidden_dim(
    metadata: Dict[str, Any],
    strategy: str,
    layer_idx: int,
    languages: Sequence[str],
    fallback_idx: Optional[np.ndarray],
) -> Optional[int]:
    """Return the hidden dimension for a given (strategy, layer) combination.

    For per-language strategies the frequency array is stored under the bare
    language code; for composite strategies (shared_only, random_*…) we fall
    back to the first available language frequency so that selected_ratio is
    computed against the true hidden dimension rather than max(idx)+1.
    """
    # Try the strategy's own key first (works for "<lang>_selected").
    key = strategy.replace("_selected", "")
    freq = metadata.get("freq", {}).get((key, layer_idx))
    if freq is not None:
        return int(len(freq))
    # Composite strategies: borrow from any language that has data for this layer.
    for lang in list(languages) + ["pooled"]:
        freq = metadata.get("freq", {}).get((normalize_lang(lang), layer_idx))
        if freq is not None:
            return int(len(freq))
    # Last resort: infer from the selected indices (underestimates, flagged below).
    if fallback_idx is not None and len(fallback_idx) > 0:
        return int(fallback_idx.max() + 1)
    return None


def write_selection_and_overlap_summaries(
    output_dir: Path,
    masks: Dict[str, Dict[str, np.ndarray]],
    layer_weights: Dict[str, Dict[int, float]],
    metadata: Dict[str, Any],
    pooling_type: str,
    languages: Sequence[str],
) -> None:
    with open(output_dir / "selection_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "layer", "n_selected", "hidden_dim", "selected_ratio", "layer_weight", "hidden_dim_exact"])
        writer.writeheader()
        for strategy, md in masks.items():
            for key, idx in sorted(md.items()):
                layer = int(key.split("_")[0].replace("layer", ""))
                hidden_dim = _get_hidden_dim(metadata, strategy, layer, languages, idx)
                # Flag when we had to fall back to max(idx)+1 (underestimate).
                exact = hidden_dim is not None and hidden_dim > (int(idx.max() + 1) if len(idx) else 0)
                writer.writerow({
                    "strategy": strategy,
                    "layer": layer,
                    "n_selected": int(len(idx)),
                    "hidden_dim": int(hidden_dim) if hidden_dim is not None else "",
                    "selected_ratio": float(len(idx) / hidden_dim) if hidden_dim else float("nan"),
                    "layer_weight": float(layer_weights.get(strategy, {}).get(layer, float("nan"))),
                    "hidden_dim_exact": exact,
                })

    rows = []
    lang_strategies = [f"{normalize_lang(l)}_selected" for l in languages]
    for i in range(len(lang_strategies)):
        for j in range(i + 1, len(lang_strategies)):
            a_name, b_name = lang_strategies[i], lang_strategies[j]
            for key in sorted(set(masks.get(a_name, {})) & set(masks.get(b_name, {}))):
                layer = int(key.split("_")[0].replace("layer", ""))
                A = set(masks[a_name][key].tolist())
                B = set(masks[b_name][key].tolist())
                obs_inter = len(A & B)
                obs_union = len(A | B)
                obs_j = obs_inter / obs_union if obs_union else float("nan")
                # Use the true hidden dim from the frequency array; fall back
                # to the larger of the two index sets if unavailable.
                hd_a = _get_hidden_dim(metadata, a_name, layer, languages, masks[a_name][key])
                hd_b = _get_hidden_dim(metadata, b_name, layer, languages, masks[b_name][key])
                d = hd_a if hd_a is not None else (hd_b if hd_b is not None else max(max(A, default=0), max(B, default=0)) + 1)
                exp_inter, exp_j, _, _ = expected_random_jaccard(len(A), len(B), d)
                rows.append({
                    "pair": f"{a_name}__{b_name}",
                    "layer": layer,
                    "n_a": len(A),
                    "n_b": len(B),
                    "hidden_dim": d,
                    "observed_intersection": obs_inter,
                    "expected_random_intersection": exp_inter,
                    "intersection_lift": obs_inter / exp_inter if exp_inter else float("nan"),
                    "observed_jaccard": obs_j,
                    "expected_random_jaccard": exp_j,
                    "jaccard_lift": obs_j / exp_j if exp_j else float("nan"),
                    "jaccard_delta": obs_j - exp_j if not math.isnan(obs_j) and not math.isnan(exp_j) else float("nan"),
                })
    with open(output_dir / "overlap_summary.csv", "w", newline="") as f:
        fieldnames = [
            "pair", "layer", "n_a", "n_b", "hidden_dim",
            "observed_intersection", "expected_random_intersection", "intersection_lift",
            "observed_jaccard", "expected_random_jaccard", "jaccard_lift", "jaccard_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_and_eval_all_strategies(
    args: argparse.Namespace,
    all_reps: Dict[str, Any],
    masks: Dict[str, Dict[str, np.ndarray]],
    layer_weights: Dict[str, Dict[int, float]],
    metadata: Dict[str, Any],
    pooling_type: str,
    output_dir: Path,
) -> Dict[str, Any]:
    print("[4/4] Training SIREN classifiers for each selection strategy...")
    # Resolve device: honour the user's request when the hardware is available;
    # fall back to CPU with a warning otherwise.
    requested = args.device
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested)
        else:
            print(f"[warn] CUDA not available; falling back to CPU (requested: {requested})")
            device = torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print(f"[warn] MPS not available; falling back to CPU (requested: {requested})")
            device = torch.device("cpu")
    else:
        print(f"[warn] Unknown device '{requested}'; falling back to CPU")
        device = torch.device("cpu")
    results: Dict[str, Any] = {}
    model_dir = ensure_dir(output_dir / "models")
    standard_strategies = [
        f"{normalize_lang(l)}_selected" for l in args.languages
    ] + [
        "pooled_selected",
        "shared_only",
        "shared_plus_all_specific",
        "random_same_size_as_pooled",
    ]

    for strategy in standard_strategies:
        if strategy not in masks or not masks[strategy]:
            print(f"  [skip] strategy={strategy}: no mask")
            continue
        print(f"\n  strategy={strategy}")
        weights = layer_weights[strategy]
        selected_layers = sorted(weights.keys())
        X_train = aggregate_standard(all_reps["train"]["representations"], pooling_type, masks[strategy], weights, selected_layers)
        X_val = aggregate_standard(all_reps["validation"]["representations"], pooling_type, masks[strategy], weights, selected_layers)
        X_test = aggregate_standard(all_reps["test"]["representations"], pooling_type, masks[strategy], weights, selected_layers)
        print(f"    feature_dim={X_train.shape[1]}")
        model, train_info = train_siren_classifier(
            X_train, all_reps["train"]["labels"], all_reps["train"]["langs"],
            X_val, all_reps["validation"]["labels"], all_reps["validation"]["langs"],
            device, args,
        )
        test_metrics = eval_classifier(
            model,
            X_test,
            all_reps["test"]["labels"],
            all_reps["test"]["langs"],
            device,
            args.eval_batch_size,
        )
        results[strategy] = {
            "strategy": strategy,
            "pooling_type": pooling_type,
            "feature_dim": int(X_train.shape[1]),
            "n_features_by_layer": {k: int(len(v)) for k, v in masks[strategy].items()},
            "train_info": train_info,
            "test": test_metrics,
        }
        if args.save_models:
            torch.save({
                "state_dict": model.state_dict(),
                "strategy": strategy,
                "pooling_type": pooling_type,
                "feature_dim": int(X_train.shape[1]),
                "mask": {k: v.tolist() for k, v in masks[strategy].items()},
                "layer_weights": {str(k): float(v) for k, v in weights.items()},
            }, model_dir / f"{strategy}.pt")
        print(
            f"    test macro_lang_f1={test_metrics['macro_over_languages_f1']:.4f} "
            f"worst={test_metrics['worst_language_macro_f1']:.4f} "
            f"gap={test_metrics['language_gap_macro_f1']:.4f}"
        )

    # Routed shared + language-specific strategy. Requires language ids at feature construction time.
    strategy = "routed_shared_specific"
    if "shared_only" in masks and metadata.get("specific"):
        print(f"\n  strategy={strategy}")
        weights = layer_weights["shared_only"]
        langs = [normalize_lang(l) for l in args.languages]
        X_train = aggregate_routed_shared_specific(
            all_reps["train"]["representations"], all_reps["train"]["langs"], pooling_type,
            masks["shared_only"], metadata["specific"], weights, langs,
        )
        X_val = aggregate_routed_shared_specific(
            all_reps["validation"]["representations"], all_reps["validation"]["langs"], pooling_type,
            masks["shared_only"], metadata["specific"], weights, langs,
        )
        X_test = aggregate_routed_shared_specific(
            all_reps["test"]["representations"], all_reps["test"]["langs"], pooling_type,
            masks["shared_only"], metadata["specific"], weights, langs,
        )
        print(f"    feature_dim={X_train.shape[1]}")
        model, train_info = train_siren_classifier(
            X_train, all_reps["train"]["labels"], all_reps["train"]["langs"],
            X_val, all_reps["validation"]["labels"], all_reps["validation"]["langs"],
            device, args,
        )
        test_metrics = eval_classifier(
            model,
            X_test,
            all_reps["test"]["labels"],
            all_reps["test"]["langs"],
            device,
            args.eval_batch_size,
        )
        results[strategy] = {
            "strategy": strategy,
            "pooling_type": pooling_type,
            "feature_dim": int(X_train.shape[1]),
            "train_info": train_info,
            "test": test_metrics,
            "note": "Feature vector uses shared block plus language-routed specific blocks; requires language id.",
        }
        if args.save_models:
            torch.save({
                "state_dict": model.state_dict(),
                "strategy": strategy,
                "pooling_type": pooling_type,
                "feature_dim": int(X_train.shape[1]),
                "shared_mask": {k: v.tolist() for k, v in masks["shared_only"].items()},
                "specific_masks": {lang: {k: v.tolist() for k, v in md.items()} for lang, md in metadata["specific"].items()},
                "layer_weights": {str(k): float(v) for k, v in weights.items()},
                # IMPORTANT: this classifier requires the input language to be
                # known at inference time so the correct language-specific block
                # can be routed.  Callers must pass language ids when building
                # the feature vector via aggregate_routed_shared_specific().
                "requires_language_id": True,
                "language_order": [normalize_lang(l) for l in args.languages],
            }, model_dir / f"{strategy}.pt")
        print(
            f"    test macro_lang_f1={test_metrics['macro_over_languages_f1']:.4f} "
            f"worst={test_metrics['worst_language_macro_f1']:.4f} "
            f"gap={test_metrics['language_gap_macro_f1']:.4f}"
        )

    return results


def write_results_csv(output_dir: Path, results: Dict[str, Any]) -> None:
    rows = []
    for strategy, res in results.items():
        test = res["test"]
        row = {
            "strategy": strategy,
            "feature_dim": res.get("feature_dim"),
            "overall_macro_f1": test["overall"]["macro_f1"],
            "overall_auroc": test["overall"].get("auroc"),
            "macro_over_languages_f1": test["macro_over_languages_f1"],
            "worst_language_macro_f1": test["worst_language_macro_f1"],
            "language_gap_macro_f1": test["language_gap_macro_f1"],
        }
        for key, val in test.items():
            if key.startswith("lang_"):
                lang = key.replace("lang_", "")
                row[f"{lang}_macro_f1"] = val["macro_f1"]
                row[f"{lang}_auroc"] = val.get("auroc")
                row[f"{lang}_binary_recall"] = val.get("binary_recall")
        rows.append(row)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(output_dir / "classifier_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--hf_dataset", type=str, default=None, help="HF dataset id for MLMA-style data")
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--local_file", type=str, default=None, help="Local CSV/JSON/JSONL alternative")
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--language_column", type=str, default=None)
    parser.add_argument("--unsafe_labels", type=str, default=",".join(sorted(DEFAULT_UNSAFE_LABELS)))
    parser.add_argument(
        "--normal_label",
        type=str,
        default=None,
        help=(
            "If set, the binarization rule becomes: 0 if label == normal_label else 1. "
            "Use --normal_label normal for nedjmaou/MLMA_hate_speech, which has compound "
            "labels like 'offensive_disrespectful' and 'hateful_normal' that are all toxic "
            "but would be missed by the default unsafe_labels set."
        ),
    )
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "fr", "ar"])
    parser.add_argument(
        "--infer_language",
        action="store_true",
        help=(
            "Detect the language of each row when the dataset has no language column. "
            "Required for nedjmaou/MLMA_hate_speech.  Uses Arabic-script detection for "
            "Arabic and langdetect for other languages (pip install langdetect)."
        ),
    )
    parser.add_argument("--max_per_lang", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--val_seed",
        type=int,
        default=None,
        help="Random seed for the val/test sub-split. Defaults to split_seed + 17 if not set.",
    )

    # Representation extraction
    parser.add_argument("--model", type=str, default="qwen3-4b")
    parser.add_argument("--extractor_class", type=str, default="Qwen3RepresentationExtractor")
    parser.add_argument("--pooling_types", type=str, nargs="+", default=["residual_mean"])
    parser.add_argument("--extract_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_texts", action="store_true")
    parser.add_argument("--reuse_representations", action="store_true")

    # Probe selection
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--c_values", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument(
        "--nonzero_eps",
        type=float,
        default=1e-4,
        help=(
            "Threshold for treating an L1-probe weight as nonzero. "
            "The default 1e-4 is intentionally conservative: numerical noise "
            "from the solver can leave near-zero weights that are not truly "
            "selected. Increase towards 1e-3 if too many neurons pass. "
            "The original value of 1e-8 is effectively no threshold."
        ),
    )
    parser.add_argument("--stability_tau", type=float, default=0.60)
    parser.add_argument("--specific_tau_low", type=float, default=0.30)
    parser.add_argument("--shared_fallback_topk", type=int, default=1)
    parser.add_argument("--layer_weight_floor", type=float, default=0.10)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--reuse_probes", action="store_true")

    # MLP classifier
    parser.add_argument("--mlp_hidden_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_patience", type=int, default=10)
    parser.add_argument("--mlp_batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.languages = [normalize_lang(l) for l in args.languages]
    output_dir = ensure_dir(args.output_dir)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("[data] Loading MLMA-style records...")
    records = load_mlma_records(args)
    records = balanced_subsample(records, args.max_per_lang if args.max_per_lang > 0 else None, args.split_seed)
    print("[data] Counts by language/label:")
    for lang in args.languages:
        for label in [0, 1]:
            n = sum(1 for r in records if r["lang"] == lang and r["label"] == label)
            print(f"  {lang} label={label}: {n}")

    splits = split_records_by_language(
        records,
        args.languages,
        args.train_ratio,
        args.val_ratio,
        args.split_seed,
        val_seed=args.val_seed,
    )
    for split_name, split in splits.items():
        print(f"[data] {split_name}: n={len(split.labels)}, pos_rate={np.mean(split.labels):.3f}")

    all_reps = extract_or_load_representations(args, splits, output_dir)
    probe_runs = train_selection_probes(args, all_reps, output_dir)

    all_pooling_results = {}
    for pooling_type in args.pooling_types:
        pooling_out = ensure_dir(output_dir / pooling_type)
        masks, layer_weights, metadata = build_selection_masks(
            args, probe_runs, pooling_type, all_reps["train"]["num_layers"]
        )
        write_selection_and_overlap_summaries(pooling_out, masks, layer_weights, metadata, pooling_type, args.languages)
        results = train_and_eval_all_strategies(
            args, all_reps, masks, layer_weights, metadata, pooling_type, pooling_out
        )
        write_results_csv(pooling_out, results)
        with open(pooling_out / "classifier_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        all_pooling_results[pooling_type] = results

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_pooling_results, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved under: {output_dir}")


if __name__ == "__main__":
    main()
