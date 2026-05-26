#!/usr/bin/env python3
"""Small helper to rank SIREN selection strategies after the MLMA experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_csv", type=str, help="Path to classifier_results.csv")
    p.add_argument("--sort_by", type=str, default="macro_over_languages_f1")
    args = p.parse_args()

    df = pd.read_csv(args.results_csv)
    cols = [
        "strategy",
        "feature_dim",
        "macro_over_languages_f1",
        "worst_language_macro_f1",
        "language_gap_macro_f1",
        "overall_macro_f1",
    ]
    lang_cols = sorted([c for c in df.columns if c.endswith("_macro_f1") and c not in cols])
    cols = [c for c in cols if c in df.columns] + lang_cols
    print(df[cols].sort_values(args.sort_by, ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
