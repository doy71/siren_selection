# Patch notes: official SIREN cumulative selection

This patch changes the selection rule in `lang_selection_siren_mlma.py`.

## Main change

Previous version:
- Final masks were based on stable `nonzero_indices`, i.e. `abs(weight) > --nonzero_eps`.

Patched version:
- Final masks are based on the official SIREN cumulative-importance rule.
- For each L1 probe:
  1. compute `weights_abs = abs(probe.weight)`
  2. sort dimensions by descending `weights_abs`
  3. keep the smallest prefix whose cumulative weight reaches `saliency_threshold * sum(weights_abs)`

`--nonzero_eps` is now diagnostic only.

## New CLI option

```bash
--saliency_thresholds 0.6 0.8
```

The script now runs all classifier comparisons separately under each threshold and writes results to:

```text
<output_dir>/<pooling_type>/saliency_0p60/
<output_dir>/<pooling_type>/saliency_0p80/
```

## Kept from previous fixed version

- Exact hypergeometric expected random Jaccard baseline.
- MLMA-friendly label/language handling.
- Reuse of cached representations/probes.
