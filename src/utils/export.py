"""CSV export utilities for experiment results.

main.py currently produces two "tabs" of results that only ever get
printed/displayed, never saved to disk:

  1. The "Final Results Summary" rich Table: one row per experiment with
     name, param count, CPU/GPU time, speedup, and the single headline test
     metric (accuracy for classification, r2 for regression).
  2. The per-experiment metrics breakdown (`print_metrics_summary`): every
     train/val/test metric for that experiment, currently only ever printed
     via a nested dict shaped like `cpu_metrics['test']['accuracy']`.

This module exports both as CSV files. Each exported row is a flat
key -> value mapping (no nested objects or lists as values): keys are
always strings, and values are `int`/`float` for numeric data or `string`
otherwise (e.g. `name`). Nested metrics like `metrics['test']['accuracy']`
are flattened into their own top-level column, e.g. `test_accuracy`.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..richlog import Logger, StepHandle, NULL_STEP


def _flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Flatten `{'train': {...}, 'val': {...}, 'test': {...}}` into a single
    flat dict with keys like `train_loss`, `val_accuracy`, `test_r2`, etc.

    This is the flattening the TODO calls for: no split of the result stays
    a nested object, each `split_metric` combination becomes its own
    top-level key mapping to a single numeric value.
    """
    flat: Dict[str, Any] = {}
    for split, split_metrics in metrics.items():
        for key, value in split_metrics.items():
            flat[f"{split}_{key}"] = value
    return flat


def build_summary_row(result: dict) -> Dict[str, Any]:
    """Build one flat row matching the "Final Results Summary" table for a
    single experiment result (as returned by `run_experiment`).

    Mirrors main.py's table exactly: whichever device actually has metrics
    (GPU if it ran, otherwise CPU) supplies the single headline test metric,
    the same way the table picks accuracy for classification tasks and r2
    for regression tasks.
    """
    cpu_time = result['cpu_time']
    gpu_time = result['gpu_time']

    test_metrics = result['gpu_metrics']['test'] if result['gpu_metrics'] else result['cpu_metrics']['test']
    metric_key = 'accuracy' if 'accuracy' in test_metrics else 'r2'

    return {
        'name': result['name'],
        'n_params': result['n_params'],
        'cpu_time': cpu_time,
        'gpu_time': gpu_time if gpu_time is not None else '',
        'speedup': (cpu_time / gpu_time) if gpu_time else '',
        'primary_metric_name': metric_key,
        'primary_metric_value': test_metrics[metric_key],
    }


def build_metrics_row(result: dict) -> Dict[str, Any]:
    """Build one flat row with every train/val/test metric for a single
    experiment result, flattened out of nesting like
    `cpu_metrics['test']['accuracy']` into top-level keys like
    `test_accuracy`.

    Uses GPU metrics when available (the same source `print_metrics_summary`
    is handed in main.py), otherwise falls back to CPU metrics.
    """
    metrics = result['gpu_metrics'] or result['cpu_metrics']

    row: Dict[str, Any] = {
        'name': result['name'],
        'n_params': result['n_params'],
        'cpu_time': result['cpu_time'],
        'gpu_time': result['gpu_time'] if result['gpu_time'] is not None else '',
    }
    row.update(_flatten_metrics(metrics))
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a CSV file.

    Column set is the union of keys across all rows, in first-seen order,
    so experiments with different metric sets (e.g. classification's
    accuracy/f1/auc_roc vs regression's mae/rmse/r2) don't break the write;
    any row missing a given column just gets an empty cell there.
    """
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


def export_results_csv(results: List[dict], save_dir: str = 'results', run_timestamp: Optional[str] = None,
                        *, log: Union[StepHandle, Logger] = NULL_STEP) -> None:
    """Export both result "tabs" as CSV files, flattened as described above:

      - `results_summary.csv`: one row per experiment, matching the
        "Final Results Summary" table (name, params, cpu/gpu time, speedup,
        primary test metric).
      - `results_metrics.csv`: one row per experiment with every
        train/val/test metric flattened into its own top-level column
        (e.g. `test_accuracy`, `val_loss`) instead of a nested dict.

    Args:
        log: open StepHandle, or the bare Logger if called after every step
            has already closed (main.py's final summary calls this the same
            way it calls `print_metrics_summary` — both just need `.block()`).
        results: list of per-experiment result dicts, as returned by
            `run_experiment` (name, n_params, cpu_time, gpu_time,
            cpu_metrics, gpu_metrics).
        save_dir: base directory results are saved under (default: 'results').
        run_timestamp: shared run timestamp (YYYY-MM-DD_HHhMM) so the CSVs
            land alongside this run's plots under `results/<run_timestamp>/`.
            If None, the files are written directly under `save_dir`.
    """
    results_path = Path(save_dir)
    if run_timestamp:
        results_path = results_path / run_timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    summary_rows = [build_summary_row(r) for r in results]
    metrics_rows = [build_metrics_row(r) for r in results]

    summary_path = results_path / 'results_summary.csv'
    metrics_path = results_path / 'results_metrics.csv'

    _write_csv(summary_path, summary_rows)
    _write_csv(metrics_path, metrics_rows)

    message = f"\nexported results:\n    {summary_path}\n    {metrics_path}"
    log.block(message, indent=3, style="dim")
