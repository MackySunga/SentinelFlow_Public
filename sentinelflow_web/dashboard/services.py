from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from django.conf import settings

try:
    from sentinelflow_utils import read_table_safely, clean_netflow_df, make_signal_table, build_segment_datasets
except Exception:  # pragma: no cover
    read_table_safely = clean_netflow_df = make_signal_table = build_segment_datasets = None

REQUIRED_ANY = [
    'FLOW_START_MILLISECONDS', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'Label', 'Attack'
]

@dataclass
class AnalysisResult:
    ok: bool
    message: str
    context: Dict[str, Any]


def model_registry_status() -> Dict[str, Any]:
    models_dir = Path(settings.MODELS_DIR)
    expected = [
        'binary_model.pt', 'multiclass_model.pt',
        'sentinelflow_binary_model.pt', 'sentinelflow_multiclass_model.pt',
        'binary_scaler.pkl', 'multiclass_scaler.pkl',
        'binary_feature_list.json', 'multiclass_feature_list.json',
        'binary_label_encoder.pkl', 'multiclass_label_encoder.pkl',
    ]
    files = {name: (models_dir / name).exists() for name in expected}
    found = [name for name, exists in files.items() if exists]
    return {
        'models_dir': str(models_dir),
        'files': files,
        'found_count': len(found),
        'has_binary_model': files.get('binary_model.pt', False) or files.get('sentinelflow_binary_model.pt', False),
        'has_multiclass_model': files.get('multiclass_model.pt', False) or files.get('sentinelflow_multiclass_model.pt', False),
        'note': 'Run the notebook pipeline first to export trained models. The app can still show dataset profiling and notebook-generated reports without model files.'
    }


def safe_percent(count: int, total: int) -> float:
    return round((count / total * 100), 2) if total else 0.0


def _top_counts(df: pd.DataFrame, col: str, limit: int = 10) -> List[Dict[str, Any]]:
    if col not in df.columns:
        return []
    vc = df[col].astype(str).value_counts(dropna=False).head(limit)
    total = max(len(df), 1)
    return [{'name': str(k), 'count': int(v), 'percent': safe_percent(int(v), total)} for k, v in vc.items()]


def analyze_uploaded_file(path: Path, max_rows: int = 50000) -> AnalysisResult:
    if read_table_safely is None:
        return AnalysisResult(False, 'SentinelFlow utility module could not be imported.', {})
    path = Path(path)
    try:
        raw = read_table_safely(path, max_rows=max_rows, show_progress=False)
    except Exception as exc:
        return AnalysisResult(False, f'Could not read uploaded file: {exc}', {})
    if raw.empty:
        return AnalysisResult(False, 'Uploaded file was read but no rows were found.', {})

    raw.columns = [str(c).strip().replace(' ', '_') for c in raw.columns]
    missing_required = [c for c in REQUIRED_ANY if c not in raw.columns]

    try:
        cleaned, clean_summary = clean_netflow_df(raw)
        signal_df, signal_mode = make_signal_table(cleaned, window_seconds=1)
        baseline_segments, fft_segments, signal_cols = build_segment_datasets(signal_df, segment_size=16, stride=4)
    except Exception as exc:
        return AnalysisResult(False, f'File was loaded, but SentinelFlow processing failed: {exc}', {})

    rows = int(len(cleaned))
    attack_count = int(cleaned.get('target_binary', pd.Series(dtype=int)).sum()) if 'target_binary' in cleaned else 0
    benign_count = rows - attack_count

    context = {
        'filename': path.name,
        'rows': rows,
        'columns': int(cleaned.shape[1]),
        'raw_columns': int(raw.shape[1]),
        'missing_required': missing_required,
        'clean_summary': clean_summary,
        'benign_count': benign_count,
        'attack_count': attack_count,
        'benign_percent': safe_percent(benign_count, rows),
        'attack_percent': safe_percent(attack_count, rows),
        'attack_distribution': _top_counts(cleaned, 'target_attack', 12),
        'signal_rows': int(len(signal_df)),
        'signal_mode': signal_mode,
        'signal_columns': signal_cols,
        'baseline_segments': int(len(baseline_segments)),
        'fft_segments': int(len(fft_segments)),
        'baseline_feature_count': int(max(0, baseline_segments.shape[1] - 4)) if len(baseline_segments) else 0,
        'fft_feature_count': int(max(0, fft_segments.shape[1] - 4)) if len(fft_segments) else 0,
        'model_status': model_registry_status(),
        'prototype_note': 'This page validates and profiles uploaded traffic. Final prediction mode should use model files exported by the notebooks.',
    }

    # Save lightweight JSON summary for dashboard refresh.
    result_dir = Path(settings.MEDIA_ROOT) / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / 'latest_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, default=str)

    return AnalysisResult(True, 'Dataset processed successfully.', context)


def load_latest_analysis() -> Dict[str, Any] | None:
    p = Path(settings.MEDIA_ROOT) / 'results' / 'latest_analysis.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def load_notebook_metrics() -> List[Dict[str, Any]]:
    candidates = [
        Path(settings.OUTPUTS_DIR) / 'metrics' / '04_model_results_expanded.csv',
        Path(settings.OUTPUTS_DIR) / '03_model_results.csv',
        Path(settings.PROJECT_ROOT) / '03_model_results.csv',
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df.replace({np.nan: None}).to_dict('records')[:50]
            except Exception:
                continue
    return []


def available_reports() -> List[Dict[str, str]]:
    reports_dir = Path(settings.REPORTS_DIR)
    out = []
    if reports_dir.exists():
        for p in sorted(reports_dir.rglob('*.html'))[:50]:
            out.append({'name': p.name, 'path': str(p)})
    return out
