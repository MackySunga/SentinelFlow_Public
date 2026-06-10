from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
EPS = 1e-12

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def stage_message(step: str, started_at=None):
    now = time.time()
    if started_at is None:
        print(f"[SentinelFlow] {step}")
    else:
        print(f"[SentinelFlow] {step} ({now-started_at:.1f}s)")
    return now


def find_project_root(start: Optional[Path] = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "src" / "sentinelflow_utils.py").exists():
            return p
    return start


def ensure_dirs(root: Path):
    for rel in [
        "data/raw", "data/processed", "data/database", "outputs", "reports", "models",
        "reports/confusion_matrices", "reports/classification_reports", "reports/figures",
        "outputs/predictions", "outputs/metrics"
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def read_table_safely(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.head(max_rows)
        return df
    for kwargs in [dict(sep="\t", engine="python"), dict(sep=",", engine="python"), dict(sep=None, engine="python")]:
        try:
            df = pd.read_csv(path, nrows=max_rows, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not read file: {path}")


def find_dataset_path(root: Path, user_path: str = "") -> Path:
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
        raise FileNotFoundError(f"DATA_PATH was set, but file not found: {p}")
    candidates = [
        root / "data/processed/sentinelflow_analysis_dataset.parquet",
        root / "data/processed/sentinelflow_cleaned_netflow.parquet",
        root / "data/raw/NF-UQ-NIDS-v2.csv",
        root / "data/raw/NF-CSE-CIC-IDS2018-v3.csv",
        root / "data/raw/netflow_sample.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No dataset found. Put CSV/TSV/Parquet in data/raw or set DATA_PATH.")


def find_raw_dataset_path(root: Path, user_path: str = "") -> Path:
    if user_path:
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
        raise FileNotFoundError(f"RAW_DATA_PATH was set, but file not found: {p}")
    for pattern in ["NF-UQ-NIDS-v2.csv", "NF-CSE-CIC-IDS2018-v3.csv", "*.csv", "*.tsv", "*.txt", "*.parquet"]:
        for p in (root / "data/raw").glob(pattern):
            if p.name.lower() != "netflow_v3_features.csv":
                return p
    raise FileNotFoundError("No raw dataset found in data/raw.")

# -----------------------------------------------------------------------------
# Cleaning and EDA
# -----------------------------------------------------------------------------

def clean_netflow_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = normalize_column_names(df)
    summary = {"initial_shape": tuple(df.shape)}

    for col in ["Label", "Attack", "label", "attack"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    protected = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack", "Label", "attack", "label", "target_attack"}
    for col in df.columns:
        if col not in protected:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_count = int(np.isinf(df[numeric_cols]).sum().sum()) if numeric_cols else 0
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    missing_before = int(df.isna().sum().sum())

    for col in numeric_cols:
        med = df[col].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        df[col] = df[col].fillna(med)
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].fillna("Unknown")

    # Binary target detection: in most NetFlow/NIDS datasets, Label is 0/1 and Attack is text.
    if "Label" in df.columns:
        numeric_label = pd.to_numeric(df["Label"], errors="coerce")
        if numeric_label.notna().sum() > 0 and set(numeric_label.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            df["target_binary"] = numeric_label.fillna(0).astype(int)
        else:
            df["target_binary"] = (df["Label"].astype(str).str.lower() != "benign").astype(int)
    elif "Attack" in df.columns:
        df["target_binary"] = (df["Attack"].astype(str).str.lower() != "benign").astype(int)
    else:
        df["target_binary"] = 0

    if "Attack" in df.columns:
        df["target_attack"] = df["Attack"].astype(str)
    elif "Label" in df.columns:
        df["target_attack"] = df["Label"].astype(str)
    else:
        df["target_attack"] = np.where(df["target_binary"] == 1, "Attack", "Benign")

    summary.update({
        "final_shape": tuple(df.shape),
        "numeric_columns": len(numeric_cols),
        "inf_values_replaced": inf_count,
        "missing_values_before_fill": missing_before,
        "missing_values_after_fill": int(df.isna().sum().sum()),
    })
    return df, summary


def class_distribution(df: pd.DataFrame, col: str = "target_attack") -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count", "percent"])
    out = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    out["percent"] = (out["count"] / max(len(df), 1) * 100).round(2)
    return out

# -----------------------------------------------------------------------------
# Time-window and FFT feature extraction
# -----------------------------------------------------------------------------

def make_signal_table(df: pd.DataFrame, window_seconds: int = 1, use_time_when_possible: bool = True):
    d = df.copy().reset_index(drop=True)
    time_col = "FLOW_START_MILLISECONDS"
    mode = "row_index_fallback"

    if use_time_when_possible and time_col in d.columns:
        ts = pd.to_datetime(pd.to_numeric(d[time_col], errors="coerce"), unit="ms", errors="coerce")
        if ts.notna().sum() > 0:
            d["__time_bin"] = ts.dt.floor(f"{window_seconds}s")
            if d["__time_bin"].nunique(dropna=True) >= 5:
                key = "__time_bin"
                mode = f"timestamp_{window_seconds}s"
            else:
                d["__time_bin"] = np.arange(len(d))
                key = "__time_bin"
        else:
            d["__time_bin"] = np.arange(len(d))
            key = "__time_bin"
    else:
        d["__time_bin"] = np.arange(len(d))
        key = "__time_bin"

    def has(c): return c in d.columns

    agg = {"flows": ("target_binary", "size"), "attack_ratio": ("target_binary", "mean"), "target_binary": ("target_binary", "max")}
    for col, out, func in [
        ("IN_BYTES", "in_bytes_sum", "sum"), ("OUT_BYTES", "out_bytes_sum", "sum"),
        ("IN_PKTS", "in_pkts_sum", "sum"), ("OUT_PKTS", "out_pkts_sum", "sum"),
        ("FLOW_DURATION_MILLISECONDS", "duration_mean", "mean"),
        ("RETRANSMITTED_IN_PKTS", "retrans_in_pkts_sum", "sum"),
        ("RETRANSMITTED_OUT_PKTS", "retrans_out_pkts_sum", "sum"),
        ("SRC_TO_DST_AVG_THROUGHPUT", "src_dst_throughput_mean", "mean"),
        ("DST_TO_SRC_AVG_THROUGHPUT", "dst_src_throughput_mean", "mean"),
        ("SRC_TO_DST_IAT_AVG", "src_dst_iat_mean", "mean"),
        ("DST_TO_SRC_IAT_AVG", "dst_src_iat_mean", "mean"),
    ]:
        if has(col):
            agg[out] = (col, func)

    grouped = d.groupby(key).agg(**agg).reset_index().rename(columns={key: "window_id"})
    grouped["unique_src_ips"] = d.groupby(key)["IPV4_SRC_ADDR"].nunique().values if has("IPV4_SRC_ADDR") else 0
    grouped["unique_dst_ips"] = d.groupby(key)["IPV4_DST_ADDR"].nunique().values if has("IPV4_DST_ADDR") else 0
    grouped["unique_dst_ports"] = d.groupby(key)["L4_DST_PORT"].nunique().values if has("L4_DST_PORT") else 0
    if has("target_attack"):
        grouped["target_attack"] = d.groupby(key)["target_attack"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).values
    else:
        grouped["target_attack"] = np.where(grouped["target_binary"] == 1, "Attack", "Benign")

    grouped["total_bytes"] = grouped.get("in_bytes_sum", 0) + grouped.get("out_bytes_sum", 0)
    grouped["total_pkts"] = grouped.get("in_pkts_sum", 0) + grouped.get("out_pkts_sum", 0)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0)
    return grouped, mode


def spectral_entropy(magnitudes: np.ndarray) -> float:
    magnitudes = np.asarray(magnitudes, dtype=float)
    total = magnitudes.sum()
    if total <= EPS:
        return 0.0
    p = magnitudes / total
    return float(-np.sum(p * np.log2(p + EPS)))


def fft_features(signal: Iterable[float], prefix: str) -> Dict[str, float]:
    x = np.asarray(list(signal), dtype=float)
    if x.size == 0:
        x = np.zeros(1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centered = x - x.mean()
    mags = np.abs(np.fft.rfft(centered))
    mags_no_dc = mags[1:] if mags.size > 1 else mags
    split = max(1, len(mags_no_dc) // 3) if len(mags_no_dc) else 1
    low = float(np.sum(mags_no_dc[:split] ** 2)) if len(mags_no_dc) else 0.0
    high = float(np.sum(mags_no_dc[split:] ** 2)) if len(mags_no_dc) > split else 0.0
    return {
        f"{prefix}_fft_energy": float(np.sum(mags_no_dc ** 2)),
        f"{prefix}_fft_max_magnitude": float(np.max(mags_no_dc)) if len(mags_no_dc) else 0.0,
        f"{prefix}_fft_mean_magnitude": float(np.mean(mags_no_dc)) if len(mags_no_dc) else 0.0,
        f"{prefix}_fft_dominant_index": float(np.argmax(mags_no_dc) + 1) if len(mags_no_dc) else 0.0,
        f"{prefix}_fft_entropy": spectral_entropy(mags_no_dc),
        f"{prefix}_fft_low_power": low,
        f"{prefix}_fft_high_power": high,
        f"{prefix}_fft_high_low_ratio": float(high / (low + EPS)),
        f"{prefix}_burstiness": float((np.max(x) - np.mean(x)) / (np.std(x) + EPS)),
        f"{prefix}_periodicity": float((np.max(mags_no_dc) if len(mags_no_dc) else 0.0) / (np.sum(mags_no_dc) + EPS)),
    }


def build_segment_datasets(signal_df: pd.DataFrame, signal_cols=None, segment_size: int = 16, stride: int = 4):
    if signal_cols is None:
        candidates = [
            "flows", "total_bytes", "total_pkts", "in_bytes_sum", "out_bytes_sum", "in_pkts_sum", "out_pkts_sum",
            "duration_mean", "retrans_in_pkts_sum", "retrans_out_pkts_sum", "src_dst_throughput_mean", "dst_src_throughput_mean",
            "unique_src_ips", "unique_dst_ips", "unique_dst_ports", "src_dst_iat_mean", "dst_src_iat_mean"
        ]
        signal_cols = [c for c in candidates if c in signal_df.columns]
    if not signal_cols:
        raise ValueError("No usable signal columns found.")
    n = len(signal_df)
    if n < 4:
        reps = int(math.ceil(4 / max(n, 1)))
        signal_df = pd.concat([signal_df] * reps, ignore_index=True)
        n = len(signal_df)
    segment_size = max(4, min(int(segment_size), n))
    stride = max(1, int(stride))
    base_rows, fft_rows = [], []

    for start in range(0, n - segment_size + 1, stride):
        seg = signal_df.iloc[start:start + segment_size]
        label = int(seg["target_binary"].iloc[-1]) if "target_binary" in seg else 0
        attack = str(seg["target_attack"].iloc[-1]) if "target_attack" in seg else "Unknown"
        base = {"segment_start": int(start), "segment_end": int(start + segment_size - 1), "target_binary": label, "target_attack": attack}
        for c in signal_cols:
            values = pd.to_numeric(seg[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values
            base[f"{c}_mean"] = float(np.mean(values))
            base[f"{c}_std"] = float(np.std(values))
            base[f"{c}_min"] = float(np.min(values))
            base[f"{c}_max"] = float(np.max(values))
            base[f"{c}_sum"] = float(np.sum(values))
        row_fft = dict(base)
        for c in signal_cols:
            values = pd.to_numeric(seg[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).values
            row_fft.update(fft_features(values, c))
        base_rows.append(base)
        fft_rows.append(row_fft)
    return pd.DataFrame(base_rows), pd.DataFrame(fft_rows), signal_cols

# -----------------------------------------------------------------------------
# Evaluation helpers: confusion matrix, reports, ROC-AUC, PR-AUC, FPR, FNR
# -----------------------------------------------------------------------------

def safe_split(X, y, test_size: float = 0.25, random_state: int = 42):
    from sklearn.model_selection import train_test_split
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    counts = y.value_counts()
    if len(counts) < 2 or len(y) < 20 or counts.min() < 2 or int(len(y) * test_size) < len(counts):
        return X, X, y, y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def prepare_numeric_xy(dataset: pd.DataFrame, target_col: str = "target_binary", max_features: int = 250):
    y = dataset[target_col]
    X = dataset.drop(columns=[c for c in ["target_binary", "target_attack", "segment_start", "segment_end"] if c in dataset.columns])
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    if X.shape[1] > max_features:
        X = X[X.var().sort_values(ascending=False).head(max_features).index.tolist()]
    return X, y


def binary_metrics(y_true, y_pred, y_score=None, model_name="model", feature_set="baseline") -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, roc_auc_score, average_precision_score
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    specificity = tn / (tn + fp + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    roc_auc = np.nan
    pr_auc = np.nan
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = np.nan
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except Exception:
            pr_auc = np.nan
    return {
        "feature_set": feature_set,
        "task": "binary",
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "support": int(len(y_true)),
    }


def multiclass_metrics(y_true, y_pred, model_name="model", feature_set="baseline") -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "feature_set": feature_set,
        "task": "multiclass",
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(macro[0]),
        "recall": float(macro[1]),
        "specificity": np.nan,
        "f1": float(macro[2]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "false_positive_rate": np.nan,
        "false_negative_rate": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "true_negative": np.nan,
        "false_positive": np.nan,
        "false_negative": np.nan,
        "true_positive": np.nan,
        "support": int(len(y_true)),
    }


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    return df


def save_confusion_matrix_plot(y_true, y_pred, labels: List[Any], title: str, path: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig_w = max(6, min(16, 0.7 * len(labels) + 4))
    fig_h = max(5, min(16, 0.65 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_pr_curves(predictions_df: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in predictions_df[["feature_set", "model"]].drop_duplicates().iterrows():
        fs, model = row["feature_set"], row["model"]
        sub = predictions_df[(predictions_df["feature_set"] == fs) & (predictions_df["model"] == model)]
        if "y_score" not in sub.columns or sub["y_true"].nunique() < 2 or sub["y_score"].isna().all():
            continue
        safe_name = f"{fs}_{model}".replace(" ", "_").replace("/", "_").replace("+", "plus")
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(sub["y_true"].astype(int), sub["y_score"].astype(float), ax=ax)
            ax.set_title(f"ROC Curve: {model} ({fs})")
            fig.tight_layout()
            p = out_dir / f"roc_{safe_name}.png"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        except Exception:
            pass
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            PrecisionRecallDisplay.from_predictions(sub["y_true"].astype(int), sub["y_score"].astype(float), ax=ax)
            ax.set_title(f"Precision-Recall Curve: {model} ({fs})")
            fig.tight_layout()
            p = out_dir / f"pr_{safe_name}.png"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
        except Exception:
            pass
    return paths

# -----------------------------------------------------------------------------
# Model training with saved predictions
# -----------------------------------------------------------------------------

def train_sklearn_models(dataset: pd.DataFrame, feature_set="baseline", max_features=250):
    """Backward-compatible simple training result output."""
    results, _, _ = train_sklearn_binary_models_with_predictions(dataset, feature_set=feature_set, max_features=max_features)
    return results


def train_sklearn_binary_models_with_predictions(dataset: pd.DataFrame, feature_set="baseline", max_features=250):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = dataset["target_binary"].astype(int)
    X, y = prepare_numeric_xy(dataset, "target_binary", max_features=max_features)
    X_train, X_test, y_train, y_test = safe_split(X, y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=80, random_state=42, class_weight="balanced", n_jobs=1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120, random_state=42, early_stopping=True if len(X_train) > 80 else False)),
    }
    results, pred_rows, reports = [], [], {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X_test)
                score = 1 / (1 + np.exp(-raw))
            else:
                score = pred.astype(float)
            results.append(binary_metrics(y_test, pred, score, name, feature_set))
            rep = classification_report_df(y_test, pred)
            reports[f"{feature_set}__{name}__binary"] = rep
            for i, (yt, yp, ys) in enumerate(zip(y_test, pred, score)):
                pred_rows.append({"feature_set": feature_set, "task": "binary", "model": name, "row_id": int(i), "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys)})
        except Exception as exc:
            results.append({"feature_set": feature_set, "task": "binary", "model": name, "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "f1": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "false_positive_rate": np.nan, "false_negative_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "true_negative": np.nan, "false_positive": np.nan, "false_negative": np.nan, "true_positive": np.nan, "support": len(y), "error": str(exc)})
    return pd.DataFrame(results), pd.DataFrame(pred_rows), reports


def train_torch_models(dataset: pd.DataFrame, feature_set="fft", epochs=8, max_features=250):
    results, _, _ = train_torch_binary_models_with_predictions(dataset, feature_set=feature_set, epochs=epochs, max_features=max_features)
    return results


def train_torch_binary_models_with_predictions(dataset: pd.DataFrame, feature_set="fft", epochs=8, max_features=250):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        err = f"PyTorch unavailable: {exc}"
        return pd.DataFrame([{"feature_set": feature_set, "task": "binary", "model": "PyTorch deep models", "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "f1": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "false_positive_rate": np.nan, "false_negative_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "support": len(dataset), "error": err}]), pd.DataFrame(), {}

    y = dataset["target_binary"].astype(int)
    X, y = prepare_numeric_xy(dataset, "target_binary", max_features=max_features)
    if len(y.unique()) < 2 or len(y) < 20:
        err = "Not enough rows/classes for deep learning. Use a larger dataset."
        return pd.DataFrame([{"feature_set": feature_set, "task": "binary", "model": "Deep models", "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "f1": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "false_positive_rate": np.nan, "false_negative_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "support": len(y), "error": err}]), pd.DataFrame(), {}

    X_train, X_test, y_train, y_test = safe_split(X, y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype("float32")
    Xte = scaler.transform(X_test).astype("float32")
    ytr = y_train.values.astype("float32")
    yte = y_test.values.astype("int64")
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    loader = DataLoader(ds, batch_size=min(128, max(8, len(ds) // 4)), shuffle=True)
    n = Xtr.shape[1]

    class MLPNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x): return self.net(x).squeeze(1)

    class CNN1DNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(16))
            self.fc = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x): return self.fc(self.conv(x.unsqueeze(1)).flatten(1)).squeeze(1)

    class LSTMNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=24, batch_first=True)
            self.fc = nn.Linear(24, 1)
        def forward(self, x):
            out, _ = self.lstm(x.unsqueeze(-1))
            return self.fc(out[:, -1, :]).squeeze(1)

    class TransformerNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            d = 16
            self.proj = nn.Linear(1, d)
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=32, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=1)
            self.fc = nn.Linear(d, 1)
        def forward(self, x):
            x = self.proj(x.unsqueeze(-1))
            x = self.encoder(x)
            return self.fc(x.mean(dim=1)).squeeze(1)

    models = {"Torch MLP": MLPNet(n), "CNN-1D": CNN1DNet(n), "LSTM": LSTMNet(n), "Transformer Encoder": TransformerNet(n)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.BCEWithLogitsLoss()
    Xte_t = torch.tensor(Xte).to(device)
    results, pred_rows, reports = [], [], {}
    for name, model in models.items():
        try:
            torch.manual_seed(42)
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            model.train()
            for _ in range(max(1, int(epochs))):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                score = torch.sigmoid(model(Xte_t)).cpu().numpy()
            pred = (score >= .5).astype(int)
            results.append(binary_metrics(yte, pred, score, name, feature_set))
            reports[f"{feature_set}__{name}__binary"] = classification_report_df(yte, pred)
            for i, (yt, yp, ys) in enumerate(zip(yte, pred, score)):
                pred_rows.append({"feature_set": feature_set, "task": "binary", "model": name, "row_id": int(i), "y_true": int(yt), "y_pred": int(yp), "y_score": float(ys)})
        except Exception as exc:
            results.append({"feature_set": feature_set, "task": "binary", "model": name, "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "f1": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "false_positive_rate": np.nan, "false_negative_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "support": len(y), "error": str(exc)})
    return pd.DataFrame(results), pd.DataFrame(pred_rows), reports


def train_multiclass_models_with_predictions(dataset: pd.DataFrame, feature_set="baseline", max_features=250, max_classes: int = 12):
    """Train robust multiclass classifiers on target_attack when enough examples exist."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if "target_attack" not in dataset.columns:
        return pd.DataFrame(), pd.DataFrame(), {}

    data = dataset.copy()
    counts = data["target_attack"].astype(str).value_counts()
    keep = counts.head(max_classes).index.tolist()
    data = data[data["target_attack"].astype(str).isin(keep)].copy()
    data["target_attack"] = data["target_attack"].astype(str)
    if data["target_attack"].nunique() < 2 or len(data) < 30:
        return pd.DataFrame(), pd.DataFrame(), {}

    X, y = prepare_numeric_xy(data, "target_attack", max_features=max_features)
    # remove classes with only one row for stratification
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    X = X[y.isin(valid_classes)].reset_index(drop=True)
    y = y[y.isin(valid_classes)].reset_index(drop=True)
    if y.nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), {}
    X_train, X_test, y_train, y_test = safe_split(X, y)

    models = {
        "Multiclass Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=1),
        "Multiclass MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=150, random_state=42, early_stopping=True if len(X_train) > 100 else False)),
    }
    results, pred_rows, reports = [], [], {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results.append(multiclass_metrics(y_test, pred, name, feature_set))
            reports[f"{feature_set}__{name}__multiclass"] = classification_report_df(y_test, pred)
            for i, (yt, yp) in enumerate(zip(y_test, pred)):
                pred_rows.append({"feature_set": feature_set, "task": "multiclass", "model": name, "row_id": int(i), "y_true": str(yt), "y_pred": str(yp), "y_score": np.nan})
        except Exception as exc:
            results.append({"feature_set": feature_set, "task": "multiclass", "model": name, "accuracy": np.nan, "balanced_accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "f1": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "false_positive_rate": np.nan, "false_negative_rate": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "support": len(y), "error": str(exc)})
    return pd.DataFrame(results), pd.DataFrame(pred_rows), reports

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------

def _metric_fmt(v):
    try:
        return "N/A" if pd.isna(v) else f"{float(v):.3f}"
    except Exception:
        return "N/A"


def make_html_dashboard(results_df: pd.DataFrame, class_dist: pd.DataFrame, summary: Dict[str, object], image_paths: Optional[List[Path]] = None) -> str:
    image_paths = image_paths or []
    best = "Not available"
    score = "N/A"
    if not results_df.empty and "macro_f1" in results_df.columns:
        valid = results_df.dropna(subset=["macro_f1"])
        if len(valid):
            r = valid.sort_values("macro_f1", ascending=False).iloc[0]
            best = str(r["model"])
            score = _metric_fmt(r["macro_f1"])
    cols = ["feature_set", "task", "model", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "weighted_f1", "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc"]
    show_cols = [c for c in cols if c in results_df.columns]
    header = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in show_cols)
    rows = ""
    for _, r in results_df.iterrows():
        rows += "<tr>" + "".join(f"<td>{_metric_fmt(r[c]) if c not in ['feature_set','task','model'] else r.get(c,'')}</td>" for c in show_cols) + "</tr>"
    crows = "".join([f"<tr><td>{r.iloc[0]}</td><td>{int(r['count'])}</td><td>{float(r['percent']):.2f}%</td></tr>" for _, r in class_dist.iterrows()]) if not class_dist.empty else ""
    img_html = "".join([f"<figure><img src='{p.as_posix()}' alt='{p.name}'><figcaption>{p.name}</figcaption></figure>" for p in image_paths])

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SentinelFlow V3 Results Dashboard</title>
<style>
:root {{ --bg:#06111f; --panel:#0d1d33; --text:#eaf3ff; --muted:#a8bdd6; --accent:#41d6ff; --ok:#8df7b0; --warn:#ffc857; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at top left,#123a63,#06111f 45%,#030812); color:var(--text); }}
.hero {{ padding:56px 7vw; background:linear-gradient(135deg,rgba(65,214,255,.18),rgba(141,247,176,.08)); border-bottom:1px solid rgba(255,255,255,.08); }}
h1 {{ font-size:clamp(2.2rem,5vw,4.8rem); margin:.1rem 0; letter-spacing:-.04em; }}
.sub {{ max-width:980px; color:var(--muted); line-height:1.65; font-size:1.08rem; }}
.wrap {{ padding:32px 7vw 80px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:18px; }}
.card {{ background:linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,.03)); border:1px solid rgba(255,255,255,.12); border-radius:24px; padding:24px; margin-bottom:24px; box-shadow:0 16px 60px rgba(0,0,0,.25); }}
.kpi {{ font-size:2rem; font-weight:800; color:var(--accent); word-break:break-word; }}
.badge {{ display:inline-block; padding:7px 11px; border-radius:999px; background:rgba(141,247,176,.14); color:var(--ok); font-weight:700; margin:0 8px 12px 0; }}
table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
th,td {{ padding:11px 12px; border-bottom:1px solid rgba(255,255,255,.1); text-align:left; vertical-align:top; }}
th {{ background:rgba(65,214,255,.12); color:#dff9ff; }}
.table-scroll {{ overflow-x:auto; }}
figure {{ margin:0 0 24px; }}
img {{ max-width:100%; border-radius:18px; border:1px solid rgba(255,255,255,.12); background:white; }}
figcaption {{ color:var(--muted); margin-top:8px; }}
.problem {{ border-left:5px solid var(--accent); }}
.footer {{ color:var(--muted); padding-top:20px; }}
</style>
</head>
<body>
<header class='hero'>
  <span class='badge'>SentinelFlow V3</span><span class='badge'>FFT + Deep Learning</span><span class='badge'>Expanded Metrics</span>
  <h1>Intrusion Detection Results Dashboard</h1>
  <p class='sub'>This dashboard summarizes baseline and FFT-enhanced intrusion detection experiments using extended metrics, confusion matrices, classification reports, ROC-AUC, PR-AUC, false positive rate, and false negative rate.</p>
</header>
<main class='wrap'>
<section class='card problem'><h2>Short Problem Statement</h2><p>Existing deep learning-based Intrusion Detection Systems often rely mainly on time-domain flow features and may miss hidden burst patterns, repeated traffic rhythms, and frequency-based attack signatures. SentinelFlow addresses this by using Fast Fourier Transform-enhanced traffic profiling to improve deep learning-based intrusion detection on large-scale network traffic datasets.</p></section>
<div class='grid'>
  <div class='card'><div class='kpi'>{summary.get('rows','N/A')}</div><p>Records analyzed</p></div>
  <div class='card'><div class='kpi'>{summary.get('features','N/A')}</div><p>Input columns</p></div>
  <div class='card'><div class='kpi'>{best}</div><p>Best model by Macro F1-score</p></div>
  <div class='card'><div class='kpi'>{score}</div><p>Best Macro F1-score</p></div>
</div>
<section class='card'><h2>Model Comparison with Expanded Metrics</h2><div class='table-scroll'><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div></section>
<section class='card'><h2>Class Distribution</h2><div class='table-scroll'><table><thead><tr><th>Class</th><th>Count</th><th>Percent</th></tr></thead><tbody>{crows}</tbody></table></div></section>
<section class='card'><h2>Confusion Matrices and Curves</h2>{img_html if img_html else '<p>No image outputs were generated yet. Run Notebook 05 after model training.</p>'}</section>
<p class='footer'>Generated by SentinelFlow. Created by Capstone Group 3: Altonaga, Sarceda, Sunga, Torres</p>
</main>
</body>
</html>"""
