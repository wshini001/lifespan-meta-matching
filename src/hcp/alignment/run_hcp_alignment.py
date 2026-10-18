"""Calculate HCP frozen-PC and reconstructed phenotype alignments."""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd


def read_table_auto(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(p, sep="\t")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.read_csv(p)


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def clean_feature_name(x: str) -> str:
    s = str(x)
    s = s.replace("_", " ")
    s = s.replace(".", " ")
    s = re.sub("\\s+", " ", s).strip()
    return s


def short_feature_name(x: str, max_len: int = 58) -> str:
    s = clean_feature_name(x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def infer_domain(block: str) -> str:
    b = str(block)
    mapping = {
        "cognition_core": "Cognition",
        "early_life_adversity": "Early life adversity",
        "lifestyle_environment_superblock": "Lifestyle / environment",
        "mental_health_core": "Mental health",
        "physiology_simple": "Physiology",
        "sleep_circadian": "Sleep / circadian",
    }
    return mapping.get(b, b)


def load_loading_matrix(loading_csv: str, pc_names: list[str], loading_value_col: str):
    load = read_table_auto(loading_csv)
    required = ["raw_col", "pc_name", loading_value_col]
    missing = [c for c in required if c not in load.columns]
    if missing:
        raise KeyError(
            f"loading file missing required columns: {missing}\nAvailable columns: {list(load.columns)}"
        )
    if "block" not in load.columns:
        load["block"] = "unknown"
    if "source_table" not in load.columns:
        load["source_table"] = ""
    load = load[load["pc_name"].isin(pc_names)].copy()
    if load.empty:
        raise RuntimeError(
            f"No loading rows matched HCP final12 PC names. pc_names={pc_names[:5]}..."
        )
    load[loading_value_col] = pd.to_numeric(load[loading_value_col], errors="coerce")
    load = load[np.isfinite(load[loading_value_col])].copy()
    missing_pc = sorted(set(pc_names) - set(load["pc_name"].unique()))
    if missing_pc:
        raise RuntimeError(f"loading file missing PCs: {missing_pc}")
    raw_meta = (
        load[["raw_col", "block", "source_table"]]
        .drop_duplicates("raw_col")
        .reset_index(drop=True)
    )
    Ldf = (
        load.pivot_table(
            index="raw_col", columns="pc_name", values=loading_value_col, aggfunc="mean"
        )
        .reindex(index=raw_meta["raw_col"].tolist(), columns=pc_names)
        .fillna(0.0)
    )
    L = Ldf.to_numpy(dtype=np.float64)
    raw_meta = raw_meta.set_index("raw_col").loc[Ldf.index].reset_index()
    raw_meta["domain"] = raw_meta["block"].map(infer_domain)
    return (load, raw_meta, Ldf, L)


def top_loading_pcs_for_raw(raw_col: str, Ldf: pd.DataFrame, k: int = 4) -> str:
    if raw_col not in Ldf.index:
        return ""
    s = Ldf.loc[raw_col].copy()
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    out = []
    for pc, val in s.head(k).items():
        if abs(float(val)) > 0:
            out.append(f"{pc}:{float(val):+.3f}")
    return "; ".join(out)


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def parse_pc_name_from_zpred(col: str) -> str:
    if not col.endswith("__zpred"):
        raise ValueError(f"not a zpred column: {col}")
    return col[:-7]


def make_pc_importance(
    pred: pd.DataFrame,
    pc: pd.DataFrame,
    best_pred_col: str,
    rsfc_col: str,
    pc_stream_col: str,
    y_col: str,
) -> pd.DataFrame:
    zcols = [c for c in pc.columns if c.endswith("__zpred")]
    if len(zcols) == 0:
        raise RuntimeError("no __zpred columns found in PC parquet")
    pred = pred.copy()
    pred["incremental_component"] = pred[best_pred_col] - pred[rsfc_col]
    targets = {
        "incremental_component": pred["incremental_component"].to_numpy(
            dtype=np.float64
        ),
        "best_transfer_prediction": pred[best_pred_col].to_numpy(dtype=np.float64),
        "pc_stream_prediction": pred[pc_stream_col].to_numpy(dtype=np.float64),
        "y_true_resid": pred[y_col].to_numpy(dtype=np.float64),
        "rsfc_baseline_prediction": pred[rsfc_col].to_numpy(dtype=np.float64),
    }
    rows = []
    for zc in zcols:
        pc_name = parse_pc_name_from_zpred(zc)
        x = pc[zc].to_numpy(dtype=np.float64)
        row = {
            "pc_name": pc_name,
            "zpred_col": zc,
            "n": int(np.isfinite(x).sum()),
            "zpred_mean": float(np.nanmean(x)),
            "zpred_sd": float(np.nanstd(x, ddof=1)),
        }
        for target_name, y in targets.items():
            r = safe_pearson(x, y)
            row[f"r_with_{target_name}"] = r
            row[f"abs_r_with_{target_name}"] = abs(r) if np.isfinite(r) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["abs_r_with_incremental_component", "abs_r_with_best_transfer_prediction"],
        ascending=[False, False],
    ).reset_index(drop=True)
    out.insert(0, "rank_by_abs_r_incremental", np.arange(1, len(out) + 1, dtype=int))
    return out


def calculate_raw_importance(X_pc, targets, raw_meta, Ldf, L):
    X_recon = X_pc @ L.T
    rows = []
    raw_cols = raw_meta["raw_col"].tolist()
    for j, raw_col in enumerate(raw_cols):
        x = X_recon[:, j]
        rows.append(
            {
                "raw_col": raw_col,
                "feature_label": clean_feature_name(raw_col),
                "feature_label_short": short_feature_name(raw_col),
                "block": raw_meta.loc[j, "block"],
                "domain": raw_meta.loc[j, "domain"],
                "source_table": raw_meta.loc[j, "source_table"],
                "n_nonzero_pc_loadings_final12": int(np.sum(np.abs(L[j, :]) > 0)),
                "max_abs_loading_final12": float(np.max(np.abs(L[j, :]))),
                "top_loading_pcs": top_loading_pcs_for_raw(raw_col, Ldf, k=4),
                "r_with_incremental_component": safe_corr(
                    x, targets["incremental_component"]
                ),
                "r_with_best_transfer_prediction": safe_corr(
                    x, targets["best_transfer_prediction"]
                ),
                "r_with_rsfc_prediction": safe_corr(x, targets["rsfc_prediction"]),
                "r_with_pc_stream_prediction": safe_corr(
                    x, targets["pc_stream_prediction"]
                ),
                "r_with_y_true_resid": safe_corr(x, targets["y_true_resid"]),
            }
        )
    imp = pd.DataFrame(rows)
    imp["abs_r_with_incremental_component"] = imp["r_with_incremental_component"].abs()
    imp = imp.sort_values(
        ["abs_r_with_incremental_component", "raw_col"], ascending=[False, True]
    ).reset_index(drop=True)
    imp["rank_by_abs_r_incremental"] = np.arange(1, len(imp) + 1)
    return imp


def align_rows(pred, pc):
    pred = pred.copy()
    pc = pc.copy()
    for frame in [pred, pc]:
        if frame["Subject"].isna().any():
            raise ValueError("Missing participant IDs")
        frame["Subject"] = frame["Subject"].astype("string").str.strip()
        if frame["Subject"].eq("").any() or frame["Subject"].duplicated().any():
            raise ValueError("Empty or duplicate participant IDs")
    pc = pc.set_index("Subject")
    if len(pred) != 1080 or len(pc) != 1080 or set(pred["Subject"]) != set(pc.index):
        raise ValueError("Expected the same 1,080 participants in both inputs")
    pc = pc.loc[pred["Subject"]].reset_index()
    pred = pred.reset_index(drop=True)
    for key in ["locked_row_idx", "fold"]:
        if key in pred and key in pc:
            if not np.array_equal(
                pd.to_numeric(pred[key]).to_numpy(), pd.to_numeric(pc[key]).to_numpy()
            ):
                raise ValueError(f"Aligned {key} values differ")
    return (pred, pc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--pc-parquet", required=True)
    config = Path(__file__).resolve().parent / "config"
    parser.add_argument("--loadings-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.predictions_csv, dtype={"Subject": "string"})
    pc = pd.read_parquet(args.pc_parquet)
    pred, pc = align_rows(pred, pc)
    best = "y_pred_rsfc_plus_pc_nested"
    base = "y_pred_rsfc_nested"
    stream = "y_pred_pc_nested"
    outcome = "y_true_resid"
    required_predictions = [best, base, stream, outcome]
    missing = [column for column in required_predictions if column not in pred]
    if missing:
        raise KeyError(f"HCP predictions are missing required columns: {missing}")
    zcols = [c for c in pc if c.endswith("__zpred")]
    expected_pcs = (config / "pc_names.txt").read_text().splitlines()
    if len(zcols) != 12 or set((c[:-7] for c in zcols)) != set(expected_pcs):
        raise ValueError("PC identities differ from the configured Final12")
    if (
        not np.isfinite(pc[zcols].to_numpy(float)).all()
        or not np.isfinite(pred[[best, base, stream, outcome]].to_numpy(float)).all()
    ):
        raise ValueError("Non-finite prediction, outcome or PC values")
    y = pred[outcome].to_numpy(float)
    importance = make_pc_importance(pred, pc, best, base, stream, outcome)
    names = [c[:-7] for c in zcols]
    _, meta, ldf, loadings = load_loading_matrix(
        args.loadings_tsv, names, "loading_aligned_to_saved_pc"
    )
    targets = {
        "incremental_component": pred[best].to_numpy(float)
        - pred[base].to_numpy(float),
        "best_transfer_prediction": pred[best].to_numpy(float),
        "rsfc_prediction": pred[base].to_numpy(float),
        "pc_stream_prediction": pred[stream].to_numpy(float),
        "y_true_resid": y,
    }
    raw = calculate_raw_importance(
        pc[zcols].to_numpy(float), targets, meta, ldf, loadings
    )
    if len(raw) != 112:
        raise ValueError("Expected 112 reconstructed raw phenotypes")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out / "hcp_pc_alignments.tsv", sep="\t", index=False)
    raw.to_csv(out / "hcp_reconstructed_alignments.tsv", sep="\t", index=False)
    print("PASS: HCP participant-level PC and reconstructed alignments calculated")


if __name__ == "__main__":
    main()
