"""Apply frozen UKB DNN checkpoints to ABCD RSFC and average five seeds.

Use the raw or PC checkpoint directory with its matching target list.
RSFC inputs are not standardized or covariate-residualized here.
"""

import argparse
import re
import json
import random
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader


def read_freeze_list(freeze_tsv_path: str) -> list[str]:
    cols = []
    with open(freeze_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            cols.append(s.split("\t")[0])
    if cols and cols[0].lower() == "col":
        cols = cols[1:]
    if len(cols) == 0:
        raise ValueError(f"freeze list empty: {freeze_tsv_path}")
    return cols


def get_edge_cols_from_header(csv_path: str) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    e_cols = []
    for c in header:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            e_cols.append((int(m.group(1)), c))
    if not e_cols:
        raise ValueError(f"No edge columns found in header: {csv_path}")
    e_cols.sort(key=lambda x: x[0])
    return [c for _, c in e_cols]


class UKBDataset(Dataset):

    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32)).float()

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return int(self.x.shape[0])


class UKB_DNN_4layer(nn.Module):

    def __init__(
        self,
        input_size: int,
        n_l1: int = 87,
        n_l2: int = 386,
        n_l3: int = 313,
        dropout: float = 0.241964,
        output_size: int = 1,
    ):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),
        )
        self.fc4 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(n_l3, output_size))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _load_abcd_for_inference(
    abcd_csv: str,
    ukb_csv: str,
    id_col: str = "src_subject_id",
    subjectkey_col: str = "subjectkey",
    date_col: str = "interview_date",
    age_col: str = "age",
    sex_col: str = "sex",
    site_col: str = "site",
    target_col: str = "intelligence_score",
    min_site_n: int = 50,
):
    ukb_edges = get_edge_cols_from_header(ukb_csv)
    abcd_edges = get_edge_cols_from_header(abcd_csv)
    if abcd_edges != ukb_edges:
        raise ValueError(
            "ABCD edge order does not match UKB edge order; inference must stop."
        )
    base_cols = [
        subjectkey_col,
        id_col,
        date_col,
        age_col,
        sex_col,
        site_col,
        target_col,
    ]
    header = pd.read_csv(abcd_csv, nrows=0).columns.tolist()
    missing = [c for c in base_cols if c not in header]
    if missing:
        raise KeyError(
            f"ABCD csv missing required columns for inference bundle: {missing}"
        )
    usecols = base_cols + abcd_edges
    df = pd.read_csv(abcd_csv, usecols=usecols)
    df["_orig_row_idx"] = np.arange(len(df), dtype=np.int64)
    vc = df[site_col].value_counts(dropna=False)
    small_sites = vc[vc < int(min_site_n)].index
    df_site = df[~df[site_col].isin(small_sites)].copy()
    df_site["_interview_dt"] = _ensure_datetime(df_site[date_col])
    df_site["_age_num"] = pd.to_numeric(df_site[age_col], errors="coerce")
    key_cols_for_missing = [date_col, age_col, sex_col, site_col, target_col]
    df_site["_key_missing_count"] = (
        df_site[key_cols_for_missing].isna().sum(axis=1).astype(np.int64)
    )
    df_sorted = df_site.sort_values(
        by=[id_col, "_interview_dt", "_age_num", "_key_missing_count", "_orig_row_idx"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).copy()
    keep_mask = ~df_sorted.duplicated(subset=[id_col], keep="first")
    df_sorted["_keep_for_stage2a"] = keep_mask
    dup_manifest = df_sorted[df_sorted.duplicated(subset=[id_col], keep=False)].copy()
    dedup = df_sorted[df_sorted["_keep_for_stage2a"]].copy()
    dedup = dedup.sort_values("_orig_row_idx", kind="mergesort").reset_index(drop=True)
    n_edges = len(abcd_edges)
    X_edges = dedup[abcd_edges].to_numpy(dtype=np.float32)
    if X_edges.shape[1] != n_edges:
        raise RuntimeError(
            f"Edge matrix width mismatch: {X_edges.shape[1]} vs expected {n_edges}"
        )
    if not np.isfinite(X_edges).all():
        raise RuntimeError(
            "ABCD edge matrix contains NaN/Inf after site filtering + dedup; inference cannot proceed."
        )
    stats = {
        "n_rows_raw": int(len(df)),
        "n_rows_after_small_site_filter": int(len(df_site)),
        "n_small_sites_removed": int(len(small_sites)),
        "small_sites_removed": {
            str(k): int(v) for k, v in vc[vc < int(min_site_n)].to_dict().items()
        },
        "n_duplicate_subject_rows_after_site_filter": int(len(dup_manifest)),
        "n_duplicate_unique_subjects_after_site_filter": (
            int(dup_manifest[id_col].nunique()) if len(dup_manifest) > 0 else 0
        ),
        "n_rows_deduped": int(len(dedup)),
        "n_edge_cols": int(n_edges),
        "id_col": id_col,
        "subjectkey_col": subjectkey_col,
        "date_col": date_col,
        "age_col": age_col,
        "sex_col": sex_col,
        "site_col": site_col,
        "target_col": target_col,
        "dedup_rule": "keep earliest interview_date; tie-break by smaller age, fewer missing key fields, then smaller original row index",
    }
    return {
        "edges": abcd_edges,
        "df_site": df_site,
        "df_sorted": df_sorted,
        "dup_manifest": dup_manifest,
        "dedup": dedup,
        "X_edges": X_edges,
        "stats": stats,
        "small_sites": small_sites.tolist(),
    }


def _infer_arch_from_state_dict(state_dict: dict) -> dict:
    try:
        w1 = state_dict["fc1.1.weight"]
        w2 = state_dict["fc2.1.weight"]
        w3 = state_dict["fc3.1.weight"]
        w4 = state_dict["fc4.1.weight"]
    except KeyError as e:
        raise KeyError(
            f"Unexpected Stage1 model state_dict layout; missing key: {e}"
        ) from e
    return {
        "input_size": int(w1.shape[1]),
        "n_l1": int(w1.shape[0]),
        "n_l2": int(w2.shape[0]),
        "n_l3": int(w3.shape[0]),
        "output_size": int(w4.shape[0]),
    }


def _load_stage1_artifacts_for_seed(model_root: str, seed: int, freeze_tsv):
    seed_dir = Path(model_root) / f"stage1_seed{seed}"
    model_path = seed_dir / f"dnn_stage1_model_stage1_seed{seed}.pth"
    norm_path = seed_dir / f"dnn_stage1_norm_stage1_seed{seed}.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Stage1 model file: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing Stage1 norm file: {norm_path}")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    norm = np.load(norm_path)
    y_mean = np.asarray(norm["y_mean"], dtype=np.float32)
    y_std = np.asarray(norm["y_std"], dtype=np.float32)
    arch = _infer_arch_from_state_dict(state_dict)
    if arch["output_size"] != y_mean.shape[0] or arch["output_size"] != y_std.shape[0]:
        raise RuntimeError(
            f"Norm/model output mismatch for seed {seed}: output_size={arch['output_size']} y_mean={y_mean.shape} y_std={y_std.shape}"
        )
    log_path = seed_dir / f"dnn_stage1_log_stage1_seed{seed}.npz"
    if not log_path.exists():
        raise FileNotFoundError(
            f"Missing training log for output-name validation: {log_path}"
        )
    with np.load(log_path, allow_pickle=False) as log:
        saved_names = log["target_names"].astype(str).tolist()
    if saved_names != read_freeze_list(freeze_tsv):
        raise RuntimeError(
            f"Training target order differs from freeze list for seed {seed}."
        )
    return {
        "seed_dir": seed_dir,
        "model_path": model_path,
        "norm_path": norm_path,
        "state_dict": state_dict,
        "y_mean": y_mean,
        "y_std": y_std,
        "arch": arch,
    }


def _predict_with_stage1_model(
    X_edges: np.ndarray, artifacts: dict, batch_size: int, device: torch.device
):
    arch = artifacts["arch"]
    model = UKB_DNN_4layer(
        input_size=arch["input_size"],
        n_l1=arch["n_l1"],
        n_l2=arch["n_l2"],
        n_l3=arch["n_l3"],
        dropout=0.241964,
        output_size=arch["output_size"],
    ).to(device)
    model.load_state_dict(artifacts["state_dict"])
    model.eval()
    if X_edges.shape[1] != arch["input_size"]:
        raise RuntimeError(
            f"Input width mismatch for inference: X has {X_edges.shape[1]} cols, model expects {arch['input_size']}"
        )
    loader = DataLoader(
        UKBDataset(X_edges), batch_size=batch_size, shuffle=False, num_workers=0
    )
    preds_z = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds_z.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
    y_pred_z = np.concatenate(preds_z, axis=0)
    y_pred_raw = y_pred_z * artifacts["y_std"][None, :] + artifacts["y_mean"][None, :]
    return (y_pred_z, y_pred_raw)


def _build_prediction_frame(
    meta_df: pd.DataFrame,
    meta_cols: Sequence[str],
    y_names: Sequence[str],
    y_pred_raw: np.ndarray,
    y_pred_z: np.ndarray,
):
    if y_pred_raw.shape[1] != len(y_names) or y_pred_z.shape[1] != len(y_names):
        raise RuntimeError("Prediction arrays do not match target_names length.")
    missing_meta = [c for c in meta_cols if c not in meta_df.columns]
    if missing_meta:
        raise KeyError(f"Missing metadata columns when saving bundle: {missing_meta}")
    out = meta_df[list(meta_cols)].copy()
    for j, name in enumerate(y_names):
        out[name] = y_pred_raw[:, j].astype(np.float32)
        out[f"{name}__zpred"] = y_pred_z[:, j].astype(np.float32)
    return out


def run_stage2a_inference(
    abcd_csv: str,
    ukb_csv: str,
    freeze_tsv: str,
    model_root: str,
    out_dir: str,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    min_site_n: int = 50,
    batch_size: int = 256,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    y_names = read_freeze_list(freeze_tsv)
    abcd = _load_abcd_for_inference(
        abcd_csv=abcd_csv, ukb_csv=ukb_csv, min_site_n=min_site_n
    )
    dedup = abcd["dedup"].copy()
    X_edges = abcd["X_edges"]
    meta_cols = [
        abcd["stats"]["subjectkey_col"],
        abcd["stats"]["id_col"],
        abcd["stats"]["date_col"],
        abcd["stats"]["age_col"],
        abcd["stats"]["sex_col"],
        abcd["stats"]["site_col"],
        abcd["stats"]["target_col"],
    ]
    dedup_manifest_csv = out_dir / "abcd_stage2a_dedup_manifest.csv"
    dedup_selected_csv = out_dir / "abcd_stage2a_selected_cohort.csv"
    stats_json = out_dir / "abcd_stage2a_metadata.json"
    dup_manifest_cols = list(meta_cols) + [
        "_orig_row_idx",
        "_interview_dt",
        "_age_num",
        "_key_missing_count",
        "_keep_for_stage2a",
    ]
    keep_dup_cols = [c for c in dup_manifest_cols if c in abcd["df_sorted"].columns]
    abcd["df_sorted"][keep_dup_cols].to_csv(dedup_manifest_csv, index=False)
    selected_cols = list(meta_cols) + ["_orig_row_idx"]
    dedup[[c for c in selected_cols if c in dedup.columns]].to_csv(
        dedup_selected_csv, index=False
    )
    metadata = {
        **abcd["stats"],
        "abcd_csv": str(abcd_csv),
        "ukb_csv": str(ukb_csv),
        "freeze_tsv": str(freeze_tsv),
        "model_root": str(model_root),
        "out_dir": str(out_dir),
        "seeds": list(map(int, seeds)),
        "batch_size": int(batch_size),
        "device": str(device),
        "target_names_count": int(len(y_names)),
        "target_names_file_order_preserved": True,
    }
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    pred_raw_all = []
    pred_z_all = []
    for seed in seeds:
        artifacts = _load_stage1_artifacts_for_seed(
            model_root=model_root, seed=int(seed), freeze_tsv=freeze_tsv
        )
        if int(artifacts["arch"]["output_size"]) != int(len(y_names)):
            raise RuntimeError(
                f"Seed {seed}: model output_size={artifacts['arch']['output_size']} but freeze_tsv has {len(y_names)} targets"
            )
        y_pred_z, y_pred_raw = _predict_with_stage1_model(
            X_edges=X_edges, artifacts=artifacts, batch_size=batch_size, device=device
        )
        pred_raw_all.append(y_pred_raw)
        pred_z_all.append(y_pred_z)
        seed_frame = _build_prediction_frame(
            dedup, meta_cols, y_names, y_pred_raw, y_pred_z
        )
        seed_parquet = out_dir / f"abcd_predicted_source_seed{seed}.parquet"
        seed_csv_gz = out_dir / f"abcd_predicted_source_seed{seed}.csv.gz"
        seed_frame.to_parquet(seed_parquet, index=False)
        seed_frame.to_csv(seed_csv_gz, index=False, compression="gzip")
        np.savez(
            out_dir / f"abcd_predicted_source_seed{seed}.npz",
            subjectkey=dedup[abcd["stats"]["subjectkey_col"]].to_numpy(dtype=object),
            src_subject_id=dedup[abcd["stats"]["id_col"]].to_numpy(dtype=object),
            target_names=np.asarray(y_names, dtype=object),
            y_pred_raw=y_pred_raw.astype(np.float32),
            y_pred_z=y_pred_z.astype(np.float32),
        )
    mean_raw = np.mean(np.stack(pred_raw_all, axis=0), axis=0)
    mean_z = np.mean(np.stack(pred_z_all, axis=0), axis=0)
    mean_frame = _build_prediction_frame(dedup, meta_cols, y_names, mean_raw, mean_z)
    mean_frame.to_parquet(out_dir / "abcd_predicted_source_mean5.parquet", index=False)
    mean_frame.to_csv(
        out_dir / "abcd_predicted_source_mean5.csv.gz", index=False, compression="gzip"
    )
    rows = []
    for seed, arr in zip(seeds, pred_raw_all):
        rows.append(
            {
                "seed": int(seed),
                "n_subjects": int(arr.shape[0]),
                "n_targets": int(arr.shape[1]),
                "global_mean_pred_raw": float(np.mean(arr)),
                "global_std_pred_raw": float(np.std(arr)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "abcd_stage2a_seed_summary.csv", index=False)
    print("[Stage2A] done")
    print(f"[Stage2A] selected cohort rows = {len(dedup)}")
    print(f"[Stage2A] output dir = {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abcd-csv", required=True)
    parser.add_argument("--ukb-csv", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-site-n", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and (not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable.")
    device = None if args.device == "auto" else torch.device(args.device)
    run_stage2a_inference(
        abcd_csv=args.abcd_csv,
        ukb_csv=args.ukb_csv,
        freeze_tsv=args.freeze_tsv,
        model_root=args.model_root,
        out_dir=args.out_dir,
        seeds=(0, 1, 2, 3, 4),
        min_site_n=args.min_site_n,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
