"""Apply frozen UKB raw-DNN checkpoints to BCP RSFC.

Uses untransformed float32 edges and averages seed predictions in float64
before storing the mean predictions as float32.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def read_freeze_list(path: str) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "feature" in df.columns:
        col = "feature"
    elif "target" in df.columns:
        col = "target"
    else:
        col = df.columns[0]
    out = df[col].astype(str).tolist()
    if len(out) == 0:
        raise RuntimeError(f"No targets found in freeze list: {path}")
    if len(out) != len(set(out)):
        raise RuntimeError("Duplicate targets found in freeze list.")
    return out


class EdgeDataset(Dataset):

    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class UKB_DNN_4layer(nn.Module):

    def __init__(self, input_size, n_l1, n_l2, n_l3, output_size, dropout=0.241964):
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

    def forward(self, x):
        return self.fc4(self.fc3(self.fc2(self.fc1(x))))


def infer_arch(state_dict):
    return {
        "input_size": int(state_dict["fc1.1.weight"].shape[1]),
        "n_l1": int(state_dict["fc1.1.weight"].shape[0]),
        "n_l2": int(state_dict["fc2.1.weight"].shape[0]),
        "n_l3": int(state_dict["fc3.1.weight"].shape[0]),
        "output_size": int(state_dict["fc4.1.weight"].shape[0]),
    }


def load_artifacts(model_root: Path, seed: int, expected_output: int, device):
    seed_dir = model_root / f"stage1_seed{seed}"
    model_path = seed_dir / f"dnn_stage1_model_stage1_seed{seed}.pth"
    norm_path = seed_dir / f"dnn_stage1_norm_stage1_seed{seed}.npz"
    log_path = seed_dir / f"dnn_stage1_log_stage1_seed{seed}.npz"
    split_path = seed_dir / f"dnn_stage1_outer_split_stage1_seed{seed}.npz"
    for p in [model_path, norm_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))
    state = torch.load(model_path, map_location=device)
    arch = infer_arch(state)
    if arch["output_size"] != expected_output:
        raise RuntimeError(
            f"Output size mismatch seed {seed}: model={arch['output_size']} freeze={expected_output}"
        )
    norm = np.load(norm_path)
    y_mean = np.asarray(norm["y_mean"], dtype=np.float32)
    y_std = np.asarray(norm["y_std"], dtype=np.float32)
    if y_mean.shape[0] != expected_output or y_std.shape[0] != expected_output:
        raise RuntimeError(
            f"Norm size mismatch seed {seed}: y_mean={y_mean.shape}, y_std={y_std.shape}"
        )
    if (
        not np.isfinite(y_mean).all()
        or not np.isfinite(y_std).all()
        or np.any(y_std <= 0)
    ):
        raise RuntimeError(f"Bad y norm values seed {seed}")
    model = UKB_DNN_4layer(**arch)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return {
        "seed": seed,
        "model": model,
        "arch": arch,
        "y_mean": y_mean,
        "y_std": y_std,
        "model_path": str(model_path),
        "norm_path": str(norm_path),
        "log_path": str(log_path),
        "split_path": str(split_path),
    }


def predict(X, art, batch_size, device):
    loader = DataLoader(
        EdgeDataset(X), batch_size=batch_size, shuffle=False, num_workers=0
    )
    outs = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            y = art["model"](xb).detach().cpu().numpy()
            outs.append(y)
    z = np.vstack(outs).astype(np.float32)
    raw = z * art["y_std"][None, :] + art["y_mean"][None, :]
    return (raw.astype(np.float32), z.astype(np.float32))


def build_frame(meta, target_names, raw, z):
    out = meta.copy()
    for j, name in enumerate(target_names):
        out[name] = raw[:, j].astype(np.float32)
        out[f"{name}__zpred"] = z[:, j].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-csv", required=True)
    ap.add_argument("--fold-manifest-csv", required=True)
    ap.add_argument("--freeze-tsv", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    per_seed_dir = out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    target_names = read_freeze_list(args.freeze_tsv)
    df = pd.read_csv(args.cohort_csv)
    df["_bcp_row"] = np.arange(len(df), dtype=np.int64)
    drop_from_cohort = [c for c in ["fold"] if c in df.columns]
    if drop_from_cohort:
        df = df.drop(columns=drop_from_cohort)
    fold = pd.read_csv(args.fold_manifest_csv)
    needed = ["_bcp_row", "scan_id", "imaging_numeric_id", "fold"]
    for c in needed:
        if c not in fold.columns:
            raise RuntimeError(f"fold manifest missing {c}")
    df = df.merge(
        fold[["_bcp_row", "scan_id", "imaging_numeric_id", "fold"]],
        on=["_bcp_row", "scan_id", "imaging_numeric_id"],
        how="left",
        validate="one_to_one",
    )
    if df["fold"].isna().any():
        raise RuntimeError("Missing fold after merge.")
    edge_cols = sorted(
        [c for c in df.columns if c.startswith("e") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )
    if len(edge_cols) != 1485:
        raise RuntimeError(f"Expected 1485 edges, got {len(edge_cols)}")
    X = df[edge_cols].to_numpy(np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("BCP edge matrix contains NaN/Inf.")
    meta_cols = ["_bcp_row", "scan_id", "imaging_numeric_id", "fold"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_cols = list(dict.fromkeys(meta_cols))
    meta = df[meta_cols].copy()
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    seed_frames = []
    artifact_rows = []
    for seed in seeds:
        print(f"[seed {seed}] loading artifacts", flush=True)
        art = load_artifacts(Path(args.model_root), seed, len(target_names), device)
        if art["arch"]["input_size"] != X.shape[1]:
            raise RuntimeError(
                f"Input size mismatch seed {seed}: model={art['arch']['input_size']} BCP={X.shape[1]}"
            )
        raw, z = predict(X, art, args.batch_size, device)
        frame = build_frame(meta, target_names, raw, z)
        seed_path = per_seed_dir / f"bcp_dnnraw_transferred_stage1_seed{seed}.parquet"
        frame.to_parquet(seed_path, index=False)
        seed_frames.append(frame)
        artifact_rows.append(
            {
                "seed": seed,
                "model_path": art["model_path"],
                "norm_path": art["norm_path"],
                "log_path": art["log_path"],
                "split_path": art["split_path"],
                **art["arch"],
                "y_std_min": float(np.min(art["y_std"])),
                "y_std_max": float(np.max(art["y_std"])),
                "pred_z_global_mean": float(np.mean(z)),
                "pred_z_global_sd": float(np.std(z)),
                "pred_raw_global_mean": float(np.mean(raw)),
                "pred_raw_global_sd": float(np.std(raw)),
            }
        )
    base = seed_frames[0][meta_cols].copy()
    mean_df = base.copy()
    summary_rows = []
    for name in target_names:
        raw_stack = np.stack([f[name].to_numpy(float) for f in seed_frames], axis=1)
        z_stack = np.stack(
            [f[f"{name}__zpred"].to_numpy(float) for f in seed_frames], axis=1
        )
        raw_mean = raw_stack.mean(axis=1)
        z_mean = z_stack.mean(axis=1)
        raw_sd = raw_stack.std(axis=1)
        z_sd = z_stack.std(axis=1)
        mean_df[name] = raw_mean.astype(np.float32)
        mean_df[f"{name}__zpred"] = z_mean.astype(np.float32)
        mean_df[f"{name}__raw_seed_sd"] = raw_sd.astype(np.float32)
        mean_df[f"{name}__zpred_seed_sd"] = z_sd.astype(np.float32)
        summary_rows.append(
            {
                "target": name,
                "mean5_raw_mean": float(np.mean(raw_mean)),
                "mean5_raw_sd": float(np.std(raw_mean)),
                "mean5_raw_min": float(np.min(raw_mean)),
                "mean5_raw_max": float(np.max(raw_mean)),
                "mean5_z_mean": float(np.mean(z_mean)),
                "mean5_z_sd": float(np.std(z_mean)),
                "mean5_z_min": float(np.min(z_mean)),
                "mean5_z_max": float(np.max(z_mean)),
                "mean_seed_sd_z": float(np.mean(z_sd)),
                "mean_seed_sd_raw": float(np.mean(raw_sd)),
            }
        )
    mean_path = out_dir / "bcp_dnnraw_transferred_mean5.parquet"
    compat_path = out_dir / "bcp_predicted_source_mean5_dnnraw.parquet"
    mean_df.to_parquet(mean_path, index=False)
    mean_df.to_parquet(compat_path, index=False)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "bcp_dnnraw_transferred_mean5_target_summary.csv", index=False
    )
    pd.DataFrame(artifact_rows).to_csv(
        out_dir / "bcp_dnnraw_transferred_artifact_summary.csv", index=False
    )
    manifest = {
        "script": "infer_raw_dnn.py",
        "created_at": datetime.now().isoformat(),
        "cohort_csv": args.cohort_csv,
        "fold_manifest_csv": args.fold_manifest_csv,
        "freeze_tsv": args.freeze_tsv,
        "model_root": args.model_root,
        "out_dir": str(out_dir),
        "seeds": seeds,
        "device": str(device),
        "n_rows": int(len(mean_df)),
        "n_edges": int(len(edge_cols)),
        "n_targets": int(len(target_names)),
        "edge_names_first_last": [edge_cols[0], edge_cols[-1]],
        "target_names": target_names,
        "outputs": {
            "mean5_parquet": str(mean_path),
            "compat_parquet": str(compat_path),
        },
    }
    with open(
        out_dir / "bcp_dnnraw_transferred_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)
    print("DONE")
    print(
        json.dumps(
            {
                "n_rows": manifest["n_rows"],
                "n_edges": manifest["n_edges"],
                "n_targets": manifest["n_targets"],
                "mean5": str(mean_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
