"""Train UKB multitask DNN source models for raw phenotypes or selected PCs.

RSFC enters unchanged. Targets are normalized within each seed's training
subset. No covariate residualization is performed here. Cohort and
missingness rules depend on the source family.
"""

import argparse
import os
import re
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader


def load_raw_source_data(
    ukb_final_csv, cov_parquet, freeze_tsv, site_col="QC_Imaging_Site"
):
    y_names = read_freeze_list(freeze_tsv)
    try:
        import pyarrow.parquet as pq

        pq_cols = pq.read_schema(cov_parquet).names
    except Exception:
        pq_cols = pd.read_parquet(cov_parquet).head(0).columns.tolist()
    pq_cols_set = set(map(str, pq_cols))
    ind_cols_exist = []
    for y in y_names:
        ind = f"{y}__is_missing"
        if ind in pq_cols_set:
            ind_cols_exist.append(ind)
    need_cols = ["eid"] + y_names + ind_cols_exist
    cov_df = pd.read_parquet(cov_parquet, columns=need_cols)
    edge_cols = get_edge_cols_from_header(ukb_final_csv)
    x_cols = ["eid"] + edge_cols
    need_site = False
    if site_col is not None:
        header = pd.read_csv(ukb_final_csv, nrows=0).columns.tolist()
        if site_col in header:
            need_site = True
            x_cols.append(site_col)
    ukb_df = pd.read_csv(ukb_final_csv, usecols=x_cols)
    cov_df["eid"] = pd.to_numeric(cov_df["eid"], errors="raise").astype(np.int64)
    ukb_df["eid"] = pd.to_numeric(ukb_df["eid"], errors="raise").astype(np.int64)
    n_cov = cov_df["eid"].nunique()
    n_ukb = ukb_df["eid"].nunique()
    merged = ukb_df.merge(cov_df, on="eid", how="inner", sort=False)
    if merged.shape[0] == 0:
        raise RuntimeError("eid join produced 0 rows; please check eid types/format.")
    n_merged = merged["eid"].nunique()
    if n_merged != n_cov or n_merged != n_ukb:
        raise RuntimeError(
            f"eid join lost samples: ukb={n_ukb}, cov={n_cov}, merged={n_merged}. Please check eid dtype/format and subject alignment."
        )
    X_edges = merged[edge_cols].to_numpy(dtype=np.float32)
    Y = merged[y_names].to_numpy(dtype=np.float32)
    mask = np.ones_like(Y, dtype=bool)
    for j, y in enumerate(y_names):
        ind = f"{y}__is_missing"
        if ind in merged.columns:
            m = merged[ind].to_numpy()
            mask[:, j] = m == 0
        else:
            mask[:, j] = True
    site_vec = None
    if need_site:
        site_vec = merged[site_col].astype(str).to_numpy()
    eid = merged["eid"].to_numpy()
    return (X_edges, Y, mask, eid, y_names, site_vec)


def load_pc_source_data(
    ukb_final_csv, cov_parquet, freeze_tsv, site_col="QC_Imaging_Site"
):
    y_names = read_freeze_list(freeze_tsv)
    try:
        import pyarrow.parquet as pq

        pq_cols = pq.read_schema(cov_parquet).names
    except Exception:
        pq_cols = pd.read_parquet(cov_parquet).head(0).columns.tolist()
    pq_cols_set = set(map(str, pq_cols))
    ind_cols_exist = []
    for y in y_names:
        ind = f"{y}__is_missing"
        if ind in pq_cols_set:
            ind_cols_exist.append(ind)
    need_cols = ["eid"] + y_names + ind_cols_exist
    cov_df = pd.read_parquet(cov_parquet, columns=need_cols)
    edge_cols = get_edge_cols_from_header(ukb_final_csv)
    x_cols = ["eid"] + edge_cols
    need_site = False
    if site_col is not None:
        header = pd.read_csv(ukb_final_csv, nrows=0).columns.tolist()
        if site_col in header:
            need_site = True
            x_cols.append(site_col)
    ukb_df = pd.read_csv(ukb_final_csv, usecols=x_cols)
    cov_df["eid"] = pd.to_numeric(cov_df["eid"], errors="raise").astype(np.int64)
    ukb_df["eid"] = pd.to_numeric(ukb_df["eid"], errors="raise").astype(np.int64)
    n_cov = cov_df["eid"].nunique()
    n_ukb = ukb_df["eid"].nunique()
    merged = ukb_df.merge(cov_df, on="eid", how="inner", sort=False)
    if merged.shape[0] == 0:
        raise RuntimeError("eid join produced 0 rows; please check eid types/format.")
    n_merged = merged["eid"].nunique()
    if n_merged != n_cov:
        raise RuntimeError(
            f"eid join lost target-bank samples: ukb={n_ukb}, cov={n_cov}, merged={n_merged}. Some subjects in cov_parquet were not found in UKB_final.csv."
        )
    if n_merged != n_ukb:
        print(
            f"[load_stage1_data] using target-bank subjects only: ukb={n_ukb}, cov={n_cov}, merged={n_merged}"
        )
    X_edges = merged[edge_cols].to_numpy(dtype=np.float32)
    Y = merged[y_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(Y)
    for j, y in enumerate(y_names):
        ind = f"{y}__is_missing"
        if ind in merged.columns:
            m = merged[ind].to_numpy()
            mask[:, j] = np.isfinite(Y[:, j]) & (m == 0)
    site_vec = None
    if need_site:
        site_vec = merged[site_col].astype(str).to_numpy()
    eid = merged["eid"].to_numpy()
    return (X_edges, Y, mask, eid, y_names, site_vec)


def read_freeze_list(freeze_tsv_path):
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


def get_edge_cols_from_header(ukb_final_csv):
    header = pd.read_csv(ukb_final_csv, nrows=0).columns.tolist()
    e_cols = []
    for c in header:
        m = re.fullmatch("e(\\d+)", str(c))
        if m:
            e_cols.append((int(m.group(1)), c))
    if not e_cols:
        raise ValueError("No RSFC edge columns found in the UKB input header.")
    e_cols.sort(key=lambda x: x[0])
    return [c for _, c in e_cols]


def torch_nanmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num = torch.where(mask, torch.zeros_like(x), torch.ones_like(x)).sum()
    value = torch.where(mask, torch.zeros_like(x), x).sum()
    if num.item() == 0:
        return value * 0.0
    return value / num


def msenanloss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if mask is None:
        mask = torch.isnan(input) | torch.isnan(target)
    safe_input = input.masked_fill(mask, 0.0)
    safe_target = target.masked_fill(mask, 0.0)
    ret = (safe_input - safe_target) ** 2
    return torch_nanmean(ret, mask)


class UKBDataset(Dataset):

    def __init__(
        self, x: np.ndarray, y: np.ndarray, valid_mask: np.ndarray | None = None
    ):
        if y.ndim == 1:
            y = y[:, None]
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.valid_mask = None
        if valid_mask is not None:
            if valid_mask.ndim == 1:
                valid_mask = valid_mask[:, None]
            self.valid_mask = torch.from_numpy(valid_mask).bool()

    def __getitem__(self, idx):
        if self.valid_mask is None:
            return (self.x[idx], self.y[idx])
        return (self.x[idx], self.y[idx], self.valid_mask[idx])

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


def compute_metrics(y_true, y_pred, y_sigma=None, valid_mask=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"compute_metrics: shape mismatch y_true={y_true.shape} y_pred={y_pred.shape}"
        )
    N, T = y_true.shape
    if valid_mask is None:
        valid_mask = np.ones((N, T), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask).astype(bool)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask[:, None]
        if valid_mask.shape != (N, T):
            raise ValueError(
                f"compute_metrics: valid_mask shape mismatch {valid_mask.shape} vs {(N, T)}"
            )
    if y_sigma is None:
        sigma = None
    else:
        sigma = np.asarray(y_sigma)
        if sigma.ndim == 0:
            sigma = float(sigma)
        else:
            sigma = sigma.astype(np.float32)
            if sigma.shape != (T,):
                raise ValueError(
                    f"compute_metrics: y_sigma shape {sigma.shape} incompatible with T={T}"
                )
    r2_arr = np.full(T, np.nan, dtype=np.float32)
    r_arr = np.full(T, np.nan, dtype=np.float32)
    cod_arr = np.full(T, np.nan, dtype=np.float32)
    rmse_z = np.full(T, np.nan, dtype=np.float32)
    mae_z = np.full(T, np.nan, dtype=np.float32)
    rmse = np.full(T, np.nan, dtype=np.float32)
    mae = np.full(T, np.nan, dtype=np.float32)
    for j in range(T):
        v = valid_mask[:, j]
        if v.sum() == 0:
            continue
        yt = y_true[v, j].astype(np.float32)
        yp = y_pred[v, j].astype(np.float32)
        mse = mean_squared_error(yt, yp)
        _rmse = float(np.sqrt(mse))
        _mae = float(np.mean(np.abs(yt - yp)))
        rmse_z[j] = _rmse
        mae_z[j] = _mae
        r2_arr[j] = float(r2_score(yt, yp))
        if np.std(yt) > 0 and np.std(yp) > 0:
            r_arr[j] = float(np.corrcoef(yt, yp)[0, 1])
        sse = float(np.sum((yt - yp) ** 2))
        if sigma is not None:
            sst_for_cod = float(np.sum(yt**2))
        else:
            sst_for_cod = float(np.sum((yt - np.mean(yt)) ** 2))
        cod_arr[j] = float(1.0 - sse / sst_for_cod) if sst_for_cod > 0 else np.nan
        if sigma is None:
            rmse[j] = _rmse
            mae[j] = _mae
        elif isinstance(sigma, float):
            rmse[j] = _rmse * sigma
            mae[j] = _mae * sigma
        else:
            rmse[j] = _rmse * float(sigma[j])
            mae[j] = _mae * float(sigma[j])
    metrics = {
        "R²_mean": float(np.nanmean(r2_arr)),
        "Pearson_r_mean": float(np.nanmean(r_arr)),
        "COD_mean": float(np.nanmean(cod_arr)),
        "RMSE_mean": float(np.nanmean(rmse)),
        "MAE_mean": float(np.nanmean(mae)),
        "RMSE_z_mean": float(np.nanmean(rmse_z)),
        "MAE_z_mean": float(np.nanmean(mae_z)),
        "R²_per_task": r2_arr,
        "Pearson_r_per_task": r_arr,
        "COD_per_task": cod_arr,
        "RMSE_per_task": rmse,
        "MAE_per_task": mae,
        "RMSE_z_per_task": rmse_z,
        "MAE_z_per_task": mae_z,
    }
    if T == 1:
        metrics_old = {
            "R²": metrics["R²_mean"],
            "Pearson_r": metrics["Pearson_r_mean"],
            "COD": metrics["COD_mean"],
            "RMSE": metrics["RMSE_mean"],
            "MAE": metrics["MAE_mean"],
        }
        if y_sigma is not None:
            metrics_old["RMSE_z"] = metrics["RMSE_z_mean"]
            metrics_old["MAE_z"] = metrics["MAE_z_mean"]
        return metrics_old
    return metrics


def _collect_loader_metrics(loader, model, device, y_sigma):
    model.eval()
    all_true = []
    all_pred = []
    all_valid = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_valid = batch
            else:
                batch_x, batch_y = batch
                batch_valid = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            y_np = batch_y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            if y_np.ndim == 1:
                y_np = y_np[:, None]
            if p_np.ndim == 1:
                p_np = p_np[:, None]
            if y_np.shape != p_np.shape:
                raise ValueError(
                    f"_collect_loader_metrics: shape mismatch y={y_np.shape} pred={p_np.shape}"
                )
            if batch_valid is None:
                v_np = np.ones_like(y_np, dtype=bool)
            else:
                v_np = batch_valid.detach().bool().cpu().numpy()
                if v_np.ndim == 1:
                    v_np = v_np[:, None]
                if v_np.shape != y_np.shape:
                    raise ValueError(
                        f"_collect_loader_metrics: valid_mask shape mismatch {v_np.shape} vs {y_np.shape}"
                    )
            all_true.append(y_np.astype(np.float32, copy=False))
            all_pred.append(p_np.astype(np.float32, copy=False))
            all_valid.append(v_np)
    if len(all_true) == 0:
        return {}
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    valid_mask = np.concatenate(all_valid, axis=0)
    return compute_metrics(y_true, y_pred, y_sigma=y_sigma, valid_mask=valid_mask)


def _fit_eval_one_outer_split_dnn_stage1(
    X_train_val,
    Y_train_val,
    valid_mask_train_val,
    X_test,
    Y_test,
    valid_mask_test,
    eid_test=None,
    y_names=None,
    site_trv=None,
    random_state=0,
    run_tag="stage1",
    fold_tag="",
    out_dir=".",
    num_workers=4,
    device=None,
    n_l1=87,
    n_l2=386,
    n_l3=313,
    dropout=0.241964,
    batch_size=128,
    epochs=300,
    lr=0.003645653,
    momentum=0.9,
    weight_decay=0.000844732,
    scheduler_decrease=75,
    patience=20,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(random_state)

    def _can_stratify(labels):
        if labels is None:
            return False
        uniq, cnt = np.unique(labels, return_counts=True)
        if uniq.size < 2:
            return False
        if (cnt * 0.2 < 1).any() or (cnt * 0.8 < 1).any():
            return False
        return True

    use_stratify_inner = _can_stratify(site_trv)
    idx_all = np.arange(X_train_val.shape[0])
    if use_stratify_inner:
        idx_tr, idx_va = train_test_split(
            idx_all, test_size=0.2, random_state=seed, stratify=site_trv
        )
    else:
        idx_tr, idx_va = train_test_split(
            idx_all, test_size=0.2, random_state=seed, shuffle=True
        )
    X_tr = X_train_val[idx_tr]
    X_va = X_train_val[idx_va]
    X_te = X_test
    y_tr = Y_train_val[idx_tr]
    y_va = Y_train_val[idx_va]
    y_te = Y_test
    m_tr = valid_mask_train_val[idx_tr]
    m_va = valid_mask_train_val[idx_va]
    m_te = valid_mask_test
    y_mean = np.zeros(y_tr.shape[1], dtype=np.float32)
    y_std = np.ones(y_tr.shape[1], dtype=np.float32)
    for j in range(y_tr.shape[1]):
        v = m_tr[:, j]
        if v.sum() == 0:
            y_mean[j] = 0.0
            y_std[j] = 1.0
        else:
            col = y_tr[v, j].astype(np.float32)
            y_mean[j] = float(col.mean())
            s = float(col.std())
            y_std[j] = 1.0 if s < 1e-12 else s
    y_tr_z = (y_tr - y_mean[None, :]) / y_std[None, :]
    y_va_z = (y_va - y_mean[None, :]) / y_std[None, :]
    y_te_z = (y_te - y_mean[None, :]) / y_std[None, :]
    train_dataset = UKBDataset(X_tr, y_tr_z, valid_mask=m_tr)
    val_dataset = UKBDataset(X_va, y_va_z, valid_mask=m_va)
    test_dataset = UKBDataset(X_te, y_te_z, valid_mask=m_te)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    input_dim = X_tr.shape[1]
    out_dim = y_tr_z.shape[1]
    model = UKB_DNN_4layer(
        input_size=input_dim,
        n_l1=n_l1,
        n_l2=n_l2,
        n_l3=n_l3,
        dropout=dropout,
        output_size=out_dim,
    ).to(device)
    criterion = msenanloss
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[scheduler_decrease, 2 * scheduler_decrease], gamma=0.1
    )
    train_losses, val_losses = ([], [])
    train_metrics_hist = []
    val_metrics_hist = []
    n_task = int(y_tr_z.shape[1])
    train_cor_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    train_cod_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    train_mae_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    val_cor_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    val_cod_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    val_mae_record = np.full((epochs, n_task), np.nan, dtype=np.float32)
    train_cor_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    train_cod_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    train_mae_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    val_cor_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    val_cod_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    val_mae_mean_record = np.full((epochs,), np.nan, dtype=np.float32)
    best_val_cod = -np.inf
    best_state = None
    no_improve = 0
    best_epoch = -1
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_valid = batch
            else:
                batch_x, batch_y = batch
                batch_valid = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if batch_valid is not None:
                batch_valid = batch_valid.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            if batch_valid is None:
                loss = criterion(pred, batch_y)
            else:
                invalid_mask = ~batch_valid
                batch_y_safe = batch_y.clone()
                batch_y_safe.masked_fill_(invalid_mask, 0.0)
                loss = criterion(pred, batch_y_safe, mask=invalid_mask)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        model.train(False)
        train_metrics = _collect_loader_metrics(
            train_loader, model, device, y_sigma=y_std
        )
        train_metrics_hist.append(train_metrics)
        train_cor_record[epoch, :] = np.asarray(
            train_metrics["Pearson_r_per_task"], dtype=np.float32
        )
        train_cod_record[epoch, :] = np.asarray(
            train_metrics["COD_per_task"], dtype=np.float32
        )
        train_mae_record[epoch, :] = np.asarray(
            train_metrics["MAE_per_task"], dtype=np.float32
        )
        train_cor_mean_record[epoch] = np.float32(train_metrics["Pearson_r_mean"])
        train_cod_mean_record[epoch] = np.float32(train_metrics["COD_mean"])
        train_mae_mean_record[epoch] = np.float32(train_metrics["MAE_mean"])
        model.eval()
        epoch_val_loss = 0.0
        val_true_batches = []
        val_pred_batches = []
        val_valid_batches = []
        n_task_expected = None
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_x, batch_y, batch_valid = batch
                else:
                    batch_x, batch_y = batch
                    batch_valid = None
                if batch_valid is None:
                    raise RuntimeError(
                        "[stage1 val] batch_valid is None; expected valid masks in Stage1"
                    )
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_valid = batch_valid.to(device)
                pred = model(batch_x)
                y_np = batch_y.detach().cpu().numpy()
                p_np = pred.detach().cpu().numpy()
                if y_np.ndim == 1:
                    y_np = y_np[:, None]
                if p_np.ndim == 1:
                    p_np = p_np[:, None]
                if y_np.shape != p_np.shape:
                    raise ValueError(
                        f"[stage1 val] shape mismatch y={y_np.shape} pred={p_np.shape}"
                    )
                if n_task_expected is None:
                    n_task_expected = y_np.shape[1]
                elif y_np.shape[1] != n_task_expected:
                    raise ValueError(
                        f"[stage1 val] inconsistent task dim across batches: got {y_np.shape[1]}, expected {n_task_expected}"
                    )
                v_np = batch_valid.detach().cpu().numpy()
                if v_np.ndim == 1:
                    v_np = v_np[:, None]
                v_np = v_np.astype(bool, copy=False)
                if v_np.shape != y_np.shape:
                    raise ValueError(
                        f"[stage1 val] valid_mask shape mismatch v={v_np.shape} y={y_np.shape}"
                    )
                invalid_mask = ~batch_valid
                batch_y_safe = batch_y.clone()
                batch_y_safe.masked_fill_(invalid_mask, 0.0)
                loss = criterion(pred, batch_y_safe, mask=invalid_mask)
                epoch_val_loss += loss.item()
                val_true_batches.append(y_np.astype(np.float32, copy=False))
                val_pred_batches.append(p_np.astype(np.float32, copy=False))
                val_valid_batches.append(v_np)
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        if len(val_true_batches) > 0:
            y_val_true = np.concatenate(val_true_batches, axis=0)
            y_val_pred = np.concatenate(val_pred_batches, axis=0)
            val_valid_mask = np.concatenate(val_valid_batches, axis=0)
            val_metrics = compute_metrics(
                y_val_true, y_val_pred, y_sigma=y_std, valid_mask=val_valid_mask
            )
        else:
            val_metrics = {}
        val_metrics_hist.append(val_metrics)
        if len(val_metrics) > 0:
            val_cor_record[epoch, :] = np.asarray(
                val_metrics["Pearson_r_per_task"], dtype=np.float32
            )
            val_cod_record[epoch, :] = np.asarray(
                val_metrics["COD_per_task"], dtype=np.float32
            )
            val_mae_record[epoch, :] = np.asarray(
                val_metrics["MAE_per_task"], dtype=np.float32
            )
            val_cor_mean_record[epoch] = np.float32(val_metrics["Pearson_r_mean"])
            val_cod_mean_record[epoch] = np.float32(val_metrics["COD_mean"])
            val_mae_mean_record[epoch] = np.float32(val_metrics["MAE_mean"])
        scheduler.step()
        current_val_cod = float(val_metrics.get("COD_mean", np.nan))
        if np.isfinite(current_val_cod) and current_val_cod > best_val_cod + 1e-12:
            best_val_cod = current_val_cod
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"[Stage1] No validation COD improvement for {patience} epochs; stopping training."
                )
                break
    epochs_logged = len(train_losses)
    if (best_epoch >= 0) != (best_state is not None):
        raise RuntimeError(
            f"[stage1] inconsistent best-model bookkeeping: best_epoch={best_epoch}, best_state_is_none={best_state is None}"
        )
    if best_epoch < 0:
        raise RuntimeError(
            "[stage1] no valid best epoch was selected; training/validation loop did not produce a usable best model."
        )
    if not (
        0 <= best_epoch < len(train_cor_mean_record)
        and 0 <= best_epoch < len(train_cod_mean_record)
        and (0 <= best_epoch < len(train_mae_mean_record))
        and (0 <= best_epoch < len(val_cor_mean_record))
        and (0 <= best_epoch < len(val_cod_mean_record))
        and (0 <= best_epoch < len(val_mae_mean_record))
    ):
        raise RuntimeError(
            f"[stage1] best_epoch out of range for recorded curves: best_epoch={best_epoch}, len(train_cor_mean_record)={len(train_cor_mean_record)}, len(val_cor_mean_record)={len(val_cor_mean_record)}"
        )
    best_epoch_1based = int(best_epoch + 1)
    best_train_cor_mean = float(train_cor_mean_record[best_epoch])
    best_train_cod_mean = float(train_cod_mean_record[best_epoch])
    best_train_mae_mean = float(train_mae_mean_record[best_epoch])
    best_val_cor_mean = float(val_cor_mean_record[best_epoch])
    best_val_cod_mean = float(val_cod_mean_record[best_epoch])
    best_val_mae_mean = float(val_mae_mean_record[best_epoch])
    best_summary_vals = np.asarray(
        [
            best_train_cor_mean,
            best_train_cod_mean,
            best_train_mae_mean,
            best_val_cor_mean,
            best_val_cod_mean,
            best_val_mae_mean,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(best_summary_vals)):
        raise RuntimeError(
            f"[stage1] non-finite best-epoch summary encountered at best_epoch={best_epoch}: {best_summary_vals}"
        )
    model.load_state_dict(best_state)
    model.eval()
    all_true_z = []
    all_pred_z = []
    all_valid = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_valid = batch
            else:
                batch_x, batch_y = batch
                batch_valid = None
            if batch_valid is None:
                raise RuntimeError(
                    "[stage1 test] batch_valid is None; expected valid masks in Stage1"
                )
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            y_np = batch_y.detach().cpu().numpy().astype(np.float32, copy=False)
            p_np = pred.detach().cpu().numpy().astype(np.float32, copy=False)
            if y_np.ndim == 1:
                y_np = y_np[:, None]
            if p_np.ndim == 1:
                p_np = p_np[:, None]
            if y_np.shape != p_np.shape:
                raise ValueError(
                    f"[stage1 test] shape mismatch y={y_np.shape} pred={p_np.shape}"
                )
            v_np = batch_valid.detach().cpu().numpy()
            if v_np.ndim == 1:
                v_np = v_np[:, None]
            v_np = v_np.astype(bool, copy=False)
            if v_np.shape != y_np.shape:
                raise ValueError(
                    f"[stage1 test] valid_mask shape mismatch v={v_np.shape} y={y_np.shape}"
                )
            all_true_z.append(y_np)
            all_pred_z.append(p_np)
            all_valid.append(v_np)
    y_test_true_z = np.concatenate(all_true_z, axis=0)
    y_test_pred_z = np.concatenate(all_pred_z, axis=0)
    valid_mask_test_out = np.concatenate(all_valid, axis=0)
    if y_mean.ndim != 1 or y_std.ndim != 1:
        raise ValueError(
            f"[stage1 test] expected y_mean/y_std to be 1D, got {y_mean.shape}, {y_std.shape}"
        )
    if (
        y_test_true_z.shape[1] != y_mean.shape[0]
        or y_test_true_z.shape[1] != y_std.shape[0]
    ):
        raise ValueError(
            f"[stage1 test] target dim mismatch: y_test_true_z={y_test_true_z.shape}, y_mean={y_mean.shape}, y_std={y_std.shape}"
        )
    y_test_true_raw = y_test_true_z * y_std[None, :] + y_mean[None, :]
    y_test_pred_raw = y_test_pred_z * y_std[None, :] + y_mean[None, :]
    final_test_metrics_z = compute_metrics(
        y_test_true_z, y_test_pred_z, y_sigma=y_std, valid_mask=valid_mask_test_out
    )
    final_test_metrics_raw = compute_metrics(
        y_test_true_raw, y_test_pred_raw, y_sigma=None, valid_mask=valid_mask_test_out
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(out_dir, f"dnn_stage1_model_{run_tag}{fold_tag}.pth"),
    )
    np.savez(
        os.path.join(out_dir, f"dnn_stage1_norm_{run_tag}{fold_tag}.npz"),
        y_mean=y_mean,
        y_std=y_std,
    )
    np.savez(
        os.path.join(out_dir, f"dnn_stage1_log_{run_tag}{fold_tag}.npz"),
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32),
        train_loss_history=np.array(train_losses, dtype=np.float32),
        val_loss_history=np.array(val_losses, dtype=np.float32),
        best_epoch=int(best_epoch),
        best_epoch_0based=np.int32(best_epoch),
        best_epoch_1based=np.int32(best_epoch_1based),
        best_val_cod=np.float32(best_val_cod),
        best_val=np.float32(best_val_cod),
        best_val_loss=np.float32(val_losses[best_epoch]),
        epochs_logged=np.int32(epochs_logged),
        training_epochs_logged=np.int32(epochs_logged),
        best_train_cor_mean=np.float32(best_train_cor_mean),
        best_train_cod_mean=np.float32(best_train_cod_mean),
        best_train_mae_mean=np.float32(best_train_mae_mean),
        best_val_cor_mean=np.float32(best_val_cor_mean),
        best_val_cod_mean=np.float32(best_val_cod_mean),
        best_val_mae_mean=np.float32(best_val_mae_mean),
        train_cor_record=train_cor_record,
        train_cod_record=train_cod_record,
        train_mae_record=train_mae_record,
        val_cor_record=val_cor_record,
        val_cod_record=val_cod_record,
        val_mae_record=val_mae_record,
        train_cor_mean_record=train_cor_mean_record,
        train_cod_mean_record=train_cod_mean_record,
        train_mae_mean_record=train_mae_mean_record,
        val_cor_mean_record=val_cor_mean_record,
        val_cod_mean_record=val_cod_mean_record,
        val_mae_mean_record=val_mae_mean_record,
        train_pearson_r_mean_record=train_cor_mean_record,
        train_cod_mean_history=train_cod_mean_record,
        train_mae_mean_history=train_mae_mean_record,
        val_pearson_r_mean_record=val_cor_mean_record,
        val_cod_mean_history=val_cod_mean_record,
        val_mae_mean_history=val_mae_mean_record,
        y_test_true_z=y_test_true_z.astype(np.float32),
        y_test_pred_z=y_test_pred_z.astype(np.float32),
        y_test_true_raw=y_test_true_raw.astype(np.float32),
        y_test_pred_raw=y_test_pred_raw.astype(np.float32),
        valid_mask_test=valid_mask_test_out.astype(np.bool_),
        final_test_y_true_z=y_test_true_z.astype(np.float32),
        final_test_y_pred_z=y_test_pred_z.astype(np.float32),
        final_test_y_true_raw=y_test_true_raw.astype(np.float32),
        final_test_y_pred_raw=y_test_pred_raw.astype(np.float32),
        final_test_valid_mask=valid_mask_test_out.astype(np.bool_),
        target_names=(
            np.asarray(y_names, dtype=str)
            if y_names is not None
            else np.asarray([], dtype=str)
        ),
        eid_test=(
            np.asarray(eid_test, dtype=np.int64)
            if eid_test is not None
            else np.asarray([], dtype=np.int64)
        ),
        final_test_target_names=(
            np.asarray(y_names, dtype=str)
            if y_names is not None
            else np.asarray([], dtype=str)
        ),
        final_test_eid=(
            np.asarray(eid_test, dtype=np.int64)
            if eid_test is not None
            else np.asarray([], dtype=np.int64)
        ),
        test_r2_mean_z=np.float32(final_test_metrics_z["R²_mean"]),
        test_pearson_r_mean_z=np.float32(final_test_metrics_z["Pearson_r_mean"]),
        test_cod_mean_z=np.float32(final_test_metrics_z["COD_mean"]),
        test_rmse_z_mean=np.float32(final_test_metrics_z["RMSE_z_mean"]),
        test_mae_z_mean=np.float32(final_test_metrics_z["MAE_z_mean"]),
        test_r2_per_task_z=np.asarray(
            final_test_metrics_z["R²_per_task"], dtype=np.float32
        ),
        test_pearson_r_per_task_z=np.asarray(
            final_test_metrics_z["Pearson_r_per_task"], dtype=np.float32
        ),
        test_cod_per_task_z=np.asarray(
            final_test_metrics_z["COD_per_task"], dtype=np.float32
        ),
        test_rmse_z_per_task=np.asarray(
            final_test_metrics_z["RMSE_z_per_task"], dtype=np.float32
        ),
        test_mae_z_per_task=np.asarray(
            final_test_metrics_z["MAE_z_per_task"], dtype=np.float32
        ),
        test_r2_mean_raw=np.float32(final_test_metrics_raw["R²_mean"]),
        test_pearson_r_mean_raw=np.float32(final_test_metrics_raw["Pearson_r_mean"]),
        test_cod_mean_raw=np.float32(final_test_metrics_raw["COD_mean"]),
        test_rmse_mean_raw=np.float32(final_test_metrics_raw["RMSE_mean"]),
        test_mae_mean_raw=np.float32(final_test_metrics_raw["MAE_mean"]),
        test_r2_per_task_raw=np.asarray(
            final_test_metrics_raw["R²_per_task"], dtype=np.float32
        ),
        test_pearson_r_per_task_raw=np.asarray(
            final_test_metrics_raw["Pearson_r_per_task"], dtype=np.float32
        ),
        test_cod_per_task_raw=np.asarray(
            final_test_metrics_raw["COD_per_task"], dtype=np.float32
        ),
        test_rmse_per_task_raw=np.asarray(
            final_test_metrics_raw["RMSE_per_task"], dtype=np.float32
        ),
        test_mae_per_task_raw=np.asarray(
            final_test_metrics_raw["MAE_per_task"], dtype=np.float32
        ),
        final_test_r2_mean_z=np.float32(final_test_metrics_z["R²_mean"]),
        final_test_pearson_r_mean_z=np.float32(final_test_metrics_z["Pearson_r_mean"]),
        final_test_cod_mean_z=np.float32(final_test_metrics_z["COD_mean"]),
        final_test_rmse_z_mean=np.float32(final_test_metrics_z["RMSE_z_mean"]),
        final_test_mae_z_mean=np.float32(final_test_metrics_z["MAE_z_mean"]),
        final_test_r2_per_task_z=np.asarray(
            final_test_metrics_z["R²_per_task"], dtype=np.float32
        ),
        final_test_pearson_r_per_task_z=np.asarray(
            final_test_metrics_z["Pearson_r_per_task"], dtype=np.float32
        ),
        final_test_cod_per_task_z=np.asarray(
            final_test_metrics_z["COD_per_task"], dtype=np.float32
        ),
        final_test_rmse_z_per_task=np.asarray(
            final_test_metrics_z["RMSE_z_per_task"], dtype=np.float32
        ),
        final_test_mae_z_per_task=np.asarray(
            final_test_metrics_z["MAE_z_per_task"], dtype=np.float32
        ),
        final_test_r2_mean_raw=np.float32(final_test_metrics_raw["R²_mean"]),
        final_test_pearson_r_mean_raw=np.float32(
            final_test_metrics_raw["Pearson_r_mean"]
        ),
        final_test_cod_mean_raw=np.float32(final_test_metrics_raw["COD_mean"]),
        final_test_rmse_mean_raw=np.float32(final_test_metrics_raw["RMSE_mean"]),
        final_test_mae_mean_raw=np.float32(final_test_metrics_raw["MAE_mean"]),
        final_test_r2_per_task_raw=np.asarray(
            final_test_metrics_raw["R²_per_task"], dtype=np.float32
        ),
        final_test_pearson_r_per_task_raw=np.asarray(
            final_test_metrics_raw["Pearson_r_per_task"], dtype=np.float32
        ),
        final_test_cod_per_task_raw=np.asarray(
            final_test_metrics_raw["COD_per_task"], dtype=np.float32
        ),
        final_test_rmse_per_task_raw=np.asarray(
            final_test_metrics_raw["RMSE_per_task"], dtype=np.float32
        ),
        final_test_mae_per_task_raw=np.asarray(
            final_test_metrics_raw["MAE_per_task"], dtype=np.float32
        ),
        training_history_keys=np.asarray(
            [
                "train_losses",
                "val_losses",
                "train_loss_history",
                "val_loss_history",
                "best_epoch",
                "best_epoch_0based",
                "best_epoch_1based",
                "best_val_cod",
                "best_val",
                "best_val_loss",
                "epochs_logged",
                "training_epochs_logged",
                "best_train_cor_mean",
                "best_train_cod_mean",
                "best_train_mae_mean",
                "best_val_cor_mean",
                "best_val_cod_mean",
                "best_val_mae_mean",
                "train_cor_record",
                "train_cod_record",
                "train_mae_record",
                "val_cor_record",
                "val_cod_record",
                "val_mae_record",
                "train_cor_mean_record",
                "train_cod_mean_record",
                "train_mae_mean_record",
                "val_cor_mean_record",
                "val_cod_mean_record",
                "val_mae_mean_record",
                "train_pearson_r_mean_record",
                "train_cod_mean_history",
                "train_mae_mean_history",
                "val_pearson_r_mean_record",
                "val_cod_mean_history",
                "val_mae_mean_history",
            ],
            dtype=str,
        ),
        final_test_keys=np.asarray(
            [
                "y_test_true_z",
                "y_test_pred_z",
                "y_test_true_raw",
                "y_test_pred_raw",
                "valid_mask_test",
                "final_test_y_true_z",
                "final_test_y_pred_z",
                "final_test_y_true_raw",
                "final_test_y_pred_raw",
                "final_test_valid_mask",
                "target_names",
                "eid_test",
                "final_test_target_names",
                "final_test_eid",
                "test_r2_mean_z",
                "test_pearson_r_mean_z",
                "test_cod_mean_z",
                "test_rmse_z_mean",
                "test_mae_z_mean",
                "test_r2_per_task_z",
                "test_pearson_r_per_task_z",
                "test_cod_per_task_z",
                "test_rmse_z_per_task",
                "test_mae_z_per_task",
                "test_r2_mean_raw",
                "test_pearson_r_mean_raw",
                "test_cod_mean_raw",
                "test_rmse_mean_raw",
                "test_mae_mean_raw",
                "test_r2_per_task_raw",
                "test_pearson_r_per_task_raw",
                "test_cod_per_task_raw",
                "test_rmse_per_task_raw",
                "test_mae_per_task_raw",
                "final_test_r2_mean_z",
                "final_test_pearson_r_mean_z",
                "final_test_cod_mean_z",
                "final_test_rmse_z_mean",
                "final_test_mae_z_mean",
                "final_test_r2_per_task_z",
                "final_test_pearson_r_per_task_z",
                "final_test_cod_per_task_z",
                "final_test_rmse_z_per_task",
                "final_test_mae_z_per_task",
                "final_test_r2_mean_raw",
                "final_test_pearson_r_mean_raw",
                "final_test_cod_mean_raw",
                "final_test_rmse_mean_raw",
                "final_test_mae_mean_raw",
                "final_test_r2_per_task_raw",
                "final_test_pearson_r_per_task_raw",
                "final_test_cod_per_task_raw",
                "final_test_rmse_per_task_raw",
                "final_test_mae_per_task_raw",
            ],
            dtype=str,
        ),
    )
    print(
        f"[Stage1] epochs_logged={epochs_logged} best_epoch={best_epoch} best_val_cod={best_val_cod:.6f} saved to {out_dir}"
    )


def train_evaluate_dnn_stage1(
    ukb_final_csv,
    cov_parquet,
    freeze_tsv,
    seeds=(0, 1, 2, 3, 4),
    test_size=0.2,
    num_workers=4,
    device=None,
    out_dir=None,
    source_kind="raw",
):
    X, Y, valid_mask, eid, y_names, site_vec = load_stage1_data(
        ukb_final_csv=ukb_final_csv,
        cov_parquet=cov_parquet,
        freeze_tsv=freeze_tsv,
        site_col="QC_Imaging_Site",
        source_kind=source_kind,
    )
    if out_dir is None:
        try:
            _base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _base_dir = os.getcwd()
        out_dir = os.path.join(_base_dir, "dnn_stage1_out")
    os.makedirs(out_dir, exist_ok=True)

    def _can_stratify(labels):
        if labels is None:
            return False
        uniq, cnt = np.unique(labels, return_counts=True)
        if uniq.size < 2:
            return False
        if (cnt * (1 - test_size) < 1).any() or (cnt * test_size < 1).any():
            return False
        return True

    for seed in seeds:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
        use_stratify_outer = _can_stratify(site_vec)
        if use_stratify_outer:
            X_trv, X_te, Y_trv, Y_te, M_trv, M_te, eid_trv, eid_te, site_trv, _ = (
                train_test_split(
                    X,
                    Y,
                    valid_mask,
                    eid,
                    site_vec,
                    test_size=test_size,
                    random_state=int(seed),
                    stratify=site_vec,
                )
            )
        else:
            X_trv, X_te, Y_trv, Y_te, M_trv, M_te, eid_trv, eid_te = train_test_split(
                X,
                Y,
                valid_mask,
                eid,
                test_size=test_size,
                random_state=int(seed),
                shuffle=True,
            )
            site_trv = None
        run_tag = f"stage1_seed{seed}"
        seed_dir = os.path.join(out_dir, run_tag)
        os.makedirs(seed_dir, exist_ok=True)
        np.savez(
            os.path.join(seed_dir, f"dnn_stage1_outer_split_{run_tag}.npz"),
            eid_train_val=np.asarray(eid_trv, dtype=np.int64),
            eid_test=np.asarray(eid_te, dtype=np.int64),
        )
        _fit_eval_one_outer_split_dnn_stage1(
            X_train_val=X_trv,
            Y_train_val=Y_trv,
            valid_mask_train_val=M_trv,
            X_test=X_te,
            Y_test=Y_te,
            valid_mask_test=M_te,
            eid_test=eid_te,
            y_names=y_names,
            site_trv=site_trv,
            random_state=int(seed),
            run_tag=run_tag,
            fold_tag="",
            out_dir=seed_dir,
            num_workers=num_workers,
            device=device,
        )


def load_stage1_data(
    ukb_final_csv,
    cov_parquet,
    freeze_tsv,
    site_col="QC_Imaging_Site",
    source_kind="raw",
):
    if source_kind not in ("raw", "pc"):
        raise ValueError("source_kind must be raw or pc")
    loader = load_raw_source_data if source_kind == "raw" else load_pc_source_data
    return loader(ukb_final_csv, cov_parquet, freeze_tsv, site_col)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-kind", choices=("raw", "pc"), required=True)
    parser.add_argument("--ukb-csv", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--freeze-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    seeds = tuple((int(s.strip()) for s in args.seeds.split(",") if s.strip()))
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("Provide a nonempty list of distinct seeds.")
    if args.device == "cuda" and (not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable.")
    device = None if args.device == "auto" else torch.device(args.device)
    train_evaluate_dnn_stage1(
        ukb_final_csv=args.ukb_csv,
        cov_parquet=args.target_parquet,
        freeze_tsv=args.freeze_tsv,
        seeds=seeds,
        test_size=args.test_size,
        num_workers=args.num_workers,
        device=device,
        out_dir=args.out_dir,
        source_kind=args.source_kind,
    )


if __name__ == "__main__":
    main()
