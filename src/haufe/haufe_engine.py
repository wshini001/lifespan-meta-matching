"""Calculate edge covariances, correlations and incident-edge node summaries."""

import numpy as np
import pandas as pd


def compute_pnf(X, y, ddof=1):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    y = y[ok]
    n = len(y)
    if n < 5:
        raise RuntimeError(f"Too few rows for PNF: {n}")
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    if ddof not in (0, 1):
        raise ValueError("Covariance ddof must be zero or one")
    cov = Xc.T @ yc / (n - ddof)
    xsd = np.sqrt(np.sum(Xc**2, axis=0) / (n - ddof))
    ysd = np.sqrt(np.sum(yc**2) / (n - ddof))
    denom = xsd * ysd
    cor = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
    return {"cov": cov, "corr": cor, "n": n}


def map_similarity(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def node_importance(edge_table, value_col, n_nodes=55):
    abs_sum = np.zeros(n_nodes, dtype=float)
    signed_sum = np.zeros(n_nodes, dtype=float)
    for _, r in edge_table.iterrows():
        i = int(r["Node1"]) - 1
        j = int(r["Node2"]) - 1
        v = float(r[value_col])
        abs_sum[i] += abs(v)
        abs_sum[j] += abs(v)
        signed_sum[i] += v
        signed_sum[j] += v
    out = pd.DataFrame(
        {
            "node": np.arange(1, n_nodes + 1),
            "abs_pnf_sum": abs_sum,
            "abs_pnf_mean": abs_sum / (n_nodes - 1),
            "signed_pnf_sum": signed_sum,
        }
    )
    return out.sort_values("abs_pnf_sum", ascending=False).reset_index(drop=True)
