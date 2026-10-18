"""Select gross-motor branches using repeated inner-OOF predictions and convex fusion."""

import numpy as np

RAW_PLATEAU = [12, 14, 15]
DISEASE_PLATEAU = [11, 12, 13, 15, 16]
DNN_M = [5, 10, 20, 40, 60, 72, 80, 88, 100, 120]
ALPHAS = [
    0.001,
    0.004,
    0.01,
    0.04,
    0.1,
    0.4,
    1.0,
    4.0,
    10.0,
    40.0,
    100.0,
    400.0,
    1000.0,
    4000.0,
    10000.0,
]
W_RAW = [0.2, 0.3, 0.4, 0.5, 0.6]
W_DIS = [0.0, 0.05, 0.1, 0.2, 0.3]
W_DNN = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]


def metscore(E, reps, fn):
    a = np.asarray([E.metrics(r["y"], fn(r))["COD"] for r in reps])
    return (float(a.mean()), float(a.std(ddof=0)), float(a.min()))


def best(rows):
    return sorted(
        rows, key=lambda r: (-r["mean"], r.get("size", 0), -r.get("alpha", 0))
    )[0]


def weighted_rank(X, y, g, p3a):
    w = p3a.child_weights(g)
    w = w / w.sum()
    xm = np.sum(X * w[:, None], axis=0)
    ym = np.sum(y * w)
    xc = X - xm
    yc = y - ym
    num = np.sum(w[:, None] * xc * yc[:, None], axis=0)
    den = np.sqrt(np.sum(w[:, None] * xc * xc, axis=0) * np.sum(w * yc * yc))
    s = np.divide(np.abs(num), den, out=np.zeros_like(num), where=den > 1e-12)
    return np.argsort(-s, kind="stable")


def make_reps(E, ref, p3a, rsfc, raw, dis, dnn, y, cov, g, tr0, k, repeats, seed0):
    reps = []
    for rr in range(repeats):
        n = len(tr0)
        yo = np.full(n, np.nan)
        pb = np.full(n, np.nan)
        rp = {(m, a): np.full(n, np.nan) for m in RAW_PLATEAU for a in ALPHAS}
        dp = {(q, a): np.full(n, np.nan) for q in DISEASE_PLATEAU for a in ALPHAS}
        da = {a: np.full(n, np.nan) for a in ALPHAS}
        dt = {(m, a): np.full(n, np.nan) for m in DNN_M for a in ALPHAS}
        for il, iv in E.grouped_folds(g[tr0], k, seed0 + 1009 * rr):
            ti, vi = (tr0[il], tr0[iv])
            yt, yv, *_ = E.residualize(y[ti], y[vi], cov[ti], cov[vi])
            yo[iv] = yv
            pb[iv] = p3a.fit_baseline(ref, rsfc[ti], yt, g[ti], rsfc[vi])
            for m in RAW_PLATEAU:
                for a in ALPHAS:
                    rp[m, a][iv] = p3a.fit_transfer(
                        raw[ti, :m], yt, g[ti], raw[vi, :m], a
                    )
            for q in DISEASE_PLATEAU:
                for a in ALPHAS:
                    dp[q, a][iv] = p3a.fit_transfer(
                        dis[ti, :q], yt, g[ti], dis[vi, :q], a
                    )
            rank = weighted_rank(dnn[ti], yt, g[ti], p3a)
            for a in ALPHAS:
                da[a][iv] = p3a.fit_transfer(dnn[ti], yt, g[ti], dnn[vi], a)
            for m in DNN_M:
                ix = rank[:m]
                for a in ALPHAS:
                    dt[m, a][iv] = p3a.fit_transfer(
                        dnn[ti][:, ix], yt, g[ti], dnn[vi][:, ix], a
                    )
        if any(
            (
                not np.isfinite(v).all()
                for v in [
                    yo,
                    pb,
                    *rp.values(),
                    *dp.values(),
                    *da.values(),
                    *dt.values(),
                ]
            )
        ):
            raise RuntimeError("incomplete inner OOF")
        reps.append(
            {"y": yo, "base": pb, "raw": rp, "dis": dp, "dnn_all": da, "dnn_top": dt}
        )
    return reps


def select(E, reps):
    ra = {}
    for m in RAW_PLATEAU:
        rows = []
        for a in ALPHAS:
            q = metscore(E, reps, lambda r, m=m, a=a: r["raw"][m, a])
            rows.append(
                {"size": m, "alpha": a, "mean": q[0], "sd": q[1], "worst": q[2]}
            )
        ra[m] = best(rows)["alpha"]
    xa = {}
    for k in DISEASE_PLATEAU:
        rows = []
        for a in ALPHAS:
            q = metscore(E, reps, lambda r, k=k, a=a: r["dis"][k, a])
            rows.append(
                {"size": k, "alpha": a, "mean": q[0], "sd": q[1], "worst": q[2]}
            )
        xa[k] = best(rows)["alpha"]
    drows = []
    for a in ALPHAS:
        q = metscore(E, reps, lambda r, a=a: r["dnn_all"][a])
        drows.append(
            {
                "repr": "all",
                "size": 120,
                "alpha": a,
                "mean": q[0],
                "sd": q[1],
                "worst": q[2],
            }
        )
    for m in DNN_M:
        for a in ALPHAS:
            q = metscore(E, reps, lambda r, m=m, a=a: r["dnn_top"][m, a])
            drows.append(
                {
                    "repr": "topm",
                    "size": m,
                    "alpha": a,
                    "mean": q[0],
                    "sd": q[1],
                    "worst": q[2],
                }
            )
    bd = best(drows)

    def branches(r):
        pr = np.mean([r["raw"][m, ra[m]] for m in RAW_PLATEAU], axis=0)
        pd = np.mean([r["dis"][k, xa[k]] for k in DISEASE_PLATEAU], axis=0)
        pn = (
            r["dnn_all"][bd["alpha"]]
            if bd["repr"] == "all"
            else r["dnn_top"][bd["size"], bd["alpha"]]
        )
        return (pr, pd, pn)

    rows = []
    for wr in W_RAW:
        for wd in W_DIS:
            for wn in W_DNN:
                if wr + wd + wn > 0.75 + 1e-12:
                    continue
                wb = 1 - wr - wd - wn
                q = metscore(
                    E,
                    reps,
                    lambda r, wb=wb, wr=wr, wd=wd, wn=wn: (
                        lambda z: wb * r["base"] + wr * z[0] + wd * z[1] + wn * z[2]
                    )(branches(r)),
                )
                rows.append(
                    {
                        "w_base": wb,
                        "w_raw": wr,
                        "w_disease": wd,
                        "w_dnn": wn,
                        "mean": q[0],
                        "sd": q[1],
                        "worst": q[2],
                    }
                )
    core = [r for r in rows if r["w_dnn"] == 0]
    joint = best(rows)
    corebest = best(core)
    return (ra, xa, bd, corebest, joint)


def fit(E, ref, p3a, rsfc, raw, dis, dnn, ytr, gtr, tr, te, ra, xa, bd, s):
    pb = p3a.fit_baseline(ref, rsfc[tr], ytr, gtr, rsfc[te])
    pr = np.mean(
        [
            p3a.fit_transfer(raw[tr, :m], ytr, gtr, raw[te, :m], ra[m])
            for m in RAW_PLATEAU
        ],
        axis=0,
    )
    pd = np.mean(
        [
            p3a.fit_transfer(dis[tr, :k], ytr, gtr, dis[te, :k], xa[k])
            for k in DISEASE_PLATEAU
        ],
        axis=0,
    )
    if bd["repr"] == "all":
        pn = p3a.fit_transfer(dnn[tr], ytr, gtr, dnn[te], bd["alpha"])
    else:
        ix = weighted_rank(dnn[tr], ytr, gtr, p3a)[: int(bd["size"])]
        pn = p3a.fit_transfer(dnn[tr][:, ix], ytr, gtr, dnn[te][:, ix], bd["alpha"])
    out = s["w_base"] * pb + s["w_raw"] * pr + s["w_disease"] * pd + s["w_dnn"] * pn
    return (out, pb, pn)
