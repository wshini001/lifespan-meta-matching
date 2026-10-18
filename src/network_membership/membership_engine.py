"""Calculate ICA spatial-magnitude memberships in Yeo-7 networks."""

from collections import Counter
import numpy as np
import pandas as pd

YEO = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention",
    5: "Limbic",
    6: "Frontoparietal Control",
    7: "Default Mode",
}


def safe_div(a, b):
    if not np.isfinite(b) or b <= 0:
        return np.nan
    return float(a / b)


def dominant_from(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).any():
        return None
    return int(np.nanargmax(x) + 1)


def calculate_membership(ica, yeo, good):
    cortical_mask = yeo > 0
    rows = []
    for node in range(1, 56):
        ica_id = int(good[node - 1])
        v = np.asarray(ica[..., ica_id - 1], dtype=float)
        v[~np.isfinite(v)] = 0.0
        a = np.abs(v)
        q = v * v
        total_abs = float(a.sum())
        total_sq = float(q.sum())
        cortical_abs = float(a[cortical_mask].sum())
        cortical_sq = float(q[cortical_mask].sum())
        abs_total = []
        abs_cort = []
        sq_total = []
        sq_cort = []
        for k in range(1, 8):
            mk = yeo == k
            av = float(a[mk].sum())
            qv = float(q[mk].sum())
            abs_total.append(safe_div(av, total_abs))
            abs_cort.append(safe_div(av, cortical_abs))
            sq_total.append(safe_div(qv, total_sq))
            sq_cort.append(safe_div(qv, cortical_sq))
        positive = a[a > 0]
        threshold_results = {}
        for pct in [90, 92, 95]:
            thr = float(np.percentile(positive, pct))
            high = a >= thr
            total_high_abs = float(a[high].sum())
            cortical_high_abs = float(a[high & cortical_mask].sum())
            frac = []
            for k in range(1, 8):
                mk = high & (yeo == k)
                frac.append(safe_div(float(a[mk].sum()), cortical_high_abs))
            threshold_results[pct] = {
                "threshold": thr,
                "coverage": safe_div(cortical_high_abs, total_high_abs),
                "fractions": frac,
                "dominant": dominant_from(frac),
            }
        dom_abs = dominant_from(abs_cort)
        dom_sq = dominant_from(sq_cort)
        doms = [
            dom_abs,
            dom_sq,
            threshold_results[90]["dominant"],
            threshold_results[92]["dominant"],
            threshold_results[95]["dominant"],
        ]
        counts = Counter((x for x in doms if x is not None))
        consensus = counts.most_common(1)[0][0] if counts else None
        consensus_n = counts[consensus] if consensus is not None else 0
        r = {
            "node": node,
            "FMRIB_ICA_component": ica_id,
            "cortical_abs_mass_fraction": safe_div(cortical_abs, total_abs),
            "cortical_sq_mass_fraction": safe_div(cortical_sq, total_sq),
            "highmag90_cortical_coverage": threshold_results[90]["coverage"],
            "highmag92_cortical_coverage": threshold_results[92]["coverage"],
            "highmag95_cortical_coverage": threshold_results[95]["coverage"],
            "dominant_abs_network": dom_abs,
            "dominant_sq_network": dom_sq,
            "dominant_thr90_network": threshold_results[90]["dominant"],
            "dominant_thr92_network": threshold_results[92]["dominant"],
            "dominant_thr95_network": threshold_results[95]["dominant"],
            "consensus_network": consensus,
            "consensus_network_name": YEO[consensus] if consensus is not None else "NA",
            "consensus_n_of_5": consensus_n,
            "consensus_fraction": consensus_n / 5.0,
        }
        for k in range(1, 8):
            r[f"abs_total_Yeo{k}"] = abs_total[k - 1]
            r[f"abs_cortical_Yeo{k}"] = abs_cort[k - 1]
            r[f"sq_total_Yeo{k}"] = sq_total[k - 1]
            r[f"sq_cortical_Yeo{k}"] = sq_cort[k - 1]
            for pct in [90, 92, 95]:
                r[f"thr{pct}_cortical_Yeo{k}"] = threshold_results[pct]["fractions"][
                    k - 1
                ]
        rows.append(r)
    return pd.DataFrame(rows)
