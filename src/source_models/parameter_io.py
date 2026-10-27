"""Whitelisted source-parameter exports and inference for audited source schemas.

KRR conversion applies only to the row-centered, L2-normalized dot-product
kernels used here. It is not an exporter for arbitrary nonlinear kernels.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np

EDGES = [f"e{i}" for i in range(1, 1486)]
PARAMETER_KEYS = {"beta", "y_mean", "y_std", "target_names", "edge_names"}
TRANSFORMS = {
    "float64_center_l2_then_float32",
    "float32_center_l2",
    "float64_center_l2",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def names(values):
    result = [str(value) for value in values]
    require(bool(result) and len(result) == len(set(result)), "Empty/duplicate names")
    require(
        all(value and len(value) < 256 and "\n" not in value for value in result),
        "Invalid names",
    )
    return result


def unitize(edges, transform):
    require(transform in TRANSFORMS, "Unsupported input transform")
    dtype = np.float32 if transform == "float32_center_l2" else np.float64
    x = np.asarray(edges, dtype=dtype)
    require(
        x.ndim == 2 and x.shape[1] == 1485 and np.isfinite(x).all(),
        "Expected finite n-by-1485 RSFC",
    )
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if transform == "float64_center_l2":
        require(
            np.all(norm > 0), "Original raw-disease inference rejects zero-norm rows"
        )
    norm[norm <= 0] = 1
    z = x / norm
    return z.astype(np.float32) if transform == "float64_center_l2_then_float32" else z


def normalize_ids(values):
    import pandas as pd

    ids = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="raise").to_numpy(
        np.float64
    )
    require(
        np.isfinite(ids).all() and np.equal(ids, np.floor(ids)).all(),
        "Invalid fit-ID encoding",
    )
    return ids.astype(np.int64).astype(str)


def disease_training_features(bundle, ukb_csv, expected_rows):
    import pandas as pd

    require(
        ukb_csv is not None,
        "Raw-disease conversion needs the original authorized UKB FC CSV locally",
    )
    df = pd.read_csv(ukb_csv, usecols=["eid"] + EDGES)
    require(len(df) == expected_rows, "UKB FC cohort-size contract differs")
    ids = normalize_ids(df["eid"])
    require(len(set(ids)) == len(ids), "UKB FC IDs are not unique")
    lookup = {value: i for i, value in enumerate(ids)}
    fit_ids = normalize_ids(bundle["fit_eid"])
    require(len(set(fit_ids)) == len(fit_ids), "Duplicate disease fit IDs")
    require(
        all(value in lookup for value in fit_ids), "Disease fit IDs absent from UKB FC"
    )
    z = unitize(df[EDGES].to_numpy(np.float64), "float64_center_l2")
    return z[[lookup[value] for value in fit_ids]]


def export_krr(
    source, output_dir, *, ukb_csv=None, expected_ukb_rows=24668, atol=1e-5, rtol=1e-5
):
    """Convert one trusted legacy bundle; fail before publishing an invalid export."""
    import joblib

    require(
        atol >= 0 and rtol >= 0 and np.isfinite([atol, rtol]).all(),
        "Invalid tolerances",
    )
    source, output_dir = Path(source), Path(output_dir)
    require(not output_dir.exists(), "Export destination already exists")
    bundle = joblib.load(source)
    require(
        isinstance(bundle, dict) and list(bundle["edge_names"]) == EDGES,
        "Unexpected bundle/edge order",
    )
    kind = bundle.get("bundle_type")
    disease = kind == "ukb_raw_disease_residualized_krr_stage1_target"
    if disease:
        require(
            bundle["feature_transform"] == "row_center_then_l2_unitize"
            and bundle["kernel_type"] == "Pearson_correlation_precomputed",
            "Unsupported disease kernel",
        )
        require(
            bundle["target_space"] == "covariate_residual_z"
            and bundle["prediction_scale"] == "z"
            and not bundle["refit_on_train_validation"],
            "Unsupported disease output contract",
        )
        targets = names([bundle["target"]])
        transform = "float64_center_l2"
        fit = disease_training_features(bundle, ukb_csv, expected_ukb_rows)
        records = [(fit, bundle["dual_coef"], bundle["y_mean"], bundle["y_std"])]
        output_space = "covariate_residual_z; scaled output remains a residual score, not diagnosis probability"
    else:
        require(
            kind
            in {"ukb_stage1_krr_pc_transfer_bundle", "ukb_stage1_krr_transfer_bundle"},
            "Unsupported legacy KRR schema",
        )
        metadata = bundle["metadata"]
        require(
            metadata["kernel_type"] == "corr_precomputed"
            and metadata["feature_transform"] == "unitize_rows",
            "Unsupported kernel/transform",
        )
        raw = kind == "ukb_stage1_krr_transfer_bundle"
        targets = names(bundle["y_cols"] if raw else bundle["target_names"])
        require(not raw or len(targets) == 1, "Raw bundle must contain one target")
        transform = "float32_center_l2" if raw else "float64_center_l2_then_float32"
        z = np.asarray(bundle["Z_train_full"])
        require(
            z.dtype == np.float32
            and z.ndim == 2
            and z.shape[1] == 1485
            and np.isfinite(z).all(),
            "Unexpected original training matrix",
        )
        records = []
        for target in targets:
            rec = bundle["target_models"][target]
            require(
                rec.get("status", "ok") == "ok", "Target was not fitted successfully"
            )
            idx = np.asarray(rec["fit_train_idx"], np.int64)
            require(
                idx.ndim == 1
                and len(idx) > 0
                and idx.min() >= 0
                and idx.max() < len(z)
                and len(set(idx)) == len(idx),
                "Invalid fit indices",
            )
            sd = float(rec["y_sigma"])
            if not raw and abs(sd) < 1e-12:
                sd = 1.0  # Match legacy PC inference.
            records.append((z[idx], rec["dual_coef"], rec["y_mu"], sd))
        output_space = "source_phenotype_z"
    probe = unitize(np.random.default_rng(20261027).normal(size=(16, 1485)), transform)
    betas, means, stds, checks = [], [], [], []
    for target, (fit, dual, mu, sd) in zip(targets, records):
        dual = np.asarray(dual, np.float64).ravel()
        require(
            dual.shape == (len(fit),) and np.isfinite(dual).all(),
            "Invalid dual coefficients",
        )
        require(np.isfinite([mu, sd]).all() and sd > 0, "Invalid target normalization")
        beta = fit.astype(np.float64).T @ dual
        original = (probe @ fit.T).astype(np.float64) @ dual
        converted = probe.astype(np.float64) @ beta
        require(
            np.isfinite(beta).all() and np.isfinite(original).all(),
            "Nonfinite conversion",
        )
        require(
            np.allclose(original, converted, atol=atol, rtol=rtol),
            "Synthetic arithmetic tolerance failed",
        )
        checks.append(
            {
                "target": target,
                "max_abs_z_difference": float(np.max(np.abs(original - converted))),
            }
        )
        betas.append(beta)
        means.append(mu)
        stds.append(sd)
    arrays = dict(
        beta=np.column_stack(betas),
        y_mean=np.asarray(means),
        y_std=np.asarray(stds),
        target_names=np.asarray(targets),
        edge_names=np.asarray(EDGES),
    )
    meta = dict(
        schema_version=1,
        model_kind="krr",
        format="CONVERTED_feature_space_parameters_NOT_original_bundle",
        input_transform=transform,
        output_space=output_space,
        source_sha256=sha256(source),
        synthetic_checks=checks,
        synthetic_checks_passed=True,
        atol=atol,
        rtol=rtol,
        retrained=False,
        original_files_modified=False,
        participant_records_exported=False,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=".parameter_export_"
    ) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        np.savez_compressed(staged / "parameters.npz", **arrays)
        meta["parameter_sha256"] = sha256(staged / "parameters.npz")
        write_json(staged / "model_metadata.json", meta)
        validate_krr(staged)
        staged.rename(output_dir)
    return meta


def validate_krr(directory):
    directory = Path(directory)
    if (directory / "model_metadata.json").is_file():
        meta = json.loads((directory / "model_metadata.json").read_text())
        require(
            meta["schema_version"] == 1 and meta["model_kind"] == "krr",
            "Unknown parameter schema",
        )
    else:
        # Compatibility with the October 2026 private review exports. These
        # are parameter-only files, never legacy joblib training bundles.
        meta = json.loads((directory / "conversion_and_validation.json").read_text())
        require(
            meta.get("synthetic_checks_passed") is True
            and str(meta.get("format", "")).startswith("CONVERTED_feature_space"),
            "Unknown review export",
        )
        meta["model_kind"] = "krr"
        meta.setdefault("input_transform", "float64_center_l2_then_float32")
    require(meta["input_transform"] in TRANSFORMS, "Unknown input transform")
    if "parameter_sha256" in meta:
        require(
            sha256(directory / "parameters.npz") == meta["parameter_sha256"],
            "Parameter checksum mismatch",
        )
    with np.load(directory / "parameters.npz", allow_pickle=False) as p:
        require(set(p.files) == PARAMETER_KEYS, "Parameter field whitelist failed")
        require(
            p["target_names"].ndim == 1 and p["target_names"].dtype.kind in "US",
            "Invalid target-name vector",
        )
        require(
            p["edge_names"].ndim == 1 and p["edge_names"].dtype.kind in "US",
            "Invalid edge-name vector",
        )
        targets = names(p["target_names"])
        require(list(p["edge_names"]) == EDGES, "Parameter edge order mismatch")
        require(
            p["beta"].shape == (1485, len(targets)) and np.isfinite(p["beta"]).all(),
            "Invalid beta matrix",
        )
        for key in ["y_mean", "y_std"]:
            require(
                p[key].shape == (len(targets),) and np.isfinite(p[key]).all(),
                "Invalid normalization",
            )
        require(np.all(p["y_std"] > 0), "Nonpositive target standard deviation")
    return meta


def predict_krr(directory, edges):
    meta = validate_krr(directory)
    z = unitize(edges, meta["input_transform"]).astype(np.float64)
    with np.load(Path(directory) / "parameters.npz", allow_pickle=False) as p:
        prediction_z = z @ p["beta"]
        return (
            p["target_names"].astype(str).tolist(),
            prediction_z,
            prediction_z * p["y_std"] + p["y_mean"],
        )


def dnn_names(seed_dir, stem):
    """Prefer the non-participant manifest; support restricted historical logs."""
    manifest = Path(seed_dir) / f"dnn_stage1_manifest_{stem}.json"
    if manifest.is_file():
        meta = json.loads(manifest.read_text())
        require(list(meta["edge_names"]) == EDGES, "DNN manifest edge order mismatch")
        return names(meta["target_names"])
    warnings.warn(
        "Legacy fallback reads target_names only from a restricted training log; export/use the standalone manifest for future inference.",
        UserWarning,
        stacklevel=2,
    )
    with np.load(
        Path(seed_dir) / f"dnn_stage1_log_{stem}.npz", allow_pickle=False
    ) as log:
        return names(log["target_names"].astype(str))


def check_dnn_state(state):
    """Check tensor keys and attached state attributes before using a weight file."""
    import torch

    expected = set()
    for i in range(1, 5):
        expected.update({f"fc{i}.1.weight", f"fc{i}.1.bias"})
        if i < 4:
            expected.update(
                f"fc{i}.3.{key}"
                for key in [
                    "weight",
                    "bias",
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                ]
            )
    require(
        isinstance(state, dict)
        and set(state) == expected
        and all(
            isinstance(v, torch.Tensor) and torch.isfinite(v).all()
            for v in state.values()
        ),
        "Unexpected DNN state fields",
    )
    require(
        set(getattr(state, "__dict__", {})) <= {"_metadata"},
        "Unexpected state attributes",
    )
    for key, value in getattr(state, "_metadata", {}).items():
        require(
            isinstance(key, str)
            and isinstance(value, dict)
            and set(value) <= {"version"}
            and all(isinstance(v, int) for v in value.values()),
            "Unexpected state module metadata",
        )


def load_dnn(directory):
    import torch
    from torch import nn

    directory = Path(directory)
    meta = json.loads((directory / "model_metadata.json").read_text())
    if "schema_version" in meta:
        require(
            meta["schema_version"] == 1 and meta["model_kind"] == "dnn",
            "Unknown DNN parameter schema",
        )
    else:
        require(
            meta.get("format") == "original_pytorch_state_dict_and_whitelisted_norm",
            "Unknown DNN review export",
        )
        meta["model_kind"] = "dnn"
        meta["weight_sha256"] = meta["original_weight_sha256"]
    require(
        meta["edge_names"] == EDGES
        and names(meta["target_names"]) == meta["target_names"],
        "Invalid DNN metadata",
    )
    arch = meta["architecture"]

    class DNN(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [1485, arch["hidden1"], arch["hidden2"], arch["hidden3"]]
            for i in range(1, 4):
                setattr(
                    self,
                    f"fc{i}",
                    nn.Sequential(
                        nn.Dropout(0.241964),
                        nn.Linear(widths[i - 1], widths[i]),
                        nn.ReLU(),
                        nn.BatchNorm1d(widths[i]),
                    ),
                )
            self.fc4 = nn.Sequential(
                nn.Dropout(0.241964), nn.Linear(widths[3], arch["outputs"])
            )

        def forward(self, x):
            return self.fc4(self.fc3(self.fc2(self.fc1(x))))

    require(
        sha256(directory / "model.pth") == meta["weight_sha256"],
        "DNN checksum mismatch",
    )
    if "normalization_sha256" in meta:
        require(
            sha256(directory / "normalization.npz") == meta["normalization_sha256"],
            "DNN norm checksum mismatch",
        )
    with np.load(directory / "normalization.npz", allow_pickle=False) as p:
        require(set(p.files) == {"y_mean", "y_std"}, "Unexpected DNN norm fields")
        require(
            p["y_mean"].shape == p["y_std"].shape == (len(meta["target_names"]),),
            "Invalid DNN norm shape",
        )
        require(
            np.isfinite(p["y_mean"]).all()
            and np.isfinite(p["y_std"]).all()
            and np.all(p["y_std"] > 0),
            "Invalid DNN norm values",
        )
    model = DNN()
    state = torch.load(directory / "model.pth", weights_only=True, map_location="cpu")
    check_dnn_state(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


def export_dnn(source, output_dir):
    import torch

    source, output_dir = Path(source), Path(output_dir)
    require(not output_dir.exists(), "Export destination already exists")
    prefix = "dnn_stage1_model_"
    require(
        source.name.startswith(prefix) and source.suffix == ".pth",
        "Expected audited DNN weight filename",
    )
    stem = source.stem[len(prefix) :]
    state = torch.load(source, weights_only=True, map_location="cpu")
    check_dnn_state(state)
    targets = dnn_names(source.parent, stem)
    with np.load(
        source.with_name(f"dnn_stage1_norm_{stem}.npz"), allow_pickle=False
    ) as p:
        require(set(p.files) == {"y_mean", "y_std"}, "Unexpected norm fields")
        mu, sd = p["y_mean"], p["y_std"]
    require(
        mu.shape == sd.shape == (len(targets),)
        and np.isfinite(mu).all()
        and np.isfinite(sd).all()
        and np.all(sd > 0),
        "DNN normalization mismatch",
    )
    arch = dict(
        hidden1=int(state["fc1.1.weight"].shape[0]),
        hidden2=int(state["fc2.1.weight"].shape[0]),
        hidden3=int(state["fc3.1.weight"].shape[0]),
        outputs=len(targets),
    )
    require(
        state["fc1.1.weight"].shape[1] == 1485
        and state["fc4.1.weight"].shape[0] == len(targets),
        "DNN dimensions differ",
    )
    meta = dict(
        schema_version=1,
        model_kind="dnn",
        format="original_state_dict_with_whitelisted_normalization",
        edge_names=EDGES,
        target_names=targets,
        architecture=arch,
        input_transform="float32_raw_RSFC_no_row_unitization",
        weight_sha256=sha256(source),
        retrained=False,
        original_files_modified=False,
        participant_records_exported=False,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=".parameter_export_"
    ) as temporary:
        staged = Path(temporary) / "model"
        staged.mkdir()
        shutil.copyfile(source, staged / "model.pth")
        np.savez_compressed(staged / "normalization.npz", y_mean=mu, y_std=sd)
        meta["normalization_sha256"] = sha256(staged / "normalization.npz")
        write_json(staged / "model_metadata.json", meta)
        model, _ = load_dnn(staged)
        with torch.no_grad():
            probe = model(torch.zeros((2, 1485), dtype=torch.float32)).numpy()
        require(
            probe.shape == (2, len(targets)) and np.isfinite(probe).all(),
            "DNN inference smoke test failed",
        )
        staged.rename(output_dir)
    return meta


def predict_dnn(directory, edges):
    import torch

    model, meta = load_dnn(directory)
    x = np.asarray(edges, dtype=np.float32)
    require(
        x.ndim == 2 and x.shape[1] == 1485 and np.isfinite(x).all(),
        "Expected finite n-by-1485 RSFC",
    )
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).numpy()
    with np.load(Path(directory) / "normalization.npz", allow_pickle=False) as p:
        require(set(p.files) == {"y_mean", "y_std"}, "Unexpected DNN norm fields")
        return meta["target_names"], pred, pred * p["y_std"] + p["y_mean"]
