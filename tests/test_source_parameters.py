"""Synthetic-only conversion, alignment, precision and publication-boundary tests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

from src.source_models import parameter_io as io

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Parameters(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rng = np.random.default_rng(5)

    def tearDown(self):
        self.temporary.cleanup()

    def krr_fixture(self, raw=False):
        transform = "float32_center_l2" if raw else "float64_center_l2_then_float32"
        z = io.unitize(self.rng.normal(size=(31, 1485)), transform)
        targets = ["synthetic_raw"] if raw else ["synthetic_pc1", "synthetic_pc2"]
        models = {}
        for i, target in enumerate(targets):
            models[target] = dict(
                status="ok",
                fit_train_idx=np.arange(23 - i),
                dual_coef=self.rng.normal(size=23 - i),
                y_mu=2.0,
                y_sigma=3.0,
            )
        b = dict(
            bundle_type=(
                "ukb_stage1_krr_transfer_bundle"
                if raw
                else "ukb_stage1_krr_pc_transfer_bundle"
            ),
            metadata=dict(
                kernel_type="corr_precomputed", feature_transform="unitize_rows"
            ),
            edge_names=io.EDGES,
            Z_train_full=z,
            eid_train_full=np.arange(31) + 1000,
            target_models=models,
        )
        b["y_cols" if raw else "target_names"] = targets
        return b, transform, targets

    def test_pc_and_raw_conversion_match_original_kernel_on_new_inputs(self):
        for raw in [False, True]:
            with self.subTest(raw=raw):
                b, transform, targets = self.krr_fixture(raw)
                source, out = self.root / f"{raw}.joblib", self.root / f"out{raw}"
                joblib.dump(b, source)
                before = io.sha256(source)
                io.export_krr(source, out)
                x = self.rng.normal(size=(9, 1485))
                actual_names, actual, scaled = io.predict_krr(out, x)
                z = io.unitize(x, transform)
                for j, target in enumerate(targets):
                    r = b["target_models"][target]
                    expected = (z @ b["Z_train_full"][r["fit_train_idx"]].T).astype(
                        np.float64
                    ) @ r["dual_coef"]
                    np.testing.assert_allclose(
                        actual[:, j], expected, atol=1e-5, rtol=1e-5
                    )
                np.testing.assert_allclose(scaled, actual * 3 + 2)
                self.assertEqual(actual_names, targets)
                self.assertEqual(io.sha256(source), before)
                with np.load(out / "parameters.npz", allow_pickle=False) as p:
                    self.assertEqual(set(p.files), io.PARAMETER_KEYS)

    def test_raw_disease_uses_ids_not_csv_row_order(self):
        x = self.rng.normal(size=(29, 1485))
        z = io.unitize(x, "float64_center_l2")
        fit_idx = np.array([20, 2, 7, 0, 18])
        dual = self.rng.normal(size=5)
        b = dict(
            bundle_type="ukb_raw_disease_residualized_krr_stage1_target",
            edge_names=io.EDGES,
            target="250.20",
            feature_transform="row_center_then_l2_unitize",
            kernel_type="Pearson_correlation_precomputed",
            target_space="covariate_residual_z",
            prediction_scale="z",
            refit_on_train_validation=False,
            fit_eid=(1000 + fit_idx).astype(str),
            dual_coef=dual,
            y_mean=0.1,
            y_std=0.3,
        )
        df = pd.DataFrame(x, columns=io.EDGES)
        df.insert(0, "eid", np.arange(29) + 1000)
        csv = self.root / "synthetic_ukb.csv"
        df.sample(frac=1, random_state=4).to_csv(csv, index=False)
        source = self.root / "disease.joblib"
        joblib.dump(b, source)
        out = self.root / "disease"
        io.export_krr(source, out, ukb_csv=csv, expected_ukb_rows=29)
        probe = self.rng.normal(size=(8, 1485))
        _, actual, _ = io.predict_krr(out, probe)
        expected = (io.unitize(probe, "float64_center_l2") @ z[fit_idx].T) @ dual
        np.testing.assert_allclose(actual[:, 0], expected, atol=1e-12, rtol=1e-12)
        df = df[df.eid != 1000 + fit_idx[0]]
        df.to_csv(csv, index=False)
        with self.assertRaisesRegex(ValueError, "fit IDs absent"):
            io.export_krr(
                source, self.root / "bad_alignment", ukb_csv=csv, expected_ukb_rows=28
            )
        self.assertFalse((self.root / "bad_alignment").exists())

    def test_rejects_nonlinear_kernel_and_unexpected_export_fields(self):
        b, _, _ = self.krr_fixture()
        b["metadata"]["kernel_type"] = "rbf"
        source = self.root / "bad.joblib"
        joblib.dump(b, source)
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            io.export_krr(source, self.root / "bad_export")
        self.assertFalse((self.root / "bad_export").exists())
        b["metadata"]["kernel_type"] = "corr_precomputed"
        joblib.dump(b, source)
        out = self.root / "good"
        io.export_krr(source, out)
        with np.load(out / "parameters.npz", allow_pickle=False) as p:
            arrays = {key: p[key] for key in p.files}
        arrays["eid_train"] = np.arange(10)
        np.savez_compressed(out / "parameters.npz", **arrays)
        meta = json.loads((out / "model_metadata.json").read_text())
        meta["parameter_sha256"] = io.sha256(out / "parameters.npz")
        io.write_json(out / "model_metadata.json", meta)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            io.validate_krr(out)

    def test_dnn_manifest_works_without_log_and_checks_edge_order(self):
        manifest = self.root / "dnn_stage1_manifest_stage1_seed0.json"
        io.write_json(manifest, dict(target_names=["synthetic_a"], edge_names=io.EDGES))
        self.assertEqual(io.dnn_names(self.root, "stage1_seed0"), ["synthetic_a"])
        io.write_json(
            manifest, dict(target_names=["synthetic_a"], edge_names=io.EDGES[::-1])
        )
        with self.assertRaisesRegex(ValueError, "edge order"):
            io.dnn_names(self.root, "stage1_seed0")

    def test_legacy_log_fallback_reads_names_only(self):
        np.savez(
            self.root / "dnn_stage1_log_synthetic.npz",
            target_names=np.array(["synthetic_a"]),
            eid_test=np.array([123456]),
            participant_object=np.array([object()], dtype=object),
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertEqual(io.dnn_names(self.root, "synthetic"), ["synthetic_a"])
        self.assertTrue(seen)

    def test_review_krr_format_and_cli_prediction_scale(self):
        b, _, _ = self.krr_fixture(raw=True)
        source = self.root / "raw.joblib"
        joblib.dump(b, source)
        out = self.root / "raw"
        meta = io.export_krr(source, out)
        meta["format"] = "CONVERTED_feature_space_weights_NOT_original_bundle"
        io.write_json(out / "conversion_and_validation.json", meta)
        (out / "model_metadata.json").unlink()
        x = self.rng.normal(size=(3, 1485))
        df = pd.DataFrame(x, columns=io.EDGES)
        df.insert(0, "synthetic_scan", ["a", "b", "c"])
        cohort = self.root / "cohort.csv"
        df.to_csv(cohort, index=False)
        # Match the actual CSV input and its array layout; float32 row reductions
        # can change at the last bit between C- and Fortran-contiguous arrays.
        _, _, expected = io.predict_krr(out, pd.read_csv(cohort)[io.EDGES].to_numpy())
        pred = self.root / "pred.csv"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.source_models.infer_source_parameters",
                "--model-dir",
                str(out),
                "--cohort-csv",
                str(cohort),
                "--id-column",
                "synthetic_scan",
                "--prediction-scale",
                "scaled",
                "--out-csv",
                str(pred),
            ],
            check=True,
            capture_output=True,
        )
        actual = pd.read_csv(pred)
        self.assertEqual(actual.synthetic_scan.tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(
            actual.iloc[:, 1:].to_numpy(), expected, atol=1e-12, rtol=1e-12
        )

    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "PyTorch is not installed; native DNN test must run in CI/authorized training environment",
    )
    def test_native_dnn_weights_are_identical_and_inference_needs_no_log(self):
        import torch
        from torch import nn

        class DNN(nn.Module):
            def __init__(self):
                super().__init__()
                widths = [1485, 3, 4, 5]
                for i in range(1, 4):
                    setattr(
                        self,
                        f"fc{i}",
                        nn.Sequential(
                            nn.Dropout(0.24),
                            nn.Linear(widths[i - 1], widths[i]),
                            nn.ReLU(),
                            nn.BatchNorm1d(widths[i]),
                        ),
                    )
                self.fc4 = nn.Sequential(nn.Dropout(0.24), nn.Linear(5, 2))

            def forward(self, x):
                return self.fc4(self.fc3(self.fc2(self.fc1(x))))

        torch.manual_seed(8)
        model = DNN().eval()
        source = self.root / "dnn_stage1_model_stage1_seed0.pth"
        torch.save(model.state_dict(), source)
        np.savez(
            self.root / "dnn_stage1_norm_stage1_seed0.npz",
            y_mean=np.array([2.0, 3.0]),
            y_std=np.array([4.0, 5.0]),
        )
        io.write_json(
            self.root / "dnn_stage1_manifest_stage1_seed0.json",
            dict(target_names=["synthetic_a", "synthetic_b"], edge_names=io.EDGES),
        )
        out = self.root / "dnn"
        io.export_dnn(source, out)
        self.assertEqual(io.sha256(source), io.sha256(out / "model.pth"))
        x = self.rng.normal(size=(4, 1485)).astype(np.float32)
        _, actual, _ = io.predict_dnn(out, x)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        np.testing.assert_array_equal(actual, expected)
        # Attached attributes survive serialization but are ignored by
        # load_state_dict; they must not bypass the parameter-only check.
        state = torch.load(out / "model.pth", weights_only=True)
        state.participant_rows = torch.zeros((2, 1485))
        torch.save(state, out / "model.pth")
        meta = json.loads((out / "model_metadata.json").read_text())
        meta["weight_sha256"] = io.sha256(out / "model.pth")
        io.write_json(out / "model_metadata.json", meta)
        with self.assertRaisesRegex(ValueError, "state attributes"):
            io.load_dnn(out)


if __name__ == "__main__":
    unittest.main()
