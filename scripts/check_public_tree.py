"""Check Git-tracked publication files; not a storage or privacy certificate."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

EXAMPLES = {
    "examples/simulated_data/simulated_cohort.csv",
    "examples/simulated_data/simulated_pc_predictions.csv",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
}
RECORD_FIELDS = {
    "eid_train",
    "eid_train_full",
    "eid_valid",
    "eid_test",
    "fit_eid",
    "Z_train_full",
    "dual_coef",
    "y_true_z",
    "y_pred_z",
    "final_test_eid",
}


def check_json(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in RECORD_FIELDS and isinstance(child, (list, dict)):
                raise ValueError("JSON contains a participant/training-array field")
            check_json(child)
    elif isinstance(value, list):
        for child in value:
            check_json(child)


def check_file(root, relative):
    path = root / relative
    if path.is_symlink():
        raise ValueError("Tracked symlink requires separate review")
    if relative in EXAMPLES:
        return
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
        ".gitignore",
        "LICENSE",
        "LICENSE.md",
        ".gitattributes",
    }:
        raise ValueError("Non-allowlisted data/model/archive/binary file")
    content = path.read_text(encoding="utf-8")
    if "\x00" in content or content.startswith(
        "version https://git-lfs.github.com/spec/v1"
    ):
        raise ValueError("Binary content or Git LFS pointer requires review")
    if path.suffix == ".json":
        check_json(json.loads(content))


def check_simulation(root):
    spec = importlib.util.spec_from_file_location(
        "simulated_generator", root / "examples/generate_simulated_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        module.generate(directory, seed=42)
        for relative in sorted(EXAMPLES):
            actual = pd.read_csv(root / relative)
            expected = pd.read_csv(Path(directory) / Path(relative).name)
            if actual.shape != expected.shape or list(actual.columns) != list(
                expected.columns
            ):
                raise ValueError("Simulation table shape/column mismatch")
            for column in expected:
                if pd.api.types.is_numeric_dtype(expected[column]):
                    if not np.allclose(
                        actual[column],
                        expected[column],
                        atol=1e-12,
                        rtol=1e-12,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            "Simulation numeric values do not match generator"
                        )
                elif not actual[column].equals(expected[column]):
                    raise ValueError(
                        "Simulation identifiers/labels do not match generator"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .strip("\0")
        .split("\0")
    )
    failures = []
    for relative in tracked:
        try:
            check_file(root, relative)
        except (ValueError, OSError, UnicodeError) as exc:
            failures.append({"path": relative, "reason": str(exc)})
    try:
        check_simulation(root)
    except (ValueError, OSError) as exc:
        failures.append({"path": "examples/simulated_data", "reason": str(exc)})
    print(
        json.dumps(
            {
                "tracked_files": len(tracked),
                "failures": failures,
                "scope": "Current tracked working-tree files and reproducibility of the two simulated tables; not Git history, releases, remote assets, storage deletion, model privacy or retention permission.",
            },
            indent=2,
        )
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
