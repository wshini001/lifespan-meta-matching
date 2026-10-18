"""Calculate Yeo-7 spatial memberships for 55 selected ICA components.

Requires ICA maps and Yeo labels on matching spatial grids.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from membership_engine import calculate_membership


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ica", required=True)
    parser.add_argument("--yeo", required=True)
    parser.add_argument(
        "--good-components",
        default=str(Path(__file__).parent / "config/good_components.txt"),
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    import nibabel as nib

    ica_image = nib.load(args.ica)
    yeo_image = nib.load(args.yeo)
    if ica_image.shape != (91, 109, 91, 100):
        raise ValueError("Expected ICA-100 grid: 91 x 109 x 91 x 100")
    if tuple(np.squeeze(yeo_image.dataobj).shape) != (91, 109, 91):
        raise ValueError("Yeo grid must match the ICA grid")
    if not np.allclose(ica_image.affine, yeo_image.affine, atol=1e-08):
        raise ValueError("ICA/Yeo affines differ")
    ica = np.asarray(ica_image.dataobj, dtype=np.float32)
    labels = np.asarray(yeo_image.dataobj).squeeze()
    if not np.isfinite(labels).all() or not np.array_equal(labels, labels.astype(int)):
        raise ValueError("Yeo labels must be finite integers")
    yeo = labels.astype(int)
    if set(np.unique(yeo)) != set(range(8)):
        raise ValueError("Expected Yeo labels 0 through 7")
    good = np.loadtxt(args.good_components, dtype=int).reshape(-1)
    if len(good) != 55 or len(set(good)) != 55 or np.any((good < 1) | (good > 100)):
        raise ValueError("Expected 55 unique, one-based ICA component IDs")
    table = calculate_membership(ica, yeo, good)
    total = table[[f"abs_total_Yeo{k}" for k in range(1, 8)]].sum(axis=1)
    cortical = table[[f"abs_cortical_Yeo{k}" for k in range(1, 8)]].sum(axis=1)
    if not np.allclose(total, table.cortical_abs_mass_fraction, atol=1e-08, rtol=0):
        raise ValueError("Total spatial memberships do not sum to cortical coverage")
    if not np.allclose(cortical, 1, atol=1e-08, rtol=0):
        raise ValueError("Cortical-normalized memberships do not sum to one")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "network_membership.csv", index=False)
    metadata = {
        "ica_sha256": file_hash(args.ica),
        "yeo_sha256": file_hash(args.yeo),
        "good_components_sha256": file_hash(args.good_components),
        "n_nodes": 55,
        "primary_membership": "absolute ICA magnitude in network / total absolute magnitude",
        "resampling": "Input Yeo atlas is already on the ICA grid",
    }
    (out / "input_contract.json").write_text(json.dumps(metadata, indent=2))
    print("PASS: ICA-to-Yeo7 network memberships calculated")


if __name__ == "__main__":
    main()
