"""Export one trusted legacy source model without participant-record fields.
"""

import argparse
import json
import os

try:
    from .parameter_io import export_dnn, export_krr
except ImportError:
    from parameter_io import export_dnn, export_krr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["krr", "dnn"], required=True)
    parser.add_argument(
        "--source", required=True, help="Trusted local legacy .joblib or .pth"
    )
    parser.add_argument("--out-dir", required=True, help="New private output directory")
    parser.add_argument(
        "--ukb-csv",
        help="Only for raw-disease KRR conversion; read locally and not copied",
    )
    parser.add_argument("--expected-ukb-rows", type=int, default=24668)
    args = parser.parse_args()
    os.umask(0o077)
    if args.kind == "dnn":
        result = export_dnn(args.source, args.out_dir)
    else:
        result = export_krr(
            args.source,
            args.out_dir,
            ukb_csv=args.ukb_csv,
            expected_ukb_rows=args.expected_ukb_rows,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
