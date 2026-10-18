"""Calculate ABCD Haufe patterns from user-supplied data and predictions."""

import argparse
from types import SimpleNamespace
import haufe_engine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--abcd-csv", required=True)
    p.add_argument("--predictions-csv", required=True)
    p.add_argument("--selected-cohort-csv", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    haufe_engine.run(
        SimpleNamespace(
            abcd_csv=a.abcd_csv,
            pred_csv=a.predictions_csv,
            selected_cohort_csv=a.selected_cohort_csv,
            out_dir=a.out_dir,
            id_col="src_subject_id",
            subjectkey_col="subjectkey",
            target_col="intelligence_score",
            expected_edges=1485,
            min_subjects=9954,
            top_edge_n=30,
            top_node_n=20,
            plot_edge_each=15,
        )
    )


if __name__ == "__main__":
    main()
