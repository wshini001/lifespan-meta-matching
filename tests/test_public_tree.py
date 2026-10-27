"""Publication guard tests use synthetic files only."""

import tempfile
from pathlib import Path
import unittest

from scripts.check_public_tree import check_file


class PublicationGuard(unittest.TestCase):
    def test_rejects_model_archive_table_and_hidden_participant_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["model.pth", "data.zip", "cohort.csv", "bank.parquet"]:
                (root / name).write_bytes(b"synthetic test")
                with self.assertRaises(ValueError):
                    check_file(root, name)
            (root / "config.json").write_text('{"fit_eid": [123456]}')
            with self.assertRaises(ValueError):
                check_file(root, "config.json")
            (root / "config.json").write_text('{"id_column": "eid", "seeds": [0, 1]}')
            check_file(root, "config.json")
            (root / "pointer.txt").write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:synthetic\n"
            )
            with self.assertRaises(ValueError):
                check_file(root, "pointer.txt")


if __name__ == "__main__":
    unittest.main()
