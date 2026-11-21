"""Checks that run on every OS without downloading model weights."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in [ROOT / "asrr6.py", ROOT / "Shenava" / "persian_itn.py"]:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_small_runtime_assets_exist(self):
        for name in ("tokens.txt", "persian_itn.py"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "Shenava" / name).is_file())

    def test_no_machine_specific_model_path(self):
        source = (ROOT / "asrr6.py").read_text(encoding="utf-8")
        self.assertNotIn(r"C:\Users\A\Desktop\Shenava", source)
        self.assertIn('os.environ.get("SHENAVA_DIR"', source)


if __name__ == "__main__":
    unittest.main()
