"""The data directory is decided when needed, so --data-dir (paths.set_data_dir) applies everywhere."""
import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autosteer import cli, paths
from autosteer.sampler import manifest
from autosteer.services.dataset_store import DatasetStore
from autosteer.web import review


class DataDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.saved = (paths._override, os.environ.get("AUTOSTEER_DATA_DIR"))

    def tearDown(self):
        paths._override = self.saved[0]
        if self.saved[1] is None:
            os.environ.pop("AUTOSTEER_DATA_DIR", None)
        else:
            os.environ["AUTOSTEER_DATA_DIR"] = self.saved[1]
        self.tmp.cleanup()

    def test_everything_follows_set_data_dir(self):
        paths.set_data_dir(self.root)
        self.assertEqual(paths.data_dir(), self.root)
        self.assertEqual(paths.templates_dir(), self.root / "templates")
        self.assertEqual(Path(manifest.manifest_path()), self.root / "manifest.json")
        self.assertEqual(review._review_dir(), self.root / "review")
        self.assertEqual(DatasetStore().data_root, self.root)
        self.assertEqual(os.environ["AUTOSTEER_DATA_DIR"], str(self.root))   # a reloader process inherits it

    def test_the_environment_variable_is_read_when_needed(self):
        paths._override = None
        os.environ["AUTOSTEER_DATA_DIR"] = str(self.root)
        self.assertEqual(paths.data_dir(), self.root)
        self.assertEqual(Path(manifest.manifest_path()), self.root / "manifest.json")

    def test_dev_with_data_dir_uses_it(self):
        seen = {}
        fake = mock.MagicMock(run=lambda *a, **k: seen.update(data_dir=paths.data_dir(), store=DatasetStore().data_root))
        args = argparse.Namespace(data_dir=str(self.root), host="127.0.0.1", port=0, reload=False)
        with mock.patch.dict(sys.modules, {"uvicorn": fake}):
            cli.run_dev(args)
        self.assertEqual(seen, {"data_dir": self.root, "store": self.root})
        self.assertTrue((self.root / "templates").is_dir())


if __name__ == "__main__":
    unittest.main()
