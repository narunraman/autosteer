"""Concurrent manifest builds: the startup warm-up and a first request used to race on one temporary file."""
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from autosteer.sampler import manifest


class ManifestConcurrencyTests(unittest.TestCase):
    def test_threads_loading_a_missing_manifest_all_succeed(self):
        with TemporaryDirectory() as d:
            path = str(Path(d) / "manifest.json")
            builds = []
            def entry(name):
                builds.append(name)
                threading.Event().wait(0.05)   # slow enough for the threads to overlap
                return {"template_question_count": 1}
            with mock.patch.object(manifest, "MANIFEST_PATH", path), \
                 mock.patch.object(manifest, "_template_names_on_disk", return_value={"a", "b"}), \
                 mock.patch.object(manifest, "_build_entry", side_effect=entry):
                errors, results = [], []
                def load():
                    try:
                        results.append(manifest.load_manifest())
                    except Exception as e:   # noqa: BLE001
                        errors.append(e)
                threads = [threading.Thread(target=load) for _ in range(8)]
                for t in threads: t.start()
                for t in threads: t.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 8)
            self.assertEqual(sorted(builds), ["a", "b"])   # built once, not once per thread
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["manifest.json"])   # no temporary files left


if __name__ == "__main__":
    unittest.main()
