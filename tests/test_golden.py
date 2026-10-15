"""Every output in tests/reference/golden_digests.json is unchanged (see tests/golden.py)."""
import json
import unittest

import pytest

import golden


@pytest.mark.golden
class GoldenTests(unittest.TestCase):
    def test_outputs_match_the_golden_digests(self):
        want = json.loads(golden.REFERENCE.read_text())
        got = golden.digests()
        changed = sorted(k for k in want if got.get(k) != want[k])
        missing, new = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        msg = (f"{len(changed)} outputs changed: {changed[:40]}\nmissing: {missing[:20]}\nnew: {new[:20]}\n"
               "Inspect with: uv run python tests/golden.py show <key>")
        self.assertEqual((changed, missing, new), ([], [], []), msg)


if __name__ == "__main__":
    unittest.main()
