"""The Review step reads a sample of templates spread over (type, level) cells in proportion to their size."""
import unittest

from autosteer.web.review import _allocate


class AllocateTests(unittest.TestCase):
    def test_proportional_with_one_per_cell(self):
        alloc = _allocate({"big": 50, "mid": 5, "small": 1}, 20)
        self.assertEqual(sum(alloc.values()), 20)
        self.assertEqual(alloc["small"], 1)  # every cell is read when the sample is large enough
        self.assertGreater(alloc["big"], alloc["mid"])

    def test_never_more_than_a_cell_has(self):
        alloc = _allocate({"a": 3, "b": 3}, 20)
        self.assertEqual(alloc, {"a": 3, "b": 3})

    def test_more_cells_than_picks(self):
        alloc = _allocate({k: 1 for k in range(30)}, 10)
        self.assertEqual(sum(alloc.values()), 10)
        self.assertTrue(all(v in (0, 1) for v in alloc.values()))


if __name__ == "__main__":
    unittest.main()
