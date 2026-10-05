import unittest

import numpy as np

from plot_scan import _shared_signed_limit


class PlotScanTests(unittest.TestCase):
    def test_all_components_use_limit_of_widest_map(self):
        fields = (("a", "A"), ("b", "B"))
        maps = {"a": np.array([[0.0, 1.0, -2.0]]),
                "b": np.array([[0.0, -4.0, np.nan]])}
        self.assertEqual(_shared_signed_limit(maps, fields, 100, 100), 400)
        self.assertEqual(_shared_signed_limit(maps, fields, 1, 100), 4)


if __name__ == "__main__":
    unittest.main()
