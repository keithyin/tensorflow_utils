from __future__ import print_function

from unittest import TestCase
import pandas as pd
import draw_auuc


class Test(TestCase):
    def test_compute_area(self):
        values = pd.Series([5., 10., 10.], index=[0, 1, 2])
        print(draw_auuc.compute_area(values))
