import unittest

class DepsTest(unittest.TestCase):
    def test(self):
        # have to pip install xgboost
        from autoviz import AutoViz_Class as AV
        AVC = AV.AutoViz_Class()
        self.assertIsNotNone(AVC)