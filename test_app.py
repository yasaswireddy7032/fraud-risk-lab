import unittest
from app import *

class FraudTests(unittest.TestCase):
    def test_reproducible(self):
        a,b=dataset(100);c,d=dataset(100);np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    def test_threshold_uses_precision(self):self.assertGreater(choose_threshold(np.array([0,1]),np.array([.1,.9]),1),.1)
    def test_no_qualifier(self):self.assertEqual(choose_threshold(np.array([0,1]),np.array([.9,.1]),1),1)
    def test_heldout_report(self):
        r=experiment();self.assertEqual(sum(map(sum,r['confusion_matrix'])),r['test_n']);self.assertTrue(0<=r['average_precision']<=1)

if __name__ == '__main__': unittest.main()
