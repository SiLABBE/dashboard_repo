import unittest

from dash import gauge

class TestGauge(unittest.TestCase):
    def test_limit_threshold(self):
        """
        Test that threshold is at 0.47 and not 0.5
        """
        proba_in = 0.49
        result = gauge(proba_in)
        self.assertEqual(result, 'Loan Decision: To be confirmed')

if __name__ == '__main__':
    unittest.main()

# python -m unittest test_gauge