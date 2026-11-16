import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import lookup_order_status  # noqa: E402


class TestLookup(unittest.TestCase):
    def test_shipped(self):
        self.assertEqual(lookup_order_status("A123")["status"], "Shipped")

    def test_delivered(self):
        self.assertEqual(lookup_order_status("A124")["status"], "Delivered")

    def test_processing(self):
        self.assertEqual(lookup_order_status("A125")["status"], "Processing")

    def test_unknown(self):
        self.assertEqual(lookup_order_status("Z999")["status"], "Not Found")

    def test_case_and_spaces(self):
        self.assertEqual(lookup_order_status("  a123 ")["status"], "Shipped")


if __name__ == "__main__":
    unittest.main()
