import unittest
from mon_package.core import Core

class TestCore(unittest.TestCase):

    def setUp(self):
        self.core = Core()

    def test_initialize(self):
        self.core.initialize()
        # Add assertions to verify the expected behavior

    def test_run(self):
        result = self.core.run()
        # Add assertions to verify the expected behavior

if __name__ == '__main__':
    unittest.main()