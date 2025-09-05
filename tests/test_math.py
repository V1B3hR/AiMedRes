import unittest

# Function to test
def add(a, b):
    return a + b

# Test case class
class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)  # Check if result is 5

    def test_add_negative(self):
        self.assertEqual(add(-1, 1), 0)  # Check if result is 0

if __name__ == '__main__':
    unittest.main()
