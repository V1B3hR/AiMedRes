"""
Advanced pytest version of basic math operations testing.
Demonstrates pytest features like parameterization and fixtures.
"""
import pytest


# Function to test (moved from original test_math.py)
def add(a, b):
    """Add two numbers together."""
    return a + b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


def divide(a, b):
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class TestMathOperations:
    """Advanced math operations tests using pytest features."""
    
    def test_add_basic(self):
        """Test basic addition functionality."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (10, -5, 5),
        (1.5, 2.5, 4.0),
        (-10, -20, -30)
    ])
    def test_add_parametrized(self, a, b, expected):
        """Test addition with multiple parameter sets."""
        assert add(a, b) == expected
    
    def test_multiply_basic(self):
        """Test basic multiplication."""
        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0
    
    @pytest.mark.parametrize("a,b,expected", [
        (3, 4, 12),
        (-2, 3, -6),
        (0, 100, 0),
        (1, 1, 1),
        (0.5, 2, 1.0)
    ])
    def test_multiply_parametrized(self, a, b, expected):
        """Test multiplication with multiple parameter sets."""
        assert multiply(a, b) == expected
    
    def test_divide_basic(self):
        """Test basic division."""
        assert divide(10, 2) == 5
        assert divide(9, 3) == 3
        assert divide(-6, 2) == -3
    
    @pytest.mark.parametrize("a,b,expected", [
        (10, 2, 5),
        (9, 3, 3),
        (-6, 2, -3),
        (1, 1, 1),
        (0, 1, 0)
    ])
    def test_divide_parametrized(self, a, b, expected):
        """Test division with multiple parameter sets."""
        result = divide(a, b)
        assert abs(result - expected) < 1e-10  # Handle floating point precision
    
    def test_divide_by_zero(self):
        """Test division by zero raises appropriate exception."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)
    
    @pytest.fixture
    def math_operations_data(self):
        """Fixture providing test data for math operations."""
        return {
            'positive_numbers': [1, 2, 3, 4, 5],
            'negative_numbers': [-1, -2, -3, -4, -5],
            'zero': 0,
            'floats': [1.1, 2.2, 3.3]
        }
    
    def test_add_with_fixture(self, math_operations_data):
        """Test addition using fixture data."""
        positive = math_operations_data['positive_numbers']
        negative = math_operations_data['negative_numbers']
        
        # Test positive + positive
        assert add(positive[0], positive[1]) == 3
        
        # Test positive + negative
        assert add(positive[0], negative[0]) == 0
        
        # Test with zero
        assert add(positive[0], math_operations_data['zero']) == positive[0]
    
    @pytest.mark.performance
    def test_performance_add(self, performance_timer):
        """Test addition performance."""
        performance_timer.start()
        
        # Perform many additions
        for i in range(10000):
            add(i, i + 1)
        
        elapsed = performance_timer.stop()
        
        # Should complete quickly
        assert elapsed < 1.0, f"Addition took too long: {elapsed}s"
    
    @pytest.mark.slow
    def test_large_number_operations(self):
        """Test operations with very large numbers."""
        large_num = 10**15
        
        assert add(large_num, 1) == large_num + 1
        assert multiply(large_num, 2) == large_num * 2
        assert divide(large_num, 2) == large_num / 2