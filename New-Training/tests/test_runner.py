"""
Test runner for HVT v3 training pipeline.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main(["-v", "--tb=short"])