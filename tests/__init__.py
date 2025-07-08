"""Test package initialization."""

import os
import sys

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_CONFIG = {
    "max_perlin_iterations": 3,
    "max_poisson_iterations": 3,
    "max_natural_iterations": 3,
    "max_uniform_angle_iterations": 5,
    "randomness_threshold": 0.3,
}
