"""Tests for generator package initialization."""

import unittest

from tree_planner.generators import (
    NaturalForestGenerator,
    PerlinNoiseGenerator,
    PoissonDiscGenerator,
    UniformAngleGenerator,
)


class TestGeneratorsInit(unittest.TestCase):
    """Test cases for generators package initialization."""

    def test_import_generators(self):
        """Test that all generators exist."""
        self.assertTrue(hasattr(PerlinNoiseGenerator, "method_name"))
        self.assertTrue(hasattr(PoissonDiscGenerator, "method_name"))
        self.assertTrue(hasattr(NaturalForestGenerator, "method_name"))
        self.assertTrue(hasattr(UniformAngleGenerator, "method_name"))


if __name__ == "__main__":
    unittest.main()
