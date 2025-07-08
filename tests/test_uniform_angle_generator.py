"""Tests for the uniform angle generator."""

import unittest
from unittest.mock import MagicMock, patch

from tests import TEST_CONFIG
from tree_planner.generators.uniform_angle_generator import UniformAngleGenerator


class TestUniformAngleGenerator(unittest.TestCase):
    """Test cases for UniformAngleGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 20.0
        self.length = 15.0
        self.tree_distance = 3.0
        self.randomness = 0.5
        self.generator = UniformAngleGenerator(
            self.width, self.length, self.tree_distance, self.randomness, TEST_CONFIG
        )

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.width, self.width)
        self.assertEqual(self.generator.length, self.length)
        self.assertEqual(self.generator.tree_distance, self.tree_distance)
        self.assertEqual(self.generator.randomness, self.randomness)
        self.assertEqual(self.generator.method_name, "uniform_angle")

    def test_constants_defined(self):
        """Test that all required constants are defined."""
        # Check that constants exist and have reasonable values
        self.assertEqual(self.generator.DEFAULT_ALPHA_0, 72.0)
        self.assertEqual(self.generator.DEFAULT_N_NEIGHBORS, 4)
        self.assertEqual(self.generator.TARGET_W_RATIO, 0.5)
        self.assertEqual(self.generator.MAX_OPTIMIZATION_ITERATIONS, 50)
        self.assertEqual(self.generator.GRID_RESOLUTION_FACTOR, 0.1)
        self.assertEqual(self.generator.BUFFER_DISTANCE_METERS, 1.0)
        self.assertEqual(self.generator.MIN_NEIGHBORS_FOR_STRUCTURAL_UNIT, 4)
        self.assertEqual(self.generator.STRUCTURAL_UNIT_SIZE, 4)
        self.assertEqual(self.generator.MIN_RANDOM_RATIO, 0.5)
        self.assertEqual(self.generator.MIN_DENSITY_RATIO, 0.5)
        self.assertEqual(self.generator.DISPLACEMENT_FACTOR, 0.25)
        self.assertEqual(self.generator.DISPLACEMENT_RANGE, 2.0)

    def test_generate_positions_basic(self):
        """Test basic position generation."""
        positions = self.generator.generate_positions()

        # Check that positions are generated
        self.assertIsInstance(positions, list)
        self.assertGreater(len(positions), 0)

        # Check that all positions are within bounds
        for x, y in positions:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, self.width)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, self.length)

    def test_generate_positions_with_callback(self):
        """Test position generation with callback."""
        callback = MagicMock()
        positions = self.generator.generate_positions(callback=callback)

        # Check that callback was called
        self.assertTrue(callback.called)
        self.assertIsInstance(positions, list)

    def test_apply_natural_displacement_zero_randomness(self):
        """Test natural displacement with zero randomness."""
        generator = UniformAngleGenerator(
            self.width, self.length, self.tree_distance, 0.0, TEST_CONFIG
        )
        x, y = 5.0, 7.0
        new_x, new_y = generator._apply_natural_displacement(x, y)

        # Should return same position when randomness is 0
        self.assertEqual(new_x, x)
        self.assertEqual(new_y, y)

    def test_apply_natural_displacement_with_randomness(self):
        """Test natural displacement with randomness."""
        x, y = 5.0, 7.0
        new_x, new_y = self.generator._apply_natural_displacement(x, y)

        # Should be within bounds
        self.assertGreaterEqual(new_x, 0)
        self.assertLessEqual(new_x, self.width)
        self.assertGreaterEqual(new_y, 0)
        self.assertLessEqual(new_y, self.length)

    def test_create_random_structural_unit_insufficient_neighbors(self):
        """Test structural unit creation with insufficient neighbors."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        available_neighbors = [(0, 1), (1, 0)]  # Only 2 neighbors, need 4
        trees = []

        result = self.generator._create_random_structural_unit(
            grid, 1, 1, available_neighbors, 1.0, trees
        )

        self.assertFalse(result)
        self.assertEqual(len(trees), 0)

    def test_get_generation_stats(self):
        """Test generation statistics retrieval."""
        # First generate some trees to have stats
        self.generator.generate_positions()
        stats = self.generator.get_generation_stats()

        # Check that stats include base statistics
        expected_keys = [
            "total_trees",
            "area",
            "density",
            "method",
            "width",
            "length",
            "tree_distance",
            "randomness",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_fill_to_meet_density_constraint_no_trees_needed(self):
        """Test density constraint filling when no additional trees needed."""
        grid = [[1, 1], [1, 1]]  # Full grid
        trees = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]  # Already enough trees
        target_trees = 2.0  # Less than current

        initial_count = len(trees)
        self.generator._fill_to_meet_density_constraint(grid, 1.0, trees, target_trees)

        # Should not add any trees
        self.assertEqual(len(trees), initial_count)

    @patch("random.choice")
    def test_create_random_structural_unit_with_mocked_pattern(self, mock_choice):
        """Test structural unit creation with mocked pattern selection."""
        mock_choice.return_value = [0, 2, 4, 6]  # First pattern

        grid = [[0 for _ in range(5)] for _ in range(5)]
        available_neighbors = [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 1),
        ]
        trees = []

        result = self.generator._create_random_structural_unit(
            grid, 2, 2, available_neighbors, 1.0, trees
        )

        # Should succeed with enough neighbors
        self.assertTrue(result or len(trees) > 0)  # Either returns True or adds trees


if __name__ == "__main__":
    unittest.main()
