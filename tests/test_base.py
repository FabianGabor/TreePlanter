"""Tests for the base TreePositionGenerator class."""

import unittest
from unittest.mock import MagicMock

from tree_planner.base import OptimizationTracker, TreePositionGenerator


class MockGenerator(TreePositionGenerator):
    """Mock generator for testing base functionality."""

    @property
    def method_name(self):
        return "mock"

    def generate_positions(self, callback=None):
        # Simple grid pattern for testing
        trees = [
            (x + self.tree_distance / 2, y + self.tree_distance / 2)
            for x in range(0, int(self.width), int(self.tree_distance))
            for y in range(0, int(self.length), int(self.tree_distance))
            if (
                x + self.tree_distance / 2 < self.width
                and y + self.tree_distance / 2 < self.length
            )
        ]
        self.trees = trees
        return trees


class TestTreePositionGenerator(unittest.TestCase):
    """Test cases for TreePositionGenerator base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 10.0
        self.length = 8.0
        self.tree_distance = 2.0
        self.randomness = 0.3

    def test_initialization_valid_inputs(self):
        """Test initialization with valid inputs."""
        generator = MockGenerator(
            self.width, self.length, self.tree_distance, self.randomness
        )

        self.assertEqual(generator.width, self.width)
        self.assertEqual(generator.length, self.length)
        self.assertEqual(generator.tree_distance, self.tree_distance)
        self.assertEqual(generator.randomness, self.randomness)
        self.assertEqual(generator.trees, [])
        self.assertEqual(generator.method_name, "mock")

    def test_initialization_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with self.assertRaises(ValueError):
            MockGenerator(-1, self.length, self.tree_distance, self.randomness)

        with self.assertRaises(ValueError):
            MockGenerator(self.width, 0, self.tree_distance, self.randomness)

    def test_initialization_invalid_tree_distance(self):
        """Test initialization with invalid tree distance."""
        with self.assertRaises(ValueError):
            MockGenerator(self.width, self.length, -1, self.randomness)

        with self.assertRaises(ValueError):
            MockGenerator(
                self.width, self.length, 15, self.randomness
            )  # Larger than smallest dimension

    def test_initialization_invalid_randomness(self):
        """Test initialization with invalid randomness."""
        with self.assertRaises(ValueError):
            MockGenerator(self.width, self.length, self.tree_distance, -0.1)

        with self.assertRaises(ValueError):
            MockGenerator(self.width, self.length, self.tree_distance, 1.1)

    def test_is_valid_position_within_bounds(self):
        """Test position validation within bounds."""
        generator = MockGenerator(
            self.width, self.length, self.tree_distance, self.randomness
        )

        # Valid position
        self.assertTrue(generator.is_valid_position((5.0, 4.0)))

        # Position outside bounds
        self.assertFalse(generator.is_valid_position((-1.0, 4.0)))
        self.assertFalse(generator.is_valid_position((5.0, -1.0)))
        self.assertFalse(generator.is_valid_position((15.0, 4.0)))
        self.assertFalse(generator.is_valid_position((5.0, 15.0)))

    def test_is_valid_position_distance_constraint(self):
        """Test position validation with distance constraints."""
        generator = MockGenerator(
            self.width, self.length, self.tree_distance, self.randomness
        )
        generator.trees = [(5.0, 4.0)]

        # Too close to existing tree
        self.assertFalse(generator.is_valid_position((5.5, 4.0)))

        # Far enough from existing tree
        self.assertTrue(generator.is_valid_position((8.0, 4.0)))

    def test_generate_positions(self):
        """Test position generation."""
        generator = MockGenerator(
            self.width, self.length, self.tree_distance, self.randomness
        )

        callback = MagicMock()
        trees = generator.generate_positions(callback)

        self.assertIsInstance(trees, list)
        self.assertTrue(len(trees) > 0)
        self.assertEqual(generator.trees, trees)

        # Check that all trees are valid positions
        for tree in trees:
            self.assertIsInstance(tree, tuple)
            self.assertEqual(len(tree), 2)
            x, y = tree
            self.assertTrue(0 <= x <= self.width)
            self.assertTrue(0 <= y <= self.length)

    def test_get_statistics(self):
        """Test statistics calculation."""
        generator = MockGenerator(
            self.width, self.length, self.tree_distance, self.randomness
        )
        generator.generate_positions()

        stats = generator.get_statistics()

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

        self.assertEqual(stats["method"], "mock")
        self.assertEqual(stats["area"], self.width * self.length)
        self.assertEqual(stats["width"], self.width)
        self.assertEqual(stats["length"], self.length)
        self.assertEqual(stats["tree_distance"], self.tree_distance)
        self.assertEqual(stats["randomness"], self.randomness)
        self.assertTrue(stats["density"] > 0)


class TestOptimizationTracker(unittest.TestCase):
    """Test cases for OptimizationTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = OptimizationTracker()

    def test_initialization(self):
        """Test tracker initialization."""
        self.assertIsNone(self.tracker.current_generator)
        self.assertEqual(self.tracker.progress_log, [])

    def test_set_generator(self):
        """Test setting generator."""
        mock_generator = MagicMock()
        self.tracker.set_generator(mock_generator)

        self.assertEqual(self.tracker.current_generator, mock_generator)
        self.assertEqual(self.tracker.progress_log, [])

    def test_add_progress(self):
        """Test adding progress data."""
        self.tracker.add_progress(1, 10, 5, True)

        self.assertEqual(len(self.tracker.progress_log), 1)
        progress = self.tracker.progress_log[0]

        expected_keys = [
            "iteration",
            "tree_count",
            "max_iterations",
            "is_best",
            "progress_percent",
        ]
        for key in expected_keys:
            self.assertIn(key, progress)

        self.assertEqual(progress["iteration"], 1)
        self.assertEqual(progress["tree_count"], 10)
        self.assertEqual(progress["max_iterations"], 5)
        self.assertTrue(progress["is_best"])
        self.assertEqual(progress["progress_percent"], 20.0)

    def test_reset(self):
        """Test resetting tracker."""
        self.tracker.add_progress(1, 10, 5, True)
        self.tracker.reset()

        self.assertEqual(self.tracker.progress_log, [])


if __name__ == "__main__":
    unittest.main()
