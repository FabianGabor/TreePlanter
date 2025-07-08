"""Tests for the TreePlanner core class."""

import io
import unittest
from unittest.mock import MagicMock, patch

from tests import TEST_CONFIG
from tree_planner.core import TreePlanner


class TestTreePlanner(unittest.TestCase):
    """Test cases for TreePlanner core class."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 10.0
        self.length = 8.0
        self.tree_distance = 2.0
        self.randomness = 0.3
        self.method = "perlin"

    def test_initialization_valid_method(self):
        """Test initialization with valid method."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            self.method,
            TEST_CONFIG,
        )

        self.assertEqual(planner.width, self.width)
        self.assertEqual(planner.length, self.length)
        self.assertEqual(planner.tree_distance, self.tree_distance)
        self.assertEqual(planner.randomness, self.randomness)
        self.assertEqual(planner.method, self.method)
        self.assertIsNotNone(planner.generator)

        # Check that the generator has the expected interface
        self.assertTrue(hasattr(planner.generator, "generate_positions"))
        self.assertTrue(hasattr(planner.generator, "get_statistics"))

    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        with self.assertRaises(ValueError):
            TreePlanner(
                self.width,
                self.length,
                self.tree_distance,
                self.randomness,
                "invalid_method",
            )

    def test_generate_tree_positions(self):
        """Test tree position generation."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            self.method,
            TEST_CONFIG,
        )

        callback = MagicMock()
        result = planner.generate_tree_positions(callback)

        # Check that positions were generated
        self.assertIsInstance(result, list)
        self.assertEqual(planner.trees, result)

        # Check that all positions are within bounds
        for x, y in result:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, self.width)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, self.length)

    @patch("tree_planner.core.plt")
    @patch("tree_planner.generators.PerlinNoiseGenerator")
    def test_generate_planting_image(self, mock_generator_class, mock_plt):
        """Test planting image generation."""
        mock_generator = MagicMock()
        mock_generator.method_name = "perlin"
        mock_generator_class.return_value = mock_generator

        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            self.method,
            TEST_CONFIG,
        )
        planner.trees = [(2, 2), (4, 4), (6, 6)]

        result = planner.generate_planting_image()

        self.assertIsInstance(result, io.BytesIO)

        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(12, 8))
        mock_ax.set_xlim.assert_called_once_with(0, self.width)
        mock_ax.set_ylim.assert_called_once_with(0, self.length)
        mock_ax.set_aspect.assert_called_once_with("equal")

    def test_get_tree_coordinates_json(self):
        """Test tree coordinates JSON generation."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            self.method,
            TEST_CONFIG,
        )
        planner.trees = [(1.23, 2.34), (3.45, 4.56)]

        result = planner.get_tree_coordinates_json()

        expected_keys = ["area", "spacing", "total_trees", "method", "coordinates"]
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertEqual(result["area"]["width"], self.width)
        self.assertEqual(result["area"]["length"], self.length)
        self.assertEqual(result["spacing"], self.tree_distance)
        self.assertEqual(result["total_trees"], 2)
        self.assertEqual(result["method"], self.method)

        # Check coordinates format
        coordinates = result["coordinates"]
        self.assertEqual(len(coordinates), 2)

        coord1 = coordinates[0]
        self.assertEqual(coord1["x"], 1.23)
        self.assertEqual(coord1["y"], 2.34)
        self.assertEqual(coord1["id"], 1)

        coord2 = coordinates[1]
        self.assertEqual(coord2["x"], 3.45)
        self.assertEqual(coord2["y"], 4.56)
        self.assertEqual(coord2["id"], 2)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            self.method,
            TEST_CONFIG,
        )

        result = planner.get_statistics()

        # Check that all expected keys are present
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
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(result["method"], self.method)
        self.assertEqual(result["width"], self.width)
        self.assertEqual(result["length"], self.length)
        self.assertEqual(result["tree_distance"], self.tree_distance)
        self.assertEqual(result["randomness"], self.randomness)
        self.assertEqual(result["area"], self.width * self.length)


if __name__ == "__main__":
    unittest.main()
