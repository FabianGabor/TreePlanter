"""Tests for core visualization and constants."""

import unittest
from unittest.mock import MagicMock, patch

from tests import TEST_CONFIG
from tree_planner.core import (
    MIN_TREES_FOR_WI_HEATMAP,
    PERLIN_GRID_SPACING_FACTOR,
    WI_COLORMAP_COLORS,
    WI_COLORMAP_N,
    WI_NEIGHBOR_COUNT,
    WI_STANDARD_ANGLE_DEGREES,
    TreePlanner,
)


class TestTreePlannerVisualization(unittest.TestCase):
    """Test cases for TreePlanner visualization methods and constants."""

    def setUp(self):
        """Set up test fixtures."""
        self.width = 20.0
        self.length = 15.0
        self.tree_distance = 3.0
        self.randomness = 0.5

    def test_constants_defined(self):
        """Test that visualization constants are properly defined."""
        # Test that constants have reasonable values
        self.assertEqual(PERLIN_GRID_SPACING_FACTOR, 0.8)
        self.assertEqual(WI_STANDARD_ANGLE_DEGREES, 72.0)
        self.assertEqual(WI_NEIGHBOR_COUNT, 4)
        self.assertEqual(MIN_TREES_FOR_WI_HEATMAP, 5)
        self.assertIsInstance(WI_COLORMAP_COLORS, list)
        self.assertEqual(len(WI_COLORMAP_COLORS), 7)
        self.assertEqual(WI_COLORMAP_N, 256)

    def test_uniform_angle_planner_initialization(self):
        """Test TreePlanner with uniform_angle method."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )

        self.assertEqual(planner.method, "uniform_angle")
        self.assertEqual(planner.generator.method_name, "uniform_angle")

    @patch("tree_planner.core.plt")
    def test_generate_planting_image_uniform_angle(self, mock_plt):
        """Test planting image generation for uniform angle method."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.colorbar.return_value = MagicMock()

        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )

        # Add some trees to test Wi heatmap generation
        planner.trees = [
            (3.0, 3.0),
            (6.0, 3.0),
            (9.0, 3.0),
            (12.0, 3.0),
            (15.0, 3.0),
            (3.0, 6.0),
            (6.0, 6.0),
            (9.0, 6.0),
            (12.0, 6.0),
            (15.0, 6.0),
        ]

        result = planner.generate_planting_image()

        # Check that an image was generated
        self.assertIsNotNone(result)

        # Verify matplotlib setup calls
        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(12, 8))
        mock_ax.set_xlim.assert_called_once_with(0, self.width)
        mock_ax.set_ylim.assert_called_once_with(0, self.length)

    def test_calculate_wi_values_insufficient_trees(self):
        """Test Wi calculation with insufficient trees."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )

        # Set trees below threshold
        planner.trees = [(3.0, 3.0), (6.0, 6.0)]  # Only 2 trees, need 5

        wi_values = planner._calculate_wi_values_for_visualization()
        self.assertEqual(wi_values, [])

    def test_calculate_wi_values_sufficient_trees(self):
        """Test Wi calculation with sufficient trees."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )

        # Set trees above threshold
        planner.trees = [
            (3.0, 3.0),
            (6.0, 3.0),
            (9.0, 3.0),
            (12.0, 3.0),
            (15.0, 3.0),
            (3.0, 6.0),
        ]

        wi_values = planner._calculate_wi_values_for_visualization()
        self.assertEqual(len(wi_values), len(planner.trees))

        # All Wi values should be between 0 and 1
        for wi in wi_values:
            self.assertGreaterEqual(wi, 0.0)
            self.assertLessEqual(wi, 1.0)

    @patch("tree_planner.core.plt")
    def test_add_simple_uniform_angle_background(self, mock_plt):
        """Test fallback uniform angle background."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )

        # Set insufficient trees to trigger fallback
        planner.trees = [(3.0, 3.0)]

        planner._add_simple_uniform_angle_background(mock_ax)

        # Should call axvline and axhline for grid lines
        self.assertTrue(mock_ax.axvline.called)
        self.assertTrue(mock_ax.axhline.called)
        self.assertTrue(mock_ax.text.called)

    def test_get_plot_title_uniform_angle(self):
        """Test plot title generation for uniform angle method."""
        planner = TreePlanner(
            self.width,
            self.length,
            self.tree_distance,
            self.randomness,
            "uniform_angle",
            TEST_CONFIG,
        )
        planner.trees = [(3.0, 3.0), (6.0, 6.0)]

        title, subtitle = planner._get_plot_title()

        self.assertIn("Uniform Angle Index Method", title)
        self.assertIn(
            "Zhang et al. (2019)", subtitle
        )  # Should be in subtitle, not title
        self.assertIn(f"{self.width}m x {self.length}m", subtitle)
        self.assertIn("2 trees", subtitle)

    def test_all_methods_have_plot_titles(self):
        """Test that all generator methods have plot titles defined."""
        methods = ["perlin", "poisson", "natural", "uniform_angle"]

        for method in methods:
            with self.subTest(method=method):
                planner = TreePlanner(
                    self.width,
                    self.length,
                    self.tree_distance,
                    self.randomness,
                    method,
                    TEST_CONFIG,
                )
                planner.trees = [(3.0, 3.0)]

                title, subtitle = planner._get_plot_title()
                self.assertIsInstance(title, str)
                self.assertIsInstance(subtitle, str)
                self.assertGreater(len(title), 0)
                self.assertGreater(len(subtitle), 0)


if __name__ == "__main__":
    unittest.main()
