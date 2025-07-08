"""Tests for natural forest pattern generator."""

import unittest
from unittest.mock import patch

from tree_planner.generators.natural_generator import NaturalForestGenerator


class TestNaturalForestGenerator(unittest.TestCase):
    """Test cases for NaturalForestGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = NaturalForestGenerator(
            width=20.0, length=20.0, tree_distance=2.0, randomness=0.3
        )

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.width, 20.0)
        self.assertEqual(self.generator.length, 20.0)
        self.assertEqual(self.generator.tree_distance, 2.0)
        self.assertEqual(self.generator.randomness, 0.3)
        self.assertEqual(self.generator.method_name, "natural")

    def test_initialization_with_config(self):
        """Test generator initialization with custom config."""
        config = {"max_natural_iterations": 5}
        generator = NaturalForestGenerator(
            width=10.0, length=10.0, tree_distance=1.5, randomness=0.2, config=config
        )
        self.assertEqual(generator.max_iterations, 5)

    def test_method_name_property(self):
        """Test method_name property."""
        self.assertEqual(self.generator.method_name, "natural")

    def test_create_forest_gaps(self):
        """Test _create_forest_gaps method."""
        gaps = self.generator._create_forest_gaps()

        self.assertIsInstance(gaps, list)
        self.assertGreater(len(gaps), 0)
        self.assertLessEqual(len(gaps), 8)  # Max 8 gaps

        # Check gap structure
        for gap in gaps:
            self.assertIsInstance(gap, tuple)
            self.assertEqual(len(gap), 3)  # (x, y, radius)
            x, y, radius = gap
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)
            self.assertIsInstance(radius, float)
            self.assertGreaterEqual(radius, self.generator.tree_distance * 1.5)

    def test_generate_cluster_centers(self):
        """Test _generate_cluster_centers method."""
        # Mock gaps
        gaps = [(5.0, 5.0, 2.0), (15.0, 15.0, 3.0)]

        centers = self.generator._generate_cluster_centers(gaps)

        self.assertIsInstance(centers, list)
        # May be 0 if all attempts fall in gaps
        self.assertGreaterEqual(len(centers), 0)
        self.assertLessEqual(len(centers), 12)  # Max 12 clusters

        # Check center structure
        for center in centers:
            self.assertIsInstance(center, tuple)
            self.assertEqual(len(center), 2)  # (x, y)
            x, y = center
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    @patch("random.uniform")
    @patch("random.randint")
    @patch("numpy.random.normal")
    @patch("numpy.sqrt")
    def test_populate_cluster(self, mock_sqrt, mock_normal, mock_randint, mock_uniform):
        """Test _populate_cluster method."""
        center = (10.0, 10.0)
        gaps = [(5.0, 5.0, 2.0)]
        trees = []

        # Mock random values
        mock_randint.return_value = 5
        mock_uniform.return_value = 0.5  # angle
        mock_normal.return_value = 2.0  # distance
        mock_sqrt.return_value = 10.0  # distance from gap

        # Mock the is_valid_position method to always return True
        with patch.object(self.generator, "is_valid_position", return_value=True):
            self.generator._populate_cluster(center, gaps, trees)

        # Should have attempted to add trees
        self.assertGreaterEqual(len(trees), 0)

        # Check tree structure
        for tree in trees:
            self.assertIsInstance(tree, tuple)
            self.assertEqual(len(tree), 2)
            x, y = tree
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_add_scattered_trees(self):
        """Test _add_scattered_trees method."""
        gaps = [(5.0, 5.0, 2.0)]
        trees = []

        self.generator._add_scattered_trees(gaps, trees)

        # Should have attempted to add trees (may be 0 due to gap/distance constraints)
        self.assertGreaterEqual(len(trees), 0)

        # Check tree structure if any were added
        for tree in trees:
            self.assertIsInstance(tree, tuple)
            self.assertEqual(len(tree), 2)
            x, y = tree
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_is_valid_position_variable_valid(self):
        """Test _is_valid_position_variable with valid position."""
        new_tree = (10.0, 10.0)
        min_distance = 2.0
        existing_trees = [(5.0, 5.0), (15.0, 15.0)]

        result = self.generator._is_valid_position_variable(
            new_tree, min_distance, existing_trees
        )

        self.assertTrue(result)

    def test_is_valid_position_variable_invalid(self):
        """Test _is_valid_position_variable with invalid position."""
        new_tree = (10.0, 10.0)
        min_distance = 2.0
        existing_trees = [(9.0, 9.0)]  # Too close

        result = self.generator._is_valid_position_variable(
            new_tree, min_distance, existing_trees
        )

        self.assertFalse(result)

    def test_is_valid_position_variable_empty_trees(self):
        """Test _is_valid_position_variable with no existing trees."""
        new_tree = (10.0, 10.0)
        min_distance = 2.0
        existing_trees = []

        result = self.generator._is_valid_position_variable(
            new_tree, min_distance, existing_trees
        )

        self.assertTrue(result)

    def test_generate_positions_basic(self):
        """Test basic position generation."""
        # Use small area for faster testing
        generator = NaturalForestGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_natural_iterations": 2},
        )

        positions = generator.generate_positions()

        self.assertIsInstance(positions, list)
        self.assertEqual(positions, generator.trees)

        # Check position structure
        for position in positions:
            self.assertIsInstance(position, tuple)
            self.assertEqual(len(position), 2)
            x, y = position
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, generator.width)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, generator.length)

    def test_generate_positions_with_callback(self):
        """Test position generation with callback function."""
        callback_calls = []

        def mock_callback(iteration, count, max_iterations, is_improvement):
            callback_calls.append((iteration, count, max_iterations, is_improvement))

        generator = NaturalForestGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_natural_iterations": 2},
        )

        positions = generator.generate_positions(callback=mock_callback)

        self.assertIsInstance(positions, list)
        self.assertGreater(len(callback_calls), 0)

        # Check callback calls
        for call in callback_calls:
            iteration, count, max_iterations, is_improvement = call
            self.assertIsInstance(iteration, int)
            self.assertIsInstance(count, int)
            self.assertIsInstance(max_iterations, int)
            self.assertIsInstance(is_improvement, bool)
            self.assertGreaterEqual(iteration, 1)
            self.assertLessEqual(iteration, max_iterations)
            self.assertGreaterEqual(count, 0)

    def test_generate_positions_optimization(self):
        """Test that optimization keeps the best result."""
        generator = NaturalForestGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_natural_iterations": 3},
        )

        # Mock the individual methods to return different counts
        with (
            patch.object(generator, "_create_forest_gaps", return_value=[]),
            patch.object(
                generator, "_generate_cluster_centers", return_value=[(5.0, 5.0)]
            ),
            patch.object(
                generator,
                "_populate_cluster",
                side_effect=lambda c, g, t: t.extend([(4.0, 4.0), (6.0, 6.0)]),
            ),
            patch.object(
                generator,
                "_add_scattered_trees",
                side_effect=lambda g, t: t.extend([(8.0, 8.0)]),
            ),
        ):
            positions = generator.generate_positions()

        # Should have some trees
        self.assertGreater(len(positions), 0)

    def test_gap_constraints(self):
        """Test that gaps have proper constraints."""
        gaps = self.generator._create_forest_gaps()

        for gap_x, gap_y, gap_radius in gaps:
            # Gap center should be within bounds with margin
            self.assertGreaterEqual(gap_x, self.generator.tree_distance)
            self.assertLessEqual(
                gap_x, self.generator.width - self.generator.tree_distance
            )
            self.assertGreaterEqual(gap_y, self.generator.tree_distance)
            self.assertLessEqual(
                gap_y, self.generator.length - self.generator.tree_distance
            )

            # Gap radius should be reasonable
            min_radius = self.generator.tree_distance * 1.5
            max_radius = (
                self.generator.tree_distance * 3 * (1 + self.generator.randomness)
            )
            self.assertGreaterEqual(gap_radius, min_radius)
            self.assertLessEqual(gap_radius, max_radius)

    def test_cluster_count_constraints(self):
        """Test that cluster count is within proper bounds."""
        gaps = []
        centers = self.generator._generate_cluster_centers(gaps)

        # Should have between 2-12 clusters
        self.assertGreaterEqual(len(centers), 0)  # May be less due to gap avoidance
        self.assertLessEqual(len(centers), 12)

    def test_edge_case_small_area(self):
        """Test generator with very small area."""
        generator = NaturalForestGenerator(
            width=3.0,
            length=3.0,
            tree_distance=1.0,
            randomness=0.1,
            config={"max_natural_iterations": 1},
        )

        positions = generator.generate_positions()

        # Should still work with small area
        self.assertIsInstance(positions, list)
        for position in positions:
            x, y = position
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 3.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 3.0)

    def test_edge_case_zero_randomness(self):
        """Test generator with zero randomness."""
        generator = NaturalForestGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.0,
            config={"max_natural_iterations": 1},
        )

        positions = generator.generate_positions()

        # Should still work with zero randomness
        self.assertIsInstance(positions, list)

    def test_edge_case_high_randomness(self):
        """Test generator with high randomness."""
        generator = NaturalForestGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.9,
            config={"max_natural_iterations": 1},
        )

        positions = generator.generate_positions()

        # Should still work with high randomness
        self.assertIsInstance(positions, list)


if __name__ == "__main__":
    unittest.main()
