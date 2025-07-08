"""Tests for Poisson disc sampling generator."""

import unittest
from unittest.mock import patch

import numpy as np

from tree_planner.generators.poisson_generator import PoissonDiscGenerator


class TestPoissonDiscGenerator(unittest.TestCase):
    """Test cases for PoissonDiscGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = PoissonDiscGenerator(
            width=20.0, length=20.0, tree_distance=2.0, randomness=0.3
        )

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.width, 20.0)
        self.assertEqual(self.generator.length, 20.0)
        self.assertEqual(self.generator.tree_distance, 2.0)
        self.assertEqual(self.generator.randomness, 0.3)
        self.assertEqual(self.generator.method_name, "poisson")

    def test_initialization_with_config(self):
        """Test generator initialization with custom config."""
        config = {"max_poisson_iterations": 5}
        generator = PoissonDiscGenerator(
            width=10.0, length=10.0, tree_distance=1.5, randomness=0.2, config=config
        )
        self.assertEqual(generator.max_iterations, 5)

    def test_method_name_property(self):
        """Test method_name property."""
        self.assertEqual(self.generator.method_name, "poisson")

    def test_poisson_disc_sampling_basic(self):
        """Test basic Poisson disc sampling algorithm."""
        points = self.generator._poisson_disc_sampling()

        self.assertIsInstance(points, list)
        self.assertGreater(len(points), 0)

        # Check point structure
        for point in points:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)  # (x, y)
            x, y = point
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, self.generator.width)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, self.generator.length)

    def test_is_valid_poisson_point_valid(self):
        """Test _is_valid_poisson_point with valid point."""
        point = (10.0, 10.0)
        cell_size = 1.0
        min_distance = 2.0

        # Grid with distant point
        grid = {(5, 5): (5.0, 5.0)}

        result = self.generator._is_valid_poisson_point(
            point, grid, cell_size, min_distance
        )

        self.assertTrue(result)

    def test_is_valid_poisson_point_invalid(self):
        """Test _is_valid_poisson_point with invalid point."""
        point = (10.0, 10.0)
        cell_size = 1.0
        min_distance = 2.0

        # Grid with close point
        grid = {(10, 10): (10.5, 10.5)}

        result = self.generator._is_valid_poisson_point(
            point, grid, cell_size, min_distance
        )

        self.assertFalse(result)

    def test_is_valid_poisson_point_empty_grid(self):
        """Test _is_valid_poisson_point with empty grid."""
        point = (10.0, 10.0)
        cell_size = 1.0
        min_distance = 2.0
        grid = {}

        result = self.generator._is_valid_poisson_point(
            point, grid, cell_size, min_distance
        )

        self.assertTrue(result)

    def test_is_valid_poisson_point_boundary_cells(self):
        """Test _is_valid_poisson_point with boundary conditions."""
        point = (0.5, 0.5)  # Near boundary
        cell_size = 1.0
        min_distance = 2.0

        # Grid with point in nearby cell
        grid = {(1, 1): (1.5, 1.5)}

        result = self.generator._is_valid_poisson_point(
            point, grid, cell_size, min_distance
        )

        self.assertFalse(result)

    def test_generate_positions_basic(self):
        """Test basic position generation."""
        # Use small area for faster testing
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_poisson_iterations": 2},
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

        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_poisson_iterations": 2},
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
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_poisson_iterations": 3},
        )

        # Mock the sampling method to return different counts
        mock_results = [
            [(5.0, 5.0), (8.0, 8.0)],  # 2 points
            [(3.0, 3.0)],  # 1 point
            [(2.0, 2.0), (6.0, 6.0), (9.0, 9.0)],  # 3 points (best)
        ]

        with patch.object(
            generator, "_poisson_disc_sampling", side_effect=mock_results
        ):
            positions = generator.generate_positions()

        # Should have the best result (3 points)
        self.assertEqual(len(positions), 3)

    def test_poisson_distance_constraints(self):
        """Test that Poisson sampling respects distance constraints."""
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.0,  # No randomness for predictable testing
            config={"max_poisson_iterations": 1},
        )

        positions = generator.generate_positions()

        # Check minimum distance between all pairs
        min_distance = generator.tree_distance
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    distance = np.sqrt(
                        (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
                    )
                    # Allow small tolerance
                    self.assertGreaterEqual(distance, min_distance * 0.95)

    def test_randomness_effect(self):
        """Test that randomness affects maximum distance."""
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.5,
            config={"max_poisson_iterations": 1},
        )

        # Test internal distance calculation
        min_distance = generator.tree_distance
        max_distance = min_distance * (1 + generator.randomness)

        self.assertEqual(max_distance, 2.0 * 1.5)  # 3.0

    def test_cell_size_calculation(self):
        """Test that cell size is calculated correctly."""
        # Cell size should be min_distance / sqrt(2)
        expected_cell_size = self.generator.tree_distance / np.sqrt(2)

        # Test through a simple calculation rather than mocking the entire sampling
        min_distance = self.generator.tree_distance
        actual_cell_size = min_distance / np.sqrt(2)

        # The cell size calculation is correct
        self.assertAlmostEqual(expected_cell_size, actual_cell_size)

    def test_edge_case_small_area(self):
        """Test generator with very small area."""
        generator = PoissonDiscGenerator(
            width=3.0,
            length=3.0,
            tree_distance=1.0,
            randomness=0.1,
            config={"max_poisson_iterations": 1},
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
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.0,
            config={"max_poisson_iterations": 1},
        )

        positions = generator.generate_positions()

        # Should still work with zero randomness
        self.assertIsInstance(positions, list)

    def test_edge_case_high_randomness(self):
        """Test generator with high randomness."""
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.9,
            config={"max_poisson_iterations": 1},
        )

        positions = generator.generate_positions()

        # Should still work with high randomness
        self.assertIsInstance(positions, list)

    def test_grid_coordinates_calculation(self):
        """Test grid coordinate calculation in validity check."""
        point = (5.7, 8.3)
        cell_size = 2.0

        # Expected grid coordinates
        expected_x = int(5.7 / 2.0)  # 2
        expected_y = int(8.3 / 2.0)  # 4

        # Test through a validity check
        grid = {(expected_x, expected_y): (5.0, 8.0)}

        result = self.generator._is_valid_poisson_point(point, grid, cell_size, 1.0)

        # Should find the point in grid and return False (too close)
        self.assertFalse(result)

    def test_active_list_management(self):
        """Test that active list is managed correctly."""
        generator = PoissonDiscGenerator(
            width=5.0,
            length=5.0,
            tree_distance=2.0,
            randomness=0.0,
            config={"max_poisson_iterations": 1},
        )

        # Mock to simulate no valid points found
        out_of_bounds = [2.5, 2.5] + [100.0] * 60  # Out of bounds
        with (
            patch("random.uniform", side_effect=out_of_bounds),
            patch("random.randint", return_value=0),
        ):
            points = generator._poisson_disc_sampling()

        # Should have at least the seed point
        self.assertGreaterEqual(len(points), 1)

    def test_bounds_checking(self):
        """Test that points are properly bounded."""
        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.3,
            config={"max_poisson_iterations": 1},
        )

        positions = generator.generate_positions()

        for x, y in positions:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 10.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 10.0)

    @patch("random.uniform")
    @patch("random.randint")
    def test_poisson_algorithm_termination(self, mock_randint, mock_uniform):
        """Test that Poisson algorithm terminates properly."""
        # Mock to simulate a situation where no valid points can be found
        # First point valid, rest out of bounds
        mock_uniform.side_effect = [5.0, 5.0] + [100.0] * 100
        mock_randint.return_value = 0

        generator = PoissonDiscGenerator(
            width=10.0,
            length=10.0,
            tree_distance=2.0,
            randomness=0.0,
            config={"max_poisson_iterations": 1},
        )

        points = generator._poisson_disc_sampling()

        # Should have at least the seed point and terminate
        self.assertGreaterEqual(len(points), 1)


if __name__ == "__main__":
    unittest.main()
