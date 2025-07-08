"""Uniform angle index method-based tree position generator."""

from collections.abc import Callable
import logging
import random

import numpy as np

from ..base import TreePositionGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class UniformAngleGenerator(TreePositionGenerator):
    """Generate tree positions using uniform angle index method.

    Based on "Designing near-natural planting patterns for plantation forests in China"
    by Zhang et al. (2019).
    """

    # Constants for the uniform angle index method
    MIN_GRID_SIZE = 3  # Minimum grid size for meaningful patterns
    MIN_NEIGHBORS_FOR_UNIT = 4  # Minimum neighbors needed for structural unit
    RANDOM_UNIT_TARGET_RATIO = 0.5  # Target ratio of random units (≥50%)
    RANDOM_CREATION_PROBABILITY = 0.7  # Probability of creating random units
    NON_RANDOM_CREATION_PROBABILITY = 0.3  # Probability of creating non-random units
    MIN_DENSITY_RATIO = 0.5  # Minimum density constraint (N ≥ 0.5N0)
    STRUCTURAL_UNIT_SIZE = 4  # Number of neighbors in a structural unit

    def __init__(
        self,
        width: float,
        length: float,
        tree_distance: float,
        randomness: float = 0.3,
        config: dict | None = None,
    ):
        super().__init__(width, length, tree_distance, randomness, config)
        self.max_iterations = self.config.get(
            "max_uniform_angle_iterations", Config.MAX_UNIFORM_ANGLE_ITERATIONS
        )
        self._last_generation_stats = {}

    @property
    def method_name(self) -> str:
        return "uniform_angle"

    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using uniform angle index method."""
        logger.info(
            f"Starting uniform angle index generation for {self.width}x{self.length}m "
            f"area with {self.tree_distance}m spacing"
        )

        self.trees = []

        # Calculate grid dimensions for planting points
        unit_size = self.tree_distance
        cols = int(self.width / unit_size) + 1
        rows = int(self.length / unit_size) + 1
        n0 = cols * rows  # Total possible planting points

        # Ensure we have enough space for meaningful patterns
        if cols < self.MIN_GRID_SIZE or rows < self.MIN_GRID_SIZE:
            logger.warning("Area too small for uniform angle method, using simple grid")
            return self._generate_simple_grid()

        # Initialize grid state: 0=empty, 1=planted, -1=destroyed
        grid = np.zeros((rows, cols), dtype=int)

        # Track statistics for objective function
        random_units_count = 0
        total_units_processed = 0

        # Algorithm from the paper: Process grid sequentially
        for row in range(1, rows - 1):  # Skip edges for complete 3x3 neighborhoods
            for col in range(1, cols - 1):
                # Step 1: Identify potential reference tree (not destroyed)
                if grid[row, col] != -1:  # Not destroyed
                    # Get 8-neighborhood positions
                    neighbors = []
                    neighbor_positions = [
                        (-1, -1),
                        (0, -1),
                        (1, -1),  # Top row
                        (-1, 0),
                        (1, 0),  # Middle row (skip center)
                        (-1, 1),
                        (0, 1),
                        (1, 1),  # Bottom row
                    ]

                    for dr, dc in neighbor_positions:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append((nr, nc))

                    # Count available neighbors (empty or planted, not destroyed)
                    available_neighbors = [
                        (nr, nc) for nr, nc in neighbors if grid[nr, nc] != -1
                    ]

                    # Need at least 4 neighbors to form a structural unit
                    if len(available_neighbors) >= self.MIN_NEIGHBORS_FOR_UNIT:
                        # Calculate current proportion of random units
                        current_random_ratio = random_units_count / max(
                            total_units_processed, 1
                        )

                        # Apply constraint: aim for ≥50% random units
                        should_create_random = (
                            total_units_processed == 0  # First unit
                            or current_random_ratio
                            < self.RANDOM_UNIT_TARGET_RATIO  # Below target
                            or random.random()
                            < self.RANDOM_CREATION_PROBABILITY  # Probabilistic
                        )

                        if should_create_random:
                            # Create random structural unit (Wi = 0.5)
                            success = self._create_random_structural_unit(
                                grid, row, col, available_neighbors, unit_size
                            )
                            if success:
                                random_units_count += 1
                                total_units_processed += 1
                        # Create even or cluster unit with lower probability
                        elif (
                            random.random() < self.NON_RANDOM_CREATION_PROBABILITY
                        ):  # Occasionally create non-random units
                            self._create_even_structural_unit(
                                grid, row, col, available_neighbors, unit_size
                            )
                            total_units_processed += 1

                # Progress callback
                if callback and total_units_processed % 5 == 0:
                    iteration = row * cols + col
                    max_iter = rows * cols
                    callback(iteration, len(self.trees), max_iter, False)

        # Apply constraint: N ≥ 0.5N0 (maintain reasonable density)
        current_density = len(self.trees) / n0
        if current_density < self.MIN_DENSITY_RATIO:
            # Fill additional spaces to meet density constraint
            self._fill_to_meet_density_constraint(
                grid, unit_size, n0 * self.MIN_DENSITY_RATIO
            )

        # Final validation and cleanup
        self.trees = self._validate_and_clean_positions(self.trees)

        # Calculate final statistics
        final_random_ratio = random_units_count / max(total_units_processed, 1)
        density_ratio = len(self.trees) / n0

        logger.info(
            f"Generated {len(self.trees)} trees with "
            f"{random_units_count}/{total_units_processed} "
            f"random units ({final_random_ratio * 100:.1f}%), "
            f"density: {density_ratio:.1f}"
        )

        if callback:
            callback(rows * cols, len(self.trees), rows * cols, True)

        # Store additional metadata
        self._last_generation_stats = {
            "random_units": random_units_count,
            "total_units": total_units_processed,
            "random_percentage": final_random_ratio * 100,
            "density_ratio": density_ratio,
            "meets_constraints": final_random_ratio >= self.RANDOM_UNIT_TARGET_RATIO
            and density_ratio >= self.MIN_DENSITY_RATIO,
        }

        return self.trees

    def _generate_simple_grid(self) -> list[tuple[float, float]]:
        """Fallback method for small areas."""
        cols = int(self.width / self.tree_distance)
        rows = int(self.length / self.tree_distance)

        trees = []
        for row in range(rows):
            for col in range(cols):
                x = col * self.tree_distance + self.tree_distance / 2
                y = row * self.tree_distance + self.tree_distance / 2
                if x < self.width and y < self.length:
                    trees.append((x, y))

        # Store simple grid metadata for consistency
        self._last_generation_stats = {
            "random_units": 0,
            "total_units": 0,
            "random_percentage": 0,
            "density_ratio": len(trees) / (cols * rows) if cols * rows > 0 else 0,
            "meets_constraints": False,
        }

        return trees

    def _create_random_structural_unit(
        self,
        grid: np.ndarray,
        center_row: int,
        center_col: int,
        available_neighbors: list[tuple[int, int]],
        unit_size: float,
    ) -> bool:
        """Create a random structural unit (Wi = 0.5) following Zhang et al. method."""
        if len(available_neighbors) < self.MIN_NEIGHBORS_FOR_UNIT:
            return False

        # The 7 possible random patterns from the paper
        random_patterns = [
            [0, 2, 4, 6],  # Diagonal cross pattern
            [1, 3, 5, 7],  # Plus pattern (rotated 45°)
            [0, 1, 4, 5],  # Two adjacent pairs (NE quadrant)
            [2, 3, 6, 7],  # Two adjacent pairs (SW quadrant)
            [0, 3, 4, 7],  # Opposite pairs
            [1, 2, 5, 6],  # Opposite pairs (rotated)
            [0, 1, 2, 4],  # Three consecutive + one
        ]

        # Map neighbor positions to pattern indices
        neighbor_map = {}
        deltas = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

        for i, (dr, dc) in enumerate(deltas):
            neighbor_pos = (center_row + dr, center_col + dc)
            if neighbor_pos in available_neighbors:
                neighbor_map[i] = neighbor_pos

        # Try to apply a random pattern
        pattern = random.choice(random_patterns)
        selected_positions = []

        selected_positions = [
            neighbor_map[pattern_idx]
            for pattern_idx in pattern
            if pattern_idx in neighbor_map
        ]

        # Need exactly 4 valid positions
        if len(selected_positions) >= self.STRUCTURAL_UNIT_SIZE:
            selected_positions = selected_positions[: self.STRUCTURAL_UNIT_SIZE]

            # Plant center tree
            x = center_col * unit_size
            y = center_row * unit_size
            if 0 <= x <= self.width and 0 <= y <= self.length:
                self.trees.append((x, y))
                grid[center_row, center_col] = 1  # Planted

            # Plant selected neighbors
            for nr, nc in selected_positions:
                x = nc * unit_size
                y = nr * unit_size
                if 0 <= x <= self.width and 0 <= y <= self.length:
                    self.trees.append((x, y))
                    grid[nr, nc] = 1  # Planted

            # Mark remaining neighbors as destroyed
            for nr, nc in available_neighbors:
                if (nr, nc) not in selected_positions and grid[nr, nc] == 0:
                    grid[nr, nc] = -1  # Destroyed

            return True

        return False

    def _create_even_structural_unit(
        self,
        grid: np.ndarray,
        center_row: int,
        center_col: int,
        available_neighbors: list[tuple[int, int]],
        unit_size: float,
    ) -> bool:
        """Create an even structural unit (Wi < 0.5) for diversity."""
        if len(available_neighbors) < self.MIN_NEIGHBORS_FOR_UNIT:
            return False

        # Select 4 neighbors with maximum angular separation
        selected_positions = available_neighbors[:4]

        # Plant center tree
        x = center_col * unit_size
        y = center_row * unit_size
        if 0 <= x <= self.width and 0 <= y <= self.length:
            self.trees.append((x, y))
            grid[center_row, center_col] = 1  # Planted

        # Plant selected neighbors
        for nr, nc in selected_positions:
            x = nc * unit_size
            y = nr * unit_size
            if 0 <= x <= self.width and 0 <= y <= self.length:
                self.trees.append((x, y))
                grid[nr, nc] = 1  # Planted

        # Mark remaining neighbors as destroyed
        for nr, nc in available_neighbors:
            if (nr, nc) not in selected_positions and grid[nr, nc] == 0:
                grid[nr, nc] = -1  # Destroyed

        return True

    def _fill_to_meet_density_constraint(
        self, grid: np.ndarray, unit_size: float, target_trees: float
    ) -> None:
        """Fill additional spaces to meet density constraint N ≥ 0.5N0."""
        current_trees = len(self.trees)
        needed_trees = int(target_trees - current_trees)

        if needed_trees <= 0:
            return

        rows, cols = grid.shape
        empty_positions = []

        # Find all empty positions
        for row in range(rows):
            for col in range(cols):
                if grid[row, col] == 0:  # Empty
                    x = col * unit_size
                    y = row * unit_size
                    if 0 <= x <= self.width and 0 <= y <= self.length:
                        empty_positions.append((row, col, x, y))

        # Randomly select positions to fill
        random.shuffle(empty_positions)
        filled = 0

        for row, col, x, y in empty_positions:
            if filled >= needed_trees:
                break

            # Check if position maintains minimum distance
            if self.is_valid_position((x, y)):
                self.trees.append((x, y))
                grid[row, col] = 1  # Planted
                filled += 1

    def _validate_and_clean_positions(
        self, trees: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Validate and clean tree positions."""
        valid_trees = []

        for tree in trees:
            x, y = tree
            # Ensure tree is within bounds
            if (
                0 <= x <= self.width
                and 0 <= y <= self.length
                and self._is_valid_position_for_cleanup(tree, valid_trees)
            ):
                valid_trees.append(tree)

        return valid_trees

    def _is_valid_position_for_cleanup(
        self, new_tree: tuple[float, float], existing_trees: list[tuple[float, float]]
    ) -> bool:
        """Check if position is valid during cleanup."""
        new_x, new_y = new_tree
        min_distance = self.tree_distance * 0.8  # Allow slightly closer spacing

        for existing_x, existing_y in existing_trees:
            distance = np.sqrt((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2)
            if distance < min_distance:
                return False
        return True

    def get_generation_stats(self) -> dict:
        """Get additional statistics specific to uniform angle method."""
        base_stats = self.get_statistics()
        base_stats.update(self._last_generation_stats)
        return base_stats
