"""Uniform angle index method-based tree position generator."""

from collections.abc import Callable
import logging
import random

from ..base import TreePositionGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class UniformAngleGenerator(TreePositionGenerator):
    """Generate tree positions using uniform angle index method.

    Based on "Designing near-natural planting patterns for plantation forests in China"
    by Zhang et al. (2019).

    Implements the uniform angle index formula:
    W_i = (1/n) * Σ_{j=1}^{n} z_ij, where
    z_ij = 1 if α_ij < α_0, 0 otherwise
    and 0 ≤ W_i ≤ 1
    """

    # Constants for the uniform angle index method
    DEFAULT_ALPHA_0 = 72.0  # Standard threshold angle in degrees (360°/5)
    DEFAULT_N_NEIGHBORS = 4  # Number of nearest neighbors to consider
    TARGET_W_RATIO = 0.5  # Target ratio of trees with W_i ≥ 0.5 (random pattern)
    MAX_OPTIMIZATION_ITERATIONS = 50  # Maximum iterations for optimization

    # Grid and positioning constants
    # Unit size = tree_distance * this factor (10x higher resolution)
    GRID_RESOLUTION_FACTOR = 0.1
    BUFFER_DISTANCE_METERS = 1.0  # Buffer distance in meters to avoid edge effects
    # Minimum neighbors needed for structural unit
    MIN_NEIGHBORS_FOR_STRUCTURAL_UNIT = 4
    # Number of neighbors selected for random structural unit
    STRUCTURAL_UNIT_SIZE = 4

    # Constraint thresholds
    # Constraint 1: N_{Wi=0.5} >= 0.5N (at least 50% random units)
    MIN_RANDOM_RATIO = 0.5
    MIN_DENSITY_RATIO = 0.5  # Constraint 2: N >= 0.5N0 (reasonable density)

    # Natural displacement constants
    # Max displacement = tree_distance * randomness * this factor
    DISPLACEMENT_FACTOR = 0.25
    DISPLACEMENT_RANGE = 2.0  # Range multiplier for random displacement

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
        self.alpha_0 = self.config.get("alpha_0", self.DEFAULT_ALPHA_0)
        self.n_neighbors = self.config.get("n_neighbors", self.DEFAULT_N_NEIGHBORS)
        self._last_generation_stats = {}

    @property
    def method_name(self) -> str:
        return "uniform_angle"

    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using uniform angle index method.

        From Zhang et al. (2019). Implements the optimization objective: max N_{Wi=0.5}
        with constraints:
        (1) N_{Wi=0.5} >= 0.5N (at least 50% random units)
        (2) N >= 0.5N0 (reasonable density)
        (3) Regular (feasible for plantation establishment)
        """
        logger.info(
            f"Starting uniform angle index optimization for "
            f"{self.width}x{self.length}m area with {self.tree_distance}m spacing"
        )

        # Create planting point grid with much higher resolution than tree spacing
        # Use very small unit size to allow natural variation and break grid alignment
        unit_size = self.tree_distance * self.GRID_RESOLUTION_FACTOR
        cols = int(self.width / unit_size) + 1
        rows = int(self.length / unit_size) + 1
        n0 = cols * rows  # Total planting points (N0)

        # Set up buffer as mentioned in the paper to avoid edge effects
        buffer_cells = max(1, int(self.BUFFER_DISTANCE_METERS / unit_size))

        best_solution = None
        best_n_wi_05 = 0  # Best number of random units (Wi = 0.5)
        best_metrics = {}

        # Run multiple iterations to find the best solution (optimization)
        max_iterations = self.MAX_OPTIMIZATION_ITERATIONS

        for iteration in range(max_iterations):
            # Initialize grid: 0=empty, 1=planted, -1=destroyed
            grid = [[0 for _ in range(cols)] for _ in range(rows)]
            trees = []
            random_units_count = 0  # N_{Wi=0.5}
            total_units = 0  # N (total structural units)

            # Process grid sequentially row by row (Zhang et al. algorithm)
            # Apply buffer: skip edge cells to avoid systematic errors
            for row in range(buffer_cells, rows - buffer_cells):
                for col in range(buffer_cells, cols - buffer_cells):
                    # Step 1: Check if this can be a reference tree
                    if grid[row][col] == -1:  # Already destroyed
                        continue

                    # Get 8 neighbors in 3x3 grid
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

                    # Count available neighbors (not destroyed)
                    available_neighbors = [
                        (nr, nc) for nr, nc in neighbors if grid[nr][nc] != -1
                    ]

                    # Need at least 4 neighbors to form a structural unit
                    if (
                        len(available_neighbors)
                        >= self.MIN_NEIGHBORS_FOR_STRUCTURAL_UNIT
                    ):
                        # Step 2: Create random structural unit (Wi = 0.5)
                        # This maximizes N_{Wi=0.5} as per the objective function
                        success = self._create_random_structural_unit(
                            grid, row, col, available_neighbors, unit_size, trees
                        )
                        if success:
                            random_units_count += 1  # Increment N_{Wi=0.5}
                        total_units += 1  # Increment N

            # Apply Zhang et al. constraints
            # Constraint 1: N_{Wi=0.5} >= 0.5N (at least 50% random units)
            random_ratio = random_units_count / max(total_units, 1)
            constraint_1_satisfied = random_ratio >= self.MIN_RANDOM_RATIO

            # Constraint 2: N >= 0.5N0 (reasonable density)
            density_ratio = len(trees) / n0
            constraint_2_satisfied = density_ratio >= self.MIN_DENSITY_RATIO

            # If density is too low, fill additional trees to meet constraint 2
            if not constraint_2_satisfied:
                self._fill_to_meet_density_constraint(
                    grid, unit_size, trees, n0 * self.MIN_DENSITY_RATIO
                )
                density_ratio = len(trees) / n0
                constraint_2_satisfied = density_ratio >= self.MIN_DENSITY_RATIO

            # Evaluate this solution based on the objective function: max N_{Wi=0.5}
            if (
                constraint_1_satisfied
                and constraint_2_satisfied
                and random_units_count > best_n_wi_05
            ):
                best_solution = trees.copy()
                best_n_wi_05 = random_units_count
                best_metrics = {
                    "iteration": iteration + 1,
                    "random_units_count": random_units_count,
                    "total_units": total_units,
                    "random_ratio": random_ratio,
                    "density_ratio": density_ratio,
                    "n0": n0,
                    "objective_value": random_units_count,  # N_{Wi=0.5}
                    "constraint_1_satisfied": constraint_1_satisfied,
                    "constraint_2_satisfied": constraint_2_satisfied,
                    "constraint_3_satisfied": True,  # Always true (regular grid-based)
                }

                if callback:
                    callback(iteration + 1, len(trees), max_iterations, True)
            elif callback:
                callback(iteration + 1, len(trees), max_iterations, False)

        # Use best solution found
        if best_solution is not None:
            self.trees = best_solution
            self._last_generation_stats = best_metrics
        else:
            # Fallback: use last iteration if no solution met all constraints
            self.trees = trees if "trees" in locals() else []
            self._last_generation_stats = {
                "iteration": max_iterations,
                "random_units_count": 0,
                "total_units": 0,
                "random_ratio": 0.0,
                "density_ratio": 0.0,
                "n0": n0,
                "objective_value": 0,
                "constraint_1_satisfied": False,
                "constraint_2_satisfied": False,
                "constraint_3_satisfied": True,
            }

        logger.info(
            f"Uniform angle optimization completed: {len(self.trees)} trees placed\n"
            f"  Objective N_{{Wi=0.5}} = {best_metrics.get('random_units_count', 0)} "
            f"(max random units)\n"
            f"  Constraint 1: N_{{Wi=0.5}}/N = "
            f"{best_metrics.get('random_ratio', 0):.3f} "
            f">= {self.MIN_RANDOM_RATIO}: "
            f"{best_metrics.get('constraint_1_satisfied', False)}\n"
            f"  Constraint 2: N/N0 = {best_metrics.get('density_ratio', 0):.3f} "
            f">= {self.MIN_DENSITY_RATIO}: "
            f"{best_metrics.get('constraint_2_satisfied', False)}\n"
            f"  Best solution found at iteration {best_metrics.get('iteration', 0)}"
        )

        if callback:
            callback(max_iterations, len(self.trees), max_iterations, True)

        return self.trees

    def _create_random_structural_unit(
        self,
        grid: list[list[int]],
        center_row: int,
        center_col: int,
        available_neighbors: list[tuple[int, int]],
        unit_size: float,
        trees: list[tuple[float, float]],
    ) -> bool:
        """Create a random structural unit (Wi = 0.5) following Zhang et al. method.

        From the paper: "the code will find 4 of 8 points to build a random
        structural unit and 'plant' trees on them; the remaining 4 planting
        points are empty and will be 'destroyed'."
        """
        if len(available_neighbors) < self.MIN_NEIGHBORS_FOR_STRUCTURAL_UNIT:
            return False

        # The 7 possible random structural units from Figure 3 (middle row)
        # Each pattern represents which 4 of 8 neighbors to select
        random_patterns = [
            [0, 2, 4, 6],  # Pattern 1: Diagonal cross
            [1, 3, 5, 7],  # Pattern 2: Plus pattern (rotated 45°)
            [0, 1, 4, 5],  # Pattern 3: Two adjacent pairs
            [2, 3, 6, 7],  # Pattern 4: Two adjacent pairs (opposite)
            [0, 3, 4, 7],  # Pattern 5: Opposite pairs
            [1, 2, 5, 6],  # Pattern 6: Opposite pairs (rotated)
            [0, 1, 2, 4],  # Pattern 7: Three consecutive + one
        ]

        # Map neighbor positions to pattern indices (3x3 grid, center excluded)
        # 8 neighbors in clockwise order starting from top-left
        neighbor_map = dict(enumerate(available_neighbors[:8]))

        # Select a random pattern
        pattern = random.choice(random_patterns)
        selected_positions = []

        # Try to apply the pattern
        selected_positions = [
            neighbor_map[idx] for idx in pattern if idx in neighbor_map
        ]

        # Need exactly 4 valid positions for Wi = 0.5
        if len(selected_positions) >= self.STRUCTURAL_UNIT_SIZE:
            selected_positions = selected_positions[: self.STRUCTURAL_UNIT_SIZE]

            # Plant center tree (reference tree) with natural displacement
            base_x = center_col * unit_size
            base_y = center_row * unit_size
            center_x, center_y = self._apply_natural_displacement(base_x, base_y)
            if (
                0 <= center_x <= self.width
                and 0 <= center_y <= self.length
                and self.is_valid_position((center_x, center_y), trees)
            ):
                trees.append((center_x, center_y))
                grid[center_row][center_col] = 1  # Planted

            # Plant 4 selected neighbors with natural displacement
            for nr, nc in selected_positions:
                base_x = nc * unit_size
                base_y = nr * unit_size
                x, y = self._apply_natural_displacement(base_x, base_y)
                if (
                    0 <= x <= self.width
                    and 0 <= y <= self.length
                    and self.is_valid_position((x, y), trees)
                ):
                    trees.append((x, y))
                    grid[nr][nc] = 1  # Planted

            # Destroy remaining neighbors (mark as -1)
            for nr, nc in available_neighbors:
                if (nr, nc) not in selected_positions and grid[nr][nc] == 0:
                    grid[nr][nc] = -1  # Destroyed

            return True

        return False

    def _fill_to_meet_density_constraint(
        self,
        grid: list[list[int]],
        unit_size: float,
        trees: list[tuple[float, float]],
        target_trees: float,
    ) -> None:
        """Fill additional trees to meet density constraint N >= 0.5N0."""
        current_trees = len(trees)
        needed_trees = int(target_trees - current_trees)

        if needed_trees <= 0:
            return

        rows, cols = len(grid), len(grid[0])
        empty_positions = []

        # Find all empty positions
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:  # Empty
                    base_x = col * unit_size
                    base_y = row * unit_size
                    # Apply natural displacement for density filling too
                    x, y = self._apply_natural_displacement(base_x, base_y)
                    if 0 <= x <= self.width and 0 <= y <= self.length:
                        empty_positions.append((row, col, x, y))

        # Randomly select positions to fill
        random.shuffle(empty_positions)
        filled = 0

        for row, col, x, y in empty_positions:
            if filled >= needed_trees:
                break

            # Check if position maintains minimum distance with stricter validation
            if self.is_valid_position((x, y), trees):
                trees.append((x, y))
                grid[row][col] = 1  # Planted
                filled += 1

    def get_generation_stats(self) -> dict:
        """Get additional statistics specific to uniform angle method."""
        base_stats = self.get_statistics()
        base_stats.update(self._last_generation_stats)
        return base_stats

    def _apply_natural_displacement(self, x: float, y: float) -> tuple[float, float]:
        """Apply natural displacement to grid position based on randomness factor."""
        if self.randomness == 0:
            return x, y

        # Maximum displacement is proportional to randomness and tree distance
        # Keep it smaller to avoid violating minimum distance constraints
        max_displacement = (
            self.tree_distance * self.randomness * self.DISPLACEMENT_FACTOR
        )

        # Apply random displacement in both x and y directions
        dx = (random.random() - 0.5) * self.DISPLACEMENT_RANGE * max_displacement
        dy = (random.random() - 0.5) * self.DISPLACEMENT_RANGE * max_displacement

        # Ensure the displaced position stays within bounds
        new_x = max(0, min(self.width, x + dx))
        new_y = max(0, min(self.length, y + dy))

        return new_x, new_y
