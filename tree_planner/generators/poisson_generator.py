"""Poisson disc sampling-based tree position generator."""

from collections.abc import Callable
import logging
import random

import numpy as np

from ..base import TreePositionGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class PoissonDiscGenerator(TreePositionGenerator):
    """Generate tree positions using Poisson disc sampling."""

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
            "max_poisson_iterations", Config.MAX_POISSON_ITERATIONS
        )

    @property
    def method_name(self) -> str:
        return "poisson"

    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using Poisson disc sampling optimization."""
        logger.info(
            f"Starting Poisson disc sampling for {self.width}x{self.length}m "
            f"area with {self.tree_distance}m spacing"
        )

        best_trees = []
        best_count = 0

        for iteration in range(self.max_iterations):
            logger.debug(f"Poisson iteration {iteration + 1}/{self.max_iterations}")

            current_trees = self._poisson_disc_sampling()
            current_count = len(current_trees)

            # Keep the best result
            if current_count > best_count:
                best_count = current_count
                best_trees = current_trees.copy()

                if callback:
                    callback(iteration + 1, current_count, self.max_iterations, True)
            elif callback:
                callback(iteration + 1, current_count, self.max_iterations, False)

        self.trees = best_trees
        logger.info(f"Poisson optimization completed: {len(self.trees)} trees placed")
        return self.trees

    def _poisson_disc_sampling(self) -> list[tuple[float, float]]:
        """Poisson disc sampling algorithm for natural-looking point distribution.

        Based on Bridson's algorithm for fast Poisson disc sampling.
        """
        min_distance = self.tree_distance
        max_distance = min_distance * (1 + self.randomness)

        # Grid for fast neighbor lookup
        cell_size = min_distance / np.sqrt(2)
        grid: dict[tuple[int, int], tuple[float, float]] = {}

        # Active list and points
        active_list = []
        points = []

        # Start with a random seed point
        first_point = (random.uniform(0, self.width), random.uniform(0, self.length))
        points.append(first_point)
        active_list.append(first_point)

        # Add to grid
        grid_x = int(first_point[0] / cell_size)
        grid_y = int(first_point[1] / cell_size)
        grid[(grid_x, grid_y)] = first_point

        attempts_per_point = 30

        while active_list:
            # Pick a random point from active list
            random_index = random.randint(0, len(active_list) - 1)
            point = active_list[random_index]

            found_valid = False

            for _ in range(attempts_per_point):
                # Generate random point in annulus around active point
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(min_distance, max_distance)

                new_x = point[0] + distance * np.cos(angle)
                new_y = point[1] + distance * np.sin(angle)

                # Check if point is within bounds
                if 0 <= new_x <= self.width and 0 <= new_y <= self.length:
                    new_point = (new_x, new_y)

                    # Check if point is far enough from existing points
                    if self._is_valid_poisson_point(
                        new_point, grid, cell_size, min_distance
                    ):
                        points.append(new_point)
                        active_list.append(new_point)

                        # Add to grid
                        grid_x = int(new_x / cell_size)
                        grid_y = int(new_y / cell_size)
                        grid[(grid_x, grid_y)] = new_point

                        found_valid = True

            # If no valid point found, remove from active list
            if not found_valid:
                active_list.pop(random_index)

        return points

    def _is_valid_poisson_point(
        self,
        point: tuple[float, float],
        grid: dict[tuple[int, int], tuple[float, float]],
        cell_size: float,
        min_distance: float,
    ) -> bool:
        """Check if a point is valid for Poisson disc sampling."""
        x, y = point
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)

        # Check surrounding grid cells
        for i in range(
            max(0, grid_x - 2), min(int(np.ceil(self.width / cell_size)), grid_x + 3)
        ):
            for j in range(
                max(0, grid_y - 2),
                min(int(np.ceil(self.length / cell_size)), grid_y + 3),
            ):
                if (i, j) in grid:
                    existing_point = grid[(i, j)]
                    distance = np.sqrt(
                        (x - existing_point[0]) ** 2 + (y - existing_point[1]) ** 2
                    )
                    if distance < min_distance:
                        return False

        return True
