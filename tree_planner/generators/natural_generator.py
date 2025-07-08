"""Natural forest pattern-based tree position generator."""

from collections.abc import Callable
import logging
import random

import numpy as np

from ..base import TreePositionGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class NaturalForestGenerator(TreePositionGenerator):
    """Generate tree positions using near-natural forest patterns."""

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
            "max_natural_iterations", Config.MAX_NATURAL_ITERATIONS
        )

    @property
    def method_name(self) -> str:
        return "natural"

    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using natural forest dynamics."""
        logger.info(
            f"Starting natural forest pattern generation for "
            f"{self.width}x{self.length}m area with {self.tree_distance}m spacing"
        )

        best_trees = []
        best_count = 0

        for iteration in range(self.max_iterations):
            logger.debug(
                f"Natural pattern iteration {iteration + 1}/{self.max_iterations}"
            )

            trees = []

            # Step 1: Create random gaps (natural forest clearings)
            gaps = self._create_forest_gaps()

            # Step 2: Generate cluster centers (seed trees or regeneration cores)
            cluster_centers = self._generate_cluster_centers(gaps)

            # Step 3: Populate clusters with variable density
            for center in cluster_centers:
                self._populate_cluster(center, gaps, trees)

            # Step 4: Add scattered individual trees in remaining space
            self._add_scattered_trees(gaps, trees)

            current_count = len(trees)

            # Keep the best result
            if current_count > best_count:
                best_count = current_count
                best_trees = trees.copy()

                if callback:
                    callback(iteration + 1, current_count, self.max_iterations, True)
            elif callback:
                callback(iteration + 1, current_count, self.max_iterations, False)

        self.trees = best_trees
        logger.info(
            f"Natural pattern optimization completed: {len(self.trees)} trees placed"
        )
        return self.trees

    def _create_forest_gaps(self) -> list[tuple[float, float, float]]:
        """Create random gaps in the forest (areas with no trees)."""
        gaps = []

        # Number of gaps based on area and randomness
        area = self.width * self.length
        gap_count = int(area / (self.tree_distance**2 * 20) * (1 + self.randomness))
        gap_count = max(1, min(gap_count, 8))  # Between 1-8 gaps

        for _ in range(gap_count):
            # Gap center
            gap_x = random.uniform(self.tree_distance, self.width - self.tree_distance)
            gap_y = random.uniform(self.tree_distance, self.length - self.tree_distance)

            # Gap radius (varies with randomness)
            min_radius = self.tree_distance * 1.5
            max_radius = self.tree_distance * 3 * (1 + self.randomness)
            gap_radius = random.uniform(min_radius, max_radius)

            gaps.append((gap_x, gap_y, gap_radius))

        return gaps

    def _generate_cluster_centers(
        self, gaps: list[tuple[float, float, float]]
    ) -> list[tuple[float, float]]:
        """Generate cluster centers avoiding gaps."""
        centers = []

        # Number of clusters based on area
        area = self.width * self.length
        cluster_count = int(area / (self.tree_distance**2 * 10))
        cluster_count = max(2, min(cluster_count, 12))  # Between 2-12 clusters

        attempts = 0
        max_attempts = cluster_count * 10

        while len(centers) < cluster_count and attempts < max_attempts:
            attempts += 1

            x = random.uniform(self.tree_distance, self.width - self.tree_distance)
            y = random.uniform(self.tree_distance, self.length - self.tree_distance)

            # Check if point is in a gap
            in_gap = False
            for gap_x, gap_y, gap_radius in gaps:
                distance = np.sqrt((x - gap_x) ** 2 + (y - gap_y) ** 2)
                if distance < gap_radius:
                    in_gap = True
                    break

            if not in_gap:
                centers.append((x, y))

        return centers

    def _populate_cluster(
        self,
        center: tuple[float, float],
        gaps: list[tuple[float, float, float]],
        trees: list[tuple[float, float]],
    ) -> None:
        """Populate a cluster around a center point with trees."""
        center_x, center_y = center

        # Cluster size varies with randomness
        base_radius = self.tree_distance * 2
        cluster_radius = base_radius * (1 + self.randomness * 2)

        # Number of trees in cluster
        cluster_density = int(8 + random.randint(0, int(12 * self.randomness)))

        for _ in range(cluster_density):
            # Generate position within cluster using normal distribution
            angle = random.uniform(0, 2 * np.pi)
            distance = np.random.normal(cluster_radius * 0.4, cluster_radius * 0.3)
            distance = max(0, min(distance, cluster_radius))

            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)

            # Ensure within bounds
            x = max(0.5, min(self.width - 0.5, x))
            y = max(0.5, min(self.length - 0.5, y))

            # Check if in gap
            in_gap = False
            for gap_x, gap_y, gap_radius in gaps:
                gap_distance = np.sqrt((x - gap_x) ** 2 + (y - gap_y) ** 2)
                if gap_distance < gap_radius:
                    in_gap = True
                    break

            if not in_gap and self.is_valid_position((x, y), trees):
                trees.append((x, y))

    def _add_scattered_trees(
        self, gaps: list[tuple[float, float, float]], trees: list[tuple[float, float]]
    ) -> None:
        """Add scattered individual trees in remaining space."""
        scatter_attempts = int(self.width * self.length / (self.tree_distance**2) * 0.3)

        for _ in range(scatter_attempts):
            x = random.uniform(0.5, self.width - 0.5)
            y = random.uniform(0.5, self.length - 0.5)

            # Check if in gap
            in_gap = False
            for gap_x, gap_y, gap_radius in gaps:
                gap_distance = np.sqrt((x - gap_x) ** 2 + (y - gap_y) ** 2)
                if gap_distance < gap_radius:
                    in_gap = True
                    break

            # Add some variability to minimum distance for scattered trees
            min_distance = self.tree_distance * (0.8 + 0.4 * random.random())

            if not in_gap and self._is_valid_position_variable(
                (x, y), min_distance, trees
            ):
                trees.append((x, y))

    def _is_valid_position_variable(
        self,
        new_tree: tuple[float, float],
        min_distance: float,
        existing_trees: list[tuple[float, float]],
    ) -> bool:
        """Check if position is valid with variable minimum distance."""
        new_x, new_y = new_tree

        for existing_x, existing_y in existing_trees:
            distance = np.sqrt((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2)
            if distance < min_distance:
                return False

        return True
