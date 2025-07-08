"""Perlin noise-based tree position generator."""

from collections.abc import Callable
import logging
import random

from perlin_noise import PerlinNoise

from ..base import TreePositionGenerator
from ..config import Config

logger = logging.getLogger(__name__)


class PerlinNoiseGenerator(TreePositionGenerator):
    """Generate tree positions using Perlin noise patterns."""

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
            "max_perlin_iterations", Config.MAX_PERLIN_ITERATIONS
        )
        self.randomness_threshold = self.config.get(
            "randomness_threshold", Config.RANDOMNESS_THRESHOLD
        )

    @property
    def method_name(self) -> str:
        return "perlin"

    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using Perlin noise optimization."""
        logger.info(
            f"Starting Perlin noise optimization for {self.width}x{self.length}m "
            f"area with {self.tree_distance}m spacing"
        )

        best_trees = []
        best_count = 0

        for iteration in range(self.max_iterations):
            logger.debug(f"Perlin iteration {iteration + 1}/{self.max_iterations}")

            current_trees = self._generate_single_iteration()
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
        logger.info(f"Perlin optimization completed: {len(self.trees)} trees placed")
        return self.trees

    def _generate_single_iteration(self) -> list[tuple[float, float]]:
        """Generate tree positions for a single iteration."""
        trees = []

        # Use a denser initial grid to ensure good coverage
        grid_spacing = self.tree_distance * 0.5
        cols = int(self.width / grid_spacing) + 1
        rows = int(self.length / grid_spacing) + 1

        # Create Perlin noise generators with random seeds
        seed_x = random.randint(1, 10000)
        seed_y = random.randint(1, 10000)
        noise_x = PerlinNoise(octaves=4, seed=seed_x)
        noise_y = PerlinNoise(octaves=4, seed=seed_y)

        scale = 0.15
        potential_positions = []

        # Generate potential positions using noise
        for row in range(rows):
            for col in range(cols):
                base_x = col * grid_spacing + grid_spacing / 2
                base_y = row * grid_spacing + grid_spacing / 2

                if base_x >= self.width or base_y >= self.length:
                    continue

                # Apply Perlin noise displacement
                x_noise_val = noise_x([col * scale, row * scale])
                y_noise_val = noise_y([col * scale + 100, row * scale + 100])

                max_displacement = self.tree_distance * self.randomness * 0.6
                displaced_x = base_x + x_noise_val * max_displacement
                displaced_y = base_y + y_noise_val * max_displacement

                # Add pure randomness for natural look
                if self.randomness > self.randomness_threshold:
                    displaced_x += (random.random() - 0.5) * self.tree_distance * 0.3
                    displaced_y += (random.random() - 0.5) * self.tree_distance * 0.3

                # Keep within bounds
                displaced_x = max(0.5, min(self.width - 0.5, displaced_x))
                displaced_y = max(0.5, min(self.length - 0.5, displaced_y))

                potential_positions.append((displaced_x, displaced_y))

        # Randomize order to avoid systematic bias
        random.shuffle(potential_positions)

        # Place trees while maintaining minimum distance
        for pos in potential_positions:
            if self.is_valid_position(pos, trees):
                trees.append(pos)

        # Try to fill remaining gaps
        self._fill_gaps(trees, noise_x, noise_y, grid_spacing, scale)

        return trees

    def _fill_gaps(
        self,
        trees: list[tuple[float, float]],
        noise_x: PerlinNoise,
        noise_y: PerlinNoise,
        grid_spacing: float,
        scale: float,
    ) -> None:
        """Fill remaining gaps with additional trees."""
        gap_fill_attempts = 30

        for _ in range(gap_fill_attempts):
            x = random.uniform(0.5, self.width - 0.5)
            y = random.uniform(0.5, self.length - 0.5)

            # Add slight noise influence to gap-filling positions
            grid_x = x / grid_spacing
            grid_y = y / grid_spacing
            x_noise = noise_x([grid_x * scale, grid_y * scale])
            y_noise = noise_y([grid_x * scale + 100, grid_y * scale + 100])

            gap_displacement = self.tree_distance * 0.2
            x += x_noise * gap_displacement
            y += y_noise * gap_displacement

            # Keep within bounds
            x = max(0.5, min(self.width - 0.5, x))
            y = max(0.5, min(self.length - 0.5, y))

            if self.is_valid_position((x, y), trees):
                trees.append((x, y))
