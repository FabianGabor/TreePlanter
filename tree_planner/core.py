"""Core tree planner class that coordinates different generators."""

from collections.abc import Callable
import io
import logging
from typing import Any

import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from perlin_noise import PerlinNoise

from .base import tracker
from .config import Config
from .generators import (
    NaturalForestGenerator,
    PerlinNoiseGenerator,
    PoissonDiscGenerator,
    UniformAngleGenerator,
)

matplotlib.use("Agg")  # Use non-interactive backend
logger = logging.getLogger(__name__)


class TreePlanner:
    """Main tree planner class that coordinates different generation methods."""

    GENERATORS = {
        "perlin": PerlinNoiseGenerator,
        "poisson": PoissonDiscGenerator,
        "natural": NaturalForestGenerator,
        "uniform_angle": UniformAngleGenerator,
    }

    def __init__(
        self,
        width: float,
        length: float,
        tree_distance: float,
        randomness: float = 0.3,
        method: str = "perlin",
        config: dict | Config | None = None,
    ):
        """Initialize the tree planner.

        Args:
            width: Width of the planting area in meters
            length: Length of the planting area in meters
            tree_distance: Approximate distance between trees in meters
            randomness: Randomness factor (0-1)
            method: Generation method ('perlin', 'poisson', 'natural', 'uniform_angle')
            config: Optional configuration dictionary
        """
        self.width = width
        self.length = length
        self.tree_distance = tree_distance
        self.randomness = randomness
        self.method = method

        # Handle config - convert Config instance to dictionary if needed
        if config is None:
            self.config = Config.get_generator_config()
        elif hasattr(config, "get_generator_config"):
            # It's a Config class instance
            self.config = config.get_generator_config()  # type: ignore
        else:
            # It's already a dictionary
            self.config = config

        # Validate method
        if method not in self.GENERATORS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(self.GENERATORS.keys())}"
            )

        # Create generator instance
        generator_class = self.GENERATORS[method]
        self.generator = generator_class(
            width, length, tree_distance, randomness, self.config
        )

        self.trees: list[tuple[float, float]] = []

    def generate_tree_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions using the selected method."""
        logger.info(f"Starting tree placement using {self.method} method")

        # Set up global tracker
        tracker.set_generator(self.generator)

        def progress_callback(
            iteration: int, count: int, max_iterations: int, is_best: bool
        ):
            tracker.add_progress(iteration, count, max_iterations, is_best)
            if callback:
                callback(iteration, count, max_iterations, is_best)

        # Generate positions
        self.trees = self.generator.generate_positions(callback=progress_callback)

        return self.trees

    def generate_planting_image(self) -> io.BytesIO:
        """Generate a visual planting plan with method-specific background."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Set up the plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_aspect("equal")

        # Generate background visualization based on method
        self._add_method_background(ax)

        # Draw the boundary
        boundary = patches.Rectangle(
            (0, 0),
            self.width,
            self.length,
            linewidth=2,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(boundary)

        # Plot trees with spacing circles
        if self.trees:
            x_coords, y_coords = zip(*self.trees, strict=False)

            # Plot trees
            ax.plot(
                x_coords,
                y_coords,
                "o",
                color="darkgreen",
                markersize=8,
                label=f"Trees ({len(self.trees)} total)",
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=1,
            )

            # Add circles around each tree showing the spacing diameter
            for x, y in self.trees:
                circle = patches.Circle(
                    (x, y),
                    self.tree_distance / 2,
                    linewidth=1,
                    edgecolor="lightgreen",
                    facecolor="none",
                    alpha=0.6,
                )
                ax.add_patch(circle)

        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Width (meters)")
        ax.set_ylabel("Length (meters)")

        # Set title based on method
        title, subtitle = self._get_plot_title()
        ax.set_title(title + subtitle)
        ax.legend()

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close()

        return img_buffer

    def _add_method_background(self, ax) -> None:
        """Add method-specific background visualization."""
        if self.method == "perlin":
            self._add_perlin_background(ax)
        elif self.method == "poisson":
            self._add_poisson_background(ax)
        elif self.method == "natural":
            self._add_natural_background(ax)
        elif self.method == "uniform_angle":
            self._add_uniform_angle_background(ax)

    def _add_perlin_background(self, ax) -> None:
        """Add Perlin noise background visualization."""
        grid_spacing = self.tree_distance * 0.8
        scale = 0.15
        resolution = 20

        x_vis = np.linspace(0, self.width, int(self.width * resolution))
        y_vis = np.linspace(0, self.length, int(self.length * resolution))
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        noise_vis = PerlinNoise(octaves=5, seed=1)
        noise_field = np.zeros_like(x_vis_grid)

        for i in range(x_vis_grid.shape[0]):
            for j in range(x_vis_grid.shape[1]):
                col_f = x_vis_grid[i, j] / grid_spacing
                row_f = y_vis_grid[i, j] / grid_spacing
                noise_field[i, j] = noise_vis([col_f * scale, row_f * scale])

        im = ax.contourf(
            x_vis_grid, y_vis_grid, noise_field, levels=15, cmap="RdYlBu_r", alpha=0.3
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Noise Displacement\n(Red = Push Away, Blue = Pull Toward)",
            rotation=270,
            labelpad=20,
        )

    def _add_poisson_background(self, ax) -> None:
        """Add Poisson disc sampling background visualization."""
        resolution = 40
        x_vis = np.linspace(0, self.width, resolution)
        y_vis = np.linspace(0, self.length, resolution)
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        distance_field = np.zeros_like(x_vis_grid)
        center_x, center_y = self.width / 2, self.length / 2

        for i in range(x_vis_grid.shape[0]):
            for j in range(x_vis_grid.shape[1]):
                x, y = x_vis_grid[i, j], y_vis_grid[i, j]
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                distance_field[i, j] = (
                    np.sin(dist_from_center / self.tree_distance * 2 * np.pi) * 0.5
                )

        im = ax.contourf(
            x_vis_grid, y_vis_grid, distance_field, levels=15, cmap="Blues", alpha=0.2
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Spacing Pattern\n(Poisson Disc Distribution)", rotation=270, labelpad=20
        )

    def _add_natural_background(self, ax) -> None:
        """Add natural forest background visualization."""
        resolution = 30
        x_vis = np.linspace(0, self.width, resolution)
        y_vis = np.linspace(0, self.length, resolution)
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        forest_field = np.zeros_like(x_vis_grid)

        # Simulate gaps (simplified version)
        num_gaps = max(1, int(self.width * self.length / 100))
        for _ in range(num_gaps):
            gap_x = np.random.uniform(0, self.width)
            gap_y = np.random.uniform(0, self.length)
            gap_radius = self.tree_distance * 2

            for i in range(x_vis_grid.shape[0]):
                for j in range(x_vis_grid.shape[1]):
                    x, y = x_vis_grid[i, j], y_vis_grid[i, j]
                    dist = np.sqrt((x - gap_x) ** 2 + (y - gap_y) ** 2)
                    if dist < gap_radius:
                        forest_field[i, j] = max(0, forest_field[i, j] - 0.5)
                    else:
                        forest_field[i, j] = min(1, forest_field[i, j] + 0.3)

        im = ax.contourf(
            x_vis_grid, y_vis_grid, forest_field, levels=15, cmap="YlGn", alpha=0.3
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Forest Structure\n(Dark = Clusters, Light = Gaps)",
            rotation=270,
            labelpad=20,
        )

    def _add_uniform_angle_background(self, ax) -> None:
        """Add uniform angle index background visualization."""
        resolution = 30
        x_vis = np.linspace(0, self.width, resolution)
        y_vis = np.linspace(0, self.length, resolution)
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        unit_field = np.zeros_like(x_vis_grid)
        unit_size = self.tree_distance

        for i in range(x_vis_grid.shape[0]):
            for j in range(x_vis_grid.shape[1]):
                x, y = x_vis_grid[i, j], y_vis_grid[i, j]

                grid_x = (x / unit_size) % 3
                grid_y = (y / unit_size) % 3

                if grid_x < 1 and grid_y < 1:
                    unit_field[i, j] = 0.8  # Random units
                elif grid_x < 1 or grid_y < 1:
                    unit_field[i, j] = 0.4  # Regular patterns
                else:
                    unit_field[i, j] = 0.1  # Background

        im = ax.contourf(
            x_vis_grid, y_vis_grid, unit_field, levels=10, cmap="viridis", alpha=0.3
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(
            "Structural Units\n(Bright = Random Wi=0.5, Dark = Regular)",
            rotation=270,
            labelpad=20,
        )

    def _get_plot_title(self) -> tuple[str, str]:
        """Get plot title and subtitle based on method."""
        method_titles = {
            "perlin": (
                "Tree Planting Plan - Perlin Noise Method\n",
                "(Colors show displacement forces)",
            ),
            "poisson": (
                "Tree Planting Plan - Poisson Disc Sampling\n",
                "(Natural distribution pattern)",
            ),
            "natural": (
                "Tree Planting Plan - Near-Natural Forest Pattern\n",
                "(Gap-cluster forest dynamics)",
            ),
            "uniform_angle": (
                "Tree Planting Plan - Uniform Angle Index Method\n",
                "Based on Zhang et al. (2019) - Structural units with "
                "≥50% random (Wi=0.5)",
            ),
        }

        title, method_desc = method_titles[self.method]
        subtitle = (
            f"{self.width}m x {self.length}m, "
            f"~{self.tree_distance}m spacing, {len(self.trees)} trees\n"
            f"Randomness: {self.randomness:.1f} {method_desc}"
        )

        return title, subtitle

    def get_tree_coordinates_json(self) -> dict[str, Any]:
        """Return tree coordinates as JSON-serializable dictionary."""
        return {
            "area": {"width": self.width, "length": self.length},
            "spacing": self.tree_distance,
            "total_trees": len(self.trees),
            "method": self.method,
            "coordinates": [
                {"x": round(x, 2), "y": round(y, 2), "id": i + 1}
                for i, (x, y) in enumerate(self.trees)
            ],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the generated plan."""
        base_stats = self.generator.get_statistics()

        # Add method-specific stats if available
        if hasattr(self.generator, "get_generation_stats"):
            base_stats.update(self.generator.get_generation_stats())

        return base_stats
