"""Core tree planner class that coordinates different generators."""

from collections.abc import Callable
import io
import logging
from typing import Any

import matplotlib
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
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

# Visualization constants
PERLIN_GRID_SPACING_FACTOR = 0.8
PERLIN_NOISE_SCALE = 0.15
PERLIN_NOISE_OCTAVES = 5
PERLIN_NOISE_SEED = 1
PERLIN_RESOLUTION = 20
PERLIN_CONTOUR_LEVELS = 15
PERLIN_ALPHA = 0.3

POISSON_RESOLUTION = 40
POISSON_ALPHA = 0.2
POISSON_CONTOUR_LEVELS = 15
POISSON_DISTANCE_FACTOR = 2.0

NATURAL_RESOLUTION = 30
NATURAL_ALPHA = 0.3
NATURAL_CONTOUR_LEVELS = 15
NATURAL_GAP_AREA_FACTOR = 100
NATURAL_GAP_RADIUS_FACTOR = 2.0
NATURAL_GAP_REDUCTION = 0.5
NATURAL_CLUSTER_BOOST = 0.3

UNIFORM_ANGLE_RESOLUTION = 50
UNIFORM_ANGLE_ALPHA = 0.6
UNIFORM_ANGLE_CONTOUR_LEVELS = 11
UNIFORM_ANGLE_GRID_FACTOR = 0.1
UNIFORM_ANGLE_GRID_SPACING = 10

# Wi classification thresholds (based on Hui and Gadow 2002)
WI_REGULAR_THRESHOLD = 0.475
WI_RANDOM_UPPER_THRESHOLD = 0.525
WI_STANDARD_ANGLE_DEGREES = 72.0
WI_NEIGHBOR_COUNT = 4

# Interpolation constants
INTERPOLATION_NEAREST_TREES = 3
INTERPOLATION_DISTANCE_THRESHOLD = 0.1
INTERPOLATION_WEIGHT_OFFSET = 0.1

# Plot styling constants
BOUNDARY_LINE_WIDTH = 2
TREE_MARKER_SIZE = 8
TREE_MARKER_ALPHA = 0.9
TREE_MARKER_EDGE_WIDTH = 1
TREE_CIRCLE_LINE_WIDTH = 2
TREE_CIRCLE_ALPHA = 0.5
GRID_ALPHA = 0.3
PLOT_DPI = 150
COLORBAR_SHRINK = 0.8
COLORBAR_PAD = 0.02
COLORBAR_UNIFORM_ANGLE_PAD = 0.05
COLORBAR_LABEL_PAD = 20
COLORBAR_UNIFORM_ANGLE_LABEL_PAD = 45

# Text annotation constants
TEXT_FONT_SIZE = 7
TEXT_FONT_SIZE_FALLBACK = 8
TEXT_BOX_PAD = 0.3
TEXT_BOX_ALPHA = 0.95
TEXT_BOX_FALLBACK_ALPHA = 0.8
TEXT_POSITION_X = 0.02
TEXT_POSITION_Y = 0.98

# Minimum trees required for Wi heatmap
MIN_TREES_FOR_WI_HEATMAP = 5

# Magic values that need to be extracted
POISSON_SINE_MULTIPLIER = 0.5
GRID_LINE_WIDTH = 0.3
GRID_LINE_ALPHA = 0.5
BBOX_ROUND_PAD = 0.3

# Color definitions
WI_COLORMAP_COLORS = [
    "#0066cc",  # Strong blue for very regular (Wi = 0)
    "#3399ff",  # Medium blue for regular (Wi = 0.25)
    "#66ccff",  # Light blue for approaching random (Wi < 0.475)
    "#00cc00",  # Green for random (Wi ≈ 0.5)
    "#66ff66",  # Light green for random variation
    "#ff6600",  # Orange for irregular (Wi = 0.75)
    "#cc0000",  # Red for very irregular/clumped (Wi = 1.0)
]
WI_COLORMAP_N = 256


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
            linewidth=BOUNDARY_LINE_WIDTH,
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
                markersize=TREE_MARKER_SIZE,
                label=f"Trees ({len(self.trees)} total)",
                alpha=TREE_MARKER_ALPHA,
                markeredgecolor="white",
                markeredgewidth=TREE_MARKER_EDGE_WIDTH,
            )

            # Add circles around each tree showing the spacing diameter
            for x, y in self.trees:
                circle = patches.Circle(
                    (x, y),
                    self.tree_distance / 2,
                    linewidth=TREE_CIRCLE_LINE_WIDTH,
                    edgecolor="darkgreen",
                    facecolor="none",
                    alpha=TREE_CIRCLE_ALPHA,
                )
                ax.add_patch(circle)

        # Add grid and labels
        ax.grid(True, alpha=GRID_ALPHA)
        ax.set_xlabel("Width (meters)")
        ax.set_ylabel("Length (meters)")

        # Set title based on method
        title, subtitle = self._get_plot_title()
        ax.set_title(title + subtitle)
        ax.legend()

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format="png", dpi=PLOT_DPI, bbox_inches="tight")
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
        grid_spacing = self.tree_distance * PERLIN_GRID_SPACING_FACTOR
        scale = PERLIN_NOISE_SCALE
        resolution = PERLIN_RESOLUTION

        x_vis = np.linspace(0, self.width, int(self.width * resolution))
        y_vis = np.linspace(0, self.length, int(self.length * resolution))
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        noise_vis = PerlinNoise(octaves=PERLIN_NOISE_OCTAVES, seed=PERLIN_NOISE_SEED)
        noise_field = np.zeros_like(x_vis_grid)

        for i in range(x_vis_grid.shape[0]):
            for j in range(x_vis_grid.shape[1]):
                col_f = x_vis_grid[i, j] / grid_spacing
                row_f = y_vis_grid[i, j] / grid_spacing
                noise_field[i, j] = noise_vis([col_f * scale, row_f * scale])

        im = ax.contourf(
            x_vis_grid,
            y_vis_grid,
            noise_field,
            levels=PERLIN_CONTOUR_LEVELS,
            cmap="RdYlBu_r",
            alpha=PERLIN_ALPHA,
        )

        cbar = plt.colorbar(im, ax=ax, shrink=COLORBAR_SHRINK, pad=COLORBAR_PAD)
        cbar.set_label(
            "Noise Displacement\n(Red = Push Away, Blue = Pull Toward)",
            rotation=270,
            labelpad=COLORBAR_LABEL_PAD,
        )

    def _add_poisson_background(self, ax) -> None:
        """Add Poisson disc sampling background visualization."""
        resolution = POISSON_RESOLUTION
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
                    np.sin(
                        dist_from_center
                        / self.tree_distance
                        * POISSON_DISTANCE_FACTOR
                        * np.pi
                    )
                    * POISSON_SINE_MULTIPLIER
                )

        im = ax.contourf(
            x_vis_grid,
            y_vis_grid,
            distance_field,
            levels=POISSON_CONTOUR_LEVELS,
            cmap="Blues",
            alpha=POISSON_ALPHA,
        )

        cbar = plt.colorbar(im, ax=ax, shrink=COLORBAR_SHRINK, pad=COLORBAR_PAD)
        cbar.set_label(
            "Spacing Pattern\n(Poisson Disc Distribution)",
            rotation=270,
            labelpad=COLORBAR_LABEL_PAD,
        )

    def _add_natural_background(self, ax) -> None:
        """Add natural forest background visualization."""
        resolution = NATURAL_RESOLUTION
        x_vis = np.linspace(0, self.width, resolution)
        y_vis = np.linspace(0, self.length, resolution)
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        forest_field = np.zeros_like(x_vis_grid)

        # Simulate gaps (simplified version)
        num_gaps = max(1, int(self.width * self.length / NATURAL_GAP_AREA_FACTOR))
        for _ in range(num_gaps):
            gap_x = np.random.uniform(0, self.width)
            gap_y = np.random.uniform(0, self.length)
            gap_radius = self.tree_distance * NATURAL_GAP_RADIUS_FACTOR

            for i in range(x_vis_grid.shape[0]):
                for j in range(x_vis_grid.shape[1]):
                    x, y = x_vis_grid[i, j], y_vis_grid[i, j]
                    dist = np.sqrt((x - gap_x) ** 2 + (y - gap_y) ** 2)
                    if dist < gap_radius:
                        forest_field[i, j] = max(
                            0, forest_field[i, j] - NATURAL_GAP_REDUCTION
                        )
                    else:
                        forest_field[i, j] = min(
                            1, forest_field[i, j] + NATURAL_CLUSTER_BOOST
                        )

        im = ax.contourf(
            x_vis_grid,
            y_vis_grid,
            forest_field,
            levels=NATURAL_CONTOUR_LEVELS,
            cmap="YlGn",
            alpha=NATURAL_ALPHA,
        )

        cbar = plt.colorbar(im, ax=ax, shrink=COLORBAR_SHRINK, pad=COLORBAR_PAD)
        cbar.set_label(
            "Forest Structure\n(Dark = Clusters, Light = Gaps)",
            rotation=270,
            labelpad=COLORBAR_LABEL_PAD,
        )

    def _add_uniform_angle_background(self, ax) -> None:
        """Add uniform angle index background visualization showing Wi values."""
        if not self.trees or len(self.trees) < MIN_TREES_FOR_WI_HEATMAP:
            # Fallback to simple grid if no trees or too few trees
            self._add_simple_uniform_angle_background(ax)
            return

        # Calculate Wi values for all trees
        wi_values = self._calculate_wi_values_for_visualization()

        if not wi_values:
            self._add_simple_uniform_angle_background(ax)
            return

        # Create interpolated heatmap of Wi values
        resolution = UNIFORM_ANGLE_RESOLUTION
        x_vis = np.linspace(0, self.width, resolution)
        y_vis = np.linspace(0, self.length, resolution)
        x_vis_grid, y_vis_grid = np.meshgrid(x_vis, y_vis)

        # Interpolate Wi values across the area
        wi_field = np.zeros_like(x_vis_grid)

        for i in range(x_vis_grid.shape[0]):
            for j in range(x_vis_grid.shape[1]):
                x_point, y_point = x_vis_grid[i, j], y_vis_grid[i, j]

                # Find the nearest trees and interpolate Wi values
                distances = []
                for k, (tree_x, tree_y) in enumerate(self.trees):
                    dist = np.sqrt((x_point - tree_x) ** 2 + (y_point - tree_y) ** 2)
                    distances.append((dist, wi_values[k]))

                # Use inverse distance weighting for interpolation
                if distances:
                    distances.sort()
                    # Use the nearest trees for interpolation
                    nearest = distances[
                        : min(INTERPOLATION_NEAREST_TREES, len(distances))
                    ]

                    total_weight = 0
                    weighted_wi = 0

                    for dist, wi in nearest:
                        # Very close to a tree
                        if dist < INTERPOLATION_DISTANCE_THRESHOLD:
                            wi_field[i, j] = wi
                            break
                        # Avoid division by zero
                        weight = 1.0 / (dist + INTERPOLATION_WEIGHT_OFFSET)
                        weighted_wi += wi * weight
                        total_weight += weight
                    else:
                        if total_weight > 0:
                            wi_field[i, j] = weighted_wi / total_weight

        # Create custom colormap for Wi values with distinct regions
        # Blue for regular (Wi < 0.475), Green for random (Wi 0.475-0.525),
        # Red for clumped (Wi > 0.525)
        # Based on Hui and Gadow (2002) classification mentioned in the paper
        cmap = LinearSegmentedColormap.from_list(
            "wi_cmap", WI_COLORMAP_COLORS, N=WI_COLORMAP_N
        )

        # Plot the Wi heatmap
        im = ax.contourf(
            x_vis_grid,
            y_vis_grid,
            wi_field,
            levels=np.linspace(0, 1, UNIFORM_ANGLE_CONTOUR_LEVELS),
            cmap=cmap,
            alpha=UNIFORM_ANGLE_ALPHA,
            extend="both",
        )

        # Add colorbar with Wi interpretation using the three main categories
        cbar = plt.colorbar(
            im, ax=ax, shrink=COLORBAR_SHRINK, pad=COLORBAR_UNIFORM_ANGLE_PAD
        )
        cbar.set_label(
            f"Uniform Angle Index (Wi)\n"
            f"Blue: Regular (Wi < {WI_REGULAR_THRESHOLD})\n"
            f"Green: Random ({WI_REGULAR_THRESHOLD} ≤ Wi ≤ "
            f"{WI_RANDOM_UPPER_THRESHOLD})\n"
            f"Red: Clumped (Wi > {WI_RANDOM_UPPER_THRESHOLD})",
            rotation=270,
            labelpad=COLORBAR_UNIFORM_ANGLE_LABEL_PAD,
        )

        # Add ticks and labels for Wi values with scientific classification
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([
            "0\n(Very Even)",
            "0.25\n(Even)",
            "0.5\n(Random)",
            "0.75\n(Irregular)",
            "1.0\n(Clumped)",
        ])

        # Add method annotation with optimization details
        stats = {}
        if hasattr(self, "generator") and hasattr(
            self.generator, "_last_generation_stats"
        ):
            stats = self.generator._last_generation_stats

        objective_value = stats.get("objective_value", 0)
        constraint_1 = stats.get("constraint_1_satisfied", "Unknown")
        constraint_2 = stats.get("constraint_2_satisfied", "Unknown")
        best_iteration = stats.get("iteration", "Unknown")

        ax.text(
            TEXT_POSITION_X,
            TEXT_POSITION_Y,
            f"Uniform Angle Index Optimization (Zhang et al. 2019)\n"
            f"Objective: max N_{{Wi=0.5}} = {objective_value}\n"
            f"α₀ = {WI_STANDARD_ANGLE_DEGREES}°, "
            f"n = {WI_NEIGHBOR_COUNT} neighbors, 1m buffer\n"
            f"Constraints: (1) ≥50% random: {constraint_1}, "
            f"(2) ≥50% density: {constraint_2}\n"
            f"Best solution: iteration {best_iteration}",
            transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE,
            verticalalignment="top",
            bbox={
                "boxstyle": f"round,pad={BBOX_ROUND_PAD}",
                "facecolor": "white",
                "alpha": TEXT_BOX_ALPHA,
            },
        )

    def _add_simple_uniform_angle_background(self, ax) -> None:
        """Fallback background for uniform angle method when insufficient trees."""
        # Create a subtle grid background
        unit_size = self.tree_distance * UNIFORM_ANGLE_GRID_FACTOR
        x_grid_lines = np.arange(
            0, self.width + unit_size, unit_size * UNIFORM_ANGLE_GRID_SPACING
        )
        y_grid_lines = np.arange(
            0, self.length + unit_size, unit_size * UNIFORM_ANGLE_GRID_SPACING
        )

        for x in x_grid_lines:
            if x <= self.width:
                ax.axvline(
                    x=x,
                    color="lightgray",
                    linewidth=GRID_LINE_WIDTH,
                    alpha=GRID_LINE_ALPHA,
                )

        for y in y_grid_lines:
            if y <= self.length:
                ax.axhline(
                    y=y,
                    color="lightgray",
                    linewidth=GRID_LINE_WIDTH,
                    alpha=GRID_LINE_ALPHA,
                )

        ax.text(
            TEXT_POSITION_X,
            TEXT_POSITION_Y,
            "Uniform Angle Index Method\n(Insufficient trees for Wi heatmap)",
            transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE_FALLBACK,
            verticalalignment="top",
            bbox={
                "boxstyle": f"round,pad={BBOX_ROUND_PAD}",
                "facecolor": "white",
                "alpha": TEXT_BOX_FALLBACK_ALPHA,
            },
        )

    def _calculate_wi_values_for_visualization(self) -> list[float]:
        """Calculate Wi values for all trees for visualization purposes."""
        if len(self.trees) < MIN_TREES_FOR_WI_HEATMAP:
            return []

        wi_values = []
        alpha_0 = WI_STANDARD_ANGLE_DEGREES  # Standard threshold angle in degrees
        n_neighbors = WI_NEIGHBOR_COUNT  # Number of nearest neighbors

        for i, (x_i, y_i) in enumerate(self.trees):
            # Find nearest neighbors
            distances = []
            for j, (x_j, y_j) in enumerate(self.trees):
                if i != j:
                    dist = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
                    distances.append((dist, (x_j, y_j)))

            # Sort by distance and take the n nearest
            distances.sort()
            neighbors = [pos for _, pos in distances[:n_neighbors]]

            if len(neighbors) < n_neighbors:
                wi_values.append(0.0)
                continue

            # Calculate angles from tree i to each neighbor
            angles = []
            for x_j, y_j in neighbors:
                angle = np.arctan2(y_j - y_i, x_j - x_i)
                if angle < 0:
                    angle += 2 * np.pi  # Normalize to [0, 2π]
                angles.append(angle)

            # Sort angles
            angles.sort()

            # Calculate consecutive angle differences and apply Wi formula
            z_sum = 0
            for k in range(len(angles)):
                next_k = (k + 1) % len(angles)
                alpha_kj = angles[next_k] - angles[k]
                if alpha_kj < 0:
                    alpha_kj += 2 * np.pi

                # Convert to degrees
                alpha_kj_deg = np.degrees(alpha_kj)

                # Apply formula: z_kj = 1 if α_kj < α_0, 0 otherwise
                if alpha_kj_deg < alpha_0:
                    z_sum += 1

            # Calculate Wi = (1/n) * Σ z_kj
            wi = z_sum / len(angles)
            wi_values.append(wi)

        return wi_values

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
