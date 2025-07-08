"""Base classes and interfaces for tree position generators."""

from abc import ABC, abstractmethod
from collections.abc import Callable
import logging

import numpy as np

logger = logging.getLogger(__name__)


class TreePositionGenerator(ABC):
    """Abstract base class for tree position generators."""

    def __init__(
        self,
        width: float,
        length: float,
        tree_distance: float,
        randomness: float = 0.3,
        config: dict | None = None,
    ):
        """Initialize the tree position generator.

        Args:
            width: Width of the planting area in meters
            length: Length of the planting area in meters
            tree_distance: Approximate distance between trees in meters
            randomness: Randomness factor (0-1)
            config: Optional configuration dictionary
        """
        self.width = width
        self.length = length
        self.tree_distance = tree_distance
        self.randomness = randomness
        self.config = config or {}
        self.trees: list[tuple[float, float]] = []

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Width and length must be positive")
        if self.tree_distance <= 0:
            raise ValueError("Tree distance must be positive")
        if self.tree_distance > min(self.width, self.length):
            raise ValueError(
                "Tree distance cannot be larger than the smallest dimension"
            )
        if not 0 <= self.randomness <= 1:
            raise ValueError("Randomness must be between 0 and 1")

    @abstractmethod
    def generate_positions(
        self, callback: Callable | None = None
    ) -> list[tuple[float, float]]:
        """Generate tree positions.

        Args:
            callback: Optional callback function for progress updates

        Returns:
            List of (x, y) tuples representing tree positions
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the generation method."""

    def is_valid_position(
        self,
        new_tree: tuple[float, float],
        existing_trees: list[tuple[float, float]] | None = None,
    ) -> bool:
        """Check if a new tree position maintains minimum distance from existing trees.

        Args:
            new_tree: (x, y) position of the new tree
            existing_trees: List of existing tree positions (defaults to self.trees)

        Returns:
            True if position is valid, False otherwise
        """
        if existing_trees is None:
            existing_trees = self.trees

        new_x, new_y = new_tree

        # Check bounds
        if not (0 <= new_x <= self.width and 0 <= new_y <= self.length):
            return False

        # Check distance from existing trees
        for existing_x, existing_y in existing_trees:
            distance = np.sqrt((new_x - existing_x) ** 2 + (new_y - existing_y) ** 2)
            if distance < self.tree_distance:
                return False

        return True

    def get_statistics(self) -> dict:
        """Get statistics about the generated tree positions."""
        area = self.width * self.length
        return {
            "total_trees": len(self.trees),
            "area": area,
            "density": len(self.trees) / area if area > 0 else 0,
            "method": self.method_name,
            "width": self.width,
            "length": self.length,
            "tree_distance": self.tree_distance,
            "randomness": self.randomness,
        }


class OptimizationTracker:
    """Track optimization progress globally."""

    def __init__(self):
        self.current_generator = None
        self.progress_log = []

    def reset(self):
        """Reset tracking state."""
        self.progress_log = []

    def set_generator(self, generator):
        """Set current generator and reset progress."""
        self.current_generator = generator
        self.reset()

    def add_progress(
        self, iteration: int, count: int, max_iterations: int, is_best: bool
    ):
        """Add progress data to the log."""
        progress_data = {
            "iteration": iteration,
            "tree_count": count,
            "max_iterations": max_iterations,
            "is_best": is_best,
            "progress_percent": (iteration / max_iterations) * 100,
        }
        self.progress_log.append(progress_data)

        if is_best:
            logger.info(f"New best result: {count} trees at iteration {iteration}")


# Global tracker instance
tracker = OptimizationTracker()
