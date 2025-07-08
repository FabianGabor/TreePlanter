"""Tree Planner Package.

A Flask application for generating tree planting plans using various algorithms.
"""

__version__ = "1.0.0"
__author__ = "Tree Planner Team"

# Import order matters for avoiding circular imports
from . import base, config, generators
from .app import create_app
from .core import TreePlanner

__all__ = ["TreePlanner", "base", "config", "create_app", "generators"]
