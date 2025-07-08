"""Configuration module for the tree planner application."""

import os
from typing import Any


class Config:
    """Base configuration class."""

    # Flask configuration
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Tree planner configuration
    MAX_PERLIN_ITERATIONS = 50
    MAX_POISSON_ITERATIONS = 50
    MAX_NATURAL_ITERATIONS = 50
    MAX_UNIFORM_ANGLE_ITERATIONS = 100

    RANDOMNESS_THRESHOLD = 0.3

    # Image generation settings
    IMAGE_DPI = 150
    IMAGE_FORMAT = "png"

    # Static directory
    STATIC_DIR = "static"
    IMAGES_DIR = "static/images"

    @classmethod
    def get_generator_config(cls) -> dict[str, Any]:
        """Get configuration for tree generators."""
        return {
            "max_perlin_iterations": cls.MAX_PERLIN_ITERATIONS,
            "max_poisson_iterations": cls.MAX_POISSON_ITERATIONS,
            "max_natural_iterations": cls.MAX_NATURAL_ITERATIONS,
            "max_uniform_angle_iterations": cls.MAX_UNIFORM_ANGLE_ITERATIONS,
            "randomness_threshold": cls.RANDOMNESS_THRESHOLD,
        }


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DEBUG = False

    # Reduced iterations for faster testing
    MAX_PERLIN_ITERATIONS = 5
    MAX_POISSON_ITERATIONS = 5
    MAX_NATURAL_ITERATIONS = 5
    MAX_UNIFORM_ANGLE_ITERATIONS = 10


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    TESTING = False


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
