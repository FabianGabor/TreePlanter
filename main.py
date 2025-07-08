#!/usr/bin/env python3
"""Main application entry point for the tree planner."""

import os

from tree_planner import create_app

if __name__ == "__main__":
    # Get configuration from environment
    config_name = os.environ.get("FLASK_ENV", "development")

    # Create app
    app = create_app(config_name)

    # Run app
    app.run(debug=app.config["DEBUG"], host="0.0.0.0", port=5000)
