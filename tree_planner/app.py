"""Flask application module for the tree planner."""

import base64
import json
import logging
import os
import time

from flask import Flask, jsonify, render_template, request, send_file

from .base import tracker
from .config import config
from .core import TreePlanner

logger = logging.getLogger(__name__)

# HTTP Status Codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500


def create_app(config_name: str = "default") -> Flask:
    """Application factory function.

    Args:
        config_name: Configuration name ('development', 'testing', 'production')

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if app.config["DEBUG"] else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Ensure static directory exists
    os.makedirs(app.config.get("IMAGES_DIR", "static/images"), exist_ok=True)

    # Register routes
    register_routes(app)

    return app


def register_routes(app: Flask) -> None:
    """Register all application routes."""

    @app.route("/")
    def index():
        """Serve the main application page."""
        return render_template("index.html")

    @app.route("/test")
    def test():
        """Simple test endpoint."""
        return jsonify({"status": "Flask app is working!", "timestamp": time.time()})

    @app.route("/generate_plan", methods=["POST"])
    def generate_plan():
        """Generate a tree planting plan."""
        try:
            # Get form data
            width = float(request.form["width"])
            length = float(request.form["length"])
            tree_distance = float(request.form["tree_distance"])
            randomness = float(request.form.get("randomness", 0.3))
            method = request.form.get("method", "perlin")

            # Validate inputs
            if width <= 0 or length <= 0 or tree_distance <= 0:
                return jsonify({
                    "error": "All dimensions must be positive numbers"
                }), HTTP_BAD_REQUEST

            if tree_distance > min(width, length):
                return jsonify({
                    "error": "Tree distance cannot be larger than the smallest "
                    "dimension"
                }), HTTP_BAD_REQUEST

            # Generate tree plan with optimization
            planner = TreePlanner(width, length, tree_distance, randomness, method)

            # Store planner globally for progress updates
            tracker.set_generator(planner.generator)

            def progress_callback(iteration, count, max_iterations, is_best):
                tracker.add_progress(iteration, count, max_iterations, is_best)

            planner.generate_tree_positions(callback=progress_callback)

            # Generate final image
            img_buffer = planner.generate_planting_image()

            # Convert image to base64 for web display
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Get coordinates data
            coordinates_data = planner.get_tree_coordinates_json()

            return jsonify({
                "success": True,
                "method": method,
                "image": img_base64,
                "coordinates": coordinates_data,
                "optimization_log": tracker.progress_log,
            })

        except ValueError as e:
            logger.warning(f"Invalid input values: {e}")
            return jsonify({
                "error": "Invalid input values. Please enter valid numbers."
            }), HTTP_BAD_REQUEST

        except Exception as e:
            logger.error(f"Error in generate_plan: {e}", exc_info=True)
            return jsonify({
                "error": f"An error occurred: {e!s}"
            }), HTTP_INTERNAL_SERVER_ERROR

    @app.route("/optimization_progress")
    def get_optimization_progress():
        """Get current optimization progress."""
        return jsonify(tracker.progress_log)

    @app.route("/generate_plan_stream", methods=["GET", "POST"])
    def generate_plan_stream():
        """Generate plan with real-time streaming updates."""
        try:
            # Get parameters from either form data (POST) or URL parameters (GET)
            if request.method == "POST":
                width = float(request.form["width"])
                length = float(request.form["length"])
                tree_distance = float(request.form["tree_distance"])
                randomness = float(request.form.get("randomness", 0.3))
                method = request.form.get("method", "perlin")
            else:  # GET request for SSE
                width = float(request.args.get("width", "0"))
                length = float(request.args.get("length", "0"))
                tree_distance = float(request.args.get("tree_distance", "0"))
                randomness = float(request.args.get("randomness", "0.3"))
                method = request.args.get("method", "perlin")

            # Validate inputs
            error_msg = None
            if width <= 0 or length <= 0 or tree_distance <= 0:
                error_msg = "All dimensions must be positive numbers"
            elif tree_distance > min(width, length):
                error_msg = "Tree distance cannot be larger than the smallest dimension"

            if error_msg:
                if request.method == "POST":
                    return jsonify({"error": error_msg}), HTTP_BAD_REQUEST
                return f"data: {json.dumps({'error': error_msg})}\\n\\n"

            def generate():
                planner = TreePlanner(width, length, tree_distance, randomness, method)
                best_image = None
                best_coordinates = None

                def progress_callback(iteration, count, max_iterations, is_best):
                    nonlocal best_image, best_coordinates

                    # Generate image for current state if it's the best so far
                    if is_best:
                        try:
                            img_buffer = planner.generate_planting_image()
                            img_base64 = base64.b64encode(
                                img_buffer.getvalue()
                            ).decode()
                            coordinates_data = planner.get_tree_coordinates_json()

                            best_image = img_base64
                            best_coordinates = coordinates_data

                            # Send update to client
                            update_data = {
                                "iteration": iteration,
                                "tree_count": count,
                                "max_iterations": max_iterations,
                                "is_best": is_best,
                                "progress_percent": (iteration / max_iterations) * 100,
                                "image": img_base64,
                                "coordinates": coordinates_data,
                            }
                            yield f"data: {json.dumps(update_data)}\\n\\n"
                        except Exception as e:
                            logger.error(
                                f"Error generating image during optimization: {e}"
                            )
                    else:
                        # Send progress update without image
                        update_data = {
                            "iteration": iteration,
                            "tree_count": count,
                            "max_iterations": max_iterations,
                            "is_best": is_best,
                            "progress_percent": (iteration / max_iterations) * 100,
                        }
                        yield f"data: {json.dumps(update_data)}\\n\\n"

                # Start optimization
                planner.generate_tree_positions(callback=progress_callback)

                # Send final result
                final_data = {
                    "completed": True,
                    "final_tree_count": len(planner.trees),
                    "image": best_image,
                    "coordinates": best_coordinates,
                }
                yield f"data: {json.dumps(final_data)}\\n\\n"

            return app.response_class(generate(), mimetype="text/event-stream")

        except Exception as e:
            logger.error(f"Error in generate_plan_stream: {e}", exc_info=True)
            error_data = {"error": f"An error occurred: {e!s}"}
            if request.method == "POST":
                return jsonify(error_data), HTTP_INTERNAL_SERVER_ERROR
            return f"data: {json.dumps(error_data)}\\n\\n"

    @app.route("/download_coordinates", methods=["POST"])
    def download_coordinates():
        """Download tree coordinates as JSON file."""
        try:
            # Get the coordinates data from the request
            data = request.get_json()

            if not data or "area" not in data:
                return jsonify({"error": "Invalid data provided"}), HTTP_BAD_REQUEST

            # Create a temporary file with coordinates
            filename = (
                f"tree_coordinates_{data['area']['width']}x"
                f"{data['area']['length']}.json"
            )
            filepath = os.path.join(app.config.get("STATIC_DIR", "static"), filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return send_file(filepath, as_attachment=True, download_name=filename)

        except Exception as e:
            logger.error(f"Download failed: {e}", exc_info=True)
            return jsonify({
                "error": f"Download failed: {e!s}"
            }), HTTP_INTERNAL_SERVER_ERROR

    @app.errorhandler(HTTP_NOT_FOUND)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({"error": "Resource not found"}), HTTP_NOT_FOUND

    @app.errorhandler(HTTP_INTERNAL_SERVER_ERROR)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), HTTP_INTERNAL_SERVER_ERROR
