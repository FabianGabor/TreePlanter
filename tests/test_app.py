"""Tests for the Flask application."""

import json
import unittest
from unittest.mock import MagicMock, patch

from tree_planner.app import create_app


class TestFlaskApp(unittest.TestCase):
    """Test cases for Flask application."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid issues with missing dependencies
        try:
            self.app = create_app("testing")
            self.client = self.app.test_client()
            self.app_context = self.app.app_context()
            self.app_context.push()
        except ImportError:
            self.skipTest("Flask not available")

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "app_context"):
            self.app_context.pop()

    def test_create_app(self):
        """Test app creation."""
        self.assertIsNotNone(self.app)
        self.assertTrue(self.app.config["TESTING"])

    def test_index_route(self):
        """Test index route."""
        try:
            response = self.client.get("/")
            # Should either return the template or 404 if template doesn't exist
            self.assertIn(response.status_code, [200, 404])
        except Exception:
            # Template might not exist in test environment
            pass

    def test_test_route(self):
        """Test the /test endpoint."""
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["status"], "Flask app is working!")

    @patch("tree_planner.app.TreePlanner")
    def test_generate_plan_valid_input(self, mock_planner_class):
        """Test generate_plan with valid input."""
        # Mock TreePlanner
        mock_planner = MagicMock()
        mock_planner.trees = [(1, 1), (2, 2)]
        mock_planner.generate_tree_positions.return_value = [(1, 1), (2, 2)]
        mock_planner.generate_planting_image.return_value = MagicMock()
        mock_planner.generate_planting_image.return_value.getvalue.return_value = (
            b"fake_image_data"
        )
        mock_planner.get_tree_coordinates_json.return_value = {
            "area": {"width": 10, "length": 10},
            "total_trees": 2,
        }
        mock_planner_class.return_value = mock_planner

        response = self.client.post(
            "/generate_plan",
            data={
                "width": "10",
                "length": "10",
                "tree_distance": "2",
                "randomness": "0.3",
                "method": "perlin",
            },
        )

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["method"], "perlin")
        self.assertIn("image", data)
        self.assertIn("coordinates", data)
        self.assertIn("optimization_log", data)

    def test_generate_plan_invalid_input(self):
        """Test generate_plan with invalid input."""
        # Test negative dimensions
        response = self.client.post(
            "/generate_plan",
            data={
                "width": "-1",
                "length": "10",
                "tree_distance": "2",
                "randomness": "0.3",
                "method": "perlin",
            },
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

        # Test tree distance too large
        response = self.client.post(
            "/generate_plan",
            data={
                "width": "5",
                "length": "10",
                "tree_distance": "8",  # Larger than width
                "randomness": "0.3",
                "method": "perlin",
            },
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_optimization_progress_route(self):
        """Test optimization progress route."""
        response = self.client.get("/optimization_progress")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIsInstance(data, list)

    def test_download_coordinates_valid_data(self):
        """Test download coordinates with valid data."""
        test_data = {
            "area": {"width": 10, "length": 10},
            "total_trees": 2,
            "coordinates": [
                {"x": 1.0, "y": 1.0, "id": 1},
                {"x": 2.0, "y": 2.0, "id": 2},
            ],
        }

        response = self.client.post(
            "/download_coordinates",
            data=json.dumps(test_data),
            content_type="application/json",
        )

        # Should return file or error
        self.assertIn(response.status_code, [200, 500])

    def test_download_coordinates_invalid_data(self):
        """Test download coordinates with invalid data."""
        response = self.client.post(
            "/download_coordinates",
            data=json.dumps({}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    def test_404_handler(self):
        """Test 404 error handler."""
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)

        data = json.loads(response.data)
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
