#!/bin/bash

# TreePlanter Flask App Startup Script for Devcontainer
echo "🌲 Starting TreePlanter Flask Application..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if not already installed
echo "📋 Checking dependencies..."
pip install -r requirements.txt --quiet

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

echo "🚀 Starting Flask development server..."
echo "📍 Application will be available at:"
echo "   - Local: http://localhost:5000"
echo "   - Container: http://0.0.0.0:5000"
echo ""
echo "🔧 To run manually: flask run --host=0.0.0.0 --port=5000"
echo "🐛 To debug: Use VS Code's 'Python: Flask App' launch configuration"
echo ""

# Start the Flask application
flask run --host=0.0.0.0 --port=5000
