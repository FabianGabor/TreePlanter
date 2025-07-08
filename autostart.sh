#!/bin/bash

# Auto-start script for TreePlanter Flask app
echo "🌲 Starting TreePlanter Flask app..."

# Wait for dependencies to be ready
sleep 2

# Start Flask app in background
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

nohup flask run --host=0.0.0.0 --port=5000 > /tmp/flask.log 2>&1 &
FLASK_PID=$!

echo "Flask app started with PID: $FLASK_PID"
echo "Access your app at: http://localhost:5000"
echo "View logs with: tail -f /tmp/flask.log"
echo "Stop the app with: kill $FLASK_PID"

# Store PID for easy stopping
echo $FLASK_PID > /tmp/flask.pid
