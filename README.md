# Tree Planner Package

A Flask application for generating tree planting plans using various algorithms including Perlin noise, Poisson disc sampling, natural forest patterns, and uniform angle index methods.

## Features

- **Multiple Generation Methods**:

  - Perlin Noise: Organic, natural-looking patterns using coherent noise algorithms
  - Poisson Disc Sampling: Evenly distributed points with controlled minimum spacing
  - Natural Forest Pattern: Simulates natural forest dynamics with gaps and clusters
  - Uniform Angle Index: Scientific method based on Zhang et al. (2019) research

- **Web Interface**: Interactive web UI for parameter configuration and visualization
- **Real-time Progress**: Live optimization progress updates
- **Export Functionality**: Download coordinates as JSON files
- **Comprehensive Testing**: Full unit test coverage

## Project Structure

```
tree_planner/
├── tree_planner/           # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── base.py            # Base classes and interfaces
│   ├── core.py            # Main TreePlanner class
│   ├── app.py             # Flask application
│   └── generators/        # Generator implementations
│       ├── __init__.py
│       ├── perlin_generator.py
│       ├── poisson_generator.py
│       ├── natural_generator.py
│       └── uniform_angle_generator.py
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_core.py
│   ├── test_app.py
│   └── generators/
├── templates/             # HTML templates
│   └── index.html
├── static/               # Static files
├── main.py              # Application entry point
├── run_tests.py         # Test runner
└── requirements.txt     # Dependencies
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python main.py
```

3. Open http://localhost:5000 in your browser

## Testing

Run the test suite:

```bash
python run_tests.py
```

Or use pytest:

```bash
pytest tests/
```

## Usage

### Programmatic API

```python
from tree_planner import TreePlanner

# Create a planner
planner = TreePlanner(
    width=20.0,           # Width in meters
    length=15.0,          # Length in meters
    tree_distance=3.0,    # Approximate spacing
    randomness=0.3,       # Randomness factor (0-1)
    method="perlin"       # Generation method
)

# Generate positions
trees = planner.generate_tree_positions()

# Get coordinates
coordinates = planner.get_tree_coordinates_json()

# Generate visualization
image_buffer = planner.generate_planting_image()
```

### Web Interface

1. Configure area dimensions (width, length)
2. Set tree spacing distance
3. Adjust randomness factor
4. Select generation method
5. Click "Generate Planting Plan"
6. View results and download coordinates

## Configuration

The application supports different configuration environments:

- `development`: Debug mode enabled, full iterations
- `testing`: Reduced iterations for faster testing
- `production`: Optimized for production deployment

Set the environment with the `FLASK_ENV` environment variable.

## Methods

### Perlin Noise Method

Uses coherent noise algorithms to create organic, natural-looking patterns with smooth variations in tree placement.

### Poisson Disc Sampling

Generates evenly distributed points with controlled minimum spacing, preventing clustering while maintaining natural randomness.

### Natural Forest Pattern

Simulates natural forest dynamics including gap creation, clustered regeneration, and variable spacing based on forest ecology principles.

### Uniform Angle Index Method

Scientific method based on Zhang et al. (2019) research using structural units to ensure ≥50% random units (Wi=0.5) for near-natural patterns.

## API Endpoints

- `GET /`: Main application interface
- `POST /generate_plan`: Generate tree planting plan
- `GET /optimization_progress`: Get current optimization progress
- `GET /generate_plan_stream`: Real-time streaming updates (SSE)
- `POST /download_coordinates`: Download coordinates as JSON
- `GET /test`: Health check endpoint

## License

This project is open source and available under the MIT License.
