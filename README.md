# Tree Planter Tool

A Python web application for generating tree planting plans with intelligent spacing using Perlin noise for natural randomness.

## Features

- **Web Interface**: Clean, modern interface for easy input
- **Intelligent Spacing**: Uses Perlin noise to create natural-looking tree placement
- **Visual Output**: Generates planting plan images with coordinates
- **Customizable Parameters**: 
  - Area dimensions (width × length)
  - Tree spacing distance
  - Randomness factor (from regular grid to natural placement)
- **Export Functionality**: Download tree coordinates as JSON file
- **Statistics**: Shows total trees, area size, and tree density

## Installation

1. Clone or download this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and go to `http://localhost:5000`

3. Enter your parameters:
   - **Width & Length**: Dimensions of your planting area in meters
   - **Tree Distance**: Approximate spacing between trees in meters
   - **Randomness**: Slider from regular grid (0) to natural placement (0.8)

4. Click "Generate Planting Plan" to create your layout

5. View the results:
   - Visual planting plan image
   - Statistics (total trees, area, density)
   - Tree coordinates list
   - Download coordinates as JSON file

## Technical Details

### Algorithm
- Uses Perlin noise to generate natural-looking randomness
- Starts with a regular grid based on specified spacing
- Applies noise-based displacement within bounds
- Ensures trees stay within the defined area

### Output Format
The coordinates JSON file contains:
```json
{
  "area": {
    "width": 50.0,
    "length": 30.0
  },
  "spacing": 5.0,
  "total_trees": 48,
  "coordinates": [
    {"x": 2.34, "y": 2.67, "id": 1},
    {"x": 7.12, "y": 2.89, "id": 2},
    ...
  ]
}
```

### Dependencies
- **Flask**: Web framework
- **NumPy**: Numerical computations
- **noise**: Perlin noise generation
- **Matplotlib**: Plotting and image generation
- **Pillow**: Image processing

## Customization

You can modify the Perlin noise parameters in `app.py`:
- `scale`: Controls noise frequency
- `octaves`: Number of noise layers
- `persistence`: How much each octave contributes
- `lacunarity`: Frequency multiplier between octaves

## Example Use Cases

- **Forestry**: Planning reforestation areas
- **Landscaping**: Designing natural-looking tree layouts
- **Agriculture**: Agroforestry planning
- **Urban Planning**: Green space design
- **Research**: Ecological studies requiring controlled tree placement

## License

This project is open source and available under the MIT License.
