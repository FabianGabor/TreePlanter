from flask import Flask, render_template, request, send_file, jsonify
import os
import numpy as np
import random
import time
from perlin_noise import PerlinNoise
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import json

app = Flask(__name__)

# Ensure static directory exists
os.makedirs('static/images', exist_ok=True)

class TreePlanner:
    def __init__(self, width, length, tree_distance, randomness=0.3):
        self.width = width  # in meters
        self.length = length  # in meters
        self.tree_distance = tree_distance  # approximate distance between trees
        self.randomness = randomness  # how much randomness to apply (0-1)
        self.trees = []
        
    def generate_tree_positions(self):
        """Generate tree positions using Perlin noise for natural randomness while maintaining minimum distance"""
        self.trees = []
        
        # Create Perlin noise generators with random seeds for each generation
        seed_x = random.randint(1, 10000)
        seed_y = random.randint(1, 10000)
        noise_x = PerlinNoise(octaves=6, seed=seed_x)
        noise_y = PerlinNoise(octaves=6, seed=seed_y)
        
        # Calculate approximate number of trees that should fit
        area = self.width * self.length
        tree_area = self.tree_distance * self.tree_distance
        max_attempts = int(area / tree_area * 3)  # Try 3x the theoretical maximum
        
        # Use sampling approach with Perlin noise
        attempts = 0
        scale = 0.02  # Noise scale for natural distribution
        
        while len(self.trees) < max_attempts and attempts < max_attempts * 2:
            attempts += 1
            
            # Generate noise-based position
            noise_val_x = noise_x([attempts * scale])
            noise_val_y = noise_y([attempts * scale * 1.3])  # Different scale for Y to avoid correlation
            
            # Map noise values (-1 to 1) to area dimensions
            x = (noise_val_x + 1) / 2 * (self.width - 1) + 0.5
            y = (noise_val_y + 1) / 2 * (self.length - 1) + 0.5
            
            # Apply randomness factor - higher randomness = more scattered
            if self.randomness > 0:
                # Add additional random displacement
                random_x = (random.random() - 0.5) * self.tree_distance * self.randomness
                random_y = (random.random() - 0.5) * self.tree_distance * self.randomness
                x += random_x
                y += random_y
            
            # Ensure position is within bounds
            x = max(0.5, min(self.width - 0.5, x))
            y = max(0.5, min(self.length - 0.5, y))
            
            # Check minimum distance constraint
            new_tree = (x, y)
            if self._is_valid_position(new_tree):
                self.trees.append(new_tree)
                
            # Early exit if we have enough trees
            if len(self.trees) >= area / tree_area * 0.8:  # Stop at 80% of theoretical maximum
                break
        
        return self.trees
    
    def _is_valid_position(self, new_tree):
        """Check if a new tree position maintains minimum distance from existing trees"""
        new_x, new_y = new_tree
        
        for existing_x, existing_y in self.trees:
            distance = np.sqrt((new_x - existing_x)**2 + (new_y - existing_y)**2)
            if distance < self.tree_distance:
                return False
        return True
    
    def generate_planting_image(self):
        """Generate a visual planting plan"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Set up the plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        
        # Draw the boundary
        boundary = patches.Rectangle((0, 0), self.width, self.length, 
                                   linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(boundary)
        
        # Plot trees with spacing circles
        if self.trees:
            x_coords, y_coords = zip(*self.trees)
            
            # Plot trees
            ax.plot(x_coords, y_coords, 'o', color='darkgreen', markersize=8,
                   label=f'Trees ({len(self.trees)} total)', alpha=0.8)
            
            # Add circles around each tree showing the spacing diameter
            for x, y in self.trees:
                circle = patches.Circle((x, y), self.tree_distance, 
                                      linewidth=1, edgecolor='lightblue', 
                                      facecolor='none', alpha=0.4)
                ax.add_patch(circle)
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Width (meters)')
        ax.set_ylabel('Length (meters)')
        ax.set_title(f'Tree Planting Plan - {self.width}m × {self.length}m\n'
                    f'~{self.tree_distance}m spacing, {len(self.trees)} trees')
        ax.legend()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def get_tree_coordinates_json(self):
        """Return tree coordinates as JSON"""
        return {
            'area': {
                'width': self.width,
                'length': self.length
            },
            'spacing': self.tree_distance,
            'total_trees': len(self.trees),
            'coordinates': [
                {'x': round(x, 2), 'y': round(y, 2), 'id': i+1} 
                for i, (x, y) in enumerate(self.trees)
            ]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        # Get form data
        width = float(request.form['width'])
        length = float(request.form['length'])
        tree_distance = float(request.form['tree_distance'])
        randomness = float(request.form.get('randomness', 0.3))
        
        # Validate inputs
        if width <= 0 or length <= 0 or tree_distance <= 0:
            return jsonify({'error': 'All dimensions must be positive numbers'}), 400
        
        if tree_distance > min(width, length):
            return jsonify({'error': 'Tree distance cannot be larger than the smallest dimension'}), 400
        
        # Generate tree plan
        planner = TreePlanner(width, length, tree_distance, randomness)
        planner.generate_tree_positions()
        
        # Generate image
        img_buffer = planner.generate_planting_image()
        
        # Convert image to base64 for web display
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Get coordinates data
        coordinates_data = planner.get_tree_coordinates_json()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'coordinates': coordinates_data
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid input values. Please enter valid numbers.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/download_coordinates', methods=['POST'])
def download_coordinates():
    try:
        # Get the coordinates data from the request
        data = request.get_json()
        
        if not data or 'area' not in data:
            return jsonify({'error': 'Invalid data provided'}), 400
        
        # Create a temporary file with coordinates
        filename = f"tree_coordinates_{data['area']['width']}x{data['area']['length']}.json"
        filepath = os.path.join('static', filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
