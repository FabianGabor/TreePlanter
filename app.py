from flask import Flask, render_template, request, send_file, jsonify
import os
import numpy as np
import random
import time
import logging
from perlin_noise import PerlinNoise
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for optimization tracking
current_planner = None
progress_log = []

# Ensure static directory exists
os.makedirs('static/images', exist_ok=True)

class TreePlanner:
    def __init__(self, width, length, tree_distance, randomness=0.3, method='perlin'):
        self.width = width  # in meters
        self.length = length  # in meters
        self.tree_distance = tree_distance  # approximate distance between trees
        self.randomness = randomness  # how much randomness to apply (0-1)
        self.method = method  # placement method: 'perlin' or 'poisson'
        self.trees = []
        
    def generate_tree_positions(self, callback=None):
        """Generate tree positions using multiple iterations to maximize tree count"""
        logging.info(f"Starting tree placement optimization using {self.method} method for {self.width}x{self.length}m area with {self.tree_distance}m spacing")
        print(f"🌲 Starting {self.method} optimization for {self.width}x{self.length}m area with {self.tree_distance}m spacing")
        
        if self.method == 'perlin':
            return self._generate_perlin_positions(callback)
        elif self.method == 'poisson':
            return self._generate_poisson_positions(callback)
        else:
            raise ValueError(f"Unknown placement method: {self.method}")
    
    def _generate_perlin_positions(self, callback=None):
        """Generate tree positions using Perlin noise (original method)"""
        best_trees = []
        best_count = 0
        max_iterations = 1000  # Try multiple random patterns
        
        for iteration in range(max_iterations):
            logging.info(f"Perlin iteration {iteration + 1}/{max_iterations} - Testing new random pattern")
            print(f"🔄 Perlin Iteration {iteration + 1}/{max_iterations} - Testing new pattern...")
            self.trees = []
            
            # Use a denser initial grid to ensure good coverage
            grid_spacing = self.tree_distance * 0.5 # Closer initial grid
            cols = int(self.width / grid_spacing) + 1
            rows = int(self.length / grid_spacing) + 1
            
            # Create Perlin noise generators with random seeds for each iteration
            seed_x = random.randint(1, 10000)
            seed_y = random.randint(1, 10000)
            noise_x = PerlinNoise(octaves=4, seed=seed_x)
            noise_y = PerlinNoise(octaves=4, seed=seed_y)
            
            # Define scale for consistent use throughout this iteration
            scale = 0.15
            
            # Collect all potential positions first
            potential_positions = []
            
            for row in range(rows):
                for col in range(cols):
                    # Base grid position
                    base_x = col * grid_spacing + grid_spacing / 2
                    base_y = row * grid_spacing + grid_spacing / 2
                    
                    # Skip if outside bounds
                    if base_x >= self.width or base_y >= self.length:
                        continue
                    
                    # Add Perlin noise displacement
                    x_noise_val = noise_x([col * scale, row * scale])
                    y_noise_val = noise_y([col * scale + 100, row * scale + 100])  # Offset for different pattern
                    
                    # Apply displacement with randomness factor
                    max_displacement = self.tree_distance * self.randomness * 0.6
                    displaced_x = base_x + x_noise_val * max_displacement
                    displaced_y = base_y + y_noise_val * max_displacement
                    
                    # Add some pure randomness for more natural look
                    if self.randomness > 0.3:
                        displaced_x += (random.random() - 0.5) * self.tree_distance * 0.3
                        displaced_y += (random.random() - 0.5) * self.tree_distance * 0.3
                    
                    # Ensure trees stay within bounds
                    displaced_x = max(0.5, min(self.width - 0.5, displaced_x))
                    displaced_y = max(0.5, min(self.length - 0.5, displaced_y))
                    
                    potential_positions.append((displaced_x, displaced_y))
            
            # Sort positions randomly to avoid systematic bias
            random.shuffle(potential_positions)
            
            # Place trees while maintaining minimum distance
            for pos in potential_positions:
                if self._is_valid_position(pos):
                    self.trees.append(pos)
            
            # Try to fill remaining gaps with additional attempts
            gap_fill_attempts = 30
            for _ in range(gap_fill_attempts):
                # Generate random position with some noise influence
                x = random.uniform(0.5, self.width - 0.5)
                y = random.uniform(0.5, self.length - 0.5)
                
                # Add slight noise influence to gap-filling positions
                grid_x = x / grid_spacing
                grid_y = y / grid_spacing
                x_noise = noise_x([grid_x * scale, grid_y * scale])
                y_noise = noise_y([grid_x * scale + 100, grid_y * scale + 100])
                
                # Small displacement for gap filling
                gap_displacement = self.tree_distance * 0.2
                x += x_noise * gap_displacement
                y += y_noise * gap_displacement
                
                # Keep within bounds
                x = max(0.5, min(self.width - 0.5, x))
                y = max(0.5, min(self.length - 0.5, y))
                
                if self._is_valid_position((x, y)):
                    self.trees.append((x, y))
            
            current_count = len(self.trees)
            logging.info(f"Perlin iteration {iteration + 1} completed: {current_count} trees placed")
            
            # Keep the best result
            if current_count > best_count:
                best_count = current_count
                best_trees = self.trees.copy()
                logging.info(f"New best result found: {best_count} trees (improvement from previous best)")
                
                # Call callback for live updates if provided
                if callback:
                    callback(iteration + 1, current_count, max_iterations, True)
            else:
                if callback:
                    callback(iteration + 1, current_count, max_iterations, False)
        
        # Use the best result found
        self.trees = best_trees
        logging.info(f"Perlin optimization completed: Final result has {len(self.trees)} trees")
        
        return self.trees
    
    def _generate_poisson_positions(self, callback=None):
        """Generate tree positions using Poisson Disc Sampling for natural distribution"""
        best_trees = []
        best_count = 0
        max_iterations = 50  # Fewer iterations for Poisson as each is more expensive
        
        for iteration in range(max_iterations):
            logging.info(f"Poisson iteration {iteration + 1}/{max_iterations} - Generating new distribution")
            print(f"🎯 Poisson Iteration {iteration + 1}/{max_iterations} - Generating distribution...")
            
            # Generate using Poisson disc sampling
            self.trees = self._poisson_disc_sampling()
            
            current_count = len(self.trees)
            logging.info(f"Poisson iteration {iteration + 1} completed: {current_count} trees placed")
            
            # Keep the best result
            if current_count > best_count:
                best_count = current_count
                best_trees = self.trees.copy()
                logging.info(f"New best result found: {best_count} trees (improvement from previous best)")
                
                # Call callback for live updates if provided
                if callback:
                    callback(iteration + 1, current_count, max_iterations, True)
            else:
                if callback:
                    callback(iteration + 1, current_count, max_iterations, False)
        
        # Use the best result found
        self.trees = best_trees
        logging.info(f"Poisson optimization completed: Final result has {len(self.trees)} trees")
        
        return self.trees
    
    def _poisson_disc_sampling(self):
        """
        Poisson Disc Sampling algorithm for natural-looking point distribution
        Based on Bridson's algorithm for fast Poisson disc sampling
        """
        min_distance = self.tree_distance
        max_distance = min_distance * (1 + self.randomness)
        
        # Grid for fast neighbor lookup
        cell_size = min_distance / np.sqrt(2)
        grid = {}
        
        # Active list of points to try extending from
        active_list = []
        points = []
        
        # Start with a random seed point
        first_point = (random.uniform(0, self.width), random.uniform(0, self.length))
        points.append(first_point)
        active_list.append(first_point)
        
        # Add to grid
        grid_x = int(first_point[0] / cell_size)
        grid_y = int(first_point[1] / cell_size)
        grid[(grid_x, grid_y)] = first_point
        
        attempts_per_point = 30  # Number of attempts to place around each active point
        
        while active_list:
            # Pick a random point from active list
            random_index = random.randint(0, len(active_list) - 1)
            point = active_list[random_index]
            
            # Try to place new points around this point
            found_valid = False
            
            for _ in range(attempts_per_point):
                # Generate random point in annulus around active point
                angle = random.uniform(0, 2 * np.pi)
                distance = random.uniform(min_distance, max_distance)
                
                new_x = point[0] + distance * np.cos(angle)
                new_y = point[1] + distance * np.sin(angle)
                
                # Check if point is within bounds
                if 0 <= new_x <= self.width and 0 <= new_y <= self.length:
                    new_point = (new_x, new_y)
                    
                    # Check if point is far enough from existing points
                    if self._is_valid_poisson_point(new_point, grid, cell_size, min_distance):
                        points.append(new_point)
                        active_list.append(new_point)
                        
                        # Add to grid
                        grid_x = int(new_x / cell_size)
                        grid_y = int(new_y / cell_size)
                        grid[(grid_x, grid_y)] = new_point
                        
                        found_valid = True
            
            # If no valid point found, remove from active list
            if not found_valid:
                active_list.pop(random_index)
        
        return points
    
    def _is_valid_poisson_point(self, point, grid, cell_size, min_distance):
        """Check if a point is valid for Poisson disc sampling"""
        x, y = point
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        
        # Check surrounding grid cells
        for i in range(max(0, grid_x - 2), min(int(np.ceil(self.width / cell_size)), grid_x + 3)):
            for j in range(max(0, grid_y - 2), min(int(np.ceil(self.length / cell_size)), grid_y + 3)):
                if (i, j) in grid:
                    existing_point = grid[(i, j)]
                    distance = np.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                    if distance < min_distance:
                        return False
        return True
    
    def _is_valid_position(self, new_tree):
        """Check if a new tree position maintains minimum distance from existing trees"""
        new_x, new_y = new_tree
        
        for existing_x, existing_y in self.trees:
            distance = np.sqrt((new_x - existing_x)**2 + (new_y - existing_y)**2)
            if distance < self.tree_distance:
                return False
        return True
    
    def generate_planting_image(self):
        """Generate a visual planting plan with noise pattern background"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Set up the plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.length)
        ax.set_aspect('equal')
        
        # Generate background visualization based on method
        if self.method == 'perlin':
            # Generate noise field for background visualization
            grid_spacing = self.tree_distance * 0.8
            scale = 0.15
            resolution = 20  # Points per unit
            x_vis = np.linspace(0, self.width, int(self.width * resolution))
            y_vis = np.linspace(0, self.length, int(self.length * resolution))
            X_vis, Y_vis = np.meshgrid(x_vis, y_vis)
            
            # Create noise field using same parameters as tree generation
            noise_vis = PerlinNoise(octaves=5, seed=1)  # Fixed seed for consistent visualization
            noise_field = np.zeros_like(X_vis)
            
            for i in range(X_vis.shape[0]):
                for j in range(X_vis.shape[1]):
                    col_f = X_vis[i, j] / grid_spacing
                    row_f = Y_vis[i, j] / grid_spacing
                    noise_field[i, j] = noise_vis([col_f * scale, row_f * scale])
            
            # Add noise pattern as background
            im = ax.contourf(X_vis, Y_vis, noise_field, levels=15, cmap='RdYlBu_r', alpha=0.3)
            
            # Add colorbar to explain noise values
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Noise Displacement\n(Red = Push Away, Blue = Pull Toward)', rotation=270, labelpad=20)
        
        elif self.method == 'poisson':
            # For Poisson disc sampling, show distance circles pattern
            resolution = 40
            x_vis = np.linspace(0, self.width, resolution)
            y_vis = np.linspace(0, self.length, resolution)
            X_vis, Y_vis = np.meshgrid(x_vis, y_vis)
            
            # Create distance field showing optimal spacing
            distance_field = np.zeros_like(X_vis)
            center_x, center_y = self.width / 2, self.length / 2
            
            for i in range(X_vis.shape[0]):
                for j in range(X_vis.shape[1]):
                    x, y = X_vis[i, j], Y_vis[i, j]
                    # Create a radial pattern centered in the middle
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    distance_field[i, j] = np.sin(dist_from_center / self.tree_distance * 2 * np.pi) * 0.5
            
            # Add distance pattern as background
            im = ax.contourf(X_vis, Y_vis, distance_field, levels=15, cmap='Blues', alpha=0.2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Spacing Pattern\n(Poisson Disc Distribution)', rotation=270, labelpad=20)
        
        # Draw the boundary
        boundary = patches.Rectangle((0, 0), self.width, self.length, 
                                   linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(boundary)
        
        # Plot trees with spacing circles
        if self.trees:
            x_coords, y_coords = zip(*self.trees)
            
            # Plot trees
            ax.plot(x_coords, y_coords, 'o', color='darkgreen', markersize=8,
                   label=f'Trees ({len(self.trees)} total)', alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            
            # Add circles around each tree showing the spacing diameter
            for x, y in self.trees:
                circle_half = patches.Circle((x, y), self.tree_distance/2, 
                                      linewidth=1, edgecolor='lightgreen', 
                                      facecolor='none', alpha=0.6)
                ax.add_patch(circle_half)
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Width (meters)')
        ax.set_ylabel('Length (meters)')
        
        # Set title based on method
        if self.method == 'perlin':
            title = 'Tree Planting Plan - Perlin Noise Method\n'
            subtitle = f'{self.width}m × {self.length}m, ~{self.tree_distance}m spacing, {len(self.trees)} trees\n'
            subtitle += f'Randomness: {self.randomness:.1f} (Colors show displacement forces)'
        else:  # poisson
            title = 'Tree Planting Plan - Poisson Disc Sampling\n'
            subtitle = f'{self.width}m × {self.length}m, ~{self.tree_distance}m spacing, {len(self.trees)} trees\n'
            subtitle += f'Randomness: {self.randomness:.1f} (Natural distribution pattern)'
        
        ax.set_title(title + subtitle)
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

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({'status': 'Flask app is working!', 'timestamp': time.time()})

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    try:
        # Get form data
        width = float(request.form['width'])
        length = float(request.form['length'])
        tree_distance = float(request.form['tree_distance'])
        randomness = float(request.form.get('randomness', 0.3))
        method = request.form.get('method', 'perlin')  # Default to perlin
        
        # Validate inputs
        if width <= 0 or length <= 0 or tree_distance <= 0:
            return jsonify({'error': 'All dimensions must be positive numbers'}), 400
        
        if tree_distance > min(width, length):
            return jsonify({'error': 'Tree distance cannot be larger than the smallest dimension'}), 400
        
        # Generate tree plan with optimization
        planner = TreePlanner(width, length, tree_distance, randomness, method)
        
        # Store planner globally for progress updates
        global current_planner, progress_log
        current_planner = planner
        progress_log = []
        
        def progress_callback(iteration, count, max_iterations, is_best):
            progress_data = {
                'iteration': iteration,
                'tree_count': count,
                'max_iterations': max_iterations,
                'is_best': is_best,
                'progress_percent': (iteration / max_iterations) * 100
            }
            progress_log.append(progress_data)
        
        planner.generate_tree_positions(callback=progress_callback)
        
        # Generate final image
        img_buffer = planner.generate_planting_image()
        
        # Convert image to base64 for web display
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Get coordinates data
        coordinates_data = planner.get_tree_coordinates_json()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'coordinates': coordinates_data,
            'optimization_log': progress_log
        })
        
    except ValueError:
        return jsonify({'error': 'Invalid input values. Please enter valid numbers.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/optimization_progress')
def get_optimization_progress():
    """Get current optimization progress"""
    global progress_log
    return jsonify(progress_log)

@app.route('/generate_plan_stream', methods=['GET', 'POST'])
def generate_plan_stream():
    """Generate plan with real-time streaming updates"""
    try:
        # Get parameters from either form data (POST) or URL parameters (GET)
        if request.method == 'POST':
            width = float(request.form['width'])
            length = float(request.form['length'])
            tree_distance = float(request.form['tree_distance'])
            randomness = float(request.form.get('randomness', 0.3))
            method = request.form.get('method', 'perlin')
        else:  # GET request for SSE
            width = float(request.args.get('width') or '0')
            length = float(request.args.get('length') or '0')
            tree_distance = float(request.args.get('tree_distance') or '0')
            randomness = float(request.args.get('randomness') or '0.3')
            method = request.args.get('method') or 'perlin'
        
        # Validate inputs
        if width <= 0 or length <= 0 or tree_distance <= 0:
            if request.method == 'POST':
                return jsonify({'error': 'All dimensions must be positive numbers'}), 400
            else:
                return f"data: {json.dumps({'error': 'All dimensions must be positive numbers'})}\n\n"
        
        if tree_distance > min(width, length):
            error_msg = 'Tree distance cannot be larger than the smallest dimension'
            if request.method == 'POST':
                return jsonify({'error': error_msg}), 400
            else:
                return f"data: {json.dumps({'error': error_msg})}\n\n"
        
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
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                        coordinates_data = planner.get_tree_coordinates_json()
                        
                        best_image = img_base64
                        best_coordinates = coordinates_data
                        
                        # Send update to client
                        update_data = {
                            'iteration': iteration,
                            'tree_count': count,
                            'max_iterations': max_iterations,
                            'is_best': is_best,
                            'progress_percent': (iteration / max_iterations) * 100,
                            'image': img_base64,
                            'coordinates': coordinates_data
                        }
                        yield f"data: {json.dumps(update_data)}\n\n"
                    except Exception as e:
                        print(f"Error generating image during optimization: {e}")
                else:
                    # Send progress update without image
                    update_data = {
                        'iteration': iteration,
                        'tree_count': count,
                        'max_iterations': max_iterations,
                        'is_best': is_best,
                        'progress_percent': (iteration / max_iterations) * 100
                    }
                    yield f"data: {json.dumps(update_data)}\n\n"
            
            # Start optimization
            planner.generate_tree_positions(callback=progress_callback)
            
            # Send final result
            final_data = {
                'completed': True,
                'final_tree_count': len(planner.trees),
                'image': best_image,
                'coordinates': best_coordinates
            }
            yield f"data: {json.dumps(final_data)}\n\n"
        
        return app.response_class(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        error_data = {'error': f'An error occurred: {str(e)}'}
        if request.method == 'POST':
            return jsonify(error_data), 500
        else:
            return f"data: {json.dumps(error_data)}\n\n"

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
