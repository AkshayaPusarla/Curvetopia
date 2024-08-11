import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import cairosvg

# Function to calculate the distance between two points
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# Function to determine if the shape is a rectangle or square and correct it
def identify_and_correct_shape(XY):
    if len(XY) != 4:
        return XY, "Irregular"

    # Calculate distances between consecutive points
    d = [distance(XY[i], XY[(i + 1) % 4]) for i in range(4)]
    diag1 = distance(XY[0], XY[2])
    diag2 = distance(XY[1], XY[3])

    # Check for square
    if np.allclose(d, d[0], atol=1e-2) and np.isclose(diag1, diag2, atol=1e-2):
        corrected_XY = correct_to_square(XY)
        return corrected_XY, "Square"
    
    # Check for rectangle
    if np.isclose(d[0], d[2], atol=1e-2) and np.isclose(d[1], d[3], atol=1e-2) and np.isclose(diag1, diag2, atol=1e-2):
        corrected_XY = correct_to_rectangle(XY)
        return corrected_XY, "Rectangle"
    
    return XY, "Irregular"

# Function to correct shape to a perfect square
def correct_to_square(XY):
    center = np.mean(XY, axis=0)
    avg_side_length = np.mean([distance(XY[i], XY[(i + 1) % 4]) for i in range(4)])
    
    corrected_XY = []
    for i in range(4):
        angle = np.pi / 4 + i * np.pi / 2
        corrected_XY.append(center + avg_side_length / np.sqrt(2) * np.array([np.cos(angle), np.sin(angle)]))
    
    return np.array(corrected_XY)

# Function to correct shape to a perfect rectangle
def correct_to_rectangle(XY):
    center = np.mean(XY, axis=0)
    width = np.mean([distance(XY[0], XY[1]), distance(XY[2], XY[3])])
    height = np.mean([distance(XY[1], XY[2]), distance(XY[3], XY[0])])
    
    corrected_XY = []
    corrected_XY.append(center + np.array([-width / 2, -height / 2]))
    corrected_XY.append(center + np.array([width / 2, -height / 2]))
    corrected_XY.append(center + np.array([width / 2, height / 2]))
    corrected_XY.append(center + np.array([-width / 2, height / 2]))
    
    return np.array(corrected_XY)

# Function to process dataset
def process_dataset(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    corrected_shapes = []
    shape_types = []

    for XY in data:
        XY = XY.reshape(-1, 2)
        corrected_XY, shape_type = identify_and_correct_shape(XY)
        corrected_shapes.append(corrected_XY)
        shape_types.append(shape_type)

    return corrected_shapes, shape_types

# Function to plot results
def plot_results(original_XYs, corrected_XYs, shape_types, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # Plot original shapes
    ax[0].set_title("Original Shapes")
    for i, XY in enumerate(original_XYs):
        ax[0].plot(np.append(XY[:, 0], XY[0, 0]), np.append(XY[:, 1], XY[0, 1]), color='blue', linewidth=2)
    ax[0].set_aspect('equal')

    # Plot corrected shapes
    ax[1].set_title("Corrected Shapes")
    for i, XY in enumerate(corrected_XYs):
        color = 'green' if shape_types[i] != "Irregular" else 'red'
        ax[1].plot(np.append(XY[:, 0], XY[0, 0]), np.append(XY[:, 1], XY[0, 1]), label=f"{shape_types[i]}", color=color, linewidth=2)
    ax[1].set_aspect('equal')

    plt.legend()
    plt.show()

# Function to export corrected shapes to SVG
def export_to_svg(corrected_XYs, shape_types, svg_file='corrected_output.svg'):
    dwg = svgwrite.Drawing(svg_file, profile='tiny')
    for i, XY in enumerate(corrected_XYs):
        color = 'green' if shape_types[i] != "Irregular" else 'red'
        path_data = "M " + " L ".join([f"{x},{y}" for x, y in XY]) + " Z"
        dwg.add(dwg.path(d=path_data, stroke=color, fill="none", stroke_width=2))
    dwg.viewbox(minx=0, miny=0, width=1000, height=1000)  # Set SVG size
    dwg.save()

    cairosvg.svg2png(url=svg_file, write_to=svg_file.replace('.svg', '.png'))

# Main function
def main():
    file_path = 'frag0.csv'  # Replace with your actual dataset file path
    
    # Step 1: Process dataset
    corrected_shapes, shape_types = process_dataset(file_path)
    
    # Step 2: Plot original and corrected shapes
    original_shapes = [np.genfromtxt(file_path, delimiter=',').reshape(-1, 2)]
    plot_results(original_shapes, corrected_shapes, shape_types, title="Shape Correction")
    
    # Step 3: Export corrected shapes to SVG
    export_to_svg(corrected_shapes, shape_types, svg_file='corrected_output.svg')

if __name__ == "__main__":
    main()