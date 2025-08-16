def read_csv(csv_path):
    np_path_XYs=np.genfromtxt(csv_path,delimiter=',')
    path_XYs=[]
    for i in np.unique(np_path_XYs[:,0]):
        npXYs=np_path_XYs[np_path_XYs[:,0]==i][:,1:]
        XYs=[]
        for j in np.unique(npXYs[:,0]):
            XY=npXYs[npXYs[:,0]==j][:,1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def plot_polylines(paths_XYs, save_path=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2, label=f'Polyline {i}')
    ax.set_aspect("equal")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, format='jpg')

    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
def fit_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept

    return slope, intercept

def draw_shape(image, shape_type, points):
    if shape_type == "line":
        pt1 = tuple(points[0][0])
        pt2 = tuple(points[1][0])
        cv2.line(image, pt1, pt2,(0, 255,0), 2)
    elif shape_type == "rectangle":
        x, y, w, h = cv2.boundingRect(points)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
    elif shape_type == "circle":
        (x, y), radius = cv2.minEnclosingCircle(points)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0,255,0), 2)
    elif shape_type == "polygon":
        cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)


        
## READING CSV, INPUT YOUR CSV FILE HERE

img = read_csv('frag0.csv') 



plot_polylines(img, 'frag0.jpg')
img = cv2.imread('frag0.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##converting the img to grey scale for edgw detection

# _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 50, 150)  ##Canny edge detection
kernel = np.ones((4, 4), np.uint8) 
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) ##Applies a closing operation to fill small gaps in the edges
contour, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) ##Finds the contours in the processed edge image

def getShapeName(approx):
    if len(approx) == 2:
        shape_name = "line"
    elif len(approx) == 3:
        shape_name = "triangle"
    elif len(approx) == 4 or len(approx) == 5:
        shape_name = "rectangle"
    elif len(approx) > 12:
        shape_name = "circle"
    else:
        shape_name = "polygon"
    return shape_name

## the below lines of code, for each contour, approximates the contour to a polygon, classifies the shape, and draws it
shape_image = np.ones_like(img)*255
circles = []
if circles:
    largest_circle = max(circles, key=lambda c: c[2])
    center = (int(largest_circle[0]), int(largest_circle[1]))
    radius = int(largest_circle[2])
    cv2.circle(shape_image, center, radius, (0,255,0), 2)
for cont in contour:
    epsilon = 0.0095 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)
    shape_name = getShapeName(approx)
    print(shape_name)
    if shape_name == 'rectangle':
        draw_shape(shape_image, 'rectangle', approx)
    elif shape_name == 'circle':
        draw_shape(shape_image, 'circle', approx)
    elif shape_name == 'polygon':
        draw_shape(shape_image, 'polygon', approx)



plt.imshow(shape_image)
plt.axis('off')
plt.show()
cv2.imwrite('resultant_shapes1.jpg', shape_image) ##img is saved

import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_lines_thinner(image_path, output_path="super_thin_lines_output.png"):
    """
    Makes the lines in the image as thin as possible while preserving the structure, colors, and angles.
    """
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale to detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Edge detection using Canny with optimized thresholds
    edges = cv2.Canny(gray, 50, 150)  # Adjusted thresholds for better edge detection
    
    # Use morphology to make lines thinner
    kernel = np.ones((1,1), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=4)  # Erode to make lines super thin
    
    # Mask the original image using the thinned edges
    thin_lines = cv2.bitwise_and(img, img, mask=edges)
    
    # Create a white background
    white_background = np.ones_like(img) * 255
    
    # Overlay thin lines onto the white background
    result = np.where(thin_lines > 0, thin_lines, white_background)
    
    # Save the output image
    cv2.imwrite(output_path, result)
    
    # Display result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Geometric Pattern with Super Thin Lines")
    plt.axis("off")
    plt.show()
    
    return output_path

image_path = "images2.png"  # Replace with actual image file path
output_path = "super_thin_lines_output.png"
make_lines_thinner(image_path, output_path)

from PIL import Image
import csv

# Load the image
image = Image.open('super_thin_lines_output.png')

# Get the dimensions (width, height)
width, height = image.size

# Define the CSV file name
csv_filename = 'image_dimensions.csv'

# Write the dimensions to the CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Width', 'Height'])  # Write header
    writer.writerow([width, height])      # Write dimensions

print(f"Image dimensions saved to {csv_filename}")

import cv2
import numpy as np
import csv

# Load the image
image = cv2.imread('super_thin_lines_output.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (adjust thresholds as needed)
edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Adjust these thresholds

# Show the edges for debugging
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect lines using Hough Line Transform (adjust parameters as needed)
lines = cv2.HoughLinesP(
    edges,
    rho=1,              # Distance resolution in pixels
    theta=np.pi / 180,  # Angle resolution in radians
    threshold=50,       # Minimum number of votes to detect a line
    minLineLength=30,   # Minimum length of a line
    maxLineGap=10       # Maximum gap between line segments
)

# Define the output CSV file name
output_filename = 'geometric_pattern.csv'

# Prepare to write to CSV
with open(output_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["x1", "y1", "x2", "y2"])  # Header

    # Check if lines were detected
    if lines is not None:
        # Draw detected lines on the original image for debugging
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines
            writer.writerow([x1, y1, x2, y2])

        # Show the image with detected lines for debugging
        cv2.imshow("Detected Lines", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"CSV file '{output_filename}' has been created with the geometric pattern data.")
    else:
        print("No lines were detected in the image. Adjust the parameters or check the image.")

def read_csv(csv_path):
    np_path_XYs=np.genfromtxt(csv_path,delimiter=',')
    path_XYs=[]
    for i in np.unique(np_path_XYs[:,0]):
        npXYs=np_path_XYs[np_path_XYs[:,0]==i][:,1:]
        XYs=[]
        for j in np.unique(npXYs[:,0]):
            XY=npXYs[npXYs[:,0]==j][:,1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs


def plot_polylines(paths_XYs, save_path=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2, label=f'Polyline {i}')
    ax.set_aspect("equal")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, format='jpg')

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
def fit_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept

    return slope, intercept

def draw_shape(image, shape_type, points):
    if shape_type == "line":
        pt1 = tuple(points[0][0])
        pt2 = tuple(points[1][0])
        cv2.line(image, pt1, pt2,(0, 255,0), 2)
    elif shape_type == "rectangle":
        x, y, w, h = cv2.boundingRect(points)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
    elif shape_type == "circle":
        (x, y), radius = cv2.minEnclosingCircle(points)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0,255,0), 2)
    elif shape_type == "polygon":
        cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)


        
## READING CSV. INPUT YOUR CSV FILE HERE

img = read_csv('geometric_pattern.csv') 



plot_polylines(img, 'frag0.jpg')
img = cv2.imread('frag0.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ##converting the img to grey scale for edgw detection

# _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(gray, 50, 150)  ##Canny edge detection
kernel = np.ones((4, 4), np.uint8) 
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) ##Applies a closing operation to fill small gaps in the edges
contour, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) ##Finds the contours in the processed edge image

def getShapeName(approx):
    if len(approx) == 2:
        shape_name = "line"
    elif len(approx) == 3:
        shape_name = "triangle"
    elif len(approx) == 4 or len(approx) == 5:
        shape_name = "rectangle"
    elif len(approx) > 12:
        shape_name = "circle"
    else:
        shape_name = "polygon"
    return shape_name

## the below lines of code, for each contour, approximates the contour to a polygon, classifies the shape, and draws it
shape_image = np.ones_like(img)*255
circles = []
if circles:
    largest_circle = max(circles, key=lambda c: c[2])
    center = (int(largest_circle[0]), int(largest_circle[1]))
    radius = int(largest_circle[2])
    cv2.circle(shape_image, center, radius, (0,255,0), 2)
for cont in contour:
    epsilon = 0.0095 * cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, epsilon, True)
    shape_name = getShapeName(approx)
    print(shape_name)
    if shape_name == 'rectangle':
        draw_shape(shape_image, 'rectangle', approx)
    elif shape_name == 'circle':
        draw_shape(shape_image, 'circle', approx)
    elif shape_name == 'polygon':
        draw_shape(shape_image, 'polygon', approx)



plt.imshow(shape_image)
plt.axis('off')
plt.show()
cv2.imwrite('resultant_shapes1.jpg', shape_image) ##img is saved
