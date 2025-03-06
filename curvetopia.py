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
