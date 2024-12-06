#
# Imports
#
import argparse
import ast
import os
import platform

# Check the operating system
if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import shutil
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import logging
import inspect
from logging.handlers import RotatingFileHandler
from logging.handlers import TimedRotatingFileHandler
import traceback
import gc

# Initialize global variables
selected_points = []
distances = []
clusters_marked = 0
lines_data = []

excel_dir = ''
contour_dir = ''
overlay_dir  = ''
Excel_filename = ''
resized_image = ''
image_copy = ''
scaling_factor = ''
cluster_centers = ''
logger = ''
measurements = []
undo_stack = []
selected_centers = []
cluster_sizes = []
show_debug = False
Measurements_stored = False
exit_flag = False
max_display_size = ''
scaling_factors = (1,1)
blends = ''
cluster_method_type = ''
xi_value = 0.05


# Function: show_image
# Date: 24-09-2024
# Parameters:
#   - caption: A brief description or title for the image.
#   - image: The image data to be displayed.
# Description:
def show_image(caption, image):
    if show_debug:
        r_image, scale = resize_image(image,max_display_size)
        cv2.imshow(caption, r_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
# Function: find_centers
# Date: 24-09-2024
# Parameters:
#   - filename: The name of the file to be processed.
#   - eps: The epsilon parameter for clustering algorithms.
# Description:
def find_centers(filename, eps) :
    
    global Excel_filename, cluster_centers, resized_image, cluster_sizes, image_copy, measurements, selected_centers, scaling_factors, exit_flag, Measurements_stored
    global xi_value, cluster_method_type

    # Load and preprocess the image
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    eps_size = eps


    # Split the file path into root and extension
    base_name, ext_name = os.path.splitext(filename)
    file_name = os.path.basename(base_name).replace('_filtered', '')
    
    # Get the path of the file
    Original_filename =  os.path.normpath(os.path.join(os.path.dirname(filename), 'original', file_name)) + ext_name
    
    # Construct the new file paths with the new extension
    Excel_filename =os.path.normpath(os.path.join(excel_dir, file_name))+ '.xlsx'
    Image_filename = os.path.normpath(os.path.join(contour_dir, file_name))+ '.tif'
    Overlay_filename = os.path.normpath(os.path.join(overlay_dir, file_name))+ '.jpg'


    # In order to extract only the contours, process the magnitude of the Sobel edges detector
    sx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    m = cv2.magnitude(sx, sy)
    m = cv2.normalize(m, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)
    show_image('Contours', m)

    # Thinning to reduce the contour thickness to as few pixels as possible
    m = cv2.ximgproc.thinning(m, None, cv2.ximgproc.THINNING_GUOHALL)
    show_image('Thinned', m)

    # Detect particles using contours
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the center of each detected particle
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])

    centers = np.array(centers)

    # Perform clustering using OPTICS
    optics = OPTICS(min_samples=2, max_eps=eps_size, cluster_method=cluster_method_type, xi=xi_value)
    labels = optics.fit_predict(centers)
    unique_labels = set(labels)

    # Dictionary to store the number of points in each cluster
    cluster_sizes = {}

    # Calculate the center of each cluster
    cluster_centers = []
    for label in unique_labels:
        if label != -1:  # Ignore noise points (label = -1)
            cluster_sizes[label] = list(labels).count(label)
            cluster_points = centers[labels == label]
            cluster_center = cluster_points.mean(axis=0)
            cluster_centers.append(cluster_center)

    # Convert list to array for easy handling
    cluster_centers = np.array(cluster_centers)

    # Visualize the results
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for center in cluster_centers:
        cv2.circle(output_image, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)

    # Assign colors to different clusters
    colors = plt.get_cmap('tab10', len(unique_labels))

    for label in unique_labels:
        if label != -1:
            class_member_mask = (labels == label)
            cluster_points = centers[class_member_mask]

            # Compute the convex hull of the cluster points
            hull = cv2.convexHull(cluster_points)

            # Draw the convex hull around the cluster
            cv2.polylines(output_image, [hull], isClosed=True, color=(int(colors(label)[0] * 255),
                                                                    int(colors(label)[1] * 255),
                                                                    int(colors(label)[2] * 255)), thickness=6)

            # Compute hull center and draw text
            if hull.shape[0] == 1:
                hull_center = hull[0]
            else:
                hull_center = np.mean(hull, axis=0).astype(int)

            hull_center = hull_center.flatten()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            font_thickness = 3
            text_color = (255, 0, 0)
            text = str(cluster_sizes[label])
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = hull_center[0] - text_size[0] // 2
            text_y = hull_center[1] - text_size[1] // 2

            cv2.putText(output_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    # Show the image with drawn contours and clusters
    show_image('Clusters with Convex Hulls', output_image)
    log_message(f"Found {len(unique_labels)} clusters")

    # Write the filtered image with circumference and point markers
    cv2.imwrite(Image_filename, output_image, [cv2.IMWRITE_JPEG_QUALITY, 65])

    # Load the original background image
    background = cv2.imread(Original_filename)
    background = cv2.normalize(background, None, 0, 255, cv2.NORM_MINMAX)

    # Blend the images
    alpha = blends[0]
    beta = blends[1]
    blended_image = cv2.addWeighted(background, alpha, output_image, beta, 0)

    # Save blended image
    cv2.imwrite(Overlay_filename, blended_image, [cv2.IMWRITE_JPEG_QUALITY, 65])

    # Resize image for display
    resized_image, scaling_factors = resize_image(blended_image, max_display_size)

    # Scale the cluster centers to match resized image
    cluster_centers = (cluster_centers * scaling_factors).astype(int)

    # Set up the mouse callback for interacting with clusters
    image_copy = resized_image.copy()
    cv2.imshow('Clusters', image_copy)
    cv2.setMouseCallback('Clusters', on_mouse)

    # Wait for user to finish (either close the window or double-click to save)
    # Display the image and handle the events
    exit_flag = False
    Measurements_stored = False

    while True:
        # Check for key press
        key = cv2.waitKey(20) & 0xFF

        # Only process regular keys (like 'q' for quit), ignore modifier keys like Shift
        if exit_flag == True:
            break
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# Function: calculate_distance
# Date: 24-09-2024
# Parameters:
#   - point1: The first point (x, y) coordinates.
#   - point2: The second point (x, y) coordinates.
# Description:
# Calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    global scaling_factors
    center1 = point1/scaling_factors
    center2 = point2/scaling_factors
    return np.linalg.norm(np.array(center1) - np.array(center2))

# Function: on_mouse
# Date: 24-09-2024
# Parameters:
#   - event: The type of mouse event.
#   - x: The x-coordinate of the mouse event.
#   - y: The y-coordinate of the mouse event.
#   - flags: Any relevant flags associated with the mouse event.
#   - param: Additional parameters for the callback.
# Description:
def on_mouse(event, x, y, flags, param):
    global selected_centers, measurements, image_copy, cluster_sizes, Measurements_stored, exit_flag

    if event in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK}:


        # left button selects the first point and then when the second point is selected it measures the distance between the two points and records it
        # unless the shift key is pressed. if it is the second point then the distance is measured to the (x,y) point and the number of gold particles recorded is -1
        if event == cv2.EVENT_LBUTTONDOWN:
                
            # shift key is NOT pressed then the measure is a distance between two cluster centers
            if not (flags & cv2.EVENT_FLAG_SHIFTKEY):

                    # Find the closest cluster center to the clicked point
                    distances = np.linalg.norm(cluster_centers - np.array([x, y]), axis=1)
                    closest_center_idx = np.argmin(distances)
                    closest_center = cluster_centers[closest_center_idx]

                    selected_centers.append((closest_center_idx, closest_center))
            else: # if shift keys is pressed it is a measure between a center and the membrane, but only as point number two
                if len(selected_centers) == 1:
                    closest_center = np.array([x,y])
                    selected_centers.append((-1, closest_center))
                    
            # selected the same point twice, which doesn't make sense
            if len(selected_centers) ==2:
                if ((selected_centers[0][0] == selected_centers[1][0])):
                    # Clear the selected centers for the next measurement
                    selected_centers = []

            # If two points are selected, measure the distance
            if (len(selected_centers) == 2): 
                # Get the two selected centers and their info
                idx1, center1 = selected_centers[0]
                idx2, center2 = selected_centers[1]
                distance = calculate_distance(center1, center2)
                
                if idx2 == -1:
                    second_cluster_size = -1
                    line_color = (0,0,255)
                else:
                    second_cluster_size = cluster_sizes[idx2]
                    line_color = (0,255,0)
                
                # Save the measurement: centers, distance, and cluster sizes
                measurements.append({
                    'Center 1': center1,
                    'Center 2': center2,
                    'Cluster Size 1': cluster_sizes[idx1],
                    'Cluster Size 2': second_cluster_size,
                    'Distance': distance
                })


            # Draw the line between the two points
                cv2.line(image_copy, tuple(center1.astype(int)), tuple(center2.astype(int)), line_color, 2)
                
                # write the distances between the two centers on the line
                put_distance_on_line(image_copy, center1,center2)                

                # Show the image with the line and the distance text
                cv2.imshow('Clusters', image_copy)

                # Clear the selected centers for the next measurement
                selected_centers = []

        # The right button removes the last recorded measurement
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to undo the last measurement
            if measurements:
                measurements.pop()  # Remove last measurement
                selected_centers = []  # Clear selections
                # Redraw the image without the last line
                image_copy[:] = resized_image.copy()  # Reset the image
                for measurement in measurements:
                    center1 = measurement['Center 1']
                    center2 = measurement['Center 2']
                    if measurement['Cluster Size 2'] == -1:
                        line_color = (0,0,255)
                    else:
                        line_color = (0,255,0)
                        
                    cv2.line(image_copy, tuple(center1.astype(int)), tuple(center2.astype(int)), line_color, 2)
                    
                    # write the distances between the two centers on the line
                    put_distance_on_line(image_copy, center1,center2)    
                                
                cv2.imshow('Clusters', image_copy)
                
        # and the left double click writes the measurements to an excel file
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            
            # Double left-click to save the measurements and exit
                
            # shift key is NOT pressed then the measure is a distance between two cluster centers
            if not (flags & cv2.EVENT_FLAG_SHIFTKEY):
                save_measurements_to_excel(measurements)
                Measurements_stored = True
                log_message(f"Made {len(measurements)} measurements") 
            else:
                log_message(f"Made no measurements") 
                
            exit_flag = True


# Function: put_distance_on_line
# Date: 24-09-2024
# Parameters:
#   - image_copy: A copy of the image on which to draw.
#   - center1: The first center point for the distance line.
#   - center2: The second center point for the distance line.
# Description:
def put_distance_on_line(image_copy, center1,center2):
                    
                # Calculate the midpoint of the line
                midpoint = ((center1 + center2) / 2).astype(int)
                
                # Find the length of the line
                distance = calculate_distance(center1, center2)

                # Offset the position to place the text slightly above the line
                offset = 10  # Adjust this value to control the distance from the line
                direction_vector = center2 - center1
                perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])  # Perpendicular to the line
                perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)  # Normalize
                text_position = midpoint + (perpendicular_vector * offset).astype(int)

                # Prepare the text for the distance
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (255, 0, 0)  # Blue color in BGR
                text = f"{distance:.0f}"

                # Get text size to ensure proper positioning
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = text_position[0] - text_size[0] // 2
                text_y = text_position[1] + text_size[1] // 2

                # Put the text (distance) on the image
                cv2.putText(image_copy, text, (text_x, text_y), font, font_scale, text_color, font_thickness)


# Function: save_measurements_to_excel
# Date: 24-09-2024
# Parameters:
#   - measurements: The data to be saved to an Excel file.
# Description:
def save_measurements_to_excel(measurements):

    if measurements:
        # Prepare data for Excel with x, y in separate cells and rounded distance
        data = []
        for measurement in measurements:
            center1_x, center1_y = measurement['Center 1']
            center2_x, center2_y = measurement['Center 2']
            cluster_size_1 = measurement['Cluster Size 1']
            cluster_size_2 = measurement['Cluster Size 2']
            rounded_distance = int(round(measurement['Distance']))  # Round and convert to integer
            
            data.append([
                (int(center1_x/scaling_factors[0])), (int(center1_y/scaling_factors[1])),
                (int(center2_x/scaling_factors[0])), (int(center2_y/scaling_factors[1])),
                cluster_size_1, cluster_size_2,
                rounded_distance
            ])

        # Create a DataFrame with separate columns for x, y, and other data
        df = pd.DataFrame(data, columns=[
            'Center 1 X', 'Center 1 Y',
            'Center 2 X', 'Center 2 Y',
            'Cluster Size 1', 'Cluster Size 2',
            'Distance'
        ])
        df.to_excel(Excel_filename, index=False)

# Function: resize_image
# Date: 24-09-2024
# Parameters:
#   - image: The image to be resized.
#   - max_size: The maximum size (width or height) for the resized image.
# Description:
# The original image is too big to disply on most PC screens, so before measuring the image is resized
def resize_image(image, max_size):
    # Original image size (height, width)
    original_height, original_width = image.shape[:2]

    # Target screen size
    screen_width, screen_height = max_size

    # Calculate aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Determine new size while maintaining aspect ratio
    if original_width / screen_width > original_height / screen_height:
        # Width is the limiting factor
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is the limiting factor
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    # Ensure new dimensions do not exceed target size
    new_width = min(new_width, screen_width)
    new_height = min(new_height, screen_height)

    # Resize the image to fit within the screen size, maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a scaling array for x and y
    scaling_factors = np.array([new_width / original_width, new_height / original_height])

    return resized_image, scaling_factors


 
# Function: create_directory
# Date: 24-09-2024
# Parameters:
#   - directory_path: The path of the directory to create.
# Description:
# do not create if already there
def create_directory(directory_path):
    directory_path = os.path.normpath(directory_path)
    if os.path.isdir(directory_path):
        print(f"Directory exists: {directory_path}")
        #print("Removing directory and its contents...")
        #shutil.rmtree(directory_path)
    else:
        print(f"Directory does not exist: {directory_path}")
        # Create the directory
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)
    return directory_path

# Function: init_logger
# Date: 24-09-2024
# Parameters:
#   - working_dir: The directory where log files will be stored.
# Description:
def init_logger(working_dir):
    global logger

    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(working_dir,'application.log')

    # Create a rotating file handler
    # rotating_handler = RotatingFileHandler(working_dir + "\\application.log", maxBytes=200000, backupCount=5)
    rotating_handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=7)

    # Create a custom log format
    formatter = logging.Formatter('%(seqnum)d | %(asctime)s | %(levelname)s | Line: %(line_number)d | %(message)s')

    # Set up formatter and add to handler
    rotating_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(rotating_handler)

    # Custom attribute for sequence number
    logger.seqnum = 0
    
    return logger



# Function: log_message
# Date: 24-09-2024
# Parameters:
#   - message (str): The message to log.
#   - severity (str): The severity level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'.
# Description:
# Logs a message to a file with a sequence number, timestamp, severity, and line number.
def log_message(message, severity="INFO"):
    
    global logger
    # Increment sequence number
    logger.seqnum += 1
    
    # Get the caller's frame info to fetch the line number
    frame_info = inspect.stack()[1]
    line_number = frame_info.lineno

    # Create an extra dictionary to hold custom attributes
    extra = {'seqnum': logger.seqnum, 'line_number': line_number}

    # Log the message with the specified severity
    if severity.upper() == "DEBUG":
        logger.debug(message, extra=extra)
    elif severity.upper() == "INFO":
        logger.info(message, extra=extra)
    elif severity.upper() == "WARNING":
        logger.warning(message, extra=extra)
    elif severity.upper() == "ERROR":
        logger.error(message, extra=extra)
    elif severity.upper() == "CRITICAL":
        logger.critical(message, extra=extra)
    else:
        logger.info(message, extra=extra)  # Default to INFO severity



# Function: parse_tuple
# Date: 24-09-2024
# Parameters:
#   - input_string: A string representation of a tuple to parse.
# Description:
def parse_tuple(input_string):
    try:
        # Use ast.literal_eval to safely evaluate the string as a tuple
        parsed_tuple = ast.literal_eval(input_string)
        if isinstance(parsed_tuple, tuple):
            return np.array(parsed_tuple)  # Convert to NumPy array if it's a tuple
        else:
            raise ValueError("Input must be a tuple")
    except (SyntaxError, ValueError) as e:
        
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {input_string}. Error: {e}")



# Function: main
# Date: date    24-09-2024
# Parameters:
#   - workdir: The working directory for the application.
#   - eps: The epsilon value for algorithms.
# Description:
def main(workdir,eps):

    global contour_dir, overlay_dir, excel_dir, logger
    
    # Set the working directories
    contour_dir = create_directory(os.path.join(workdir,'contour'))
    overlay_dir = create_directory(os.path.join(workdir,'overlay'))
    excel_dir = create_directory(os.path.join(workdir,'excel'))
    logs_dir = create_directory(os.path.join(workdir,'logs'))
    processed_dir = create_directory(os.path.join(workdir,'processed'))
    
    logger = init_logger(logs_dir)


    log_message("Processing begins.")
    # Get a list of all .tif files in the directory
    tif_files = glob.glob(os.path.join(workdir, '*.tif'))

    # Process the list of .tif files
    try:

        for file in tif_files:
            
            print(file)
            log_message(f"Processing file: {file}")

            find_centers(file, eps)   
            if Measurements_stored :
               log_message(f"Moved {file} to processed") 
               a = os.path.join(workdir,os.path.basename(file))
               b =os.path.join(processed_dir,os.path.basename(file))
               shutil.move(a,b)
            
            # Force garbage collection
            gc.collect()

    except Exception as e:
        # Get the exception information and log it
        log_message(f"Exception occurred: {e}", severity = 'ERROR')

        # Log with traceback including the line number
        log_message(f"Traceback : {traceback.format_exc()}", severity = 'ERROR')


#
# Parse and read all commandline variables and assign defaults to variables not specified
#

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    
    # Define the flag arguments
    parser.add_argument('--debug', action='store_true', help="Run the image display functions", default = False)
    parser.add_argument('--eps', type=int, help="Distance in px", default = 400)
    parser.add_argument('--method', type=str, help="Clustering method for OPTICS, either 'xi' or 'dbscan'", default = 'dbscan')
    parser.add_argument('--xi', type=float, help="Gradient change 0.05", default = 0.05)
    parser.add_argument(
        '--display',
        type=parse_tuple,
        help='Input resolution as a tuple in the format (x, y), e.g. (800, 800)',
        default = "(800,800)"
    )
    parser.add_argument(
        '--blends',
        type=parse_tuple,
        help='blending between foreground and background image the format (a, b), e.g. (0.5,0.5)',
        default = "(0.6,0.4)"
    )

    # Define the additional arguments
    parser.add_argument('work_dir', type=str, help="Working directory")


    # Parse the arguments
    args = parser.parse_args()
    
    max_display_size = args.display
    blends = args.blends
    show_debug = args.debug
    cluster_method_type = args.method
    xi_value = args.xi
        
    main(args.work_dir,args.eps)
