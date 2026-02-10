import tensorflow as tf
import cv2
import numpy as np
import time

# Define the model directory and path
model_dir = 'C:/Users/siddh/OneDrive/Desktop/SoItBegins/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model'

# Load the TensorFlow model
detect_fn = tf.saved_model.load(model_dir)
detect_fn = detect_fn.signatures['serving_default']

# Load the COCO labels for the car class (ID: 3)
car_label_id = 3

# Global variables for car counting
car_count_left = 0
car_count_right = 0

# Function to run object detection and update car counts
def run_object_detection(video_source='video.mp4'):
    global car_count_left, car_count_right

    # Initialize variables
    counted_objects = []
    last_seen_time = {}

    # Initialize video capture
    cap = cv2.VideoCapture(video_source)

    # Check if video capture was successful
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define the ROI (more vertically bigger and fits the horizontal ends)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x = 0  # Start from the left edge
    roi_y = int(frame_height * 0.5)  # Lower the y-position
    roi_width = frame_width  # Full width
    roi_height = int(frame_height * 0.4)  # Increase the height

    # Define time threshold for car departure (2 seconds)
    departure_threshold = 2.0

    # Define the line positions
    line_y = 475
    line_color = (0, 0, 255)  # Red color
    line_thickness = 2

    # Create a named window and resize it
    cv2.namedWindow('Junction Car Counting', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Junction Car Counting', 1280, 720)  # Resize to desired dimensions

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Draw the ROI rectangle
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)  # Blue color

        # Draw the counting lines
        frame_height, frame_width, _ = frame.shape
        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), line_color, line_thickness)
        cv2.line(frame, (0, line_y), (frame_width, line_y), line_color, line_thickness)

        # Resize the frame to 320x320 for model input and ensure it's uint8
        input_frame = cv2.resize(frame, (320, 320))
        input_frame = input_frame.astype(np.uint8)  # Convert to uint8
        input_tensor = tf.convert_to_tensor(input_frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = detect_fn(input_tensor)

        # Extract detection results
        num_detections = int(detections.pop('num_detections').numpy())
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Print detection results for debugging
        # print(f"Detections: {detections}")

        # Filter out car detections
        scores = detections['detection_scores']
        classes = detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']

        current_counted_objects = []

        for i in range(num_detections):
            if scores[i] > 0.05 and classes[i] == car_label_id:  # Filter out low-confidence detections

                # Get bounding box coordinates
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                xmin, xmax, ymin, ymax = int(xmin * frame_width), int(xmax * frame_width), int(ymin * frame_height), int(ymax * frame_height)

                # Calculate the center of the bounding box
                centroid = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

                # Print centroid for debugging
                # print(f"Centroid: {centroid}")

                # Check if the centroid is within the ROI
                if roi_x <= centroid[0] <= roi_x + roi_width and roi_y <= centroid[1] <= roi_y + roi_height:
                    # Check if the object has already been counted
                    if not any(np.linalg.norm(np.array(centroid) - np.array(c)) < 50 for c in counted_objects):
                        # Check if the vehicle is on the left or right of the vertical center line of the ROI
                        if centroid[0] < roi_x + roi_width // 2:
                            car_count_left += 1
                        else:
                            car_count_right += 1
                        # Add the centroid to the counted objects buffer and update last seen time
                        counted_objects.append(centroid)
                        last_seen_time[centroid] = time.time()

                    current_counted_objects.append(centroid)

                # Check if the object has already been counted
                if centroid not in counted_objects:
                    # Check if the center of the bounding box crosses the horizontal line
                    if ymin < line_y < ymax:
                        # Check if the vehicle is on the left or right of the vertical center line
                        if centroid[0] < frame_width // 2:
                            car_count_left += 1
                        else:
                            car_count_right += 1
                        # Add the centroid to the counted objects buffer
                        counted_objects.append(centroid)

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"Car: {scores[i]:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the counted objects buffer to remove old objects based on departure threshold
        current_time = time.time()
        for obj in list(counted_objects):
            if obj in last_seen_time and current_time - last_seen_time[obj] > departure_threshold:
                counted_objects.remove(obj)
                if obj[0] < roi_x + roi_width // 2:
                    car_count_left -= 1
                else:
                    car_count_right -= 1

        # Display car counts
        cv2.putText(frame, f"Cars Left: {car_count_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Cars Right: {car_count_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Junction Car Counting', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # print(f"Final Cars Counted - Left: {car_count_left}, Right: {car_count_right}")

# Example of how to use this module:
if __name__ == "__main__":
    run_object_detection()
