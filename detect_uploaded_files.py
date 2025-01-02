import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# Check if files exist
weights_path = os.path.join(os.path.dirname(__file__), "backup", "yolov3_custom_last.weights")
config_path = os.path.join(os.path.dirname(__file__), "yolov3_custom.cfg")

if not os.path.exists(weights_path) or not os.path.exists(config_path):
    raise FileNotFoundError(f"Weight or config file not found: {weights_path}, {config_path}")

# Load YOLO model
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except cv2.error as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

classes = []
with open("obj.names", "r") as f:
    classes = f.read().splitlines()

# Function to process and display the image
def process_image(image_path):
    frame = cv2.imread(image_path)
    original_height, original_width, _ = frame.shape

    # Resize the frame to 416x416 for YOLO processing
    frame_resized = cv2.resize(frame, (416, 416))

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame_resized, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO predictions
    layer_names = net.getLayerNames()
    try:
        unconnected_out_layers = net.getUnconnectedOutLayers()
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    except IndexError as e:
        print(unconnected_out_layers)
        raise RuntimeError(f"Error processing output layers: {e}")

    outputs = net.forward(output_layers)

    # Parse predictions
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # Filter weak detections
                center_x = int(detection[0] * original_width)
                center_y = int(detection[1] * original_height)
                w = int(detection[2] * original_width)
                h = int(detection[3] * original_height)

                # Box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Defect detected: {classes[class_id]} with confidence {confidence:.2f}")

    # Resize the frame to fit within a quarter of the screen
    screen_width = 1920  # Example screen width, adjust as needed
    screen_height = 1080  # Example screen height, adjust as needed
    max_width = screen_width // 2
    max_height = screen_height // 2

    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    frame_resized_display = cv2.resize(frame, (new_width, new_height))

    # Display the frame
    cv2.imshow("Frame", frame_resized_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create a file dialog to select an image
root = Tk()
root.withdraw()  # Hide the root window
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if image_path:
    process_image(image_path)
else:
    print("No image selected.")