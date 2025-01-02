import cv2
import numpy as np
import os
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

# Check if files exist
weights_path = os.path.join(os.path.dirname(__file__), "backup", "yolov3_custom_last.weights")
config_path = os.path.join(os.path.dirname(__file__), "yolov3_custom.cfg")
names_path = os.path.join(os.path.dirname(__file__), "obj.names")

if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
    raise FileNotFoundError(f"Weight, config, or names file not found: {weights_path}, {config_path}, {names_path}")

# Load YOLO model
try:
    net = cv2.dnn.readNet(weights_path, config_path)
except cv2.error as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Load class names
classes = []
with open(names_path, "r") as f:
    classes = f.read().splitlines()

# Set up video capture
url = 'http://192.168.8.101:8080/video'
cap = cv2.VideoCapture(url)  # Use 0 for webcam, or provide video file path

if not cap.isOpened():
    raise RuntimeError("Error opening video stream or file")

def take_picture():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return

    # Save the captured frame to a file
    cv2.imwrite("captured_frame.jpg", frame)
    print("Image captured and saved as captured_frame.jpg")

    # Allow the user to select an ROI
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi

    if w > 0 and h > 0:
        cropped_frame = frame[y:y+h, x:x+w]

        # Resize the cropped frame to 416x416 for YOLO processing
        frame_resized = cv2.resize(cropped_frame, (416, 416))

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
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:  # Filter weak detections
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)

                    # Box coordinates
                    box_x = int(center_x - box_w / 2)
                    box_y = int(center_y - box_h / 2)

                    boxes.append([box_x, box_y, box_w, box_h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        for i in indices:
            box = boxes[i]
            box_x, box_y, box_w, box_h = box
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw the bounding box on the cropped frame
            cv2.rectangle(cropped_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
            cv2.putText(cropped_frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Defect detected: {classes[class_ids[i]]} with confidence {confidences[i]:.2f}")

        # Display the cropped frame with bounding boxes
        cv2.imshow("Cropped Frame", cropped_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Cropped Frame")

    cv2.destroyWindow("Select ROI")

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))  # Resize to fit within the smaller window
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Create a simple GUI with a button to take a picture
root = Tk()
root.title("PCB Defect Detection")
root.geometry("640x520")  # Set the size of the window to fit within a quarter of the display

video_label = Label(root)
video_label.pack()

button = Button(root, text="Take Picture", command=take_picture)
button.pack()

# Start video streaming
update_frame()

# Run the GUI loop
root.mainloop()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()