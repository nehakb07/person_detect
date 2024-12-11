import os
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('weights/best.pt')

# Set the input and output folders
input_folder = 'input'
output_folder = 'output'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Set confidence threshold
conf_threshold = 0.5  # Confidence threshold for detection

# Define target size for resizing
target_size = (640, 640)

# Loop through all the images in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    
    # Load the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not read image {img_name}. Skipping...")
        continue

    # Resize the image to 640x640
    img = cv2.resize(img, target_size)

    # Perform inference
    results = model(img)

    # Initialize counter for persons detected
    person_count = 0

    # Collect predictions and draw bounding boxes
    for box in results[0].boxes:
        if box.conf[0] >= conf_threshold:
            class_id = int(box.cls[0])
            
            # Check for person (class_id = 0)
            if class_id == 0:  # Class for 'person'
                person_count += 1
                
                # Get bounding box coordinates for the person
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                # Draw the bounding box around the person
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for persons
                cv2.putText(img, f'Person {person_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the total number of people detected in the top-left corner
    cv2.putText(img, f'Total People: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Print the number of people detected for the image
    print(f"{img_name}: {person_count} people detected")

    # Save the processed image with bounding boxes and person count
    output_img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_img_path, img)

print("Processing complete.")
