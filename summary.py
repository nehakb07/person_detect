import os
import cv2
import numpy as np
from ultralytics import YOLO

# Function to convert bounding box format to YOLO format (x_center, y_center, width, height)
def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

# Function to check if predicted box is within a margin of error (10%)
def within_margin_error(pred_box, gt_box, margin_error=0.1):
    pred_width = pred_box[2] - pred_box[0]
    pred_height = pred_box[3] - pred_box[1]
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]

    # Check for zero-width or zero-height boxes to avoid division by zero
    if gt_width == 0 or gt_height == 0 or pred_width == 0 or pred_height == 0:
        return False  # If any box has zero width or height, it's invalid, so return False
    
    width_error = abs(pred_width - gt_width) / gt_width <= margin_error
    height_error = abs(pred_height - gt_height) / gt_height <= margin_error
    
    return width_error and height_error

# Function to resize the image while maintaining aspect ratio
def resize_image(img, target_width=1240):
    img_height, img_width = img.shape[:2]
    scale_ratio = target_width / img_width
    new_height = int(img_height * scale_ratio)
    resized_img = cv2.resize(img, (target_width, new_height))
    return resized_img

# Function to process validation images and save YOLO model predictions in YOLO format
def run_predictions_and_save_labels(valid_folder, labels_folder, model, conf_threshold=0.5):
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    # Process each image in the validation folder
    for img_file in os.listdir(valid_folder):
        img_path = os.path.join(valid_folder, img_file)
        img = cv2.imread(img_path)

        # Resize the image
        img = resize_image(img, target_width=1240)

        img_height, img_width, _ = img.shape
        pred_boxes = []

        # Predict using the YOLO model on the full image
        results = model(img, conf=conf_threshold)

        # Extract predicted bounding boxes
        for result in results:
            for box in result.boxes:
                box_xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                pred_boxes.append(box_xyxy)

        # Save the predictions to a .txt file in YOLO format
        label_file_name = img_file.replace('.jpg', '.txt')  # Adjust according to your image format
        label_file_path = os.path.join(labels_folder, label_file_name)

        with open(label_file_path, 'w') as label_file:
            for box in pred_boxes:
                yolo_box = convert_to_yolo_format(box, img_width, img_height)
                # Write to the label file; assuming class 0 (adjust if needed)
                label_file.write(f"0 {' '.join(map(str, yolo_box))}\n")

    print(f"Predictions saved in {labels_folder}.")
# Adjusted function to calculate metrics with new accuracy formula
def calculate_adjusted_metrics(TP, FP, FN, total_ground_truths, total_predictions):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (total_ground_truths) if (total_ground_truths + total_predictions - TP) > 0 else 0

    return precision, recall, f1, accuracy

# Update evaluate_predictions function to include new accuracy formula
def evaluate_predictions(predicted_folder, ground_truth_folder, iou_threshold=0.55, margin_error=0.1):  
    TP, FP, FN = 0, 0, 0
    total_ground_truths = 0
    total_predictions = 0

    
    for pred_file in os.listdir(predicted_folder):
        pred_file_path = os.path.join(predicted_folder, pred_file)
        gt_file_path = os.path.join(ground_truth_folder, pred_file.replace('.txt', '.txt'))

        # Debugging line to check if ground truth file exists
        if not os.path.exists(gt_file_path):
            print(f"Ground truth file {gt_file_path} not found.")
        
        # Load predicted boxes
        pred_boxes = []
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r') as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:5]))  # Ignore class id
                    pred_boxes.append(coords)
                    total_predictions += 1  # Count each predicted bounding box

        # Load ground truth boxes
        gt_boxes = []
        if os.path.exists(gt_file_path):
            with open(gt_file_path, 'r') as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:5]))  # Ignore class id
                    gt_boxes.append(coords)
                    total_ground_truths += 1  # Count each ground truth bounding box

        matched_gt = []
        
        # Check for true positives and false positives
        for pred_box in pred_boxes:
            match_found = False
            for gt_box in gt_boxes:
                if gt_box in matched_gt:
                    continue  # Ensure each ground truth box is only matched once
                if (calculate_iou(pred_box, gt_box) >= iou_threshold or 
                    within_margin_error(pred_box, gt_box, margin_error)):
                    TP += 1
                    matched_gt.append(gt_box)
                    match_found = True
                    break
            if not match_found:
                FP += 1
        
        # Check for false negatives
        for gt_box in gt_boxes:
            if gt_box not in matched_gt:
                FN += 1

    # Calculate adjusted metrics with new accuracy formula
    precision, recall, f1, accuracy = calculate_adjusted_metrics(TP, FP, FN, total_ground_truths, total_predictions)

    print(f"Total Ground Truth Bounding Boxes: {total_ground_truths}")
    print(f"Total Predicted Bounding Boxes: {total_predictions}")
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    return total_ground_truths, total_predictions, TP, FP, FN

valid_folder = "dataset/valid/images"
labels_folder = "predicted_labels/"
ground_truth_folder = "dataset/valid/labels/"
model = YOLO('weights/best.pt')

# Run predictions and save labels
run_predictions_and_save_labels(valid_folder, labels_folder, model)

# Evaluate predictions
evaluate_predictions(labels_folder, ground_truth_folder)
