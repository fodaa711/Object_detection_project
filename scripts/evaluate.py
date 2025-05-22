#!/usr/bin/env python
"""
Evaluation script for YOLOv8 object detection model.
This script evaluates the performance of the YOLOv8 model on a dataset.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from models.detector import YOLODetector
from utils.visualization import visualize_class_distribution, plot_confusion_matrix
from utils.processing import load_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection Evaluation')
    
    # Input arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to directory containing images')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to annotations file (JSON format)')
    parser.add_argument('--output', type=str, default='./evaluation',
                        help='Path to save evaluation results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda or cpu)')
    
    # Evaluation arguments
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching detections to ground truth')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Resize image to this size before detection')
    
    return parser.parse_args()


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        box1: First bounding box as [x1, y1, x2, y2]
        box2: Second bounding box as [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou


def evaluate_detections(detections, ground_truth, iou_threshold=0.5):
    """
    Evaluate detection results against ground truth.
    
    Args:
        detections: Detection results from YOLODetector.detect()
        ground_truth: Ground truth annotations
        iou_threshold: IoU threshold for matching detections to ground truth
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Initialize confusion matrix
    all_classes = set(detections['class_names']).union(set(gt['class'] for gt in ground_truth))
    class_to_idx = {cls: i for i, cls in enumerate(sorted(all_classes))}
    confusion_matrix = np.zeros((len(class_to_idx), len(class_to_idx)), dtype=int)
    
    # Track matched ground truth boxes
    matched_gt = [False] * len(ground_truth)
    
    # Match detections to ground truth
    for i, (box, class_name, confidence) in enumerate(zip(
        detections['boxes'],
        detections['class_names'],
        detections['confidences']
    )):
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth box
        for j, gt in enumerate(ground_truth):
            if matched_gt[j]:
                continue
                
            gt_box = gt['bbox']  # [x1, y1, x2, y2]
            
            # Calculate IoU
            iou = calculate_iou(box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if match is valid
        if best_iou >= iou_threshold:
            # True positive
            true_positives += 1
            matched_gt[best_gt_idx] = True
            
            # Update confusion matrix
            gt_class = ground_truth[best_gt_idx]['class']
            confusion_matrix[class_to_idx[gt_class], class_to_idx[class_name]] += 1
        else:
            # False positive
            false_positives += 1
            
            # Update confusion matrix (background class)
            if 'background' in class_to_idx:
                bg_idx = class_to_idx['background']
            else:
                # Add background class
                bg_idx = len(class_to_idx)
                class_to_idx['background'] = bg_idx
                confusion_matrix = np.pad(confusion_matrix, ((0, 1), (0, 0)), mode='constant')
                
            confusion_matrix[bg_idx, class_to_idx[class_name]] += 1
    
    # Count false negatives
    for j, matched in enumerate(matched_gt):
        if not matched:
            false_negatives += 1
            
            # Update confusion matrix
            gt_class = ground_truth[j]['class']
            if 'background' in class_to_idx:
                bg_idx = class_to_idx['background']
            else:
                # Add background class
                bg_idx = len(class_to_idx)
                class_to_idx['background'] = bg_idx
                confusion_matrix = np.pad(confusion_matrix, ((0, 0), (0, 1)), mode='constant')
                
            confusion_matrix[class_to_idx[gt_class], bg_idx] += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Return evaluation metrics
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': confusion_matrix,
        'class_mapping': class_to_idx
    }


def load_annotations(annotations_path):
    """
    Load annotations from file.
    
    Args:
        annotations_path: Path to annotations file
        
    Returns:
        Dictionary mapping image filenames to ground truth annotations
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def plot_metrics(metrics, output_dir):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot precision, recall, and F1 score
    plt.figure(figsize=(10, 6))
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    
    plt.bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)
    
    plt.ylim(0, 1.1)
    plt.title('Detection Performance Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
    plt.close()
    
    # Plot confusion matrix
    class_mapping = metrics['class_mapping']
    confusion_matrix = metrics['confusion_matrix']
    
    # Get class names in order
    class_names = [None] * len(class_mapping)
    for cls, idx in class_mapping.items():
        class_names[idx] = cls
    
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize detector
    detector = YOLODetector(model_path=args.model, device=args.device)
    
    # Load annotations
    annotations = load_annotations(args.annotations)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize metrics
    all_metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'images_processed': 0,
        'inference_time': 0
    }
    
    # Process each image
    image_paths = list(Path(args.data_dir).glob('*.jpg')) + list(Path(args.data_dir).glob('*.png'))
    
    for image_path in tqdm(image_paths, desc="Evaluating"):
        image_name = image_path.name
        
        # Skip if no annotations for this image
        if image_name not in annotations:
            print(f"Warning: No annotations found for {image_name}")
            continue
        
        # Load image
        image = load_image(str(image_path))
        
        # Resize image if specified
        if args.img_size:
            from utils.processing import resize_image
            image = resize_image(image, max_size=args.img_size, keep_aspect_ratio=True)
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(
            image,
            conf_threshold=args.conf_threshold
        )
        inference_time = time.time() - start_time
        
        # Update inference time
        all_metrics['inference_time'] += inference_time
        all_metrics['images_processed'] += 1
        
        # Evaluate detections
        image_metrics = evaluate_detections(
            detections,
            annotations[image_name],
            iou_threshold=args.iou_threshold
        )
        
        # Update metrics
        all_metrics['true_positives'] += image_metrics['true_positives']
        all_metrics['false_positives'] += image_metrics['false_positives']
        all_metrics['false_negatives'] += image_metrics['false_negatives']
    
    # Calculate overall metrics
    if all_metrics['images_processed'] > 0:
        # Calculate precision, recall, and F1 score
        precision = all_metrics['true_positives'] / (all_metrics['true_positives'] + all_metrics['false_positives']) if (all_metrics['true_positives'] + all_metrics['false_positives']) > 0 else 0
        recall = all_metrics['true_positives'] / (all_metrics['true_positives'] + all_metrics['false_negatives']) if (all_metrics['true_positives'] + all_metrics['false_negatives']) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_metrics['precision'] = precision
        all_metrics['recall'] = recall
        all_metrics['f1_score'] = f1_score
        
        # Calculate average inference time
        all_metrics['avg_inference_time'] = all_metrics['inference_time'] / all_metrics['images_processed']
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Average Inference Time: {all_metrics['avg_inference_time']:.4f} seconds")
        
        # Plot metrics
        plot_metrics(image_metrics, args.output)
        
        # Save metrics to file
        with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=4)
    else:
        print("No images processed.")


if __name__ == "__main__":
    main()
