"""
Visualization utilities for object detection.
This module provides functions for visualizing object detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import random


def draw_boxes(
    image: np.ndarray,
    detections: Dict,
    color_map: Optional[Dict] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_labels: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes on an image based on detection results.
    
    Args:
        image: Image as numpy array (RGB format)
        detections: Detection results from YOLODetector.detect()
        color_map: Dictionary mapping class names to colors (RGB)
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for labels
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        Image with drawn bounding boxes
    """
    # Make a copy of the image to avoid modifying the original
    image_with_boxes = image.copy()
    
    # Convert to BGR for OpenCV if it's in RGB
    if len(image_with_boxes.shape) == 3 and image_with_boxes.shape[2] == 3:
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
    
    # Generate color map if not provided
    if color_map is None:
        color_map = {}
        for class_name in set(detections['class_names']):
            color_map[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
    
    # Draw each bounding box
    for i, (box, class_name, confidence) in enumerate(zip(
        detections['boxes'],
        detections['class_names'],
        detections['confidences']
    )):
        # Get coordinates
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color for this class
        color = color_map.get(class_name, (0, 255, 0))  # Default to green if class not in color_map
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if show_labels and show_confidence:
            label = f"{class_name}: {confidence:.2f}"
        elif show_labels:
            label = class_name
        elif show_confidence:
            label = f"{confidence:.2f}"
        else:
            label = ""
        
        # Draw label background
        if label:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                image_with_boxes,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                image_with_boxes,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness
            )
    
    # Convert back to RGB
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    return image_with_boxes


def plot_detection_results(
    image: np.ndarray,
    detections: Dict,
    figsize: Tuple[int, int] = (12, 10),
    show_stats: bool = True
) -> None:
    """
    Plot detection results with matplotlib.
    
    Args:
        image: Image as numpy array (RGB format)
        detections: Detection results from YOLODetector.detect()
        figsize: Figure size (width, height)
        show_stats: Whether to show detection statistics
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw bounding boxes on image
    image_with_boxes = draw_boxes(image, detections)
    
    # Show image with detections
    plt.imshow(image_with_boxes)
    plt.axis('off')
    
    # Add title with detection count
    plt.title(f"Detected {detections['count']} objects", fontsize=14)
    
    # Show detection statistics if requested
    if show_stats and detections['count'] > 0:
        # Count objects by class
        class_counts = {}
        for class_name in detections['class_names']:
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # Create text for statistics
        stats_text = "Detection Statistics:\n"
        for class_name, count in class_counts.items():
            stats_text += f"- {class_name}: {count}\n"
        
        # Add text box with statistics
        plt.figtext(
            0.02, 0.02, stats_text,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=12
        )
    
    plt.tight_layout()
    plt.show()


def visualize_class_distribution(detections: Dict) -> None:
    """
    Visualize the distribution of detected classes.
    
    Args:
        detections: Detection results from YOLODetector.detect()
    """
    if detections['count'] == 0:
        print("No objects detected.")
        return
    
    # Count objects by class
    class_counts = {}
    for class_name in detections['class_names']:
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    # Sort by count (descending)
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_counts)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha='center',
            fontsize=10
        )
    
    plt.title('Distribution of Detected Classes', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def visualize_confidence_distribution(detections: Dict, bins: int = 10) -> None:
    """
    Visualize the distribution of confidence scores.
    
    Args:
        detections: Detection results from YOLODetector.detect()
        bins: Number of bins for histogram
    """
    if detections['count'] == 0:
        print("No objects detected.")
        return
    
    # Get confidence scores
    confidences = detections['confidences']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Confidence Scores', fontsize=14)
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_color_map(class_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Create a color map for visualization.
    
    Args:
        class_names: List of class names
        
    Returns:
        Dictionary mapping class names to colors (RGB)
    """
    color_map = {}
    
    # Predefined colors for common classes
    predefined_colors = {
        'person': (0, 128, 255),    # Orange
        'car': (0, 200, 0),         # Green
        'truck': (0, 150, 0),       # Dark Green
        'bus': (0, 100, 0),         # Darker Green
        'motorcycle': (200, 0, 0),  # Red
        'bicycle': (150, 0, 0),     # Dark Red
        'traffic light': (255, 0, 255),  # Magenta
        'stop sign': (255, 0, 0),   # Bright Red
        'dog': (128, 128, 0),       # Olive
        'cat': (128, 0, 128)        # Purple
    }
    
    # Assign colors to classes
    for class_name in class_names:
        if class_name in predefined_colors:
            color_map[class_name] = predefined_colors[class_name]
        else:
            # Generate random color for other classes
            color_map[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
    
    return color_map


def save_visualization(
    image: np.ndarray,
    detections: Dict,
    output_path: str,
    color_map: Optional[Dict] = None,
    dpi: int = 300
) -> None:
    """
    Save visualization of detection results to a file.
    
    Args:
        image: Image as numpy array (RGB format)
        detections: Detection results from YOLODetector.detect()
        output_path: Path to save the visualization
        color_map: Dictionary mapping class names to colors (RGB)
        dpi: DPI for saved image
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw bounding boxes on image
    image_with_boxes = draw_boxes(image, detections, color_map)
    
    # Show image with detections
    plt.imshow(image_with_boxes)
    plt.axis('off')
    
    # Add title with detection count
    plt.title(f"Detected {detections['count']} objects", fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
