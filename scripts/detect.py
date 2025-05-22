#!/usr/bin/env python
"""
Object detection script using YOLOv8.
This script performs object detection on images or videos using the YOLOv8 model.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from models.detector import YOLODetector
from utils.visualization import draw_boxes, save_visualization
from utils.processing import load_image, resize_image, save_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    
    # Input arguments
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video file, or directory')
    parser.add_argument('--output', type=str, default='./output',
                        help='Path to save detection results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Path to YOLOv8 model weights')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda or cpu)')
    
    # Detection arguments
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--classes', type=str, default=None,
                        help='Filter by class, comma-separated list of class names')
    
    # Image processing arguments
    parser.add_argument('--img-size', type=int, default=None,
                        help='Resize image to this size before detection')
    
    # Video processing arguments
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Number of frames to skip between detections')
    
    # Output arguments
    parser.add_argument('--save-txt', action='store_true',
                        help='Save detection results to text file')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidence scores in text file')
    parser.add_argument('--hide-labels', action='store_true',
                        help='Hide labels in output images')
    parser.add_argument('--hide-conf', action='store_true',
                        help='Hide confidence scores in output images')
    
    return parser.parse_args()


def process_image(image_path, detector, args):
    """
    Process a single image.
    
    Args:
        image_path: Path to the image file
        detector: YOLODetector instance
        args: Command line arguments
        
    Returns:
        Detection results
    """
    print(f"Processing image: {image_path}")
    
    # Load image
    image = load_image(image_path)
    
    # Resize image if specified
    if args.img_size:
        image = resize_image(image, max_size=args.img_size, keep_aspect_ratio=True)
    
    # Filter classes if specified
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    
    # Perform detection
    start_time = time.time()
    detections = detector.detect(
        image,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        classes=classes
    )
    inference_time = time.time() - start_time
    
    print(f"Detected {detections['count']} objects in {inference_time:.3f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Save detection results
    output_path = os.path.join(args.output, os.path.basename(image_path))
    
    # Draw bounding boxes on image
    image_with_boxes = draw_boxes(
        image,
        detections,
        show_labels=not args.hide_labels,
        show_confidence=not args.hide_conf
    )
    
    # Save image with detections
    save_image(image_with_boxes, output_path)
    
    # Save detection results to text file if requested
    if args.save_txt:
        txt_path = os.path.join(args.output, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
        with open(txt_path, 'w') as f:
            for i, (box, class_name, confidence) in enumerate(zip(
                detections['boxes'],
                detections['class_names'],
                detections['confidences']
            )):
                x1, y1, x2, y2 = box
                if args.save_conf:
                    f.write(f"{class_name} {confidence:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
                else:
                    f.write(f"{class_name} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    
    return detections


def process_video(video_path, detector, args):
    """
    Process a video file.
    
    Args:
        video_path: Path to the video file
        detector: YOLODetector instance
        args: Command line arguments
    """
    print(f"Processing video: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(args.output, os.path.basename(video_path))
    
    # Filter classes if specified
    classes = None
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    
    # Process video
    detector.detect_video(
        video_path,
        output_path=output_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        classes=classes,
        skip_frames=args.skip_frames
    )


def process_directory(directory_path, detector, args):
    """
    Process all images in a directory.
    
    Args:
        directory_path: Path to the directory
        detector: YOLODetector instance
        args: Command line arguments
    """
    print(f"Processing directory: {directory_path}")
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(directory_path).glob(f'*{ext}')))
        image_paths.extend(list(Path(directory_path).glob(f'*{ext.upper()}')))
    
    if not image_paths:
        print(f"No images found in {directory_path}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Process each image
    for image_path in image_paths:
        process_image(str(image_path), detector, args)


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize detector
    detector = YOLODetector(model_path=args.model, device=args.device)
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"Error: Source {args.source} does not exist")
        return
    
    # Process based on source type
    if os.path.isdir(args.source):
        # Process directory
        process_directory(args.source, detector, args)
    elif os.path.isfile(args.source):
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if any(args.source.lower().endswith(ext) for ext in video_extensions):
            # Process video
            process_video(args.source, detector, args)
        else:
            # Process single image
            process_image(args.source, detector, args)
    else:
        print(f"Error: Source {args.source} is not a valid file or directory")


if __name__ == "__main__":
    main()
