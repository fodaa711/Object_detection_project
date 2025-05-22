"""
YOLOv8 detector module for object detection.
This module provides a wrapper around the YOLOv8 model for easy use in object detection tasks.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Union, Optional


class YOLODetector:
    """
    A wrapper class for YOLOv8 object detection model.
    """
    def __init__(self, model_path: str = 'yolov8m.pt', device: Optional[str] = None):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            model_path: Path to the YOLOv8 model weights (.pt file)
            device: Device to run the model on ('cuda' or 'cpu'). If None, automatically selects.
        """
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load the model
        self.model = self._load_model()
        
        # COCO class names
        self.class_names = self.model.names
        
    def _load_model(self) -> YOLO:
        """
        Load the YOLOv8 model.
        
        Returns:
            YOLO model instance
        """
        try:
            model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect(self, 
               image: Union[str, np.ndarray], 
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               classes: Optional[List[int]] = None) -> Dict:
        """
        Perform object detection on an image.
        
        Args:
            image: Path to image file or numpy array of image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of classes to detect (None for all classes)
            
        Returns:
            Dictionary containing detection results
        """
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image, 
                            conf=conf_threshold, 
                            iou=iou_threshold, 
                            classes=classes)
        
        return self._process_results(results, image)
    
    def _process_results(self, results, original_image: np.ndarray) -> Dict:
        """
        Process the raw results from YOLOv8 model.
        
        Args:
            results: Raw results from YOLOv8 model
            original_image: Original image as numpy array
            
        Returns:
            Dictionary with processed detection results
        """
        result = results[0]  # Get the first result (batch size is 1)
        
        # Extract boxes, confidence scores, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Create a dictionary with detection results
        detections = {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids,
            'class_names': [self.class_names[class_id] for class_id in class_ids],
            'image_shape': original_image.shape,
            'count': len(boxes)
        }
        
        return detections
    
    def detect_video(self, 
                    video_path: str, 
                    output_path: Optional[str] = None,
                    conf_threshold: float = 0.25,
                    iou_threshold: float = 0.45,
                    classes: Optional[List[int]] = None,
                    skip_frames: int = 0) -> None:
        """
        Perform object detection on a video.
        
        Args:
            video_path: Path to video file
            output_path: Path to save the output video (None to not save)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of classes to detect (None for all classes)
            skip_frames: Number of frames to skip between detections
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if specified
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    if writer:
                        writer.write(frame)
                    continue
                
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                results = self.model(frame_rgb, 
                                    conf=conf_threshold, 
                                    iou=iou_threshold, 
                                    classes=classes)
                
                # Convert back to BGR for OpenCV
                result_frame = results[0].plot()
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to output video
                if writer:
                    writer.write(result_frame)
                
                # Display progress
                if frame_count % 10 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")
        
        finally:
            # Release resources
            cap.release()
            if writer:
                writer.release()
            
            print(f"Video processing complete. Output saved to {output_path}")
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get the mapping of class IDs to class names.
        
        Returns:
            Dictionary mapping class IDs to class names
        """
        return self.class_names
    
    def count_objects(self, detections: Dict) -> Dict[str, int]:
        """
        Count the number of detected objects by class.
        
        Args:
            detections: Detection results from detect() method
            
        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for class_id, class_name in zip(detections['class_ids'], detections['class_names']):
            if class_name in counts:
                counts[class_name] += 1
            else:
                counts[class_name] = 1
        
        return counts
    
    def filter_detections(self, 
                         detections: Dict, 
                         classes: List[str] = None, 
                         min_conf: float = 0.0) -> Dict:
        """
        Filter detections by class and confidence.
        
        Args:
            detections: Detection results from detect() method
            classes: List of class names to keep (None for all)
            min_conf: Minimum confidence threshold
            
        Returns:
            Filtered detection results
        """
        if classes is None and min_conf <= 0:
            return detections
        
        # Initialize lists for filtered results
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_class_names = []
        
        for i, (box, conf, class_id, class_name) in enumerate(zip(
            detections['boxes'], 
            detections['confidences'], 
            detections['class_ids'], 
            detections['class_names']
        )):
            # Check confidence threshold
            if conf < min_conf:
                continue
                
            # Check class filter
            if classes is not None and class_name not in classes:
                continue
                
            # Add to filtered results
            filtered_boxes.append(box)
            filtered_confidences.append(conf)
            filtered_class_ids.append(class_id)
            filtered_class_names.append(class_name)
        
        # Create filtered detections dictionary
        filtered_detections = {
            'boxes': np.array(filtered_boxes) if filtered_boxes else np.array([]),
            'confidences': np.array(filtered_confidences) if filtered_confidences else np.array([]),
            'class_ids': np.array(filtered_class_ids) if filtered_class_ids else np.array([]),
            'class_names': filtered_class_names,
            'image_shape': detections['image_shape'],
            'count': len(filtered_boxes)
        }
        
        return filtered_detections
