"""
Image processing utilities for object detection.
This module provides functions for processing images before and after object detection.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Union, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (RGB format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image using OpenCV (in BGR format)
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = None,
    max_size: int = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Image as numpy array
        target_size: Target size as (width, height)
        max_size: Maximum size for the largest dimension
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if target_size is None and max_size is None:
        return image
    
    h, w = image.shape[:2]
    
    if max_size is not None:
        # Resize based on max dimension
        if max(h, w) <= max_size:
            return image
        
        if keep_aspect_ratio:
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))
        else:
            new_h = new_w = max_size
    
    elif target_size is not None:
        # Resize to target size
        new_w, new_h = target_size
        
        if keep_aspect_ratio:
            # Calculate scaling factor
            scale = min(new_w / w, new_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
    
    # Perform resize
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1].
    
    Args:
        image: Image as numpy array
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image pixel values from [0, 1] to [0, 255].
    
    Args:
        image: Normalized image as numpy array
        
    Returns:
        Denormalized image
    """
    return (image * 255.0).astype(np.uint8)


def pad_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: Union[int, Tuple[int, int, int]] = 0
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to target size.
    
    Args:
        image: Image as numpy array
        target_size: Target size as (width, height)
        pad_value: Value to pad with
        
    Returns:
        Padded image and padding offsets (x, y)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate padding
    pad_x = max(0, target_w - w)
    pad_y = max(0, target_h - h)
    
    # Calculate padding on each side
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    
    # Pad image
    if len(image.shape) == 3:  # Color image
        padded_image = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )
    else:  # Grayscale image
        padded_image = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )
    
    return padded_image, (pad_left, pad_top)


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0
) -> np.ndarray:
    """
    Crop image based on bounding box.
    
    Args:
        image: Image as numpy array
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Additional padding around the bounding box
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop image
    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0,
    contrast: float = 0
) -> np.ndarray:
    """
    Apply brightness and contrast adjustments to an image.
    
    Args:
        image: Image as numpy array
        brightness: Brightness adjustment (-1 to 1)
        contrast: Contrast adjustment (-1 to 1)
        
    Returns:
        Adjusted image
    """
    # Ensure image is in the correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply brightness
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    # Apply contrast
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        image: Image as numpy array (RGB format)
        
    Returns:
        Grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Image as numpy array
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def save_image(
    image: np.ndarray,
    output_path: str,
    quality: int = 95
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array (RGB format)
        output_path: Path to save the image
        quality: JPEG quality (0-100)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert from RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Set JPEG quality
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    # Save image
    cv2.imwrite(output_path, image, params)
    
    print(f"Image saved to {output_path}")


def create_mosaic(
    images: List[np.ndarray],
    rows: int = None,
    cols: int = None,
    padding: int = 5,
    fill_value: Union[int, Tuple[int, int, int]] = (0, 0, 0)
) -> np.ndarray:
    """
    Create a mosaic from multiple images.
    
    Args:
        images: List of images as numpy arrays
        rows: Number of rows in the mosaic
        cols: Number of columns in the mosaic
        padding: Padding between images
        fill_value: Value to fill padding with
        
    Returns:
        Mosaic image
    """
    if not images:
        raise ValueError("No images provided")
    
    # Determine grid size
    n = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    
    # Ensure all images have the same shape
    shapes = [img.shape for img in images]
    h = max(shape[0] for shape in shapes)
    w = max(shape[1] for shape in shapes)
    channels = max(shape[2] if len(shape) > 2 else 1 for shape in shapes)
    
    # Create mosaic image
    mosaic_h = h * rows + padding * (rows - 1)
    mosaic_w = w * cols + padding * (cols - 1)
    
    if channels == 1:
        mosaic = np.full((mosaic_h, mosaic_w), fill_value, dtype=np.uint8)
    else:
        mosaic = np.full((mosaic_h, mosaic_w, channels), fill_value, dtype=np.uint8)
    
    # Place images in the mosaic
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        r = i // cols
        c = i % cols
        
        y1 = r * (h + padding)
        x1 = c * (w + padding)
        y2 = y1 + h
        x2 = x1 + w
        
        # Resize image if necessary
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        
        # Handle grayscale images
        if len(img.shape) == 2 and channels > 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Place image in mosaic
        mosaic[y1:y2, x1:x2] = img
    
    return mosaic
