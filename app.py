Main Flask application for YOLOv8 Object Detection web interface.
"""

import os
import sys
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from models.detector import YOLODetector
from utils.visualization import draw_boxes
from utils.processing import load_image, save_image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize YOLOv8 detector
detector = None


def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_detector():
    """Get or initialize the YOLOv8 detector."""
    global detector
    if detector is None:
        detector = YOLODetector(model_path='yolov8m.pt')
    return detector


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform object detection."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('File type not allowed')
        return redirect(request.url)
    
    # Get confidence threshold from form
    conf_threshold = float(request.form.get('confidence', 0.25))
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save uploaded file
    file.save(filepath)
    
    try:
        # Load image
        image = load_image(filepath)
        
        # Get detector
        detector = get_detector()
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(image, conf_threshold=conf_threshold)
        inference_time = time.time() - start_time
        
        # Draw bounding boxes on image
        image_with_boxes = draw_boxes(image, detections)
        
        # Save result image
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        save_image(image_with_boxes, result_filepath)
        
        # Count objects by class
        object_counts = {}
        for class_name in detections['class_names']:
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1
        
        # Render results page
        return render_template(
            'results.html',
            original_image=url_for('static', filename='uploads/' + filename),
            result_image=url_for('static', filename='results/' + result_filename),
            detection_count=detections['count'],
            object_counts=object_counts,
            inference_time=inference_time,
            confidence=conf_threshold
        )
    
    except Exception as e:
        # Handle errors
        flash(f'Error processing image: {str(e)}')
        return redirect(url_for('index'))


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for object detection."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get confidence threshold from form
    conf_threshold = float(request.form.get('confidence', 0.25))
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save uploaded file
    file.save(filepath)
    
    try:
        # Load image
        image = load_image(filepath)
        
        # Get detector
        detector = get_detector()
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(image, conf_threshold=conf_threshold)
        inference_time = time.time() - start_time
        
        # Draw bounding boxes on image
        image_with_boxes = draw_boxes(image, detections)
        
        # Save result image
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        save_image(image_with_boxes, result_filepath)
        
        # Prepare response
        response = {
            'success': True,
            'detection_count': detections['count'],
            'inference_time': inference_time,
            'original_image': url_for('static', filename='uploads/' + filename, _external=True),
            'result_image': url_for('static', filename='results/' + result_filename, _external=True),
            'detections': []
        }
        
        # Add detection details
        for i, (box, class_name, confidence) in enumerate(zip(
            detections['boxes'],
            detections['class_names'],
            detections['confidences']
        )):
            x1, y1, x2, y2 = box.tolist()
            response['detections'].append({
                'id': i,
                'class': class_name,
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            })
        
        return jsonify(response)
    
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
