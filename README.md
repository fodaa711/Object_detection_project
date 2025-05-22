# YOLOv8 Object Detection for Self-Driving Cars

A comprehensive implementation of YOLOv8 object detection for autonomous vehicle applications, providing real-time detection of vehicles, pedestrians, traffic signs, and other road elements.

## Overview

This project implements the YOLOv8 (You Only Look Once) object detection model for self-driving car applications. YOLOv8 is a state-of-the-art, real-time object detection system that processes images in a single pass, making it ideal for autonomous vehicle perception systems where speed and accuracy are critical.

The implementation includes:

- Pre-trained YOLOv8 model integration
- Image and video processing pipeline
- Real-time object detection and visualization
- Support for multiple object classes relevant to driving scenarios
- Performance metrics and evaluation tools
- Web-based user interface for interactive detection

## Features

- **Real-time Detection**: Process images and video streams with minimal latency
- **Multi-class Detection**: Identify and classify vehicles, pedestrians, cyclists, traffic signs, and more
- **Bounding Box Visualization**: Clear visual representation of detected objects with confidence scores
- **Customizable Confidence Thresholds**: Adjust detection sensitivity based on application needs
- **Performance Metrics**: Evaluate detection accuracy and processing speed
- **Web Interface**: User-friendly UI for uploading images and viewing detection results

## Requirements

- Python 3.8+
- PyTorch 1.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Flask (for web interface)

A complete list of dependencies is available in the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/object_detection_project.git
   cd object_detection_project
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model weights:
   ```
   # The weights will be downloaded automatically when running the model for the first time
   # Or you can manually download from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   ```

## Usage

### Command Line Interface

Run object detection on a single image:

```python
python scripts/detect.py --source path/to/image.jpg --output path/to/output
```

Run object detection on a video:

```python
python scripts/detect.py --source path/to/video.mp4 --output path/to/output
```

### Web Interface

Start the web application:

```python
python app.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

The web interface allows you to:
- Upload images for object detection
- Adjust confidence thresholds
- View detection results with bounding boxes and labels
- Download processed images with detections

## Project Structure

```
object_detection_project/
├── data/
│   └── README.md                # Instructions for dataset preparation
├── models/
│   ├── __init__.py
│   └── detector.py              # YOLOv8 model wrapper
├── utils/
│   ├── __init__.py
│   ├── visualization.py         # Visualization utilities
│   └── processing.py            # Image processing utilities
├── scripts/
│   ├── detect.py                # Script for running detection
│   └── evaluate.py              # Evaluation script
├── static/                      # Static files for web interface
│   ├── css/
│   ├── js/
│   └── images/
├── templates/                   # HTML templates for web interface
├── app.py                       # Flask web application
├── notebooks/
│   └── object_detection.ipynb   # Jupyter notebook with examples
├── examples/                    # Example images and results
├── requirements.txt             # Project dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Model Details

This project uses YOLOv8m, a medium-sized variant of the YOLOv8 architecture, which offers a good balance between speed and accuracy. The model is pre-trained on the COCO dataset, which includes 80 different object classes.

Key features of YOLOv8:
- Single-stage object detection
- Anchor-free detection
- Advanced backbone network
- Multi-scale feature fusion
- Optimized for real-time performance

## Examples

![Detection Example](examples/detection_example.jpg)

The model can detect various objects relevant to driving scenarios:
- Vehicles (cars, trucks, buses, motorcycles)
- Pedestrians
- Cyclists
- Traffic lights
- Traffic signs
- Road elements

## Performance

YOLOv8m offers the following performance metrics:
- Inference speed: ~35 FPS on GPU (NVIDIA RTX 3080)
- mAP (mean Average Precision): 50.2% on COCO val2017
- Size: 49.7 MB

## Web Interface

The web interface provides an intuitive way to interact with the object detection system:

1. **Upload**: Drag and drop or select images for detection
2. **Configure**: Adjust confidence threshold and other parameters
3. **Detect**: Process the image and view results
4. **Explore**: Interact with detection results
5. **Download**: Save processed images with detection annotations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- The original YOLO paper authors for their groundbreaking work in object detection
- The self-driving car research community for datasets and benchmarks
