# NEFELI

NEFELI is a deep-learning detection and tracking pipeline designed to enhance autonomy in Advanced Air Mobility (AAM). The system utilizes optical sensors to detect and track non-cooperative aerial vehicles, integrating innovative techniques to improve detection capabilities for distant, small aerial objects.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Example Images](#example-images)
- [Citation](#citation)

## Introduction

NEFELI is designed to provide efficient detection and accurate collision estimation for non-cooperative aerial vehicles, a crucial aspect for the realization of fully autonomous aircraft in AAM. The system employs an enhanced YOLOv5-large model and integrates a unique sliced inference step to improve detection of distant and small objects. The tracking component uses a large-scale re-identification (Re-ID) dataset of aerial objects and combines a deep learning appearance model with a Kalman Filter-based motion model for precise tracking.

## Features

- **Enhanced YOLOv5-large Model**: Improved detection capabilities with a sliced inference step.
- **Large-Scale Re-ID Dataset**: First dataset of aerial objects used to train the appearance model.
- **Fused Tracking Model**: Combines deep learning appearance model with Kalman Filter-based motion model.
- **Edge Implementation**: Designed for efficient real-time implementation on Nvidia Jetson Endge GPUs.

## System Architecture

The NEFELI system architecture is designed to minimize latency and maximize processing efficiency for real-time detection and tracking of aerial objects. The architecture includes several key components:

### NEFELI System Pipeline


1. Input Buffering: Images are captured from the camera and stored in a series of ring buffers to handle the processing speed disparity between the camera capture rate and NEFELI’s processing speed.

2. Preprocessing: The images undergo preprocessing steps including color correction, channel reordering, and slicing into smaller sub-images to enhance detection accuracy for small objects.

3. Detection: The preprocessed images are fed into the YOLOv5-large detector, which uses GPU acceleration to perform inference on each image slice. The slices are then reassembled into the original image dimensions.

4. Tracking: Detected objects are passed to the tracking module. High-confidence detections use the deep learning appearance Re-ID model, while low-confidence detections use the Kalman Filter-based motion model.

5. Annotation: The final output frames are annotated with bounding boxes and track IDs, and the annotated images are sent over the network or stored for further analysis.

![nefeli_pipeline](https://github.com/hellenic-drones/nefeli/assets/24351757/2097e45d-b407-432c-8a4f-66bb66367ab8)


## Installation

### Prerequisites
```
Python 3.8+
PyTorch
OpenCV
NVIDIA CUDA
ONNX Runtime
```
### Clone the Repository
```
git clone [https://github.com/yourusername/nefeli.git](https://github.com/hellenic-drones/nefeli)
cd nefeli
```

### Download the pretrained Models

Download the pre-trained models from the following links and place them in the `models` directories respectively.

- [Nefeli Models]()

## Usage

### Running the NEFELI Pipeline

To run the entire NEFELI pipeline on a sample video, use the following command:

```
python nefeli_pipeline.py --input path_to_input_video --output path_to_output_video
```

### Real-Time Detection and Tracking

For real-time detection and tracking using a connected camera:

```
python nefeli_pipeline.py --camera 0
```

## Example Images
- Sliced Inference Example
![sliced_real](https://github.com/hellenic-drones/nefeli/assets/24351757/1f4e7511-b211-41d0-99d3-cee8e3931f7c)

- Real life Examples

Direct Sunlight            |  Close Distance | Side Approach 
:-------------------------:|:-------------------------:|:-------------------------:
  ![](https://github.com/hellenic-drones/nefeli/assets/24351757/d5f895e6-a7d7-46c7-b686-5afabb6d5070") | ![](https://github.com/hellenic-drones/nefeli/assets/24351757/6850c6b0-1dce-42ec-98cf-d33811f7b10c) | ![](https://github.com/hellenic-drones/nefeli/assets/24351757/e06e763a-fd8f-44bc-b355-596e13e523b5)


## Citation
If you find Nefeli usefull, please use the following to cite the original [Nefeli Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4674579)
```
@article{arsenos2023nefeli,
  title={NEFELI: A Deep-Learning Detection and Tracking Pipeline for Enhancing Autonomy in Advanced Air Mobility},
  author={Anastasios Arsenos, Evangelos Petrongonas, Orfeas Filippopoulos, Christos Skliros, Dimitrios Kollias, Stefanos Kollias},
  journal={Aerospace Science and Technology},
  year={2023},
  url={https://dx.doi.org/10.2139/ssrn.4674579}
}
``
