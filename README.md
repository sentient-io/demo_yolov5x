[SoyNet](https://soynet.io/) is an inference optimizing solution for AI models.

This section describes the process of performing a demo running Yolov5x, one of most famous object detection models.

## SoyNet Overview

### Core technology of SoyNet

- Accelerate model inference by maximizing the utilization of numerous cores on the GPU without compromising accuracy (2x to 5x compared to Tensorflow)
- Minimize GPU memory usage (1/5~1/15 level compared to Tensorflow)

### Benefit of SoyNet

- can support customer to  provide AI applications and AI services in time (Time to Market)
- can help application developers to easily execute AI projects without additional technical AI knowledge and experience
- can help customer to reduce H/W (GPU, GPU server) or Cloud Instance cost for the same AI execution (inference)
- can support customer to respond to real-time environments that require very low latency in AI inference

### Features of SoyNet

- Dedicated engine for inference of deep learning models
- Supports NVIDIA and non-NVIDIA GPUs (based on technologies such as CUDA and OpenCL, respectively)
- library files to be easiliy integrated with customer applications
dll file (Windows), so file (Linux) with header or *.lib for building in C/C++

### Folder Structure

```
   ├─data             : sample data
   ├─lib              : library files
   ├─mgmt             : SoyNet execution env
   │  ├─configs       : model definitions (*.cfg) and trial license
   │  ├─engines       : SoyNet engine files (it's made at the first time execution.
   │  │                 It requires about 30 sec)
   │  ├─logs          : SoyNet log files
   │  └─weights       : weight files for AI models
   ├─samples          : folder to build sample demo
   └─weight_extractor : weight extract files

```

### Demo of Image-Super-Resolution with Yolov5x 

### Prerequisites

### 1.H/W

- GPU : NVIDIA GPU with PASCAL architecture or higher

### 2.S/W

- OS: Ubuntu 18.04LTS
- NVIDIA development environment: CUDA 11.0 / cuDNN 8.0.4 / TensorRT 7.1.3.x
    - For CUDA 11.0, Nvidia-driver 450.36 or higher must be installed
- Others: OpenCV (for reading video files and outputting the screen)

### Run SoyNet Demo

### 1.clone repository

```
$ git clone https://github.com/zaiin4050/demo_yolov5x demo_yolov5x
```

### 2.download pre-trained weight files

Pre-converted SoyNet weight can be downloaded at [HERE](https://github.com/zaiin4050/demo_yolov5x/releases/download/v0.1/yolov5x.weights)

- If you have your own custom pytorch weight file, you can convert pytorch weight file to SoyNet weight file with weight_extractor.py. 
```
$ cd demo_yolov5x/weight_extractor
$ python3 weight_extrator.py
```
* To run weightb_extractor.py, it requires pytorch ([PyTorch Install](https://pytorch.org/get-started/previous-versions/))  

### 3.Demo code Build and Run (Python)

It takes time to create the engine file when it is first executed, and it is loaded immediately after that.

```
$ pip install numpy opencv-python 

$ cd demo_yolov5x/samples && python3 inference.py 
```

