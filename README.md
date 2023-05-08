# YOLOv8 inference using Python

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8) 
implemented on [Python](https://www.python.org) via [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html).

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_python.git`
* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`

## Run

Execute:

```
python3 object_detector.py
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes of all  objects detected on it.