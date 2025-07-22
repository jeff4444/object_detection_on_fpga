from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys
import argparse

# Load .xmodel and create DPU runner
def get_dpu_runner(model_path):
    graph = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1
    runner = vart.Runner.create_runner(subgraphs[0], "run")
    return runner

# Extract DPU subgraph
def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    sub = root.toposort_child_subgraph()
    return [s for s in sub if s.has_attr("device") and s.get_attr("device") == "DPU"]

# Preprocess frame for classification model (resize, normalize)
def preprocess(img, input_height, input_width):
    resized = cv2.resize(img, (input_width, input_height))
    resized = resized.astype(np.float32)
    resized -= np.array([123.68, 116.78, 103.94], dtype=np.float32)  # ImageNet mean subtraction
    return resized

# Postprocess output (assuming softmax output for classification)
def postprocess(output_data):
    softmax = np.exp(output_data - np.max(output_data))
    softmax = softmax / np.sum(softmax)
    top_class = np.argmax(softmax)
    confidence = softmax[top_class]
    return top_class, confidence

def main():
    parser = argparse.ArgumentParser(description="Run video through DPU model and display output.")
    parser.add_argument('--model', required=True, help='Path to the .xmodel file')
    parser.add_argument('--input', default='/dev/video0', help='Video input source (e.g., /dev/video0 or 0)')
    args = parser.parse_args()

    model_path = args.model
    video_source = int(args.input) if args.input.isdigit() else args.input

    runner = get_dpu_runner(model_path)
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    # Get input shape
    input_shape = input_tensors[0].dims
    batch_size = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]
    input_channels = input_shape[3]

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        img = preprocess(frame, input_height, input_width)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.transpose(img, (0, 3, 1, 2))  # NHWC to NCHW

        input_data = [np.array(img, dtype=np.float32)]
        output_data = [np.empty(output_tensors[0].dims, dtype=np.float32)]

        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)

        label, conf = postprocess(output_data[0].flatten())
        label_text = f"Label: {label}, Conf: {conf:.2f}"
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("DPU Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
