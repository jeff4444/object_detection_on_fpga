#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import vart
import xir

# Helper: extract the DPU subgraph from the compiled .xmodel
def get_child_subgraph_dpu(graph: "xir.Graph"):
    root = graph.get_root_subgraph()
    if root.is_leaf:
        return []
    subs = root.toposort_child_subgraph()
    return [s for s in subs if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


def main():
    # ----- Command-line arguments -----
    parser = argparse.ArgumentParser(
        description="Run live DPU inference on ZCU104 USB camera input and display via HDMI"
    )
    parser.add_argument(
        "-d", "--device",
        default="/dev/video0",
        help="Video input device (e.g., /dev/video0 or 0 for index)"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the .xmodel file compiled for the DPU"
    )
    args = parser.parse_args()

    # Determine capture source
    source = args.device
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open camera at {source}")
        return

    # ----- Load DPU model and create runner -----
    graph = xir.Graph.deserialize(args.model)
    dpu_subgraphs = get_child_subgraph_dpu(graph)
    if len(dpu_subgraphs) != 1:
        raise RuntimeError("Expected exactly one DPU subgraph in the model.")
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

    # Prepare I/O tensors
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    in_shape = in_tensors[0].dims

    # Determine model input layout
    if len(in_shape) == 4 and (in_shape[-1] in (1, 3)):
        batch, height, width, channels = in_shape
        nchw = False
    else:
        batch, channels, height, width = in_shape
        nchw = True

    # Allocate buffers
    input_data = [np.empty(tuple(in_shape), dtype=np.int8)]
    output_data = [np.empty(tuple(out_tensors[0].dims), dtype=np.int8)]

    print("Starting video stream... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, exiting.")
            break

        # Preprocess
        frame_resized = cv2.resize(frame, (width, height))
        # Convert to NCHW or NHWC
        if nchw:
            arr = np.transpose(frame_resized, (2, 0, 1)).reshape(input_data[0].shape)
        else:
            arr = frame_resized.reshape(input_data[0].shape)
        input_data[0][...] = arr.astype(np.int8)

        # Inference
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)

        # Postprocess (classification example)
        result = output_data[0]
        scores = result[0] if result.ndim == 2 else result.flatten()
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        label = f"Class: {class_id}, Score: {conf:.2f}"
        cv2.putText(frame_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        # Display
        cv2.imshow("Inference", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
