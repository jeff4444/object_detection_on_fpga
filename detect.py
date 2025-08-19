from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import argparse
import sys


# ---------------- Helper functions ----------------
def get_dpu_runner(model_path):
    graph = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1, "Expected 1 DPU subgraph, got {}".format(len(subgraphs))
    runner = vart.Runner.create_runner(subgraphs[0], "run")
    return runner


def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    if root.has_attr('device') and root.get_attr('device') == 'DPU':
        return [root]
    sub = root.toposort_child_subgraph()
    return [s for s in sub if s.has_attr("device") and s.get_attr("device") == "DPU"]


def np_dtype(xir_dtype):
    # Try to map XIR dtype to numpy dtype
    mapping = {
        "INT8": np.int8,
        "UINT8": np.uint8,
        "FLOAT32": np.float32,
        "BF16": np.float16
    }
    if isinstance(xir_dtype, str):
        return mapping.get(xir_dtype, np.float32)
    # fallback if enum int
    return np.int8 if int(xir_dtype) == 1 else np.float32


def make_io_buffers(runner):
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    input_bufs = [np.empty(tuple(t.dims), dtype=np_dtype(t.dtype), order="C") for t in in_tensors]
    output_bufs = [np.empty(tuple(t.dims), dtype=np_dtype(t.dtype), order="C") for t in out_tensors]
    return in_tensors, out_tensors, input_bufs, output_bufs


def get_layout(tensor):
    dims = tensor.dims
    if len(dims) == 4:
        if dims[1] == 3:
            return "NCHW"
        if dims[3] == 3:
            return "NHWC"
    try:
        return tensor.get_attr("data_format")
    except:
        return "NCHW"


def quantize_if_needed(img_f32, in_tensor):
    dt = str(in_tensor.dtype)
    if "INT8" in dt or (isinstance(in_tensor.dtype, int) and int(in_tensor.dtype) == 1):
        try:
            fix = in_tensor.get_attr("fix_point")
            scale = 2 ** fix
        except:
            try:
                qinfo = in_tensor.get_attr("quantize_info")
                scale = qinfo.get("scale", 1.0)
                zero = qinfo.get("zero_point", 0)
                return np.clip(np.round(img_f32 / scale + zero), -128, 127).astype(np.int8)
            except:
                scale = 1.0
        return np.clip(np.round(img_f32 * scale), -128, 127).astype(np.int8)
    return img_f32.astype(np.float32)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Run video through DPU model and display output.")
    parser.add_argument('--model', required=True, help='Path to the .xmodel file')
    parser.add_argument('--input', default='/dev/video0', help='Video input source (e.g., /dev/video0 or 0)')
    args = parser.parse_args()

    model_path = args.model
    video_source = int(args.input) if args.input.isdigit() else args.input

    # Create runner
    runner = get_dpu_runner(model_path)
    in_tensors, out_tensors, input_bufs, output_bufs = make_io_buffers(runner)

    # Debug prints: input/output tensors
    print("========== MODEL IO INFO ==========")
    for i, t in enumerate(in_tensors):
        print(f"INPUT[{i}]: name={t.name}, dims={t.dims}, dtype={t.dtype}")
        try:
            print("   fix_point:", t.get_attr("fix_point"))
        except:
            pass
        try:
            print("   quantize_info:", t.get_attr("quantize_info"))
        except:
            pass
    for i, t in enumerate(out_tensors):
        print(f"OUTPUT[{i}]: name={t.name}, dims={t.dims}, dtype={t.dtype}")
    print("===================================")

    in_t = in_tensors[0]
    layout = get_layout(in_t)
    print("Detected input layout:", layout)

    # Get sizes
    if layout == "NCHW":
        N, C, H, W = in_t.dims
    else:
        N, H, W, C = in_t.dims

    # Video input
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_source}")
        return
    print("Video source opened successfully")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            break

        # Resize to model input size
        img = cv2.resize(frame, (W, H))
        img = img.astype(np.float32)
        # Convert BGR->RGB if model trained that way (uncomment if needed)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simple preprocessing (adapt as per training)
        img = img - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        # Layout handling
        if layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))  # HWC->CHW
            batched = np.expand_dims(img, axis=0).copy(order="C")
        else:
            batched = np.expand_dims(img, axis=0).copy(order="C")

        # Quantize if required
        batched = quantize_if_needed(batched, in_t)

        # Copy into runner input
        input_bufs[0][...] = batched

        # Debug before run
        print("Feeding input buf shape:", input_bufs[0].shape, "dtype:", input_bufs[0].dtype)

        try:
            job_id = runner.execute_async(input_bufs, output_bufs)
            runner.wait(job_id)
        except Exception as e:
            print("Runner execution failed:", repr(e))
            break

        # Debug outputs
        for i, out in enumerate(output_bufs):
            print(f"Output[{i}] shape={out.shape}, dtype={out.dtype}, "
                  f"min={out.min()}, max={out.max()}, mean={out.mean()}")

        # For classification-like models (single output)
        logits = output_bufs[0].reshape(-1).astype(np.float32)
        top = int(np.argmax(logits))
        conf = float(np.exp(logits[top] - np.max(logits)) / np.sum(np.exp(logits - np.max(logits))))

        label_text = f"Label: {top}, Conf: {conf:.2f}"
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("DPU Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
