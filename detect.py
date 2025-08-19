from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import argparse


# ---------------- Helper functions ----------------
def safe_has_attr(x, name: str) -> bool:
    try:
        return x.has_attr(name)
    except Exception:
        return False


def safe_get_attr(x, name: str, default=None):
    try:
        if safe_has_attr(x, name):
            return x.get_attr(name)
    except Exception:
        pass
    return default


def get_dpu_runner(model_path):
    graph = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1, f"Expected 1 DPU subgraph, got {len(subgraphs)}"
    runner = vart.Runner.create_runner(subgraphs[0], "run")
    return runner


def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    if safe_has_attr(root, "device") and root.get_attr("device") == "DPU":
        return [root]
    sub = root.toposort_child_subgraph()
    return [s for s in sub if safe_has_attr(s, "device") and s.get_attr("device") == "DPU"]


def np_dtype(xir_dtype):
    # Map common XIR dtypes to numpy
    mapping = {
        "INT8": np.int8,
        "xint8": np.int8,      # observed in logs
        "UINT8": np.uint8,
        "FLOAT32": np.float32,
        "xfloat32": np.float32,
        "BF16": np.float16,
    }
    if isinstance(xir_dtype, str):
        return mapping.get(xir_dtype, np.float32)
    # Fallback when dtype is enum-like int (not reliable, but safer than crashing)
    return np.float32


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
    fmt = safe_get_attr(tensor, "data_format", None)
    return fmt if fmt in ("NCHW", "NHWC") else "NCHW"


def quantize_if_needed(img_f32, in_tensor):
    dt = str(in_tensor.dtype)
    is_int8 = ("INT8" in dt.upper()) or ("XINT8" in dt.lower())
    if is_int8:
        # Prefer fix_point if present; otherwise fall back to quantize_info if present.
        fix = safe_get_attr(in_tensor, "fix_point", None)
        if fix is not None:
            scale = 2 ** fix  # int8 = float * 2^fix (Vitis/LRM convention)
            return np.clip(np.round(img_f32 * scale), -128, 127).astype(np.int8)
        qinfo = safe_get_attr(in_tensor, "quantize_info", None)
        if isinstance(qinfo, dict) and "scale" in qinfo:
            scale = qinfo.get("scale", 1.0)
            zero = qinfo.get("zero_point", 0)
            return np.clip(np.round(img_f32 / scale + zero), -128, 127).astype(np.int8)
        # Last resort (shouldn’t happen here since you have fix_point)
        return np.clip(np.round(img_f32), -128, 127).astype(np.int8)
    return img_f32.astype(np.float32)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Run video through DPU model and display output.")
    parser.add_argument('--model', required=True, help='Path to the .xmodel file')
    parser.add_argument('--input', default='/dev/video0', help='Video input source (e.g., /dev/video0 or 0)')
    args = parser.parse_args()

    model_path = args.model
    video_source = int(args.input) if args.input.isdigit() else args.input

    # Create runner & I/O
    runner = get_dpu_runner(model_path)
    in_tensors, out_tensors, input_bufs, output_bufs = make_io_buffers(runner)

    # --- Safe debug: model IO info (no fatal get_attr calls) ---
    print("========== MODEL IO INFO ==========")
    for i, t in enumerate(in_tensors):
        print(f"INPUT[{i}]: name={t.name}, dims={t.dims}, dtype={t.dtype}")
        fx = safe_get_attr(t, "fix_point", None)
        if fx is not None:
            print("   fix_point:", fx)
        if safe_has_attr(t, "quantize_info"):
            # Only print a summary; skip full dict to avoid noise
            qi = safe_get_attr(t, "quantize_info", {})
            keys = list(qi.keys()) if isinstance(qi, dict) else qi
            print("   quantize_info keys:", keys)
    for i, t in enumerate(out_tensors):
        print(f"OUTPUT[{i}]: name={t.name}, dims={t.dims}, dtype={t.dtype}")
    print("===================================")

    in_t = in_tensors[0]
    layout = get_layout(in_t)
    print("Detected input layout:", layout)

    # Get H, W, C
    if layout == "NCHW":
        N, C, H, W = in_t.dims
    else:
        N, H, W, C = in_t.dims

    # Open video
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

        # Resize to model input size (use RGB if your model expects it)
        img = cv2.resize(frame, (W, H))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # uncomment if needed

        # Preprocess (adjust to your training pipeline!)
        img = img.astype(np.float32)
        img = img - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        # Layout
        if layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            batched = np.expand_dims(img, axis=0).copy(order="C")
        else:
            batched = np.expand_dims(img, axis=0).copy(order="C")  # NHWC

        # Quantize if int8
        batched = quantize_if_needed(batched, in_t)

        # Copy into runner input
        input_bufs[0][...] = batched

        # Debug before run
        print("Feeding input:",
              "expected", in_t.dims, str(in_t.dtype),
              "| actual", input_bufs[0].shape, str(input_bufs[0].dtype))

        try:
            job_id = runner.execute_async(input_bufs, output_bufs)
            runner.wait(job_id)
        except Exception as e:
            print("Runner execution failed:", repr(e))
            break

        # Debug outputs
        for i, out in enumerate(output_bufs):
            try:
                print(f"Output[{i}] shape={out.shape}, dtype={out.dtype}, "
                      f"min={out.min()}, max={out.max()}, mean={out.mean()}")
            except Exception as e:
                print(f"Output[{i}] print failed:", repr(e))

        # Placeholder overlay (real YOLOv5 needs proper postprocess; here we just show the stream)
        cv2.putText(frame, "Inference OK", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("DPU Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
