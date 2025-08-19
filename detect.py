from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import argparse

# --------- tiny helpers (crash-safe attr access) ----------
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
    return vart.Runner.create_runner(subgraphs[0], "run")

def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    assert root is not None
    if safe_has_attr(root, "device") and root.get_attr("device") == "DPU":
        return [root]
    subs = root.toposort_child_subgraph()
    return [s for s in subs if safe_has_attr(s, "device") and s.get_attr("device") == "DPU"]

def get_layout(tensor):
    dims = tensor.dims
    if len(dims) == 4:
        if dims[1] == 3: return "NCHW"
        if dims[3] == 3: return "NHWC"
    fmt = safe_get_attr(tensor, "data_format", None)
    return fmt if fmt in ("NCHW","NHWC") else "NCHW"

def quantize_if_needed(img_f32, in_tensor):
    # your input is xint8 with fix_point on this model
    dt = str(in_tensor.get_tensor().get_data_type()) if hasattr(in_tensor, "get_tensor") else str(in_tensor.dtype)
    is_int8 = ("INT8" in dt.upper()) or ("XINT8" in dt.lower())
    if is_int8:
        fix = safe_get_attr(in_tensor.get_tensor() if hasattr(in_tensor,"get_tensor") else in_tensor, "fix_point", None)
        if fix is not None:
            scale = 2 ** fix  # Vitis convention
            return np.clip(np.round(img_f32 * scale), -128, 127).astype(np.int8)
        # fallback: leave as-is (shouldn't hit for your model)
        return np.clip(np.round(img_f32), -128, 127).astype(np.int8)
    return img_f32.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Run video through DPU model and display output.")
    parser.add_argument('--model', required=True, help='Path to the .xmodel file')
    parser.add_argument('--input', default='/dev/video0', help='Video input source (e.g., /dev/video0 or 0)')
    args = parser.parse_args()

    runner = get_dpu_runner(args.model)

    # IMPORTANT: use TensorBuffers from the runner (not raw numpy)
    inputs_tb  = runner.get_inputs()
    outputs_tb = runner.get_outputs()

    # Crash-safe model I/O info
    print("========== MODEL IO INFO ==========")
    for i, tb in enumerate(inputs_tb):
        t = tb.get_tensor()
        print(f"INPUT[{i}]: name={t.get_name()}, dims={t.get_shape()}, dtype={t.get_data_type()}")
        fx = safe_get_attr(t, "fix_point", None)
        if fx is not None:
            print("   fix_point:", fx)
        if safe_has_attr(t, "quantize_info"):
            qi = safe_get_attr(t, "quantize_info", {})
            keys = list(qi.keys()) if isinstance(qi, dict) else qi
            print("   quantize_info keys:", keys)
    for i, tb in enumerate(outputs_tb):
        t = tb.get_tensor()
        print(f"OUTPUT[{i}]: name={t.get_name()}, dims={t.get_shape()}, dtype={t.get_data_type()}")
    print("===================================")

    # Determine layout and shape
    in_tensor = inputs_tb[0].get_tensor()
    layout = get_layout(in_tensor)
    print("Detected input layout:", layout)
    shape = in_tensor.get_shape()
    if layout == "NCHW":
        N, C, H, W = shape
    else:
        N, H, W, C = shape

    # Create NumPy views over the TensorBuffers (this is how we read/write)
    in_views  = [np.asarray(tb) for tb in inputs_tb]
    out_views = [np.asarray(tb) for tb in outputs_tb]

    # Open video
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Failed to open video source: {src}")
        return
    print("Video source opened successfully")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to grab frame.")
            break

        # Preprocess (adjust to your training if needed)
        img = cv2.resize(frame, (W, H))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # uncomment if model expects RGB
        img = img.astype(np.float32)
        img = img - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        if layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            batched = np.expand_dims(img, 0).copy(order="C")
        else:
            batched = np.expand_dims(img, 0).copy(order="C")  # NHWC

        # Quantize for INT8 using fix_point
        batched = quantize_if_needed(batched, inputs_tb[0])

        # Write into the TensorBuffer memory via the NumPy view
        in_views[0][...] = batched

        # Debug before run
        print("Feeding input:",
              "expected", shape, str(in_tensor.get_data_type()),
              "| actual", in_views[0].shape, str(in_views[0].dtype))

        try:
            job_id = runner.execute_async(inputs_tb, outputs_tb)  # <-- pass TensorBuffers
            runner.wait(job_id)
        except Exception as e:
            print("Runner execution failed:", repr(e))
            break

        # Output stats (won’t crash if an output is empty)
        for i, v in enumerate(out_views):
            try:
                print(f"Output[{i}] shape={v.shape}, dtype={v.dtype}, "
                      f"min={v.min()}, max={v.max()}, mean={v.mean()}")
            except Exception as e:
                print(f"Output[{i}] print failed:", repr(e))

        # Simple overlay to confirm we’re alive (proper YOLO postprocess can be added later)
        cv2.putText(frame, "Inference OK", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("DPU Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
