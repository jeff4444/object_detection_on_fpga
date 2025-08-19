#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import xir
import vart
import sys
import traceback
import math

# ---------- Utilities ----------
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

def dtype_to_numpy(xdtype):
    # mapping observed on your board: 'xint8' -> np.int8
    s = str(xdtype).lower()
    if "int8" in s: return np.int8
    if "uint8" in s: return np.uint8
    if "float32" in s: return np.float32
    return np.float32

def print_array_info(name, arr):
    try:
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, itemsize={arr.itemsize}, "
              f"contiguous_c={arr.flags['C_CONTIGUOUS']}, contiguous_f={arr.flags['F_CONTIGUOUS']}, "
              f"strides={arr.strides}, addr=0x{arr.__array_interface__['data'][0]:x}")
    except Exception as e:
        print(f"{name}: (failed to print array info): {e}")

def quantize_with_fixpoint(img_float32, tensor):
    # use fix_point when available (your model showed fix_point:6)
    fix = safe_get_attr(tensor, "fix_point", None)
    if fix is not None:
        scale = 2 ** fix
        q = np.clip(np.round(img_float32 * scale), -128, 127).astype(np.int8)
        return q
    # fallback: direct rounding
    return np.clip(np.round(img_float32), -128, 127).astype(np.int8)

# --- Config: adjust if your model differs ---
ANCHORS = [
    (10,13, 16,30, 33,23),   # scale 0 (80x80)
    (30,61, 62,45, 59,119),  # scale 1 (40x40)
    (116,90, 156,198, 373,326)  # scale 2 (20x20)
]
NUM_CLASSES = 6            # inferred from 33 -> 3*(5+nc) => nc=6
CONF_THRESH = 0.1          # Lowered from 0.3 to catch more detections
NMS_THRESH = 0.45

# ---- utils ----
def dequantize_tensor(int8_arr, xir_tensor_obj):
    """Dequantize int8 array to float32 using fix_point if present."""
    fix = None
    try:
        fix = xir_tensor_obj.get_attr("fix_point")
    except Exception:
        pass
    if fix is None:
        # try the tensor object under the wrapper if available
        try:
            t = xir_tensor_obj
            fix = t.get_attr("fix_point")
        except Exception:
            fix = None
    if fix is not None:
        scale = 2 ** fix
        return int8_arr.astype(np.float32) / float(scale)
    # no fix_point: just cast
    return int8_arr.astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh_to_xyxy(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2

def nms(boxes, scores, iou_threshold):
    """Simple NMS for boxes: boxes = [N,4] in x1,y1,x2,y2 order."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# ---- decode function ----
def decode_yolov5_outputs(output_arrays, out_tensors, input_img_shape):
    """
    output_arrays: list of 3 numpy arrays (int8) from model: shapes:
        [1,80,80,33], [1,40,40,33], [1,20,20,33]
    out_tensors: corresponding tensor metadata objects (to read fix_point)
    input_img_shape: (H, W) of model input (e.g., (640,640))
    Returns list of detections: (class_id, score, x1,y1,x2,y2) in pixel coords.
    """
    H_in, W_in = input_img_shape
    detections = []
    total_candidates = 0
    high_score_candidates = 0

    # For each scale
    for scale_idx, out in enumerate(output_arrays):
        # Dequantize
        tmeta = out_tensors[scale_idx]
        out_f = dequantize_tensor(out, tmeta)  # shape (1, gh, gw, 33)
        out_f = out_f[0]  # remove batch dim -> (gh, gw, 33)
        gh, gw, ch = out_f.shape
        assert ch == 3 * (5 + NUM_CLASSES), f"unexpected channels {ch}"

        print(f"Scale {scale_idx}: grid={gh}x{gw}, dequantized range=[{out_f.min():.3f}, {out_f.max():.3f}]")

        # stride = input_size / grid_size
        stride_h = H_in / gh
        stride_w = W_in / gw
        stride = (stride_w + stride_h) / 2.0  # usually same if square

        # anchors for this scale: 3 pairs
        anchors = ANCHORS[scale_idx]
        anchor_pairs = [(anchors[i*2], anchors[i*2+1]) for i in range(3)]

        # reshape to (gh, gw, 3, 5+nc)
        out_reshaped = out_f.reshape(gh, gw, 3, 5 + NUM_CLASSES)

        for i in range(gh):
            for j in range(gw):
                for a in range(3):
                    vx = out_reshaped[i, j, a]  # (5+nc)
                    tx = vx[0]; ty = vx[1]; tw = vx[2]; th = vx[3]; to = vx[4]
                    cls_logits = vx[5:]  # nc
                    # activations
                    cx = (sigmoid(tx) + j) * stride_w
                    cy = (sigmoid(ty) + i) * stride_h
                    bw = np.exp(tw) * anchor_pairs[a][0]
                    bh = np.exp(th) * anchor_pairs[a][1]
                    objectness = sigmoid(to)
                    class_probs = sigmoid(cls_logits)  # apply sigmoid to class logits
                    class_id = int(np.argmax(class_probs))
                    class_score = float(class_probs[class_id]) * float(objectness)
                    
                    total_candidates += 1
                    if class_score >= CONF_THRESH:
                        high_score_candidates += 1
                        x1, y1, x2, y2 = xywh_to_xyxy(cx, cy, bw, bh)
                        # clip to image
                        x1 = max(0.0, min(W_in - 1.0, x1))
                        y1 = max(0.0, min(H_in - 1.0, y1))
                        x2 = max(0.0, min(W_in - 1.0, x2))
                        y2 = max(0.0, min(H_in - 1.0, y2))
                        detections.append((class_id, class_score, x1, y1, x2, y2))
                        
                        # Print first few detections for debugging
                        if len(detections) <= 3:
                            print(f"  Detection: class={class_id}, score={class_score:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    print(f"Total candidates: {total_candidates}, High score candidates: {high_score_candidates}")

    # now run class-wise NMS
    final_dets = []
    if len(detections) == 0:
        return final_dets

    detections = np.array(detections, dtype=object)  # each row is tuple
    # group per class
    for cls in range(NUM_CLASSES):
        inds = [i for i, d in enumerate(detections) if int(d[0]) == cls]
        if len(inds) == 0:
            continue
        boxes = [tuple(detections[i])[2:] for i in inds]
        scores = [float(detections[i][1]) for i in inds]
        keep = nms(boxes, scores, NMS_THRESH)
        for k in keep:
            idx = inds[k]
            final_dets.append(tuple(detections[idx]))

    # sort by score descending
    final_dets.sort(key=lambda x: x[1], reverse=True)
    return final_dets

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .xmodel")
    p.add_argument("--input", default="0", help="Video source or device")
    args = p.parse_args()

    print("Loading model:", args.model)
    graph = xir.Graph.deserialize(args.model)
    sub = None
    root = graph.get_root_subgraph()
    if safe_has_attr(root, 'device') and root.get_attr('device') == 'DPU':
        sub = [root]
    else:
        sub = root.toposort_child_subgraph()
        sub = [s for s in sub if safe_has_attr(s, "device") and s.get_attr("device") == "DPU"]
    assert len(sub) == 1, "Expecting exactly one DPU subgraph"
    runner = vart.Runner.create_runner(sub[0], "run")

    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    print("========== MODEL IO INFO ==========")
    for i, t in enumerate(in_tensors):
        print(f"INPUT[{i}] name={t.name} dims={t.dims} dtype={t.dtype}")
        fx = safe_get_attr(t, "fix_point", None)
        if fx is not None:
            print("   fix_point:", fx)
    for i, t in enumerate(out_tensors):
        print(f"OUTPUT[{i}] name={t.name} dims={t.dims} dtype={t.dtype}")
    print("===================================")

    # Prepare numpy input/output arrays exactly matching tensor dims & dtype
    input_arrays = []
    output_arrays = []
    for t in in_tensors:
        npdtype = dtype_to_numpy(t.dtype)
        arr = np.empty(tuple(t.dims), dtype=npdtype, order="C")
        input_arrays.append(arr)
    for t in out_tensors:
        npdtype = dtype_to_numpy(t.dtype)
        arr = np.empty(tuple(t.dims), dtype=npdtype, order="C")
        output_arrays.append(arr)

    # Print low-level info to debug
    for i, arr in enumerate(input_arrays):
        print_array_info(f"input_arrays[{i}]", arr)
    for i, arr in enumerate(output_arrays):
        print_array_info(f"output_arrays[{i}]", arr)

    # Determine layout (NHWC vs NCHW) using dims heuristic
    in_t = in_tensors[0]
    dims = in_t.dims
    if len(dims) == 4 and dims[3] == 3:
        layout = "NHWC"
        N, H, W, C = dims
    elif len(dims) == 4 and dims[1] == 3:
        layout = "NCHW"
        N, C, H, W = dims
    else:
        layout = safe_get_attr(in_t, "data_format", "NHWC")
        if layout == "NHWC":
            N, H, W, C = dims
        else:
            N, C, H, W = dims

    print(f"Detected layout={layout} shape={dims}")

    # Open capture
    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open video source:", args.input)
        # we proceed anyway with a single dummy frame to exercise DPU without camera
        use_dummy = True
    else:
        use_dummy = False
        print("Video source opened")

    # single-loop test iteration (to keep prints concise)
    for iteration in range(1):  # do one run; you can expand to continuous later
        if not use_dummy:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera; using a zero frame instead")
                frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        # resize & preprocess
        img = cv2.resize(frame, (W, H)).astype(np.float32)
        # uncomment if model expects RGB:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # example normalization (adapt to training pipeline)
        img = img - np.array([123.68, 116.78, 103.94], dtype=np.float32)

        if layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))
            batched = np.expand_dims(img, 0).copy(order="C")
        else:
            batched = np.expand_dims(img, 0).copy(order="C")

        # Quantize using tensor fix_point (your model has fix_point=6)
        batched_q = quantize_with_fixpoint(batched, in_t)

        # Ensure dtypes match expected numpy dtype
        expected_np_dtype = dtype_to_numpy(in_t.dtype)
        if batched_q.dtype != expected_np_dtype:
            try:
                batched_q = batched_q.astype(expected_np_dtype)
            except Exception as e:
                print("Failed to cast batched_q to expected dtype:", e)

        # final contiguity
        batched_q = np.ascontiguousarray(batched_q)

        # Copy to input array(s)
        input_arrays[0][...] = batched_q
        print_array_info("prepared_input", input_arrays[0])

        # print a small data sample
        flat = input_arrays[0].flatten()
        print("prepared_input sample (first 16 values):", flat[:16].tolist())

        # Attempt synchronous execute() first (older bindings sometimes implement this)
        try:
            print("Trying runner.execute(...) (synchronous) ...")
            if hasattr(runner, "execute"):
                res = runner.execute(input_arrays, output_arrays)
                print("runner.execute returned:", res)
                used_sync = True
            else:
                print("runner.execute not present; will use execute_async")
                used_sync = False
        except Exception as e:
            print("runner.execute raised exception (caught):", repr(e))
            used_sync = False

        if not used_sync:
            # Try execute_async fallback
            try:
                print("Trying runner.execute_async(input_arrays, output_arrays) ...")
                jobid = runner.execute_async(input_arrays, output_arrays)
                print("execute_async returned jobid:", jobid)
                runner.wait(jobid)
                print("wait() finished")
            except Exception as e:
                # Might not catch fatal segfaults, but any Python exception will be printed here
                print("runner.execute_async raised/caught exception:", repr(e))
                # print traceback
                traceback.print_exc()
                print(">>> If you see a kernel crash / segfault here, copy the terminal output up to the crash")
                # stop here
                if not use_dummy and cap.isOpened():
                    cap.release()
                cv2.destroyAllWindows()
                return

        # If we reached here, outputs should be filled — print stats & small sample
        for i, out in enumerate(output_arrays):
            try:
                print(f"OUTPUT[{i}] info:")
                print_array_info(f"output_arrays[{i}]", out)
                flat = out.flatten()
                print(f"output_arrays[{i}] sample (first 32):", flat[:32].tolist())
            except Exception as e:
                print(f"Failed to print OUTPUT[{i}]: {e}")

        # Process YOLOv5 outputs to get detections
        print("\n========== YOLOv5 DETECTIONS ==========")
        try:
            # Print fix_point info for output tensors
            print("Output tensor fix_point info:")
            for i, t in enumerate(out_tensors):
                fx = safe_get_attr(t, "fix_point", None)
                print(f"  OUTPUT[{i}] fix_point: {fx}")
            
            model_input_shape = (H, W)
            dets = decode_yolov5_outputs(output_arrays, out_tensors, model_input_shape)
            print(f"Found {len(dets)} detections:")
            for cls_id, score, x1, y1, x2, y2 in dets:
                print(f"Class {cls_id} score={score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        except Exception as e:
            print(f"Failed to process YOLOv5 outputs: {e}")
            traceback.print_exc()
        print("=====================================\n")

    if not use_dummy and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Done single iteration")

if __name__ == "__main__":
    main()
