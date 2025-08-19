#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import xir
import vart
import sys
import traceback

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

    if not use_dummy and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Done single iteration")

if __name__ == "__main__":
    main()
