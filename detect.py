import cv2
import numpy as np
import xir
import vart
import argparse

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
    assert len(subgraphs) == 1
    return vart.Runner.create_runner(subgraphs[0], "run")

def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    if safe_has_attr(root, "device") and root.get_attr("device") == "DPU":
        return [root]
    subs = root.toposort_child_subgraph()
    return [s for s in subs if safe_has_attr(s,"device") and s.get_attr("device")=="DPU"]

def np_dtype(xir_dtype):
    mapping = {
        "INT8": np.int8, "xint8": np.int8,
        "UINT8": np.uint8,
        "FLOAT32": np.float32, "xfloat32": np.float32
    }
    if isinstance(xir_dtype,str):
        return mapping.get(xir_dtype,np.float32)
    return np.float32

def make_numpy_io(runner):
    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    input_data  = [np.empty(tuple(t.dims), dtype=np_dtype(t.dtype), order="C") for t in in_tensors]
    output_data = [np.empty(tuple(t.dims), dtype=np_dtype(t.dtype), order="C") for t in out_tensors]
    return in_tensors, out_tensors, input_data, output_data

def get_layout(tensor):
    dims = tensor.dims
    if len(dims)==4:
        if dims[1]==3: return "NCHW"
        if dims[3]==3: return "NHWC"
    fmt = safe_get_attr(tensor,"data_format",None)
    return fmt if fmt in ("NCHW","NHWC") else "NCHW"

def quantize_if_needed(img_f32, in_tensor):
    dt = str(in_tensor.dtype)
    if "INT8" in dt.upper() or "XINT8" in dt.lower():
        fix = safe_get_attr(in_tensor,"fix_point",None)
        if fix is not None:
            scale = 2**fix
            return np.clip(np.round(img_f32*scale),-128,127).astype(np.int8)
        return np.clip(np.round(img_f32),-128,127).astype(np.int8)
    return img_f32.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="/dev/video0")
    args = parser.parse_args()

    runner = get_dpu_runner(args.model)
    in_tensors, out_tensors, input_data, output_data = make_numpy_io(runner)

    print("========== MODEL IO INFO ==========")
    for i,t in enumerate(in_tensors):
        print(f"INPUT[{i}] name={t.name}, dims={t.dims}, dtype={t.dtype}")
        fx = safe_get_attr(t,"fix_point",None)
        if fx is not None: print("   fix_point:",fx)
    for i,t in enumerate(out_tensors):
        print(f"OUTPUT[{i}] name={t.name}, dims={t.dims}, dtype={t.dtype}")
    print("===================================")

    in_t = in_tensors[0]
    layout = get_layout(in_t)
    shape = in_t.dims
    if layout=="NCHW":
        N,C,H,W = shape
    else:
        N,H,W,C = shape

    cap = cv2.VideoCapture(0 if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print("Failed to open video source",args.input)
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        img = cv2.resize(frame,(W,H)).astype(np.float32)
        img = img - np.array([123.68,116.78,103.94],dtype=np.float32)

        if layout=="NCHW":
            img = np.transpose(img,(2,0,1))
            batched = np.expand_dims(img,0).copy(order="C")
        else:
            batched = np.expand_dims(img,0).copy(order="C")

        batched = quantize_if_needed(batched,in_t)
        input_data[0][...] = batched

        print("Feeding input:", shape, str(in_t.dtype), "| actual", input_data[0].shape, input_data[0].dtype)

        try:
            jid = runner.execute_async(input_data, output_data)
            runner.wait(jid)
        except Exception as e:
            print("Runner failed:",repr(e))
            break

        for i,out in enumerate(output_data):
            print(f"Output[{i}] shape={out.shape}, dtype={out.dtype}, "
                  f"min={out.min()}, max={out.max()}, mean={out.mean()}")

        cv2.putText(frame,"Inference OK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("DPU Output",frame)
        if cv2.waitKey(1)==ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
