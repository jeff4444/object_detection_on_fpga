#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <xir/graph/graph.hpp>
#include <vart/runner.hpp>

// For simplicity, assume float inputs/outputs. Adjust types (e.g. int8) if needed.
int main(int argc, char** argv) {
    // Path to compiled XMODEL file (pass as argument or hardcode)
    std::string model_file = (argc > 1 ? argv[1] : "model.xmodel");
    // Deserialize XIR graph from the .xmodel file
    auto graph = xir::Graph::deserialize(model_file); 
    auto root = graph->get_root_subgraph();
    // Assume the DPU subgraph is the first child of the root (compiled for DPU)
    // (In practice, use xir utilities to select the correct subgraph.)
    auto subgraphs = xir::Subgraph::get_children(root); 
    if (subgraphs.empty()) {
        std::cerr << "Error: no subgraphs found in model.\n";
        return -1;
    }
    const xir::Subgraph* dpu_subgraph = subgraphs[0];

    // Create a VART runner for the DPU subgraph (mode "run") [oai_citation:4‡docs.amd.com](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Programming-with-VART#:~:text=5,object%2C%20call%20the%20following%3A%20function).
    auto runner = vart::Runner::create_runner(dpu_subgraph, "run");
    if (!runner) {
        std::cerr << "Error: failed to create DPU runner.\n";
        return -1;
    }

    // Get input and output tensor descriptors from the runner [oai_citation:5‡docs.amd.com](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Programming-with-VART#:~:text=Query%20the%20DpuRunner%20for%20the,Tensor%20format%20it%20expects).
    auto input_tensors = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();
    // Check we have at least one input tensor
    if (input_tensors.empty()) {
        std::cerr << "Error: no input tensor found.\n";
        return -1;
    }

    // Prepare CPU-side buffers for inputs. Assume one input tensor (batch=1).
    std::vector<std::unique_ptr<vart::TensorBuffer>> input_buffers;
    std::vector<vart::TensorBuffer*> input_buffer_ptrs;
    // Create a buffer for each input tensor
    for (auto tensor : input_tensors) {
        // Compute total size (in elements) of the tensor
        int64_t num_elements = 1;
        auto shape = tensor->get_shape();  // e.g. [1,3,H,W] for NCHW
        for (auto dim : shape) num_elements *= dim;
        // Allocate a flat float buffer on the CPU
        float* data_ptr = new float[num_elements];
        // Create a VART tensor buffer wrapper around this pointer [oai_citation:6‡docs.amd.com](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Programming-with-VART#:~:text=Query%20the%20DpuRunner%20for%20the,Tensor%20format%20it%20expects).
        input_buffers.emplace_back(
            std::make_unique<vart::CpuFlatTensorBuffer>(static_cast<void*>(data_ptr), tensor));
        input_buffer_ptrs.push_back(input_buffers.back().get());
    }

    // Prepare CPU-side buffers for outputs (one per output tensor).
    std::vector<std::unique_ptr<vart::TensorBuffer>> output_buffers;
    std::vector<vart::TensorBuffer*> output_buffer_ptrs;
    for (auto tensor : output_tensors) {
        int64_t num_elements = 1;
        auto shape = tensor->get_shape();
        for (auto dim : shape) num_elements *= dim;
        float* data_ptr = new float[num_elements];
        output_buffers.emplace_back(
            std::make_unique<vart::CpuFlatTensorBuffer>(static_cast<void*>(data_ptr), tensor));
        output_buffer_ptrs.push_back(output_buffers.back().get());
    }

    // Open the default camera (/dev/video0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open camera\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Assume the model expects input in NCHW format (channels-first) [oai_citation:7‡docs.amd.com](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Programming-with-VART#:~:text=3,TensorFormat%20get_tensor_format).
        // Resize frame to model’s expected width/height.
        int width  = input_tensors[0]->get_shape()[3];  // W dimension
        int height = input_tensors[0]->get_shape()[2];  // H dimension
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(width, height));

        // Copy image data into input buffer (assuming BGR input, no normalization).
        // Input buffer is flat in channel-first order.
        float* input_data = static_cast<float*>(input_buffers[0]->data().first);
        int channels = input_tensors[0]->get_shape()[1]; // typically 3
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                cv::Vec3b pixel = resized.at<cv::Vec3b>(h, w);
                for (int c = 0; c < channels; ++c) {
                    // BGR channel c at location (h,w) -> index in flat buffer
                    input_data[c * (height*width) + h * width + w] = static_cast<float>(pixel[c]);
                }
            }
        }

        // Run inference: asynchronous launch and wait [oai_citation:8‡docs.amd.com](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Programming-with-VART#:~:text=1,input).
        auto job_id = runner->execute_async(input_buffer_ptrs, output_buffer_ptrs).first;
        runner->wait(job_id, -1);

        // Retrieve and process output tensor data.
        // (Model-dependent: parse output_buffers[*]->data().first into detections.)
        // Placeholder example: suppose we have one output with [num_detections, 6] (x, y, w, h, score, class).
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        // TODO: implement real parsing of output buffers here.
        // For example (pseudo-code):
        // float* out_data = static_cast<float*>(output_buffers[0]->data().first);
        // int num_dets = ...; for each detection fill boxes, confidences, class_ids.

        // For demonstration, draw a dummy box if any output exists.
        if (!boxes.empty()) {
            for (size_t i = 0; i < boxes.size(); ++i) {
                cv::rectangle(frame, boxes[i], cv::Scalar(0,255,0), 2);
                cv::putText(frame, std::to_string(class_ids[i]) + ":" + std::to_string(confidences[i]),
                            boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
            }
        }

        // Display the result
        cv::imshow("Object Detection", frame);
        if (cv::waitKey(1) == 27) {  // exit on ESC key
            break;
        }
    }

    return 0;
}
