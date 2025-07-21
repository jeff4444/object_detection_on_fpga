#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vart/runner.hpp>
#include <vart/runner_helper.hpp> // Needed for CpuFlatTensorBuffer
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <xir/subgraph.hpp>
#include <xir/graph/subgraph.hpp>

#include <runner_helper.hpp>  // Optional, depending on install

std::vector<std::unique_ptr<vart::Runner>> get_runners(const std::string& model_path) {
    auto graph = xir::Graph::deserialize(model_path);
    auto root = graph.get_root_subgraph();

    // Get child subgraphs
    auto children = root->get_children();
    std::vector<std::unique_ptr<vart::Runner>> runners;

    for (auto& child : children) {
        if (child->has_attr("device") && child->get_attr<std::string>("device") == "DPU") {
            runners.emplace_back(vart::Runner::create_runner(child.get(), "run"));
        }
    }

    return runners;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model.xmodel>\n";
        return -1;
    }

    std::string model_path = argv[1];

    auto runners = get_runners(model_path);
    if (runners.empty()) {
        std::cerr << "Error: No DPU subgraph found in model.\n";
        return -1;
    }

    auto& runner = runners[0];
    auto input_tensors = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();

    auto input_tensor = input_tensors[0];
    int height = input_tensor->get_shape()[1];
    int width = input_tensor->get_shape()[2];

    std::cout << "Model Input: " << width << "x" << height << "\n";

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera\n";
        return -1;
    }

    cv::Mat frame, resized;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, resized, cv::Size(width, height));

        std::vector<uchar> input_data(resized.total() * resized.elemSize());
        std::memcpy(input_data.data(), resized.data, input_data.size());

        // Allocate input/output buffer
        auto input_buffer = std::make_unique<vart::CpuFlatTensorBuffer>(
            reinterpret_cast<void*>(input_data.data()), input_tensor);

        std::vector<vart::TensorBuffer*> inputs = {input_buffer.get()};
        std::vector<std::unique_ptr<vart::TensorBuffer>> output_buffers;
        std::vector<vart::TensorBuffer*> outputs;

        for (auto& ot : output_tensors) {
            auto shape = ot->get_shape();
            size_t size = ot->get_element_num() * ot->get_data_type().bit_width / 8;
            auto output_data = new uint8_t[size];
            auto buffer = std::make_unique<vart::CpuFlatTensorBuffer>(
                reinterpret_cast<void*>(output_data), ot);
            outputs.push_back(buffer.get());
            output_buffers.emplace_back(std::move(buffer));
        }

        // Run inference
        auto job_id = runner->execute_async(inputs, outputs);
        runner->wait(job_id.first, -1);

        // Handle output (print first 5 values for now)
        float* result = reinterpret_cast<float*>(outputs[0]->data().first);
        std::cout << "Output: ";
        for (int i = 0; i < 5; ++i) std::cout << result[i] << " ";
        std::cout << std::endl;

        cv::imshow("Camera", frame);
        if (cv::waitKey(10) == 27) break; // Press ESC to quit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
