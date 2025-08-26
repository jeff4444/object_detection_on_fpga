#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <model_name> <image_file>" << endl;
        return -1;
    }

    string model_name = argv[1];
    string image_file = argv[2];

    // Load YOLO model
    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    // Load image
    Mat img = imread(image_file);
    if (img.empty()) {
        cerr << "Error: could not load image " << image_file << endl;
        return -1;
    }

    // Run detection
    auto results = yolo->run(img);

    // Draw bounding boxes
    for (auto &box : results.bboxes) {
        int label = box.label;
        float xmin = box.x * img.cols + 1;
        float ymin = box.y * img.rows + 1;
        float xmax = xmin + box.width * img.cols;
        float ymax = ymin + box.height * img.rows;

        if (xmin < 0.) xmin = 1.;
        if (ymin < 0.) ymin = 1.;
        if (xmax > img.cols) xmax = img.cols;
        if (ymax > img.rows) ymax = img.rows;

        float confidence = box.score;

        cout << "RESULT: label=" << label
             << " xmin=" << xmin
             << " ymin=" << ymin
             << " xmax=" << xmax
             << " ymax=" << ymax
             << " score=" << confidence << endl;

        // Pick color based on class
        Scalar color;
        switch (label % 4) { // cycle through 4 colors
            case 0: color = Scalar(0, 255, 0); break;   // green
            case 1: color = Scalar(255, 0, 0); break;   // blue
            case 2: color = Scalar(0, 0, 255); break;   // red
            case 3: color = Scalar(0, 255, 255); break; // yellow
        }

        rectangle(img, Point(xmin, ymin), Point(xmax, ymax), color, 2);

        // Optionally add label text
        string label_text = "ID: " + to_string(label) +
                            " (" + to_string(int(confidence * 100)) + "%)";
        putText(img, label_text, Point(xmin, ymin - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    // Save output
    string out_file = "result_" + image_file;
    imwrite(out_file, img);
    cout << "Result saved to " << out_file << endl;

    return 0;
}
