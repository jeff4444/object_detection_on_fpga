#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <model_name>" << endl;
        return -1;
    }

    string model_name = argv[1];

    // Load YOLO model
    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    string pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink";

    // Open camera (/dev/video0)
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Error: could not open camera /dev/video0" << endl;
        return -1;
    }

    cout << "Running raw inference benchmark. Press Ctrl+C to stop." << endl;

    int frame_count = 0;
    double t_start = (double)getTickCount();
    double fps = 0.0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Run YOLO inference only
        auto results = yolo->run(frame);

        // Increment frame counter
        frame_count++;

        // Calculate FPS every 1 second
        double t_now = (double)getTickCount();
        double elapsed = (t_now - t_start) / getTickFrequency();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            cout << "Current FPS: " << fps << endl;
            frame_count = 0;
            t_start = t_now;
        }
    }

    cap.release();
    return 0;
}