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

    // Open camera (/dev/video0)
    VideoCapture cap(0); // 0 usually maps to /dev/video0
    if (!cap.isOpened()) {
        cerr << "Error: could not open camera /dev/video0" << endl;
        return -1;
    }

    cout << "Press 'q' to quit." << endl;

    double fps = 0.0;
    int frame_count = 0;
    double t_start = (double)getTickCount();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Run detection
        auto results = yolo->run(frame);

        // Draw bounding boxes
        for (auto &box : results.bboxes) {
            int label = box.label;
            float xmin = box.x * frame.cols + 1;
            float ymin = box.y * frame.rows + 1;
            float xmax = xmin + box.width * frame.cols;
            float ymax = ymin + box.height * frame.rows;

            if (xmin < 0.) xmin = 1.;
            if (ymin < 0.) ymin = 1.;
            if (xmax > frame.cols) xmax = frame.cols;
            if (ymax > frame.rows) ymax = frame.rows;

            float confidence = box.score;

            // Pick color based on class
            Scalar color;
            switch (label % 4) {
                case 0: color = Scalar(0, 255, 0); break;
                case 1: color = Scalar(255, 0, 0); break;
                case 2: color = Scalar(0, 0, 255); break;
                case 3: color = Scalar(0, 255, 255); break;
            }

            rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), color, 2);

            // Label text
            string label_text = "ID: " + to_string(label) +
                                " (" + to_string(int(confidence * 100)) + "%)";
            putText(frame, label_text, Point(xmin, ymin - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // FPS calculation
        frame_count++;
        double t_now = (double)getTickCount();
        double elapsed = (t_now - t_start) / getTickFrequency();
        if (elapsed >= 1.0) { // update fps every second
            fps = frame_count / elapsed;
            frame_count = 0;
            t_start = t_now;
        }

        // Show FPS on frame
        string fps_text = "FPS: " + to_string(int(fps));
        putText(frame, fps_text, Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        // Show frame
        imshow("YOLO Detection", frame);

        // Exit on 'q'
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}