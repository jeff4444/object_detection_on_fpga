#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>

#include <linux/fb.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

using namespace std;
using namespace cv;

static void bgr_to_rgb565(const Mat& src, uint16_t* dst, int fb_width, int fb_height) {
    // Resize if needed to match HDMI resolution
    Mat resized;
    if (src.cols != fb_width || src.rows != fb_height) {
        resize(src, resized, Size(fb_width, fb_height));
    } else {
        resized = src;
    }

    // Convert each pixel BGR888 → RGB565
    for (int y = 0; y < fb_height; y++) {
        for (int x = 0; x < fb_width; x++) {
            Vec3b bgr = resized.at<Vec3b>(y, x);
            uint8_t b = bgr[0] >> 3;
            uint8_t g = bgr[1] >> 2;
            uint8_t r = bgr[2] >> 3;
            dst[y * fb_width + x] = (r << 11) | (g << 5) | b;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <model_name>" << endl;
        return -1;
    }

    string model_name = argv[1];
    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    string pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink";
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Error: could not open camera /dev/video0" << endl;
        return -1;
    }

    // Open HDMI framebuffer (/dev/fb0)
    int fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd < 0) {
        perror("Error opening /dev/fb0");
        return -1;
    }

    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo);
    ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo);

    int fb_width = vinfo.xres;
    int fb_height = vinfo.yres;
    size_t screensize = fb_width * fb_height * vinfo.bits_per_pixel / 8;

    uint16_t* fbp = (uint16_t*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if ((intptr_t)fbp == -1) {
        perror("mmap failed");
        close(fb_fd);
        return -1;
    }

    cout << "HDMI output ready (" << fb_width << "x" << fb_height << ", " 
         << vinfo.bits_per_pixel << " bpp)" << endl;

    double fps = 0.0;
    int frame_count = 0;
    double t_start = (double)getTickCount();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Run YOLO detection
        auto results = yolo->run(frame);

        // Draw bounding boxes
        for (auto &box : results.bboxes) {
            int label = box.label;
            float xmin = box.x * frame.cols;
            float ymin = box.y * frame.rows;
            float xmax = xmin + box.width * frame.cols;
            float ymax = ymin + box.height * frame.rows;

            rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0,255,0), 2);
            putText(frame, "ID:" + to_string(label), Point(xmin, ymin - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
        }

        // FPS calculation
        frame_count++;
        double t_now = (double)getTickCount();
        double elapsed = (t_now - t_start) / getTickFrequency();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            t_start = t_now;
            cout << "FPS: " << fps << endl;
        }

        // Convert to framebuffer format (RGB565)
        bgr_to_rgb565(frame, fbp, fb_width, fb_height);
    }

    munmap(fbp, screensize);
    close(fb_fd);
    cap.release();
    return 0;
}