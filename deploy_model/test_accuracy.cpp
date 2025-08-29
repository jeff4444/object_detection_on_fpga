#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/yolov3.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

struct Box {
    int label;
    float xmin, ymin, xmax, ymax;
    float score; // prediction confidence
};

// Compute IoU between two boxes
float iou(const Box& a, const Box& b) {
    float xx1 = max(a.xmin, b.xmin);
    float yy1 = max(a.ymin, b.ymin);
    float xx2 = min(a.xmax, b.xmax);
    float yy2 = min(a.ymax, b.ymax);

    float w = max(0.0f, xx2 - xx1);
    float h = max(0.0f, yy2 - yy1);
    float inter = w * h;

    float areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    return inter / (areaA + areaB - inter + 1e-6);
}

// Read YOLO-format ground truth labels
vector<Box> read_labels(const string& label_file, int img_w, int img_h) {
    vector<Box> gts;
    ifstream f(label_file);
    string line;
    while (getline(f, line)) {
        stringstream ss(line);
        int cls; float cx, cy, w, h;
        ss >> cls >> cx >> cy >> w >> h;

        Box b;
        b.label = cls;
        b.xmin = (cx - w/2) * img_w;
        b.ymin = (cy - h/2) * img_h;
        b.xmax = (cx + w/2) * img_w;
        b.ymax = (cy + h/2) * img_h;
        b.score = 1.0f; // ground truth has score 1
        gts.push_back(b);
    }
    return gts;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <model_name> <dataset_dir>" << endl;
        return -1;
    }

    string model_name = argv[1];
    string dataset_dir = argv[2];

    auto yolo = vitis::ai::YOLOv3::create(model_name, true);

    float iou_threshold = 0.5;

    string images_dir = dataset_dir + "/images";
    string labels_dir = dataset_dir + "/labels";

    // per-class storage
    map<int, vector<float>> class_confidences;
    map<int, vector<int>> class_tp_fp;
    map<int, int> class_fn;

    for (auto& entry : fs::directory_iterator(images_dir)) {
        if (entry.path().extension() == ".jpg") {
            string img_path = entry.path().string();
            string label_path = labels_dir + "/" + entry.path().stem().string() + ".txt";

            Mat img = imread(img_path);
            if (img.empty()) continue;

            auto gts = read_labels(label_path, img.cols, img.rows);
            auto results = yolo->run(img);

            vector<Box> preds;
            for (auto& b : results.bboxes) {
                Box pb;
                pb.label = b.label;
                pb.xmin = b.x * img.cols;
                pb.ymin = b.y * img.rows;
                pb.xmax = pb.xmin + b.width * img.cols;
                pb.ymax = pb.ymin + b.height * img.rows;
                pb.score = b.score;
                preds.push_back(pb);
            }

            // Sort predictions by confidence descending
            sort(preds.begin(), preds.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

            vector<bool> matched(gts.size(), false);

            for (auto& p : preds) {
                bool found_match = false;
                for (size_t i = 0; i < gts.size(); i++) {
                    if (p.label == gts[i].label && !matched[i] && iou(p, gts[i]) >= iou_threshold) {
                        matched[i] = true;
                        found_match = true;
                        break;
                    }
                }
                class_confidences[p.label].push_back(p.score);
                class_tp_fp[p.label].push_back(found_match ? 1 : 0); // 1=TP,0=FP
            }

            // count unmatched ground truth boxes as FN
            for (size_t i = 0; i < gts.size(); i++) {
                if (!matched[i]) class_fn[gts[i].label]++;
            }
        }
    }

    // Compute per-class precision, recall, F1
    float sum_ap = 0;
    int num_classes = class_confidences.size();

    cout << "Per-class metrics:" << endl;
    for (auto& kv : class_confidences) {
        int cls = kv.first;
        auto& scores = kv.second;
        auto& tp_fp = class_tp_fp[cls];

        int TP = count(tp_fp.begin(), tp_fp.end(), 1);
        int FP = count(tp_fp.begin(), tp_fp.end(), 0);
        int FN = class_fn[cls];

        float precision = TP / float(TP + FP + 1e-6);
        float recall = TP / float(TP + FN + 1e-6);
        float f1 = 2 * precision * recall / (precision + recall + 1e-6);

        cout << "Class " << cls << ": Precision=" << precision
             << ", Recall=" << recall << ", F1=" << f1 << endl;

        // VOC mAP approximation: AP = TP / (TP + FP + FN)
        float ap = TP / float(TP + FP + FN + 1e-6);
        sum_ap += ap;
    }

    float mAP = sum_ap / num_classes;
    cout << "VOC-style mAP @0.5 IoU: " << mAP << endl;

    return 0;
}