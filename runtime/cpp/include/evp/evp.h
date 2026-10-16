// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// evp runtime: meta-driven pre/post-processing around an ncnn model.
// Mirrors evp/deploy/{meta,preprocess,postprocess}.py; any change there must be made here too
// (tests/test_cpp_consistency.py compares both on the same images).
#ifndef EVP_EVP_H
#define EVP_EVP_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "net.h"

namespace evp {

struct InputSpec
{
    int w = 0, h = 0;
    std::string blob = "in0";
    std::string color = "RGB";      // RGB | BGR (channel order fed to the network)
    std::string resize = "letterbox"; // letterbox | stretch | center_crop
    int pad_value = 114;
    float mean[3] = {0.f, 0.f, 0.f};
    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
};

struct OutputSpec
{
    std::string name;
    int stride = 0;
    std::vector<int> shape;         // (c, h, w) or (w)
};

struct PostSpec
{
    std::string method = "nms";     // nms | e2e_topk | topk | none
    double conf = 0.25, iou = 0.45;
    int max_det = 100;
    bool agnostic = false;
    int topk = 5;
    bool softmax = false;
};

struct Meta
{
    int schema_version = 0;
    std::string name, family;       // family: yolo | classify | raw
    InputSpec input;
    int reg_max = 16;
    int num_classes = 0;
    std::vector<OutputSpec> outputs;
    PostSpec post;
    std::vector<std::string> labels;
    std::map<std::string, std::pair<std::string, std::string> > files; // precision -> (param, bin)

    bool is_detect() const { return family == "yolo"; }
};

// Parse model.meta (flat key=value). Returns false and fills *err on failure.
// Numbers are parsed in the "C" locale regardless of the process locale (the writer always uses '.').
bool load_meta(const std::string& path, Meta& meta, std::string* err);
bool parse_meta(const std::string& text, Meta& meta, std::string* err);
// Locale-independent strict number parsing (whole string must be consumed; ints must fit in int).
bool parse_double(const std::string& s, double& v);
bool parse_int(const std::string& s, int& v);

// Largest accepted image (bytes of interleaved BGR); keeps w*h*3 and ncnn's int strides from overflowing.
const long long kMaxImageBytes = 1LL << 30;
bool image_size_ok(int w, int h);
// Binary PPM (P6, maxval 255) -> BGR interleaved.
bool load_ppm(const std::string& path, std::vector<unsigned char>& bgr, int& w, int& h, std::string* err = 0);

struct Geometry
{
    int src_w = 0, src_h = 0, dst_w = 0, dst_h = 0;
    int resize_w = 0, resize_h = 0;
    double scale_x = 1.0, scale_y = 1.0;
    int pad_left = 0, pad_top = 0, pad_right = 0, pad_bottom = 0;
    int crop_x = 0, crop_y = 0;
};

Geometry letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h);
Geometry stretch_geometry(int src_w, int src_h, int dst_w, int dst_h);
Geometry center_crop_geometry(int src_w, int src_h, int dst_w, int dst_h);
bool make_geometry(const InputSpec& spec, int src_w, int src_h, Geometry& g);

// BGR uint8 interleaved (stride = w*3) -> normalized CHW float Mat. Returns 0 on success.
int preprocess(const unsigned char* bgr, int w, int h, const InputSpec& spec, ncnn::Mat& out, Geometry& g);

struct Detection
{
    float x1, y1, x2, y2, score;
    int cls;
};

// outputs: raw head tensors in meta order. Detections are in source-image pixels.
int postprocess_detect(const std::vector<ncnn::Mat>& outputs, const Meta& meta, const PostSpec& post,
                       const Geometry& g, std::vector<Detection>& dets);
int postprocess_classify(const ncnn::Mat& output, const PostSpec& post, std::vector<std::pair<int, float> >& topk);

class Model
{
public:
    Model();
    ~Model();
    // Loads <bundle_dir>/model.meta and the param/bin of `precision` (fp32 | int8).
    int load(const std::string& bundle_dir, const std::string& precision = "fp32", int num_threads = 1,
             bool use_fp16 = false);
    const Meta& meta() const { return meta_; }
    PostSpec& post() { return post_; }
    const std::string& error() const { return err_; }

    int forward(const ncnn::Mat& in, std::vector<ncnn::Mat>& outputs);
    int detect(const unsigned char* bgr, int w, int h, std::vector<Detection>& dets, Geometry* g = 0);
    int classify(const unsigned char* bgr, int w, int h, std::vector<std::pair<int, float> >& topk);

private:
    Meta meta_;
    PostSpec post_;
    ncnn::Net net_;
    std::string err_;
};

} // namespace evp

#endif // EVP_EVP_H
