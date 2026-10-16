// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
#include "evp/evp.h"

namespace evp {

static std::string join(const std::string& dir, const std::string& name)
{
    if (dir.empty()) return name;
    char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + name : dir + "/" + name;
}

Model::Model() {}

Model::~Model()
{
    net_.clear();
}

int Model::load(const std::string& bundle_dir, const std::string& precision, int num_threads, bool use_fp16)
{
    err_.clear();
    if (!load_meta(join(bundle_dir, "model.meta"), meta_, &err_)) return -1;
    std::map<std::string, std::pair<std::string, std::string> >::const_iterator it = meta_.files.find(precision);
    if (it == meta_.files.end() || it->second.first.empty() || it->second.second.empty())
    {
        err_ = "bundle has no '" + precision + "' model files";
        return -2;
    }
    post_ = meta_.post;
    net_.clear();
    net_.opt.num_threads = num_threads;
    net_.opt.use_vulkan_compute = false;
    // fp16 off by default: Cortex-A7 has no fp16 arithmetic and results then match the verified fp32 reference
    net_.opt.use_fp16_packed = use_fp16;
    net_.opt.use_fp16_storage = use_fp16;
    net_.opt.use_fp16_arithmetic = use_fp16;
    net_.opt.use_bf16_storage = false;
    if (net_.load_param(join(bundle_dir, it->second.first).c_str()) != 0)
    {
        err_ = "failed to load " + it->second.first;
        return -3;
    }
    if (net_.load_model(join(bundle_dir, it->second.second).c_str()) != 0)
    {
        err_ = "failed to load " + it->second.second;
        return -4;
    }
    return 0;
}

int Model::forward(const ncnn::Mat& in, std::vector<ncnn::Mat>& outputs)
{
    ncnn::Extractor ex = net_.create_extractor();
    if (ex.input(meta_.input.blob.c_str(), in) != 0)
    {
        err_ = "input blob not found: " + meta_.input.blob;
        return -1;
    }
    outputs.resize(meta_.outputs.size());
    for (size_t k = 0; k < meta_.outputs.size(); k++)
    {
        if (ex.extract(meta_.outputs[k].name.c_str(), outputs[k]) != 0)
        {
            err_ = "extract failed: " + meta_.outputs[k].name;
            return -2;
        }
    }
    return 0;
}

int Model::detect(const unsigned char* bgr, int w, int h, std::vector<Detection>& dets, Geometry* geom)
{
    if (!meta_.is_detect())
    {
        err_ = "model is not a detector";
        return -1;
    }
    ncnn::Mat in;
    Geometry g;
    if (preprocess(bgr, w, h, meta_.input, in, g) != 0)
    {
        err_ = "preprocess failed";
        return -2;
    }
    std::vector<ncnn::Mat> outs;
    if (forward(in, outs) != 0) return -3;
    if (geom) *geom = g;
    if (postprocess_detect(outs, meta_, post_, g, dets) != 0)
    {
        err_ = "output shapes do not match model.meta";
        return -4;
    }
    return 0;
}

int Model::classify(const unsigned char* bgr, int w, int h, std::vector<std::pair<int, float> >& topk)
{
    if (meta_.family != "classify")
    {
        err_ = "model is not a classifier";
        return -1;
    }
    ncnn::Mat in;
    Geometry g;
    if (preprocess(bgr, w, h, meta_.input, in, g) != 0)
    {
        err_ = "preprocess failed";
        return -2;
    }
    std::vector<ncnn::Mat> outs;
    if (forward(in, outs) != 0) return -3;
    if (postprocess_classify(outs[0], post_, topk) != 0)
    {
        err_ = "bad classifier output";
        return -4;
    }
    return 0;
}

} // namespace evp
