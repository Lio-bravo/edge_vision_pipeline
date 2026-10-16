// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// evp_demo: run a deploy bundle on images and print one JSON object per image.
//
//   evp_demo <bundle_dir> <image.ppm|image.jpg ...> [--precision fp32|int8] [--threads N] [--conf X]
//            [--iou X] [--fp16] [--bench N] [--dump-input DIR]
//
// .ppm (P6) is always supported (used by the Python/C++ consistency test so both sides see identical
// pixels). Other formats need ncnn built with NCNN_SIMPLEOCV (stb-based imread).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "evp/evp.h"
#include "platform.h"
#if NCNN_SIMPLEOCV
#include "simpleocv.h"
#endif

static bool ends_with(const std::string& s, const char* suf)
{
    size_t n = strlen(suf);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; i++)
    {
        char a = s[s.size() - n + i], b = suf[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

static bool load_image(const std::string& path, std::vector<unsigned char>& bgr, int& w, int& h)
{
    if (ends_with(path, ".ppm"))
    {
        std::string err;
        if (evp::load_ppm(path, bgr, w, h, &err)) return true;
        fprintf(stderr, "%s\n", err.c_str());
        return false;
    }
#if NCNN_SIMPLEOCV
    cv::Mat m = cv::imread(path, 1);
    if (m.empty() || m.channels() != 3 || !evp::image_size_ok(m.cols, m.rows)) return false;
    w = m.cols;
    h = m.rows;
    bgr.assign(m.data, m.data + (size_t)w * h * 3);
    return true;
#else
    fprintf(stderr, "only .ppm images are supported in this build (ncnn without NCNN_SIMPLEOCV)\n");
    return false;
#endif
}

static void json_str(const std::string& s)
{
    putchar('"');
    for (size_t i = 0; i < s.size(); i++)
    {
        char c = s[i];
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if ((unsigned char)c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

static bool dump_input(const std::string& dir, size_t idx, const ncnn::Mat& m)
{
    char name[64];
    snprintf(name, sizeof(name), "/input_%04d.f32", (int)idx);
    FILE* fp = fopen((dir + name).c_str(), "wb");
    if (!fp) return false;
    bool ok = true;
    for (int c = 0; c < m.c && ok; c++)
        ok = fwrite(m.channel(c).data, sizeof(float), (size_t)m.w * m.h, fp) == (size_t)m.w * m.h;
    fclose(fp);
    return ok;
}

static void usage()
{
    fprintf(stderr, "usage: evp_demo <bundle_dir> <image...> [--precision fp32|int8] [--threads N] [--conf X] "
                    "[--iou X] [--fp16] [--bench N] [--dump-input DIR]\n");
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        usage();
        return 2;
    }
    std::string bundle = argv[1], precision = "fp32", dump_dir;
    std::vector<std::string> images;
    int threads = 1, bench = 0;
    bool fp16 = false;
    double conf = -1, iou = -1;
    for (int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        bool has_val = i + 1 < argc;
        bool bad = false;
        if (a == "--precision" && has_val) precision = argv[++i];
        else if (a == "--threads" && has_val) bad = !evp::parse_int(argv[++i], threads) || threads < 1;
        else if (a == "--conf" && has_val) bad = !evp::parse_double(argv[++i], conf) || conf < 0 || conf > 1;
        else if (a == "--iou" && has_val) bad = !evp::parse_double(argv[++i], iou) || iou < 0 || iou > 1;
        else if (a == "--bench" && has_val) bad = !evp::parse_int(argv[++i], bench) || bench < 0;
        else if (a == "--dump-input" && has_val) dump_dir = argv[++i];
        else if (a == "--fp16") fp16 = true;
        else if (a.compare(0, 2, "--") == 0)
        {
            fprintf(stderr, "unknown or incomplete option %s\n", a.c_str());
            usage();
            return 2;
        }
        else images.push_back(a);
        if (bad)
        {
            fprintf(stderr, "invalid value for %s: %s (conf/iou in [0,1], threads >= 1, bench >= 0)\n", a.c_str(),
                    argv[i]);
            return 2;
        }
    }
    if (images.empty() || threads < 1)
    {
        usage();
        return 2;
    }

    evp::Model model;
    if (model.load(bundle, precision, threads, fp16) != 0)
    {
        fprintf(stderr, "load failed: %s\n", model.error().c_str());
        return 1;
    }
    if (conf >= 0) model.post().conf = conf;
    if (iou >= 0) model.post().iou = iou;
    const evp::Meta& meta = model.meta();

    int failures = 0;
    for (size_t idx = 0; idx < images.size(); idx++)
    {
        std::vector<unsigned char> bgr;
        int w = 0, h = 0;
        if (!load_image(images[idx], bgr, w, h))
        {
            fprintf(stderr, "cannot read image %s\n", images[idx].c_str());
            failures++;
            continue;
        }
        ncnn::Mat in;
        evp::Geometry g;
        if (evp::preprocess(bgr.data(), w, h, meta.input, in, g) != 0)
        {
            fprintf(stderr, "preprocess failed for %s\n", images[idx].c_str());
            failures++;
            continue;
        }
        if (!dump_dir.empty() && !dump_input(dump_dir, idx, in)) fprintf(stderr, "dump-input failed\n");
        std::vector<ncnn::Mat> outs;
        if (model.forward(in, outs) != 0)
        {
            fprintf(stderr, "forward failed: %s\n", model.error().c_str());
            failures++;
            continue;
        }
        double ms = -1;
        if (bench > 0)
        {
            std::vector<ncnn::Mat> tmp;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < bench; r++) model.forward(in, tmp);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / bench;
        }
        printf("{\"image\":");
        json_str(images[idx]);
        printf(",\"w\":%d,\"h\":%d,\"precision\":\"%s\"", w, h, precision.c_str());
        if (ms >= 0) printf(",\"forward_ms\":%.3f", ms);
        if (meta.is_detect())
        {
            std::vector<evp::Detection> dets;
            if (evp::postprocess_detect(outs, meta, model.post(), g, dets) != 0)
            {
                fprintf(stderr, "postprocess failed (output shape mismatch)\n");
                failures++;
                printf("}\n");
                continue;
            }
            printf(",\"detections\":[");
            for (size_t k = 0; k < dets.size(); k++)
            {
                const evp::Detection& d = dets[k];
                printf("%s[%.9g,%.9g,%.9g,%.9g,%.9g,%d]", k ? "," : "", d.x1, d.y1, d.x2, d.y2, d.score, d.cls);
            }
            printf("]");
        }
        else if (meta.family == "classify")
        {
            std::vector<std::pair<int, float> > topk;
            evp::postprocess_classify(outs[0], model.post(), topk);
            printf(",\"topk\":[");
            for (size_t k = 0; k < topk.size(); k++) printf("%s[%d,%.9g]", k ? "," : "", topk[k].first, topk[k].second);
            printf("]");
        }
        else
        {
            printf(",\"outputs\":[");
            for (size_t k = 0; k < outs.size(); k++)
                printf("%s[%d,%d,%d]", k ? "," : "", outs[k].c, outs[k].h, outs[k].w);
            printf("]");
        }
        printf("}\n");
        fflush(stdout);
    }
    return failures ? 1 : 0;
}
