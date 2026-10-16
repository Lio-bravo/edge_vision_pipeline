// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// Mirrors evp/deploy/preprocess.py. Geometry in double with std::nearbyint (round-half-even, same as
// Python round) so letterbox padding matches Ultralytics exactly; pixel ops are ncnn's, as in Python.
#include "evp/evp.h"

#include <algorithm>
#include <cmath>

namespace evp {

static int rnd(double v)
{
    return (int)std::nearbyint(v);
}

Geometry letterbox_geometry(int src_w, int src_h, int dst_w, int dst_h)
{
    Geometry g;
    g.src_w = src_w;
    g.src_h = src_h;
    g.dst_w = dst_w;
    g.dst_h = dst_h;
    double r = std::min((double)dst_h / src_h, (double)dst_w / src_w);
    int new_w = std::max(1, rnd(src_w * r)), new_h = std::max(1, rnd(src_h * r));
    double dw = (dst_w - new_w) / 2.0, dh = (dst_h - new_h) / 2.0;
    int top = rnd(dh - 0.1), bottom = rnd(dh + 0.1);
    int left = rnd(dw - 0.1), right = rnd(dw + 0.1);
    if (new_w + left + right != dst_w) right = dst_w - new_w - left;
    if (new_h + top + bottom != dst_h) bottom = dst_h - new_h - top;
    g.resize_w = new_w;
    g.resize_h = new_h;
    g.scale_x = g.scale_y = r;
    g.pad_left = left;
    g.pad_top = top;
    g.pad_right = right;
    g.pad_bottom = bottom;
    return g;
}

Geometry stretch_geometry(int src_w, int src_h, int dst_w, int dst_h)
{
    Geometry g;
    g.src_w = src_w;
    g.src_h = src_h;
    g.dst_w = g.resize_w = dst_w;
    g.dst_h = g.resize_h = dst_h;
    g.scale_x = (double)dst_w / src_w;
    g.scale_y = (double)dst_h / src_h;
    return g;
}

Geometry center_crop_geometry(int src_w, int src_h, int dst_w, int dst_h)
{
    Geometry g;
    g.src_w = src_w;
    g.src_h = src_h;
    g.dst_w = dst_w;
    g.dst_h = dst_h;
    int size = dst_w;
    int new_w, new_h;
    if (dst_w == dst_h)
    {
        bool portrait = src_w <= src_h;
        int shorter = portrait ? src_w : src_h, longer = portrait ? src_h : src_w;
        // Python: int(size * long_ / short) -> exact integer product, correctly rounded division, truncation
        int new_long = (int)((double)((long long)size * longer) / shorter);
        new_w = portrait ? size : new_long;
        new_h = portrait ? new_long : size;
    }
    else
    {
        double r = std::max((double)dst_w / src_w, (double)dst_h / src_h);
        new_w = std::max(dst_w, rnd(src_w * r));
        new_h = std::max(dst_h, rnd(src_h * r));
    }
    g.resize_w = new_w;
    g.resize_h = new_h;
    g.scale_x = (double)new_w / src_w;
    g.scale_y = (double)new_h / src_h;
    g.crop_x = rnd((new_w - dst_w) / 2.0);
    g.crop_y = rnd((new_h - dst_h) / 2.0);
    return g;
}

bool make_geometry(const InputSpec& spec, int src_w, int src_h, Geometry& g)
{
    if (src_w <= 0 || src_h <= 0) return false;
    if (spec.resize == "letterbox") g = letterbox_geometry(src_w, src_h, spec.w, spec.h);
    else if (spec.resize == "stretch") g = stretch_geometry(src_w, src_h, spec.w, spec.h);
    else if (spec.resize == "center_crop") g = center_crop_geometry(src_w, src_h, spec.w, spec.h);
    else return false;
    return true;
}

int preprocess(const unsigned char* bgr, int w, int h, const InputSpec& spec, ncnn::Mat& out, Geometry& g)
{
    if (!bgr || !image_size_ok(w, h)) return -1;
    if (!make_geometry(spec, w, h, g)) return -2;
    int pix = spec.color == "RGB" ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR;
    ncnn::Mat m = ncnn::Mat::from_pixels_resize(bgr, pix, w, h, g.resize_w, g.resize_h);
    if (m.empty()) return -3;
    if (spec.resize == "letterbox" && (g.pad_left || g.pad_top || g.pad_right || g.pad_bottom))
    {
        ncnn::Mat b;
        ncnn::copy_make_border(m, b, g.pad_top, g.pad_bottom, g.pad_left, g.pad_right, ncnn::BORDER_CONSTANT,
                               (float)spec.pad_value);
        m = b;
    }
    else if (spec.resize == "center_crop" && (g.resize_w != spec.w || g.resize_h != spec.h))
    {
        ncnn::Mat c;
        ncnn::copy_cut_border(m, c, g.crop_y, g.resize_h - spec.h - g.crop_y, g.crop_x,
                              g.resize_w - spec.w - g.crop_x);
        m = c;
    }
    if (m.w != spec.w || m.h != spec.h) return -4;
    m.substract_mean_normalize(spec.mean, spec.norm);
    out = m;
    return 0;
}

} // namespace evp
