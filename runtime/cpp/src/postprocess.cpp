// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// Mirrors evp/deploy/postprocess.py (same iteration order, double precision, stable sorts).
#include "evp/evp.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace evp {

static const double MAX_WH = 7680.0;
static const size_t MAX_NMS = 30000;
static const int E2E_MIN_K = 300;

namespace {

struct Cand
{
    double box[4];
    std::vector<double> scores; // full class vector (needed by e2e multi-class top-k)
    int best;
    double best_score;
};

struct Row
{
    double box[4];
    double score;
    int cls;
};

inline double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double inv_sigmoid(double p)
{
    p = std::min(std::max(p, 1e-7), 1 - 1e-7);
    return std::log(p / (1 - p));
}

double iou(const double* a, const double* b)
{
    double ix1 = std::max(a[0], b[0]), iy1 = std::max(a[1], b[1]);
    double ix2 = std::min(a[2], b[2]), iy2 = std::min(a[3], b[3]);
    double inter = std::max(ix2 - ix1, 0.0) * std::max(iy2 - iy1, 0.0);
    double ua = (a[2] - a[0]) * (a[3] - a[1]), ub = (b[2] - b[0]) * (b[3] - b[1]);
    double uni = ua + ub - inter;
    return uni > 0 ? inter / std::max(uni, 1e-12) : 0.0;
}

int gather(const std::vector<ncnn::Mat>& outputs, const Meta& meta, double conf, std::vector<Cand>& cands)
{
    const int nc = meta.num_classes, rm = meta.reg_max, box_ch = 4 * rm;
    // Python compares float32 logits against a float32-cast threshold (NumPy weak-scalar rule)
    const float thr = conf > 0 ? (float)inv_sigmoid(conf) : -std::numeric_limits<float>::infinity();
    if (outputs.size() != meta.outputs.size()) return -1;
    std::vector<double> bins(box_ch), dist(4);
    for (size_t k = 0; k < outputs.size(); k++)
    {
        const ncnn::Mat& m = outputs[k];
        const OutputSpec& o = meta.outputs[k];
        if (m.dims != 3 || m.c != box_ch + nc || m.h != o.shape[1] || m.w != o.shape[2]) return -2;
        const double s = o.stride;
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                float mx = -std::numeric_limits<float>::infinity();
                for (int c = 0; c < nc; c++) mx = std::max(mx, m.channel(box_ch + c).row(y)[x]);
                if (!(mx > thr)) continue;
                for (int c = 0; c < box_ch; c++) bins[c] = m.channel(c).row(y)[x];
                if (rm > 1)
                {
                    for (int side = 0; side < 4; side++)
                    {
                        const double* b = &bins[side * rm];
                        double bmax = b[0];
                        for (int j = 1; j < rm; j++) bmax = std::max(bmax, b[j]);
                        double sum = 0, acc = 0;
                        for (int j = 0; j < rm; j++)
                        {
                            double e = std::exp(b[j] - bmax);
                            sum += e;
                            acc += e * j;
                        }
                        dist[side] = acc / sum;
                    }
                }
                else
                {
                    for (int side = 0; side < 4; side++) dist[side] = bins[side];
                }
                Cand cd;
                double ax = x + 0.5, ay = y + 0.5;
                cd.box[0] = (ax - dist[0]) * s;
                cd.box[1] = (ay - dist[1]) * s;
                cd.box[2] = (ax + dist[2]) * s;
                cd.box[3] = (ay + dist[3]) * s;
                cd.scores.resize(nc);
                cd.best = 0;
                cd.best_score = -1;
                for (int c = 0; c < nc; c++)
                {
                    double sc = sigmoid((double)m.channel(box_ch + c).row(y)[x]);
                    cd.scores[c] = sc;
                    if (sc > cd.best_score)
                    {
                        cd.best_score = sc;
                        cd.best = c;
                    }
                }
                if (cd.best_score > conf) cands.push_back(cd);
            }
        }
    }
    return 0;
}

void nms(std::vector<Row>& rows, double iou_thr, bool agnostic, int max_det, std::vector<Row>& keep)
{
    std::vector<size_t> order(rows.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rows[a].score > rows[b].score; });
    if (order.size() > MAX_NMS) order.resize(MAX_NMS);
    std::vector<Row> d(order.size());
    std::vector<double> boxes(order.size() * 4);
    for (size_t i = 0; i < order.size(); i++)
    {
        d[i] = rows[order[i]];
        double off = agnostic ? 0.0 : d[i].cls * MAX_WH;
        for (int j = 0; j < 4; j++) boxes[i * 4 + j] = d[i].box[j] + off;
    }
    std::vector<char> suppressed(d.size(), 0);
    keep.clear();
    for (size_t i = 0; i < d.size(); i++)
    {
        if (suppressed[i]) continue;
        keep.push_back(d[i]);
        if ((int)keep.size() >= max_det) break;
        for (size_t j = i + 1; j < d.size(); j++)
            if (!suppressed[j] && iou(&boxes[i * 4], &boxes[j * 4]) > iou_thr) suppressed[j] = 1;
    }
}

void e2e_topk(const std::vector<Cand>& c, double conf, int max_det, bool agnostic, std::vector<Row>& out)
{
    out.clear();
    if (c.empty()) return;
    size_t k = (size_t)std::max(max_det, E2E_MIN_K);
    std::vector<size_t> a_order(c.size());
    for (size_t i = 0; i < a_order.size(); i++) a_order[i] = i;
    std::stable_sort(a_order.begin(), a_order.end(),
                     [&](size_t a, size_t b) { return c[a].best_score > c[b].best_score; });
    if (a_order.size() > k) a_order.resize(k);
    if (agnostic)
    {
        for (size_t i = 0; i < a_order.size() && (int)out.size() < max_det; i++)
        {
            const Cand& cd = c[a_order[i]];
            Row r;
            std::copy(cd.box, cd.box + 4, r.box);
            r.score = cd.best_score;
            r.cls = cd.best;
            out.push_back(r);
        }
        return;
    }
    const size_t nc = c[0].scores.size();
    std::vector<size_t> flat(a_order.size() * nc);
    for (size_t i = 0; i < flat.size(); i++) flat[i] = i;
    auto score_of = [&](size_t p) { return c[a_order[p / nc]].scores[p % nc]; };
    std::stable_sort(flat.begin(), flat.end(), [&](size_t a, size_t b) { return score_of(a) > score_of(b); });
    if (flat.size() > k) flat.resize(k);
    for (size_t i = 0; i < flat.size(); i++)
    {
        double s = score_of(flat[i]);
        if (s <= conf || (int)out.size() >= max_det) break;
        const Cand& cd = c[a_order[flat[i] / nc]];
        Row r;
        std::copy(cd.box, cd.box + 4, r.box);
        r.score = s;
        r.cls = (int)(flat[i] % nc);
        out.push_back(r);
    }
}

} // namespace

int postprocess_detect(const std::vector<ncnn::Mat>& outputs, const Meta& meta, const PostSpec& post,
                       const Geometry& g, std::vector<Detection>& dets)
{
    dets.clear();
    std::vector<Cand> cands;
    int ret = gather(outputs, meta, post.conf, cands);
    if (ret != 0) return ret;
    std::vector<Row> rows;
    if (post.method == "e2e_topk")
    {
        e2e_topk(cands, post.conf, post.max_det, post.agnostic, rows);
    }
    else
    {
        std::vector<Row> all(cands.size());
        for (size_t i = 0; i < cands.size(); i++)
        {
            std::copy(cands[i].box, cands[i].box + 4, all[i].box);
            all[i].score = cands[i].best_score;
            all[i].cls = cands[i].best;
        }
        nms(all, post.iou, post.agnostic, post.max_det, rows);
    }
    for (size_t i = 0; i < rows.size(); i++)
    {
        const Row& r = rows[i];
        double x1 = (r.box[0] - g.pad_left + g.crop_x) / g.scale_x;
        double y1 = (r.box[1] - g.pad_top + g.crop_y) / g.scale_y;
        double x2 = (r.box[2] - g.pad_left + g.crop_x) / g.scale_x;
        double y2 = (r.box[3] - g.pad_top + g.crop_y) / g.scale_y;
        Detection d;
        d.x1 = (float)std::min(std::max(x1, 0.0), (double)g.src_w);
        d.y1 = (float)std::min(std::max(y1, 0.0), (double)g.src_h);
        d.x2 = (float)std::min(std::max(x2, 0.0), (double)g.src_w);
        d.y2 = (float)std::min(std::max(y2, 0.0), (double)g.src_h);
        d.score = (float)r.score;
        d.cls = r.cls;
        dets.push_back(d);
    }
    return 0;
}

int postprocess_classify(const ncnn::Mat& output, const PostSpec& post, std::vector<std::pair<int, float> >& topk)
{
    topk.clear();
    if (output.empty() || output.elempack != 1) return -1;
    // reshape drops per-channel cstep padding so the data is contiguous
    ncnn::Mat flat = output.dims == 1 ? output : output.reshape(output.w * output.h * output.d * output.c);
    const int n = flat.w;
    const float* p = (const float*)flat.data;
    std::vector<double> v(p, p + n);
    if (post.softmax)
    {
        double mx = *std::max_element(v.begin(), v.end()), sum = 0;
        for (int i = 0; i < n; i++) sum += (v[i] = std::exp(v[i] - mx));
        for (int i = 0; i < n; i++) v[i] /= sum;
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });
    for (int i = 0; i < std::min(post.topk, n); i++) topk.push_back(std::make_pair(order[i], (float)v[order[i]]));
    return 0;
}

} // namespace evp
