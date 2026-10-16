// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// model.meta parser (see evp/deploy/meta.py:to_flat for the writer).
#include "evp/evp.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <locale>
#include <sstream>

namespace evp {

static bool set_err(std::string* err, const std::string& msg)
{
    if (err) *err = msg;
    return false;
}

// istringstream imbued with the classic locale: '.' decimal point even after setlocale(LC_ALL, "de_DE")
bool parse_double(const std::string& s, double& v)
{
    if (s.empty() || s[0] == ' ' || s[0] == '+') return false;
    std::istringstream ss(s);
    ss.imbue(std::locale::classic());
    double x;
    ss >> x;
    if (ss.fail() || !ss.eof() || !std::isfinite(x)) return false;
    v = x;
    return true;
}

bool parse_int(const std::string& s, int& v)
{
    if (s.empty() || s.size() > 12) return false;
    size_t i = s[0] == '-' ? 1 : 0;
    if (i == s.size()) return false;
    long long x = 0;
    for (; i < s.size(); i++)
    {
        if (s[i] < '0' || s[i] > '9') return false;
        x = x * 10 + (s[i] - '0');
    }
    if (s[0] == '-') x = -x;
    if (x < INT_MIN || x > INT_MAX) return false;
    v = (int)x;
    return true;
}

static bool to_int(const std::string& s, int& v) { return parse_int(s, v); }
static bool to_double(const std::string& s, double& v) { return parse_double(s, v); }

static bool split_floats(const std::string& s, float* out, int n)
{
    std::stringstream ss(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, ','))
    {
        double v;
        if (i >= n || !to_double(item, v)) return false;
        out[i++] = (float)v;
    }
    return i == n;
}

static bool split_ints(const std::string& s, std::vector<int>& out)
{
    std::stringstream ss(s);
    std::string item;
    out.clear();
    while (std::getline(ss, item, ','))
    {
        int v;
        if (!to_int(item, v)) return false;
        out.push_back(v);
    }
    return !out.empty();
}

bool parse_meta(const std::string& text, Meta& m, std::string* err)
{
    std::map<std::string, std::string> kv;
    std::stringstream ss(text);
    std::string line;
    int ln = 0;
    while (std::getline(ss, line))
    {
        ln++;
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos)
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "model.meta line %d: missing '='", ln);
            return set_err(err, buf);
        }
        std::string k = line.substr(0, eq);
        if (kv.count(k)) return set_err(err, "model.meta duplicate key " + k);
        kv[k] = line.substr(eq + 1);
    }

    std::string missing;
    const char* required[] = {"schema_version", "name", "family", "input.w", "input.h", "input.blob", "input.color",
                              "input.resize", "input.pad_value", "input.mean", "input.norm", "reg_max", "num_classes",
                              "output.count", "post.method", "post.conf", "post.iou", "post.max_det", "post.agnostic",
                              "post.topk", "post.softmax", "labels.count"};
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++)
        if (!kv.count(required[i])) return set_err(err, std::string("model.meta missing key ") + required[i]);

    int iv;
    double dv;
    if (!to_int(kv["schema_version"], m.schema_version) || m.schema_version != 1)
        return set_err(err, "unsupported model.meta schema_version " + kv["schema_version"]);
    m.name = kv["name"];
    m.family = kv["family"];
    if (m.family != "yolo" && m.family != "classify" && m.family != "raw")
        return set_err(err, "unknown family " + m.family);
    InputSpec& in = m.input;
    if (!to_int(kv["input.w"], in.w) || !to_int(kv["input.h"], in.h) || in.w <= 0 || in.h <= 0)
        return set_err(err, "invalid input size");
    in.blob = kv["input.blob"];
    in.color = kv["input.color"];
    if (in.color != "RGB" && in.color != "BGR") return set_err(err, "input.color must be RGB/BGR");
    in.resize = kv["input.resize"];
    if (in.resize != "letterbox" && in.resize != "stretch" && in.resize != "center_crop")
        return set_err(err, "invalid input.resize " + in.resize);
    if (!to_int(kv["input.pad_value"], in.pad_value) || in.pad_value < 0 || in.pad_value > 255)
        return set_err(err, "invalid input.pad_value");
    if (!split_floats(kv["input.mean"], in.mean, 3) || !split_floats(kv["input.norm"], in.norm, 3))
        return set_err(err, "input.mean/input.norm need 3 floats");
    if (!to_int(kv["reg_max"], m.reg_max) || !to_int(kv["num_classes"], m.num_classes) || m.reg_max < 1
            || m.reg_max > 1024 || m.num_classes < 0)
        return set_err(err, "invalid reg_max/num_classes");

    int n_out;
    if (!to_int(kv["output.count"], n_out) || n_out <= 0) return set_err(err, "invalid output.count");
    m.outputs.clear();
    for (int k = 0; k < n_out; k++)
    {
        char key[64];
        OutputSpec o;
        snprintf(key, sizeof(key), "output.%d.name", k);
        if (!kv.count(key)) return set_err(err, std::string("missing ") + key);
        o.name = kv[key];
        snprintf(key, sizeof(key), "output.%d.stride", k);
        if (!kv.count(key) || !to_int(kv[key], o.stride)) return set_err(err, std::string("bad ") + key);
        snprintf(key, sizeof(key), "output.%d.shape", k);
        if (!kv.count(key) || !split_ints(kv[key], o.shape)) return set_err(err, std::string("bad ") + key);
        m.outputs.push_back(o);
    }

    PostSpec& p = m.post;
    p.method = kv["post.method"];
    if (p.method != "nms" && p.method != "e2e_topk" && p.method != "topk" && p.method != "none")
        return set_err(err, "invalid post.method " + p.method);
    if (!to_double(kv["post.conf"], p.conf) || !to_double(kv["post.iou"], p.iou) || p.conf < 0 || p.conf > 1
            || p.iou < 0 || p.iou > 1)
        return set_err(err, "post.conf/post.iou must be numbers in [0,1]");
    if (!to_int(kv["post.max_det"], p.max_det) || p.max_det < 1) return set_err(err, "bad post.max_det");
    if (!to_int(kv["post.agnostic"], iv)) return set_err(err, "bad post.agnostic");
    p.agnostic = iv != 0;
    if (!to_int(kv["post.topk"], p.topk) || p.topk < 1) return set_err(err, "bad post.topk");
    if (!to_int(kv["post.softmax"], iv)) return set_err(err, "bad post.softmax");
    p.softmax = iv != 0;
    (void)dv;

    int n_lab;
    if (!to_int(kv["labels.count"], n_lab) || n_lab < 0) return set_err(err, "bad labels.count");
    m.labels.clear();
    for (int k = 0; k < n_lab; k++)
    {
        char key[32];
        snprintf(key, sizeof(key), "label.%d", k);
        if (!kv.count(key)) return set_err(err, std::string("missing ") + key);
        m.labels.push_back(kv[key]);
    }
    if (m.num_classes != n_lab) return set_err(err, "num_classes must equal labels.count");
    m.files.clear();
    for (std::map<std::string, std::string>::const_iterator it = kv.begin(); it != kv.end(); ++it)
    {
        const std::string& k = it->first;
        if (k.compare(0, 5, "file.") != 0) continue;
        size_t dot = k.find('.', 5);
        if (dot == std::string::npos) continue;
        std::string prec = k.substr(5, dot - 5), what = k.substr(dot + 1);
        if (what == "param") m.files[prec].first = it->second;
        else if (what == "bin") m.files[prec].second = it->second;
    }

    if (m.is_detect())
    {
        if (p.method != "nms" && p.method != "e2e_topk") return set_err(err, "detect model needs post.method nms|e2e_topk");
        if (m.num_classes <= 0 || m.reg_max < 1) return set_err(err, "detect model needs num_classes>0, reg_max>=1");
        for (size_t k = 0; k < m.outputs.size(); k++)
        {
            const OutputSpec& o = m.outputs[k];
            if (o.shape.size() != 3 || (long long)o.shape[0] != 4LL * m.reg_max + m.num_classes || o.stride <= 0
                    || (long long)o.shape[1] * o.stride != in.h || (long long)o.shape[2] * o.stride != in.w)
                return set_err(err, "detect output " + o.name + " shape/stride inconsistent with input/reg_max/nc");
        }
    }
    if (m.family == "classify")
    {
        if (m.outputs.size() != 1 || p.method != "topk")
            return set_err(err, "classify model needs 1 output and post.method=topk");
        if (!m.labels.empty() && m.outputs[0].shape.back() != (int)m.labels.size())
            return set_err(err, "classify output size != number of labels");
    }
    return true;
}

bool load_meta(const std::string& path, Meta& m, std::string* err)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return set_err(err, "cannot open " + path);
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, n);
    fclose(fp);
    return parse_meta(text, m, err);
}

} // namespace evp
