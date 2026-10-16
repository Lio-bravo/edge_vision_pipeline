// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// Minimal self-contained C++ tests (no framework). Exit code = number of failures.
// Geometry expectations were generated from evp/deploy/preprocess.py.
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "evp/evp.h"

static int g_fail = 0;
#define CHECK(cond)                                                     \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                   \
        }                                                               \
    } while (0)

struct GeoCase
{
    char mode;
    int sw, sh, dw, dh, rw, rh, pl, pt, pr, pb, cx, cy;
    double sx, sy;
};

static const GeoCase GEO[] = {
    {'L', 810, 1080, 320, 320, 240, 320, 40, 0, 40, 0, 0, 0, 0.29629629629629628, 0.29629629629629628},
    {'L', 640, 481, 320, 320, 320, 240, 0, 40, 0, 40, 0, 0, 0.5, 0.5},
    {'L', 640, 482, 320, 320, 320, 241, 0, 39, 0, 40, 0, 0, 0.5, 0.5},
    {'L', 1, 1, 320, 320, 320, 320, 0, 0, 0, 0, 0, 0, 320, 320},
    {'L', 4000, 10, 320, 320, 320, 1, 0, 159, 0, 160, 0, 0, 0.080000000000000002, 0.080000000000000002},
    {'L', 333, 777, 640, 384, 165, 384, 237, 0, 238, 0, 0, 0, 0.49420849420849422, 0.49420849420849422},
    {'S', 640, 480, 320, 256, 320, 256, 0, 0, 0, 0, 0, 0, 0.5, 0.53333333333333333},
    {'C', 640, 480, 224, 224, 298, 224, 0, 0, 0, 0, 37, 0, 0.46562500000000001, 0.46666666666666667},
    {'C', 481, 640, 224, 224, 224, 298, 0, 0, 0, 0, 0, 37, 0.46569646569646572, 0.46562500000000001},
    {'C', 300, 301, 224, 224, 224, 224, 0, 0, 0, 0, 0, 0, 0.7466666666666667, 0.7441860465116279},
    {'C', 640, 480, 224, 160, 224, 168, 0, 0, 0, 0, 0, 4, 0.34999999999999998, 0.34999999999999998},
};

static void test_geometry()
{
    for (size_t i = 0; i < sizeof(GEO) / sizeof(GEO[0]); i++)
    {
        const GeoCase& c = GEO[i];
        evp::Geometry g = c.mode == 'L'   ? evp::letterbox_geometry(c.sw, c.sh, c.dw, c.dh)
                          : c.mode == 'S' ? evp::stretch_geometry(c.sw, c.sh, c.dw, c.dh)
                                          : evp::center_crop_geometry(c.sw, c.sh, c.dw, c.dh);
        bool ok = g.resize_w == c.rw && g.resize_h == c.rh && g.pad_left == c.pl && g.pad_top == c.pt
                  && g.pad_right == c.pr && g.pad_bottom == c.pb && g.crop_x == c.cx && g.crop_y == c.cy
                  && g.scale_x == c.sx && g.scale_y == c.sy;
        if (!ok) fprintf(stderr, "geometry case %d (%c %dx%d->%dx%d) mismatch\n", (int)i, c.mode, c.sw, c.sh, c.dw, c.dh);
        CHECK(ok);
    }
}

static std::string meta_text(const char* extra_post = "post.method=nms\n")
{
    std::string s =
        "schema_version=1\nname=t\nfamily=yolo\ninput.w=64\ninput.h=64\ninput.blob=in0\ninput.color=RGB\n"
        "input.resize=letterbox\ninput.pad_value=114\ninput.mean=0.0,0.0,0.0\n"
        "input.norm=0.00392156862745098,0.00392156862745098,0.00392156862745098\nreg_max=1\nnum_classes=2\n"
        "output.count=1\noutput.0.name=out0\noutput.0.stride=8\noutput.0.shape=6,8,8\n";
    s += extra_post;
    s += "post.conf=0.25\npost.iou=0.45\npost.max_det=100\npost.agnostic=0\npost.topk=5\npost.softmax=0\n"
         "labels.count=2\nlabel.0=a\nlabel.1=b c\nfile.fp32.param=t.fp32.param\nfile.fp32.bin=t.fp32.bin\n";
    return s;
}

static void test_meta()
{
    evp::Meta m;
    std::string err;
    CHECK(evp::parse_meta(meta_text(), m, &err));
    CHECK(m.num_classes == 2 && m.reg_max == 1 && m.outputs.size() == 1 && m.outputs[0].stride == 8);
    CHECK(m.labels.size() == 2 && m.labels[1] == "b c");
    CHECK(m.files["fp32"].first == "t.fp32.param");
    CHECK(std::fabs(m.input.norm[0] - 1 / 255.f) < 1e-9);

    evp::Meta bad;
    CHECK(!evp::parse_meta(meta_text("post.method=bogus\n"), bad, &err));
    CHECK(!evp::parse_meta("schema_version=2\n", bad, &err));
    CHECK(!evp::parse_meta(meta_text() + "name=dup\n", bad, &err));
    std::string wrong_shape = meta_text();
    wrong_shape.replace(wrong_shape.find("6,8,8"), 5, "7,8,8");
    CHECK(!evp::parse_meta(wrong_shape, bad, &err));
    CHECK(!evp::parse_meta("no equals sign\n", bad, &err));
}

// One 8x8 grid, stride 8, reg_max 1, 2 classes. Two overlapping class-0 boxes and one class-1 box.
static void test_detect_nms_e2e()
{
    evp::Meta m;
    std::string err;
    CHECK(evp::parse_meta(meta_text(), m, &err));
    ncnn::Mat out(8, 8, 6);
    out.fill(0.f);
    for (int c = 4; c < 6; c++) out.channel(c).fill(-10.f);
    struct Cell { int x, y; float l, t, r, b, s0, s1; };
    const Cell cells[] = {{2, 2, 1, 1, 1, 1, 3.f, -10.f}, {2, 3, 1, 1.5f, 1, 1, 2.f, -10.f}, {6, 6, 1, 1, 1, 1, -10.f, 1.f}};
    for (size_t i = 0; i < 3; i++)
    {
        const Cell& c = cells[i];
        float v[6] = {c.l, c.t, c.r, c.b, c.s0, c.s1};
        for (int ch = 0; ch < 6; ch++) out.channel(ch).row(c.y)[c.x] = v[ch];
    }
    std::vector<ncnn::Mat> outs(1, out);
    evp::Geometry g = evp::letterbox_geometry(64, 64, 64, 64);
    std::vector<evp::Detection> dets;
    CHECK(evp::postprocess_detect(outs, m, m.post, g, dets) == 0);
    CHECK(dets.size() == 2);
    if (dets.size() == 2)
    {
        CHECK(dets[0].cls == 0 && std::fabs(dets[0].x1 - 12.f) < 1e-4 && std::fabs(dets[0].x2 - 28.f) < 1e-4);
        CHECK(dets[1].cls == 1);
        CHECK(dets[0].score > dets[1].score);
    }
    evp::PostSpec e2e = m.post;
    e2e.method = "e2e_topk";
    CHECK(evp::postprocess_detect(outs, m, e2e, g, dets) == 0);
    CHECK(dets.size() == 3); // no NMS: the overlapping box survives
    evp::PostSpec strict = m.post;
    strict.conf = 0.99;
    CHECK(evp::postprocess_detect(outs, m, strict, g, dets) == 0 && dets.empty());
    std::vector<ncnn::Mat> wrong(1, ncnn::Mat(8, 8, 5));
    CHECK(evp::postprocess_detect(wrong, m, m.post, g, dets) != 0);
}

static void test_classify()
{
    ncnn::Mat v(4);
    float vals[4] = {0.1f, 2.f, -1.f, 2.f};
    for (int i = 0; i < 4; i++) v[i] = vals[i];
    evp::PostSpec p;
    p.topk = 3;
    p.softmax = true;
    std::vector<std::pair<int, float> > top;
    CHECK(evp::postprocess_classify(v, p, top) == 0);
    CHECK(top.size() == 3 && top[0].first == 1 && top[1].first == 3 && top[2].first == 0); // stable on ties
    float sum = 0;
    p.topk = 4;
    evp::postprocess_classify(v, p, top);
    for (size_t i = 0; i < top.size(); i++) sum += top[i].second;
    CHECK(std::fabs(sum - 1.f) < 1e-5);
}

static void test_preprocess_letterbox()
{
    evp::InputSpec s;
    s.w = 32;
    s.h = 32;
    std::vector<unsigned char> img(16 * 8 * 3, 200);
    ncnn::Mat out;
    evp::Geometry g;
    CHECK(evp::preprocess(img.data(), 16, 8, s, out, g) == 0);
    CHECK(out.w == 32 && out.h == 32 && out.c == 3);
    CHECK(g.pad_top == 8 && g.pad_bottom == 8);
    CHECK(std::fabs(out.channel(0).row(0)[0] - 114 / 255.f) < 1e-6);   // border
    CHECK(std::fabs(out.channel(0).row(16)[16] - 200 / 255.f) < 1e-6); // image
    CHECK(evp::preprocess(0, 16, 8, s, out, g) != 0);
    CHECK(evp::preprocess(img.data(), 0, 8, s, out, g) != 0);
}

static void test_number_parsing()
{
    int i = 0;
    double d = 0;
    CHECK(evp::parse_int("123", i) && i == 123);
    CHECK(evp::parse_int("-7", i) && i == -7);
    CHECK(!evp::parse_int("9999999999", i));   // would wrap on ILP32
    CHECK(!evp::parse_int("2147483648", i));
    CHECK(evp::parse_int("2147483647", i) && i == 2147483647);
    CHECK(!evp::parse_int("12a", i) && !evp::parse_int("", i) && !evp::parse_int("-", i) && !evp::parse_int(" 1", i));
    CHECK(evp::parse_double("0.25", d) && d == 0.25);
    CHECK(evp::parse_double("1e-05", d) && d == 1e-05);
    CHECK(evp::parse_double("0.00392156862745098", d) && d == 0.00392156862745098);
    CHECK(!evp::parse_double("0,25", d) && !evp::parse_double("abc", d) && !evp::parse_double("", d));
    CHECK(!evp::parse_double("0.25x", d) && !evp::parse_double("nan", d) && !evp::parse_double("1e999", d));

    // a comma-decimal process locale must not change how model.meta is read
    const char* locs[] = {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "ru_RU.UTF-8"};
    const char* used = 0;
    for (size_t k = 0; k < 4 && !used; k++)
        if (setlocale(LC_ALL, locs[k])) used = locs[k];
    if (used)
    {
        CHECK(evp::parse_double("0.25", d) && d == 0.25);
        evp::Meta m;
        std::string err;
        CHECK(evp::parse_meta(meta_text(), m, &err) && m.post.conf == 0.25);
        setlocale(LC_ALL, "C");
    }
    else
    {
        printf("note: no comma-decimal locale installed; locale test ran in C locale only\n");
    }
}

static std::string replace(std::string s, const char* from, const char* to)
{
    s.replace(s.find(from), strlen(from), to);
    return s;
}

static void test_meta_validation()
{
    evp::Meta m;
    std::string err;
    CHECK(!evp::parse_meta(replace(meta_text(), "post.conf=0.25", "post.conf=-0.1"), m, &err));
    CHECK(!evp::parse_meta(replace(meta_text(), "post.iou=0.45", "post.iou=1.5"), m, &err));
    CHECK(!evp::parse_meta(replace(meta_text(), "num_classes=2", "num_classes=3"), m, &err));
    CHECK(!evp::parse_meta(replace(meta_text(), "input.w=64", "input.w=9999999999"), m, &err));
    CHECK(!evp::parse_meta(replace(meta_text(), "output.0.stride=8", "output.0.stride=1073741824"), m, &err));
    std::string cls =
        "schema_version=1\nname=c\nfamily=classify\ninput.w=8\ninput.h=8\ninput.blob=in0\ninput.color=RGB\n"
        "input.resize=center_crop\ninput.pad_value=114\ninput.mean=0,0,0\ninput.norm=1,1,1\nreg_max=16\n"
        "num_classes=2\noutput.count=1\noutput.0.name=out0\noutput.0.stride=0\noutput.0.shape=3\n"
        "post.method=topk\npost.conf=0.25\npost.iou=0.45\npost.max_det=1\npost.agnostic=0\npost.topk=1\n"
        "post.softmax=1\nlabels.count=2\nlabel.0=a\nlabel.1=b\n";
    CHECK(!evp::parse_meta(cls, m, &err));                                   // 3 outputs vs 2 labels
    CHECK(evp::parse_meta(replace(cls, "output.0.shape=3", "output.0.shape=2"), m, &err));
}

static std::string tmp_path(const char* name)
{
    const char* d = getenv("TMPDIR");
    return std::string(d && *d ? d : "/tmp") + "/" + name;
}

static void write_file(const std::string& path, const std::string& bytes)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return;
    fwrite(bytes.data(), 1, bytes.size(), fp);
    fclose(fp);
}

static void test_ppm()
{
    std::vector<unsigned char> bgr;
    int w = 0, h = 0;
    std::string err;
    std::string px("\x01\x02\x03\x0a\x0b\x0c", 6);   // 2x1 RGB; second pixel starts with '\n' (0x0a)
    std::string p = tmp_path("evp_ut_a.ppm");
    write_file(p, "P6\n2 1\n255\n" + px);
    CHECK(evp::load_ppm(p, bgr, w, h, &err) && w == 2 && h == 1 && bgr.size() == 6);
    CHECK(bgr.size() == 6 && bgr[0] == 3 && bgr[2] == 1 && bgr[3] == 0x0c && bgr[5] == 0x0a);
    write_file(p, "P6\r\n# comment\r\n2 1\r\n255\r\n" + px);                   // CRLF header
    CHECK(evp::load_ppm(p, bgr, w, h, &err) && bgr.size() == 6 && bgr[0] == 3 && bgr[5] == 0x0a);
    write_file(p, "P6\n2 1\n255\r" + px);                                     // lone CR delimiter, data starts 0x01
    CHECK(evp::load_ppm(p, bgr, w, h, &err) && bgr.size() == 6 && bgr[0] == 3);
    write_file(p, "P6\n65536 65536\n255\n");                                  // would wrap size_t on ARM32
    CHECK(!evp::load_ppm(p, bgr, w, h, &err));
    write_file(p, "P6\n40000 40000\n255\n");
    CHECK(!evp::load_ppm(p, bgr, w, h, &err));
    write_file(p, "P6\n2 1\n255\n\x01\x02");                                  // truncated
    CHECK(!evp::load_ppm(p, bgr, w, h, &err));
    write_file(p, "P6\n2 1\n65535\n" + px);                                   // 16-bit unsupported
    CHECK(!evp::load_ppm(p, bgr, w, h, &err));
    write_file(p, "P3\n2 1\n255\n1 2 3 4 5 6\n");
    CHECK(!evp::load_ppm(p, bgr, w, h, &err));
    CHECK(!evp::load_ppm(tmp_path("evp_ut_missing.ppm"), bgr, w, h, &err));
    remove(p.c_str());

    CHECK(evp::image_size_ok(1, 1) && evp::image_size_ok(8192, 8192));
    CHECK(!evp::image_size_ok(0, 5) && !evp::image_size_ok(65536, 65536) && !evp::image_size_ok(-1, 4));
    evp::InputSpec s;
    s.w = s.h = 32;
    unsigned char one[3] = {0, 0, 0};
    ncnn::Mat out;
    evp::Geometry g;
    CHECK(evp::preprocess(one, 100000, 100000, s, out, g) != 0);           // rejected before touching pixels
}

int main()
{
    test_geometry();
    test_meta();
    test_detect_nms_e2e();
    test_classify();
    test_preprocess_letterbox();
    test_number_parsing();
    test_meta_validation();
    test_ppm();
    if (g_fail == 0) printf("all C++ unit tests passed\n");
    return g_fail;
}
