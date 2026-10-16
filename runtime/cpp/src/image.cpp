// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 edge_vision_pipeline contributors
// Image helpers: binary PPM (P6) loader and size limits shared by the demo and the board integration.
#include "evp/evp.h"

#include <cstdio>

namespace evp {

bool image_size_ok(int w, int h)
{
    if (w <= 0 || h <= 0) return false;
    long long bytes = (long long)w * (long long)h * 3;
    return bytes <= kMaxImageBytes;
}

static int skip_ws_comments(FILE* fp)
{
    int c = fgetc(fp);
    while (c != EOF)
    {
        if (c == '#')
        {
            while (c != EOF && c != '\n') c = fgetc(fp);
        }
        else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            c = fgetc(fp);
        }
        else
        {
            break;
        }
    }
    return c;
}

// Reads a decimal header field; returns the delimiter that ended it (or EOF / -2 on error).
static int read_ppm_int(FILE* fp, int& v)
{
    int c = skip_ws_comments(fp);
    if (c < '0' || c > '9') return -2;
    v = 0;
    while (c >= '0' && c <= '9')
    {
        v = v * 10 + (c - '0');
        if (v > (1 << 20)) return -2;
        c = fgetc(fp);
    }
    return c;
}

static bool fail(std::string* err, const std::string& msg)
{
    if (err) *err = msg;
    return false;
}

bool load_ppm(const std::string& path, std::vector<unsigned char>& bgr, int& w, int& h, std::string* err)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return fail(err, "cannot open " + path);
    char magic[2] = {0, 0};
    int maxval = 0;
    bool ok = fread(magic, 1, 2, fp) == 2 && magic[0] == 'P' && magic[1] == '6';
    int d = ok ? read_ppm_int(fp, w) : -2;
    ok = ok && d >= 0;
    if (ok) ungetc(d, fp);
    d = ok ? read_ppm_int(fp, h) : -2;
    ok = ok && d >= 0;
    if (ok) ungetc(d, fp);
    // exactly one whitespace byte follows maxval (it is the delimiter read_ppm_int consumed)
    d = ok ? read_ppm_int(fp, maxval) : -2;
    ok = ok && (d == ' ' || d == '\t' || d == '\r' || d == '\n') && maxval == 255;
    if (!ok)
    {
        fclose(fp);
        return fail(err, "not a binary P6 PPM with maxval 255: " + path);
    }
    if (!image_size_ok(w, h))
    {
        fclose(fp);
        return fail(err, "PPM size out of range: " + path);
    }
    size_t need = (size_t)w * (size_t)h * 3;
    if (d == '\r')
    {
        // tolerate a CRLF header only when the file size proves the extra '\n' is not pixel data
        long pos = ftell(fp);
        if (pos >= 0 && fseek(fp, 0, SEEK_END) == 0)
        {
            long end = ftell(fp);
            fseek(fp, pos, SEEK_SET);
            if (end >= 0 && (unsigned long)(end - pos) == need + 1)
            {
                int c = fgetc(fp);
                if (c != '\n') ungetc(c, fp);
            }
        }
    }
    bgr.resize(need);
    ok = fread(&bgr[0], 1, need, fp) == need;
    fclose(fp);
    if (!ok) return fail(err, "truncated PPM pixel data: " + path);
    for (size_t i = 0; i + 2 < need; i += 3)
    {
        unsigned char t = bgr[i];
        bgr[i] = bgr[i + 2];
        bgr[i + 2] = t;
    }
    return true;
}

} // namespace evp
