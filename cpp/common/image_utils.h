#ifndef BARCODE_SEGMENTATION_IMAGE_UTILS_H
#define BARCODE_SEGMENTATION_IMAGE_UTILS_H

#include <HalideBuffer.h>

namespace ImageUtils {

void save_activation_maps(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path);
void save_normalized(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path);

void save_pidrt(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path);

Halide::Runtime::Buffer<uint8_t> normalize2D(Halide::Runtime::Buffer<float> buffer);

static uint8_t jet_r[256] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 2, 6, 10, 14, 18, 22, 26, 30,
        34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82,
        86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134,
        138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186,
        190, 194, 198, 202, 206, 210, 214, 218, 222, 226, 230, 234, 238,
        242, 246, 250, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 252, 248, 244, 240, 236, 232, 228, 224, 220, 216,
        212, 208, 204, 200, 196, 192, 188, 184, 180, 176, 172, 168, 164,
        160, 156, 152, 148, 144, 140, 136, 132, 128
};
static uint8_t jet_g[256] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24,
        28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
        80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128,
        132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
        184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232,
        236, 240, 244, 248, 252, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 252, 248, 244, 240, 236, 232, 228, 224, 220,
        216, 212, 208, 204, 200, 196, 192, 188, 184, 180, 176, 172, 168,
        164, 160, 156, 152, 148, 144, 140, 136, 132, 128, 124, 120, 116,
        112, 108, 104, 100, 96, 92, 88, 84, 80, 76, 72, 68, 64,
        60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12,
        8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
};
static uint8_t jet_b[256] = {
        128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176,
        180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228,
        232, 236, 240, 244, 248, 252, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 254, 250, 246, 242, 238, 234, 230, 226,
        222, 218, 214, 210, 206, 202, 198, 194, 190, 186, 182, 178, 174,
        170, 166, 162, 158, 154, 150, 146, 142, 138, 134, 130, 126, 122,
        118, 114, 110, 106, 102, 98, 94, 90, 86, 82, 78, 74, 70,
        66, 62, 58, 54, 50, 46, 42, 38, 34, 30, 26, 22, 18,
        14, 10, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0
};

}

#endif //BARCODE_SEGMENTATION_IMAGE_UTILS_H