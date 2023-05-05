
#include "image_utils.h"
#include <halide_image_io.h>
#include <iostream>


void ImageUtils::save_pidrt(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path) {
   Halide::Runtime::Buffer<float, 2> output;
   // 3D to 2D
   int h = buffer.dim(0).extent();
   int w = buffer.dim(1).extent();
   int c = buffer.dim(2).extent();
   output = Halide::Runtime::Buffer<float>(h, w * c);
   for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++)
         for (int k = 0; k < c; k++)
            output(i, k * w + j) = buffer(i, j, k);
   Halide::Runtime::Buffer output_uint8 = normalize2D(output);
   Halide::Tools::save_image(output_uint8, path);
}

void ImageUtils::save_activation_maps(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path) {
   int d = buffer.dim(0).extent();
   int h = buffer.dim(1).extent();
   int w = buffer.dim(2).extent();
   for (int channel = 0; channel < d; channel++) {
      Halide::Runtime::Buffer slice = Halide::Runtime::Buffer<float>(h, w);
      for (int i = 0; i < h; i++)
         for (int j = 0; j < w; j++)
            slice(i, j) = buffer(channel, i, j);
      Halide::Runtime::Buffer normalized = normalize2D(slice);
      std::string filename = path + "_" + std::to_string(channel) + ".png";
      Halide::Tools::save_image(normalized, filename);
   }
}

void ImageUtils::save_normalized(Halide::Runtime::Buffer<int16_t> buffer, const std::string &path) {
   int h = buffer.dim(0).extent();
   int w = buffer.dim(1).extent();
   Halide::Runtime::Buffer slice = Halide::Runtime::Buffer<float>(h, w);
   for (int i = 0; i < h; i++)
      for (int j = 0; j < w; j++)
         slice(i, j) = buffer(i, j);
   Halide::Runtime::Buffer normalized = normalize2D(slice);
   std::string filename = path + ".png";
   Halide::Tools::save_image(normalized, filename);
}

Halide::Runtime::Buffer<uint8_t> ImageUtils::normalize2D(Halide::Runtime::Buffer<float> buffer) {
   Halide::Runtime::Buffer buffer_uint8 = Halide::Runtime::Buffer<uint8_t>(buffer.dim(0).extent(),
                                                                           buffer.dim(1).extent());
   float min_val = std::numeric_limits<float>::max();
   float max_val = std::numeric_limits<float>::min();
   for (int i = 0; i < buffer.dim(0).extent(); i++) {
      for (int j = 0; j < buffer.dim(1).extent(); j++) {
         float val = buffer(i, j);
         if (min_val > val) {
            min_val = val;
         }
         if (max_val < val) {
            max_val = val;
         }
      }
   }
   std::cout << "max="<<max_val << std::endl;
   std::cout << "min="<<min_val << std::endl;
   for (int i = 0; i < buffer.dim(0).extent(); i++)
      for (int j = 0; j < buffer.dim(1).extent(); j++)
         buffer_uint8(i, j) = (uint8_t) (255.0f * (buffer(i, j) - min_val) / (max_val - min_val));
//         buffer_uint8(i, j) = (uint8_t) buffer(i, j);
   return buffer_uint8;
}
