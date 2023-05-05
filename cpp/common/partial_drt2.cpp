#include "partial_drt2.h"
#include "image_utils.h"
#include "pdrt2_v.h"
#include "pdrt2_h.h"
#include "pdrt2_bar_detector.h"
#include "pdrt2_threshold_jet.h"

namespace PDRT2 {
int n_squares = 512;
int n_slopes_drt = 3;

Halide::Runtime::Buffer<int16_t> drt_v(1024, n_slopes_drt, n_squares);
Halide::Runtime::Buffer<int16_t> drt_h(1024, n_slopes_drt, n_squares);
Halide::Runtime::Buffer<int16_t> intensities(n_squares, n_squares);
Halide::Runtime::Buffer<int16_t> slopes(n_squares, n_squares);
Halide::Runtime::Buffer<uint8_t> output_image(n_squares, n_squares, 3);
Halide::Runtime::Buffer<uint8_t> jetr(ImageUtils::jet_r);
Halide::Runtime::Buffer<uint8_t> jetg(ImageUtils::jet_g);
Halide::Runtime::Buffer<uint8_t> jetb(ImageUtils::jet_b);

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input) {
   pdrt2_v(input, drt_v);
   pdrt2_h(input, drt_h);
   pdrt2_bar_detector(drt_h, drt_v, intensities, slopes);
   pdrt2_threshold_jet(intensities, slopes, jetr, jetg, jetb, output_image);
   return output_image;
}

}