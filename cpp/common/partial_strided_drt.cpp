#include "partial_strided_drt.h"
#include "ps_drt_v.h"
#include "ps_drt_h.h"
#include "ps_bar_detector.h"
#include "ps_threshold_jet.h"
#include "image_utils.h"


namespace PSDRT {
int n_squares = 497;
int n_slopes_drt = 63;

Halide::Runtime::Buffer<int16_t> drt_v(1024, n_slopes_drt, n_squares);
Halide::Runtime::Buffer<int16_t> drt_h(1024, n_slopes_drt, n_squares);
Halide::Runtime::Buffer<int16_t> intensities(n_squares, n_squares);
Halide::Runtime::Buffer<int16_t> slopes(n_squares, n_squares);
Halide::Runtime::Buffer<uint8_t> output_image(n_squares, n_squares, 3);
Halide::Runtime::Buffer<uint8_t> jetr(ImageUtils::jet_r);
Halide::Runtime::Buffer<uint8_t> jetg(ImageUtils::jet_g);
Halide::Runtime::Buffer<uint8_t> jetb(ImageUtils::jet_b);

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input) {
   ps_drt_v(input, drt_v);
   ps_drt_h(input, drt_h);
   ps_bar_detector(drt_h, drt_v, intensities, slopes);
//   ImageUtils::save_normalized(slopes, std::string(OUTPUT_DIR) + std::string("/slopes"));
   ps_threshold_jet(intensities, slopes, jetr, jetg, jetb, output_image);
   return output_image;
}

}

