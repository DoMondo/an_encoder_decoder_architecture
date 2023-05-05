#include "Halide.h"

namespace {

class PDRT2ThresholdJet_generator : public Halide::Generator<PDRT2ThresholdJet_generator> {
private:
   const int n_slopes = 3 * 2 - 1;
   const int n_squares = 512;
   const float threshold = 0.029f;

public:
   Var x_square{"y_square"};
   Var y_square{"x_square"};
   Var slope{"slope"};
   Input <Buffer<int16_t>> intensities{"intensities", 2};
   Input <Buffer<int16_t>> slopes{"slopes", 2};
   Input <Buffer<uint8_t>> jet_r{"jet_lookup_r", 1};
   Input <Buffer<uint8_t>> jet_g{"jet_lookup_g", 1};
   Input <Buffer<uint8_t>> jet_b{"jet_lookup_b", 1};
   Output <Buffer<uint8_t>> output{"output", 3};
   Func mask{"mask"};
   Func indices{"indices"};

   void generate() {
      using namespace Halide::ConciseCasts;
      RDom slope_dom(0, n_slopes);
      RDom intensities_dom(0, n_squares, 0, n_squares);
      Expr max_intensity = maximum(intensities_dom, intensities(intensities_dom.x, intensities_dom.y));

      // Threshold
      mask(x_square, y_square) = u8(
              select(f32(intensities(x_square, y_square)) / f32(max_intensity) > threshold, 1, 0));

      indices(x_square, y_square) = u8(255.0f * f32(slopes(x_square, y_square)) / n_slopes);

      // Jet-colorspace
      Var color_channel;
      output(x_square, y_square, color_channel) = select(color_channel == 0,
                                                         jet_b(indices(x_square, y_square)) * mask(x_square, y_square),
                                                         color_channel == 1,
                                                         jet_g(indices(x_square, y_square)) * mask(x_square, y_square),
                                                         jet_r(indices(x_square, y_square)) * mask(x_square, y_square));
   }

   void schedule() {
      if (using_autoscheduler()) {
         intensities.dim(0).set_estimate(0, n_squares);
         intensities.dim(1).set_estimate(0, n_squares);
         slopes.dim(0).set_estimate(0, n_squares);
         slopes.dim(1).set_estimate(0, n_squares);
         jet_r.dim(0).set_estimate(0, 256);
         jet_g.dim(0).set_estimate(0, 256);
         jet_b.dim(0).set_estimate(0, 256);
         output.dim(0).set_estimate(0, n_squares);
         output.dim(1).set_estimate(0, n_squares);
         output.dim(2).set_estimate(0, 3);
      } else {
         output.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(PDRT2ThresholdJet_generator, pdrt2_threshold_jet)
