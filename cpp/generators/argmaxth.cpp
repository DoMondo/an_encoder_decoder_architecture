#include "Halide.h"

namespace {

class Argmaxth_generator : public Halide::Generator<Argmaxth_generator> {
private:
   const int n_slopes = 30;

public:
   Var x_square{"y_square"};
   Var y_square{"x_square"};
   Var slope{"slope"};
   Input <Buffer<int16_t>> activations{"activations", 3};
   Input <Buffer<uint8_t>> jet_r{"jet_lookup_r", 1};
   Input <Buffer<uint8_t>> jet_g{"jet_lookup_g", 1};
   Input <Buffer<uint8_t>> jet_b{"jet_lookup_b", 1};
   Input <float> threshold{"threshold", 8.0f};
   Output <Buffer<uint8_t>> output{"output", 3};

   void generate() {
      using namespace Halide::ConciseCasts;
      RDom slope_dom(0, n_slopes);

      // Arg max
      Tuple tupl = argmax(slope_dom, activations(clamp(slope_dom, 0, n_slopes - 1),
                                                 clamp(x_square, 0, 511),
                                                 clamp(y_square, 0, 511)));
      Expr angles = cast<uint8_t>((255 * tupl[0]) / n_slopes);
      Expr intensities;
      intensities = f32(tupl[1]);
      RDom intensities_dom(0, 512, 0, 512);
      intensities = intensities / threshold;
      // Threshold
      intensities = select(intensities > 1, 1, 0);

      // Jet-colorspace
      Var color_channel;
      output(x_square, y_square, color_channel) = cast<uint8_t>(angles);
      output(x_square, y_square, color_channel) = u8(select(color_channel == 0,
                                                         jet_b(angles) * intensities,
                                                         color_channel == 1,
                                                         jet_g(angles) * intensities,
                                                         jet_r(angles) * intensities));
//      output(x_square, y_square, color_channel) = cast<uint8_t> (angles * 255);
   }

   void schedule() {
      if (using_autoscheduler()) {
         activations.dim(0).set_estimate(0, n_slopes);
         activations.dim(1).set_estimate(0, 512);
         activations.dim(2).set_estimate(0, 512);
         jet_r.dim(0).set_estimate(0, 256);
         jet_g.dim(0).set_estimate(0, 256);
         jet_b.dim(0).set_estimate(0, 256);
         output.dim(0).set_estimate(0, 512);
         output.dim(1).set_estimate(0, 512);
         output.dim(2).set_estimate(0, 3);
         threshold.set_estimate(0.06f);
      } else {
         output.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(Argmaxth_generator, argmaxth)
