#include "Halide.h"

namespace {

class Convolutions_generator : public Halide::Generator<Convolutions_generator> {
private:
   const int VAL_N = 1024;
   // Convolutions generator is called for stages 1 to 4
   const int16_t n_slopes_stages[5] = {-1, 30, 30, 30, 62};


public:
   Var x_square{"y_square"};
   Var y_square{"x_square"};
   Var slope{"slope"};
   Input <Buffer<int16_t>> activations{"activations", 3};
   Output <Buffer<int16_t>> filter_vhd{"filter_vhd", 3};
   GeneratorParam <uint8_t> stage{"stage", 0};
   Func filter_v{"filter_v"};
   Func filter_vh{"filter_vh"};
   Func filter_v2{"filter_v"};
   Func filter_vh2{"filter_vh"};


   void generate() {
      using namespace Halide::ConciseCasts;
      int tile_size = (2 << (stage.value() - 1));
      int stride = (2 << (stage.value() - 1));
      int stage_size = 1 << stage.value();
      int n_slopes = n_slopes_stages[stage.value()];
      int n_squares = (VAL_N - std::min(stage_size, tile_size)) / std::min(stage_size, stride) + 1;

      Func clamped = Halide::BoundaryConditions::mirror_image(activations);

      filter_v(slope, x_square, y_square) =
              clamped(slope, x_square, y_square - 1) / 3 +
              clamped(slope, x_square, y_square) / 3 +
              clamped(slope, x_square, y_square + 1) / 3;

      filter_vh(slope, x_square, y_square) =
              filter_v(slope, x_square - 1, y_square) / 3 +
              filter_v(slope, x_square, y_square) / 3 +
              filter_v(slope, x_square + 1, y_square) / 3 ;

      filter_v2(slope, x_square, y_square) =
         filter_vh(slope, x_square, y_square - 1) / 3 +
         filter_vh(slope, x_square, y_square) / 3 +
         filter_vh(slope, x_square, y_square + 1) / 3;

      filter_vh2(slope, x_square, y_square) =
         filter_v2(slope, x_square - 1, y_square) / 3 +
         filter_v2(slope, x_square, y_square) / 3 +
         filter_v2(slope, x_square + 1, y_square) / 3 ;

      filter_vhd(slope, x_square, y_square) = \
        (filter_vh2((slope - 1) % n_slopes, x_square, y_square)) / 4 + \
        (filter_vh2(slope, x_square, y_square)) / 2 + \
        (filter_vh2((slope + 1) % n_slopes, x_square, y_square)) / 4;
   }

   void schedule() {
      if (using_autoscheduler()) {
         int n_slopes = n_slopes_stages[stage.value()];
         int tile_size = (2 << (stage.value() - 1));
         int stride = (2 << (stage.value() - 1));
         int stage_size = 1 << stage.value();
         int n_squares = (VAL_N - std::min(stage_size, tile_size)) / std::min(stage_size, stride) + 1;
         activations.dim(0).set_estimate(0, n_slopes);
         activations.dim(1).set_estimate(0, n_squares);
         activations.dim(2).set_estimate(0, n_squares);
         filter_vhd.dim(0).set_estimate(0, n_slopes);
         filter_vhd.dim(1).set_estimate(0, n_squares);
         filter_vhd.dim(2).set_estimate(0, n_squares);
      } else {
         filter_vhd.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(Convolutions_generator, convolutions)
