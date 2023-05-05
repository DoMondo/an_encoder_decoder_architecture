#include "Halide.h"
#include <math.h>

namespace {

class PDRT32BarDetector_generator : public Halide::Generator<PDRT32BarDetector_generator> {
private:
   const int VAL_N = 1024;
   const int tile_size = 32;
   const int stride = 32;
   const int n_squares = 32;
   const int n_slopes = 63;

public:
   Var x_square{"y_square"};
   Var y_square{"x_square"};
   Var slope{"slope"};
   Input <Buffer<int16_t>> pidrt_h{"pidrt_h", 3};
   Input <Buffer<int16_t>> pidrt_v{"pidrt_v", 3};
   Output <Buffer<int16_t>> intensities{"intensities", 2};
   Output <Buffer<int16_t>> slopes{"slopes", 2};
   Func is_horizontal;

   void generate() {
      using namespace Halide::ConciseCasts;
      Expr y_central = y_square * stride + tile_size / 2;
      Expr x_central = x_square * stride + tile_size / 2;
      Expr signed_slope = i32(slope) - tile_size + 1;
      RDom displ_dom(-(tile_size >> 1), ((tile_size >> 1) << 1) - 1);
      Expr disp_h = y_central + displ_dom - signed_slope / 2;
      Expr disp_v = x_central + displ_dom + signed_slope / 2;
      Func clamped_pidrt_h = Halide::BoundaryConditions::repeat_edge(pidrt_h);
      Func clamped_pidrt_v = Halide::BoundaryConditions::repeat_edge(pidrt_v);
      Var dx, dy, dz;
      Func diff_h, diff_v;
      diff_h(dx, dy, dz) = abs(clamped_pidrt_h(dx + 1, dy, dz) - clamped_pidrt_h(dx, dy, dz));
      diff_v(dx, dy, dz) = abs(clamped_pidrt_v(dx + 1, dy, dz) - clamped_pidrt_v(dx, dy, dz));
      Expr std_h = sum(displ_dom, diff_h(disp_h, signed_slope + tile_size - 1, x_square));
      Expr std_v = sum(displ_dom, diff_v(disp_v, -signed_slope + tile_size - 1, y_square));
      Func V{"V"};
      V(slope, x_square, y_square) = abs(i16(std_h) - i16(std_v));
      is_horizontal(slope, x_square, y_square) = std_h > std_v;
      RDom slope_dom(0, n_slopes);
      Tuple res = Halide::argmax(slope_dom, V(slope_dom,
                                              clamp(y_square, 0, n_squares - 1),
                                              clamp(x_square, 0, n_squares - 1)));
      slopes(y_square, x_square) = i16(
              select(is_horizontal(clamp(res[0], 0, n_slopes - 1), y_square, x_square), res[0],
                     n_slopes + res[0]));
//      slopes(y_square, x_square) = i16(res[0]);
      intensities(y_square, x_square) = i16(res[1]);
   }

   void schedule() {
      if (using_autoscheduler()) {
         pidrt_h.dim(0).set_estimate(0, n_squares);
         pidrt_h.dim(1).set_estimate(0, n_slopes);
         pidrt_h.dim(2).set_estimate(0, VAL_N);
         pidrt_v.dim(0).set_estimate(0, n_squares);
         pidrt_v.dim(1).set_estimate(0, n_slopes);
         pidrt_v.dim(2).set_estimate(0, VAL_N);
         slopes.dim(0).set_estimate(0, n_squares);
         slopes.dim(1).set_estimate(0, n_squares);
         intensities.dim(0).set_estimate(0, n_squares);
         intensities.dim(1).set_estimate(0, n_squares);
      } else {
         std::cout << " manual_sched " << std::endl;
         slopes.compute_root();
         intensities.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(PDRT32BarDetector_generator, pdrt32_bar_detector)
