#include "Halide.h"
#include <math.h>

namespace {

class PSBarDetector_generator : public Halide::Generator<PSBarDetector_generator> {
private:
   const int VAL_N = 1024;
   const int tile_size = 32;
   const int stride = 2;
   const int n_squares = 497;
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
      } else if (get_target().has_feature(Halide::Target::OpenCL)) {
//         auto pidrt_h_im = get_pipeline().get_func(0);
//         auto lambda_0 = get_pipeline().get_func(1);
//         auto repeat_edge = get_pipeline().get_func(2);
//         auto f0 = get_pipeline().get_func(2);
//         auto sum = get_pipeline().get_func(3);
//         auto pidrt_v_im = get_pipeline().get_func(4);
//         auto lambda_1 = get_pipeline().get_func(5);
//         auto repeat_edge$1 = get_pipeline().get_func(6);
//         auto f1 = get_pipeline().get_func(7);
//         auto sum$1 = get_pipeline().get_func(8);
//         auto V = get_pipeline().get_func(9);
         auto out = get_pipeline().get_func(11);
         auto out_stage0 = out.update(0);
         auto out_v0 = Halide::Var(out_stage0.get_schedule().dims()[0].var);
         auto out_v1 = Halide::Var(out_stage0.get_schedule().dims()[1].var);
         auto out_v2 = Halide::Var(out_stage0.get_schedule().dims()[2].var);
         std::cout << out.name() << std::endl;
         out.bound(out_v0, 0, n_slopes * 2)
                 .bound(out_v1, 0, n_squares)
                 .bound(out_v2, 0, n_squares);
         out_stage0
                 .gpu_blocks(out_v2)
                 .gpu_threads(out_v1);
      } else {
         std::cout << " manual_sched " << std::endl;
         for (int i = 0; i < 12; i++)
            get_pipeline().get_func(i).compute_root();
         slopes.compute_root();
         intensities.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(PSBarDetector_generator, ps_bar_detector)
