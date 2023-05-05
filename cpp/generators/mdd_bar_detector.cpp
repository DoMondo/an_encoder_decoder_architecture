#include "Halide.h"
#include <math.h>

namespace {

class MDDBarDetector_generator : public Halide::Generator<MDDBarDetector_generator> {
private:
   const int VAL_N = 1024;

public:
   Var x_square{"y_square"};
   Var y_square{"x_square"};
   Var slope{"slope"};
   Input <Buffer<int16_t>> pidrt_h{"pidrt_h", 3};
   Input <Buffer<int16_t>> pidrt_v{"pidrt_v", 3};
   Output <Buffer<int16_t>> output{"out", 3};
   GeneratorParam <uint8_t> stage{"stage", 0};

   void generate() {
      using namespace Halide::ConciseCasts;
      int tile_size = (2 << (stage.value() - 1));
      int stride = (2 << (stage.value() - 1));
      int stage_size = 1 << stage.value();
      int n_squares = (VAL_N - std::min(stage_size, tile_size)) / std::min(stage_size, stride) + 1;
      int n_slopes = 2 * stage_size - 1;
      Expr y_central = y_square * stride + tile_size / 2;
      Expr x_central = x_square * stride + tile_size / 2;
      Expr signed_slope = slope - tile_size + 1;
      RDom dom(-(tile_size >> 1), ((tile_size >> 1) << 1) - 1);
      Expr disp_h = y_central + dom - signed_slope / 2;
      Expr disp_v = x_central + dom + signed_slope / 2;
      Func clamped_pidrt_h = Halide::BoundaryConditions::repeat_edge(pidrt_h);
      Func clamped_pidrt_v = Halide::BoundaryConditions::repeat_edge(pidrt_v);
      Var dx, dy, dz;
      Func diff_h, diff_v;
      diff_h(dx, dy, dz) = abs(clamped_pidrt_h(dx + 1, dy, dz) - clamped_pidrt_h(dx, dy, dz));
      diff_v(dx, dy, dz) = abs(clamped_pidrt_v(dx + 1, dy, dz) - clamped_pidrt_v(dx, dy, dz));
      Expr std_h = sum(dom, diff_h(disp_h, signed_slope + tile_size - 1, x_square));
      Expr std_v = sum(dom, diff_v(disp_v, - signed_slope + tile_size - 1, y_square));
      Func V{"V"};
      V(slope, x_square, y_square) = i16(std_h) - i16(std_v);
      Var output_slope{"Output Slope"};
      output(output_slope, x_square, y_square) = select(output_slope < n_slopes,
                                                        V(clamp(output_slope, 0, n_slopes - 1),
                                                          clamp(x_square, 0, n_squares - 1),
                                                          clamp(y_square, 0, n_squares - 1)),
                                                        -V(clamp(output_slope - n_slopes, 0, n_slopes - 1),
                                                           clamp(x_square, 0, n_squares - 1),
                                                           clamp(y_square, 0, n_squares - 1)));
      output(output_slope, 0, y_square) = i16(0);
      output(output_slope, x_square, 0) = i16(0);
      output(output_slope, n_squares - 1, y_square) = i16(0);
      output(output_slope, x_square, n_squares - 1) = i16(0);
   }

   void schedule() {
      if (using_autoscheduler()) {
         int tile_size = (2 << (stage.value() - 1));
         int stride = (2 << (stage.value() - 1));
         int stage_size = 1 << stage.value();
         int n_squares = (VAL_N - std::min(stage_size, tile_size)) / std::min(stage_size, stride) + 1;
         int n_slopes = 2 * stage_size - 1;
         pidrt_h.dim(0).set_estimate(0, n_squares);
         pidrt_h.dim(1).set_estimate(0, n_slopes);
         pidrt_h.dim(2).set_estimate(0, VAL_N);
         pidrt_v.dim(0).set_estimate(0, n_squares);
         pidrt_v.dim(1).set_estimate(0, n_slopes);
         pidrt_v.dim(2).set_estimate(0, VAL_N);
         output.dim(0).set_estimate(0, n_slopes * 2);
         output.dim(1).set_estimate(0, n_squares);
         output.dim(2).set_estimate(0, n_squares);
      } else if (get_target().has_feature(Halide::Target::OpenCL)) {
         int tile_size = (2 << (stage.value() - 1));
         int stride = (2 << (stage.value() - 1));
         int stage_size = 1 << stage.value();
         int n_squares = (VAL_N - std::min(stage_size, tile_size)) / std::min(stage_size, stride) + 1;
         int n_slopes = 2 * stage_size - 1;
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
         std::cout<< out.name() <<std::endl;
         out.bound(out_v0, 0, n_slopes * 2)
            .bound(out_v1, 0, n_squares)
            .bound(out_v2, 0, n_squares);
         out_stage0
            .gpu_blocks(out_v2)
            .gpu_threads(out_v1);
      } else{
         std::cout << " manual_sched " << std::endl;
         for (int i = 0; i < 12; i++)
            get_pipeline().get_func(i).compute_root();
         output.compute_root();
      }
   }
};

} // namespace

HALIDE_REGISTER_GENERATOR(MDDBarDetector_generator, mdd_bar_detector)
