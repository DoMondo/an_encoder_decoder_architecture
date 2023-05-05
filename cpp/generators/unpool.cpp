#include "Halide.h"

class UnpoolGenerator : public Halide::Generator<UnpoolGenerator> {

private:
   const int VAL_N = 1024;
   // Unpool is called for stages 1 to 4
   const int16_t coarse_slope_size[5] = {126, 62, 30, 30, 30};
   const int16_t fine_slope_size[5] = {-1, 62, 30, 14, 6};
   const int16_t fine_wh_size[5] = {-1, 64, 128, 256, 512};
public:
   Input <Buffer<int16_t>> coarse_activations{"coarse_activations", 3};
   Input <Buffer<int16_t>> fine_activations{"fine_activations", 3};
   Input<float> weight_new{"weight_new", 1.0f};
   Input<float> weight_original{"weight_original", 1.0f};
   Output <Buffer<int16_t>> new_fine_activations{"new_fine_activations", 3};
   GeneratorParam <uint8_t> stage{"stage", 0};
   Var x_square{"x_square"};
   Var y_square{"y_square"};
   Var slope{"slope"};


   void generate() {
      using namespace Halide::ConciseCasts;
      int16_t n_squares_fine = fine_wh_size[stage.value()];
      int16_t n_slopes_fine = fine_slope_size[stage.value()];
      int16_t n_slopes_coarse = coarse_slope_size[stage.value() - 1];
      int16_t n_slopes_output = coarse_slope_size[stage.value()];
      int slope_ratio = n_slopes_coarse / n_slopes_fine;
      float new_activations_slope_ratio = (float) n_slopes_coarse / (float) n_slopes_output;
      RDom slope_dom(0, n_slopes_coarse);
      Tuple tuple = argmax(slope_dom,
                           coarse_activations(clamp(slope_dom, 0, n_slopes_coarse - 1),
                                              clamp(i32(x_square) / 2, 0, n_squares_fine / 2 - 1),
                                              clamp(i32(y_square) / 2, 0, n_squares_fine / 2 - 1)));
      Expr max_slope_indices = tuple[0];
      Expr values = tuple[1];
      Expr fine_activations_coarser_slope = (cast<int>(max_slope_indices) / slope_ratio) % n_slopes_fine;
      RDom ij(0, 2, 0, 2);
      Expr x_square_rounded = u16(x_square * 0.5f) * 2;
      Expr y_square_rounded = u16(y_square * 0.5f) * 2;
      Tuple second_tuple = argmax(
         ij, fine_activations(
            clamp(fine_activations_coarser_slope, 0, n_slopes_fine - 1),
            clamp(u16(x_square_rounded + ij.x), 0, n_squares_fine - 1),
            clamp(u16(y_square_rounded + ij.y), 0, n_squares_fine - 1)));
      Expr jj = second_tuple[0];
      Expr ii = second_tuple[1];

      Expr output_slope = round(max_slope_indices / new_activations_slope_ratio) % n_slopes_output;
      new_fine_activations(slope, x_square, y_square) = select(
         (slope == output_slope) &&
         (x_square == (x_square_rounded + jj)) &&
         (y_square == (y_square_rounded + ii)),
         values,
         0
      );
      // add_original_activations
      int add_slope_ratio = n_slopes_output / n_slopes_fine;
      new_fine_activations(slope, x_square, y_square) = i16(new_fine_activations(slope, x_square, y_square) * weight_new +
                                                        fine_activations(clamp(
                                                           (i32(slope) / i32(add_slope_ratio)) % i32(n_slopes_fine), 0,
                                                           n_slopes_fine), x_square, y_square) * weight_original);
   }

   void schedule() {
      if (using_autoscheduler()) {
         int n_squares_fine = fine_wh_size[stage.value()];
         int n_slopes_fine = fine_slope_size[stage.value()];
         int n_slopes_coarse = coarse_slope_size[stage.value() - 1];
         int n_slopes_output = coarse_slope_size[stage.value()];
         coarse_activations.dim(0).set_estimate(0, n_slopes_coarse);
         coarse_activations.dim(1).set_estimate(0, n_squares_fine / 2);
         coarse_activations.dim(2).set_estimate(0, n_squares_fine / 2);
         fine_activations.dim(0).set_estimate(0, n_slopes_fine);
         fine_activations.dim(1).set_estimate(0, n_squares_fine);
         fine_activations.dim(2).set_estimate(0, n_squares_fine);
         new_fine_activations.dim(0).set_estimate(0, n_slopes_output);
         new_fine_activations.dim(1).set_estimate(0, n_squares_fine);
         new_fine_activations.dim(2).set_estimate(0, n_squares_fine);
         weight_new.set_estimate(1.0f);
         weight_original.set_estimate(1.0f);
      } else {
         new_fine_activations.compute_root();
      }
   } // schedule
};

HALIDE_REGISTER_GENERATOR(UnpoolGenerator, unpool)
