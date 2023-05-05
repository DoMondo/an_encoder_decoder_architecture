#include "Halide.h"

class PDRT32Generator : public Halide::Generator<PDRT32Generator> {
private:
   Var i, j;
   const int32_t STRIDE = 32;
   const int32_t TILE_SIZE = 32;
   const int32_t VAL_N = 1024;
   const int32_t stride_bits = std::log2(STRIDE);
   const int32_t tile_size_bits = std::log2(TILE_SIZE);
   Var x{"ySquareMp1"}, y{"_slope"}, c{"writeIdx"};
   Halide::Func fm[6];

public:
   Input <Buffer<uint8_t>> in{"in", 2};
   Output <Buffer<int16_t>> fm_5{"out", 3};
   GeneratorParam<bool> transpose{"transpose", false};

   void generate() {
      Var ySquareMp1 = x;
      Var _slope = y;
      Var writeIdx = c;
      Var readIdx = c;
      if (transpose) {
         fm[0](writeIdx, _slope, ySquareMp1) = cast<int16_t>(
            in(clamp(ySquareMp1, 0, VAL_N - 1), clamp(writeIdx, 0, VAL_N - 1)));
      } else {
         fm[0](writeIdx, _slope, ySquareMp1) = cast<int16_t>(
            in(clamp(writeIdx, 0, VAL_N - 1), clamp(ySquareMp1, 0, VAL_N - 1)));
      }
      for (int32_t m = 0; m < tile_size_bits; m++) {
         int32_t M = 1 << m;
         int32_t Mp1 = 1 << (m + 1);
         int32_t nSquaresMp1 = ((VAL_N - std::min(Mp1, TILE_SIZE)) / std::min(Mp1, STRIDE)) + 1;
         int32_t in_slope_size = 2 * M - 1;
         Expr slope = _slope - Mp1 + 1;
         Expr abs_s = abs(slope);
         Expr s2 = abs_s >> 1; // floor (half of the absolute slope)
         Expr rs = abs_s - 2 * s2; // Remainder of the absolute slope
         Expr s_sign = select(slope < 0, -1, 1);
         Expr slopeM = M - 1 + s2 * s_sign;
         Expr incIndB = s_sign * (s2 + rs);
         Expr A = select((readIdx >= VAL_N) || (readIdx < 0),
                         0,
                         m < stride_bits,
                         fm[m](clamp(readIdx, 0, VAL_N - 1),
                               clamp(slopeM, 0, in_slope_size - 1),
                               clamp(ySquareMp1 << 1, 0, nSquaresMp1 * 2 - 2)),
                         fm[m](clamp(readIdx, 0, VAL_N - 1),
                               clamp(slopeM, 0, in_slope_size - 1),
                               clamp(ySquareMp1, 0, nSquaresMp1 - 1)
                         ));

         Expr B = select((readIdx + incIndB < 0) || (readIdx + incIndB >= VAL_N),
                         0,
                         m < stride_bits,
                         fm[m](clamp(readIdx + incIndB, 0, VAL_N - 1),
                               clamp(slopeM, 0, in_slope_size - 1),
                               clamp((ySquareMp1 << 1) + 1, 0, (nSquaresMp1 << 1) - 1)),
                         fm[m](clamp(readIdx + incIndB, 0, VAL_N - 1),
                               clamp(slopeM, 0, in_slope_size - 1),
                               clamp(ySquareMp1 + m - stride_bits + 1, 0, nSquaresMp1 + m - stride_bits)
                         ));
         fm[m + 1](writeIdx, _slope, ySquareMp1) = A + B;
      }
      fm_5 = fm[5];
   }

   void schedule() {
      if (using_autoscheduler()) {
         in.dim(0).set_estimate(0, VAL_N);
         in.dim(1).set_estimate(0, VAL_N);
         fm[1].set_estimate(x, 0, 512)
            .set_estimate(y, 0, 3)
            .set_estimate(c, 0, 1024);
         fm[2].set_estimate(x, 0, 256)
            .set_estimate(y, 0, 7)
            .set_estimate(c, 0, 1024);
         fm[3].set_estimate(x, 0, 128)
            .set_estimate(y, 0, 15)
            .set_estimate(c, 0, 1024);
         fm[4].set_estimate(x, 0, 64)
            .set_estimate(y, 0, 31)
            .set_estimate(c, 0, 1024);
         fm_5.set_estimate(x, 0, 32)
            .set_estimate(y, 0, 63)
            .set_estimate(c, 0, 1024);
      } else {
         fm_5.compute_root();
      }
   } // schedule
};

HALIDE_REGISTER_GENERATOR(PDRT32Generator, pdrt32)
