#include "multiscale_domain_detector_drt.h"

#include "mdd_drt_v.h"
#include "mdd_drt_h.h"
#include "mdd_bar_detector_0.h"
#include "mdd_bar_detector_1.h"
#include "mdd_bar_detector_2.h"
#include "mdd_bar_detector_3.h"
#include "mdd_bar_detector_4.h"
#include "unpool_0.h"
#include "unpool_1.h"
#include "unpool_2.h"
#include "unpool_3.h"
#include "convolutions_0.h"
#include "convolutions_1.h"
#include "convolutions_2.h"
#include "convolutions_3.h"
#include "argmaxth.h"
#include "image_utils.h"

namespace MDDDRT {

Halide::Runtime::Buffer<int16_t> drt_v_0(1024, 3, 512);
Halide::Runtime::Buffer<int16_t> drt_v_1(1024, 7, 256);
Halide::Runtime::Buffer<int16_t> drt_v_2(1024, 15, 128);
Halide::Runtime::Buffer<int16_t> drt_v_3(1024, 31, 64);
Halide::Runtime::Buffer<int16_t> drt_v_4(1024, 63, 32);

Halide::Runtime::Buffer<int16_t> drt_h_0(1024, 3, 512);
Halide::Runtime::Buffer<int16_t> drt_h_1(1024, 7, 256);
Halide::Runtime::Buffer<int16_t> drt_h_2(1024, 15, 128);
Halide::Runtime::Buffer<int16_t> drt_h_3(1024, 31, 64);
Halide::Runtime::Buffer<int16_t> drt_h_4(1024, 63, 32);

Halide::Runtime::Buffer<int16_t> encoder_0(6, 512, 512);
Halide::Runtime::Buffer<int16_t> encoder_1(14, 256, 256);
Halide::Runtime::Buffer<int16_t> encoder_2(30, 128, 128);
Halide::Runtime::Buffer<int16_t> encoder_3(62, 64, 64);
Halide::Runtime::Buffer<int16_t> encoder_4(126, 32, 32);

Halide::Runtime::Buffer<int16_t> unpool_buffer_3(62, 64, 64);
Halide::Runtime::Buffer<int16_t> unpool_buffer_2(30, 128, 128);
Halide::Runtime::Buffer<int16_t> unpool_buffer_1(30, 256, 256);
Halide::Runtime::Buffer<int16_t> unpool_buffer_0(30, 512, 512);

Halide::Runtime::Buffer<int16_t> convolutions_buffer_3(62, 64, 64);
Halide::Runtime::Buffer<int16_t> convolutions_buffer_2(30, 128, 128);
Halide::Runtime::Buffer<int16_t> convolutions_buffer_1(30, 256, 256);
Halide::Runtime::Buffer<int16_t> convolutions_buffer_0(30, 512, 512);


Halide::Runtime::Buffer<uint8_t> jetr(ImageUtils::jet_r);
Halide::Runtime::Buffer<uint8_t> jetg(ImageUtils::jet_g);
Halide::Runtime::Buffer<uint8_t> jetb(ImageUtils::jet_b);
Halide::Runtime::Buffer<uint8_t> output_image(512, 512, 3);

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input,
                                     double w_orig_3, double w_orig_2, double w_orig_1,
                                     double w_orig_0, double w_new_3, double w_new_2,
                                     double w_new_1, double w_new_0, double threshold) {
   mdd_drt_v(input, drt_v_0, drt_v_1, drt_v_2, drt_v_3, drt_v_4);
   mdd_drt_h(input, drt_h_0, drt_h_1, drt_h_2, drt_h_3, drt_h_4);
   mdd_bar_detector_0(drt_h_0, drt_v_0, encoder_0);
   mdd_bar_detector_1(drt_h_1, drt_v_1, encoder_1);
   mdd_bar_detector_2(drt_h_2, drt_v_2, encoder_2);
   mdd_bar_detector_3(drt_h_3, drt_v_3, encoder_3);
   mdd_bar_detector_4(drt_h_4, drt_v_4, encoder_4);
   unpool_3(encoder_4, encoder_3, w_new_3, w_orig_3, unpool_buffer_3);
   convolutions_3(unpool_buffer_3, convolutions_buffer_3);
   unpool_2(convolutions_buffer_3, encoder_2, w_new_2, w_orig_2, unpool_buffer_2);
   convolutions_2(unpool_buffer_2, convolutions_buffer_2);
   unpool_1(convolutions_buffer_2, encoder_1, w_new_1, w_orig_1, unpool_buffer_1);
   convolutions_1(unpool_buffer_1, convolutions_buffer_1);
   unpool_0(convolutions_buffer_1, encoder_0, w_new_0, w_orig_0, unpool_buffer_0);
   convolutions_0(unpool_buffer_0, convolutions_buffer_0);
   argmaxth(convolutions_buffer_0, jetr, jetg, jetb, threshold, output_image);
   return output_image;
}

}