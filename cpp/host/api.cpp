#include <iostream>

#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "../common/image_utils.h"
#include "../common/multiscale_domain_detector_drt.h"
#include "../common/partial_strided_drt.h"
#include "../common/partial_drt32.h"
#include "../common/partial_drt2.h"


extern "C"
uint8_t *run_mdd_drt(uint8_t *input_data,
                   double w_orig_3, double w_orig_2, double w_orig_1, double w_orig_0,
                   double w_new_3, double w_new_2, double w_new_1, double w_new_0, double threshold) {
   Halide::Runtime::Buffer<uint8_t> input(input_data, 1024, 1024);
   auto output_image = MDDDRT::run(input, w_orig_3, w_orig_2, w_orig_1, w_orig_0, w_new_3, w_new_2, w_new_1, w_new_0,
                                   threshold);
   return output_image.data();
}

extern "C"
uint8_t *run_ps_drt(uint8_t *input_data) {
   Halide::Runtime::Buffer<uint8_t> input(input_data, 1024, 1024);
   auto output_image = PSDRT::run(input);
   return output_image.data();
}

extern "C"
uint8_t *run_pdrt2(uint8_t *input_data) {
   Halide::Runtime::Buffer<uint8_t> input(input_data, 1024, 1024);
   auto output_image = PDRT2::run(input);
   return output_image.data();
}
extern "C"
uint8_t *run_pdrt32(uint8_t *input_data) {
   Halide::Runtime::Buffer<uint8_t> input(input_data, 1024, 1024);
   auto output_image = PDRT32::run(input);
   return output_image.data();
}
