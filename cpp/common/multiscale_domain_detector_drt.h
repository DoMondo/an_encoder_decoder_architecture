#ifndef BARCODE_SEGMENTATION_MULTISCALE_DOMAIN_DETECTOR_DRT_H
#define BARCODE_SEGMENTATION_MULTISCALE_DOMAIN_DETECTOR_DRT_H


#include <HalideRuntime.h>
#include <HalideBuffer.h>

namespace MDDDRT {

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input,
                                     double w_orig_3 = 1.0, double w_orig_2 = 1.0, double w_orig_1 = 1.0,
                                     double w_orig_0 = 1.0, double w_new_3 = 1.0, double w_new_2 = 1.0,
                                     double w_new_1 = 1.0, double w_new_0 = 1.0, double threshold = 0.05);

}

#endif //BARCODE_SEGMENTATION_MULTISCALE_DOMAIN_DETECTOR_DRT_H
