#ifndef BARCODE_SEGMENTATION_PARTIAL_STRIDED_DRT_H
#define BARCODE_SEGMENTATION_PARTIAL_STRIDED_DRT_H

#include <HalideRuntime.h>
#include <HalideBuffer.h>

namespace PSDRT {

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input);

}

#endif //BARCODE_SEGMENTATION_PARTIAL_STRIDED_DRT_H
