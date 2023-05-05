#ifndef BARCODE_SEGMENTATION_PARTIAL_DRT32_H
#define BARCODE_SEGMENTATION_PARTIAL_DRT32_H

#include <HalideRuntime.h>
#include <HalideBuffer.h>


namespace PDRT32 {

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input);

}
#endif //BARCODE_SEGMENTATION_PARTIAL_DRT32_H
