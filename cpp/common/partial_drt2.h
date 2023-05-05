#ifndef BARCODE_SEGMENTATION_PARTIAL_DRT2_H
#define BARCODE_SEGMENTATION_PARTIAL_DRT2_H

#include <HalideRuntime.h>
#include <HalideBuffer.h>


namespace PDRT2 {

Halide::Runtime::Buffer<uint8_t> run(Halide::Runtime::Buffer<uint8_t> &input);

}
#endif //BARCODE_SEGMENTATION_PARTIAL_DRT2_H
