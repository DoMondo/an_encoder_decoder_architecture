#include <iostream>

#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "../common/image_utils.h"
#include "../common/multiscale_domain_detector_drt.h"
#include "../common/partial_strided_drt.h"
#include "../common/partial_drt2.h"
#include "../common/partial_drt32.h"

std::string path = std::string(INPUT_DIR) + "cluttered.jpg";

void test_mdd() {
   Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image(path);
   std::cout << "test_mdd " << path.c_str() << std::endl;
   double time_mdd = Halide::Tools::benchmark(2, 100, [&]() {
      MDDDRT::run(input);
   });
   std::cout << "Time_mdd: " << time_mdd * 1e3 << " ms." << std::endl;
   auto output_image_mdd = MDDDRT::run(input);
   Halide::Tools::save_image(output_image_mdd, std::string(OUTPUT_DIR) + "output_image_mdd.png");
}

void test_ps() {
   Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image(path);
   std::cout << "test_ps " << path.c_str() << std::endl;
   double time_ps = Halide::Tools::benchmark(2, 100, [&]() {
      PSDRT::run(input);
   });
   std::cout << "Time_ps: " << time_ps * 1e3 << " ms." << std::endl;
   auto output_image_ps = PSDRT::run(input);
   Halide::Tools::save_image(output_image_ps, std::string(OUTPUT_DIR) + "output_image_ps.png");
}

void test_pdrt2() {
   Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image(path);
   std::cout << "test_pdrt2 " << path.c_str() << std::endl;
   double time_ps = Halide::Tools::benchmark(2, 100, [&]() {
      PDRT2::run(input);
   });
   std::cout << "Time_pdrt2: " << time_ps * 1e3 << " ms." << std::endl;
   auto output_image_ps = PDRT2::run(input);
   Halide::Tools::save_image(output_image_ps, std::string(OUTPUT_DIR) + "output_image_pdrt2.png");
}


void test_pdrt32() {
   Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image(path);
   std::cout << "test_pdrt32 " << path.c_str() << std::endl;
   double time_ps = Halide::Tools::benchmark(2, 100, [&]() {
      PDRT32::run(input);
   });
   std::cout << "Time_pdrt32: " << time_ps * 1e3 << " ms." << std::endl;
   auto output_image_ps = PDRT32::run(input);
   Halide::Tools::save_image(output_image_ps, std::string(OUTPUT_DIR) + "output_image_pdrt32.png");
}

void test_all() {
   test_pdrt2();
   test_pdrt32();
   test_mdd();
   test_ps();
}

int main() {
   test_all();
   return 0;
}
