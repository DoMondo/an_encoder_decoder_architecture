cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
find_package(Halide 15 REQUIRED)
set(CMAKE_CXX_EXTENSIONS NO)

set(TARGET arm-64-android)

add_halide_generator(mdd_drt_${TARGET}.generator
        SOURCES ../generators/mdd_drt.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_drt_${TARGET}.generator
        SOURCES ../generators/ps_drt.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2_${TARGET}.generator
        SOURCES ../generators/pdrt2.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32_${TARGET}.generator
        SOURCES ../generators/pdrt32.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(mdd_bar_detector_${TARGET}.generator
        SOURCES ../generators/mdd_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_bar_detector_${TARGET}.generator
        SOURCES ../generators/ps_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2_bar_detector_${TARGET}.generator
        SOURCES ../generators/pdrt2_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32_bar_detector_${TARGET}.generator
        SOURCES ../generators/pdrt32_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_threshold_jet_${TARGET}.generator
        SOURCES ../generators/ps_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2_threshold_jet_${TARGET}.generator
        SOURCES ../generators/pdrt2_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32_threshold_jet_${TARGET}.generator
        SOURCES ../generators/pdrt32_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(unpool_${TARGET}.generator
        SOURCES ../generators/unpool.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(convolutions_${TARGET}.generator
        SOURCES ../generators/convolutions.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(argmaxth_${TARGET}.generator
        SOURCES ../generators/argmaxth.cpp
        LINK_LIBRARIES Halide::Tools)


add_custom_target(
        build_and_run_android_executable
        COMMAND cd ${CMAKE_SOURCE_DIR}/arm64-android &&
                BUILD_DIR=${CMAKE_BINARY_DIR}/arm64-android
                SRC_DIR=${CMAKE_SOURCE_DIR}/arm64-android
                TARGET=${TARGET}
                Halide_DIR=${Halide_DIR}/../../.. make
        DEPENDS argmaxth_${TARGET}.generator
                ps_drt_${TARGET}.generator
                pdrt2_${TARGET}.generator
                pdrt32_${TARGET}.generator
                mdd_drt_${TARGET}.generator
                mdd_drt_${TARGET}.generator
                pdrt2_bar_detector_${TARGET}.generator
                pdrt32_bar_detector_${TARGET}.generator
                ps_bar_detector_${TARGET}.generator
                ps_threshold_jet_${TARGET}.generator
                pdrt2_threshold_jet_${TARGET}.generator
                pdrt32_threshold_jet_${TARGET}.generator
                mdd_bar_detector_${TARGET}.generator
                unpool_${TARGET}.generator
                convolutions_${TARGET}.generator
)

add_executable(dummy
        main.cpp
        ../common/image_utils.cpp
        ../common/image_utils.h
        ../generators/ps_drt.cpp
        ../generators/mdd_drt.cpp
        ../generators/pdrt2.cpp
        ../generators/pdrt32.cpp
        ../generators/pdrt2_bar_detector.cpp
        ../generators/pdrt32_bar_detector.cpp
        ../generators/pdrt2_threshold_jet.cpp
        ../generators/pdrt32_threshold_jet.cpp
        ../generators/ps_bar_detector.cpp
        ../generators/mdd_bar_detector.cpp
        ../generators/unpool.cpp
        ../generators/convolutions.cpp
        ../generators/argmaxth.cpp
        ../common/multiscale_domain_detector_drt.cpp
        ../common/multiscale_domain_detector_drt.h
        ../common/partial_strided_drt.cpp
        ../common/partial_strided_drt.h
        ../common/partial_drt2.cpp
        ../common/partial_drt2.h
        ../common/partial_drt32.cpp
        ../common/partial_drt32.h
        )

target_link_libraries(dummy
        PRIVATE
        Halide::Halide
        Halide::ImageIO
        Halide::Tools
        ps_drt_h
        ps_drt_v
        pdrt2_h
        pdrt2_v
        pdrt32_h
        pdrt32_v
        mdd_drt_h
        mdd_drt_v
        pdrt2_bar_detector
        pdrt32_bar_detector
        ps_bar_detector
        ps_threshold_jet
        pdrt2_threshold_jet
        pdrt32_threshold_jet
        mdd_bar_detector_0
        mdd_bar_detector_1
        mdd_bar_detector_2
        mdd_bar_detector_3
        mdd_bar_detector_4
        unpool_0
        unpool_1
        unpool_2
        unpool_3
        convolutions_0
        convolutions_1
        convolutions_2
        convolutions_3
        argmaxth
        )
