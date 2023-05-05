cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
find_package(Halide 15 REQUIRED)
set(CMAKE_CXX_EXTENSIONS NO)
#set(autoscheduler_name Mullapudi2016)
set(autoscheduler_name Adams2019)
#set(autoscheduler_name Li2018)

add_halide_generator(mdd_drt.generator
        SOURCES ../generators/mdd_drt.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_drt.generator
        SOURCES ../generators/ps_drt.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2.generator
        SOURCES ../generators/pdrt2.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32.generator
        SOURCES ../generators/pdrt32.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(mdd_bar_detector.generator
        SOURCES ../generators/mdd_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_bar_detector.generator
        SOURCES ../generators/ps_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2_bar_detector.generator
        SOURCES ../generators/pdrt2_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32_bar_detector.generator
        SOURCES ../generators/pdrt32_bar_detector.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(ps_threshold_jet.generator
        SOURCES ../generators/ps_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt2_threshold_jet.generator
        SOURCES ../generators/pdrt2_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(pdrt32_threshold_jet.generator
        SOURCES ../generators/pdrt32_threshold_jet.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(unpool.generator
        SOURCES ../generators/unpool.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(convolutions.generator
        SOURCES ../generators/convolutions.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_generator(argmaxth.generator
        SOURCES ../generators/argmaxth.cpp
        LINK_LIBRARIES Halide::Tools)

add_halide_library(ps_drt_h FROM ps_drt.generator
        GENERATOR ps_drt
        PARAMS transpose=true
        SCHEDULE ps_drt_h_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(ps_drt_v FROM ps_drt.generator
        GENERATOR ps_drt
        PARAMS transpose=false
        SCHEDULE ps_drt_v_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_drt_h FROM mdd_drt.generator
        GENERATOR mdd_drt
        PARAMS transpose=true
        SCHEDULE mdd_drt_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_drt_v FROM mdd_drt.generator
        GENERATOR mdd_drt
        PARAMS transpose=false
        SCHEDULE mdd_drt_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt2_h FROM pdrt2.generator
        GENERATOR pdrt2
        PARAMS transpose=true
        SCHEDULE pdrt2_h_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt2_v FROM pdrt2.generator
        GENERATOR pdrt2
        PARAMS transpose=false
        SCHEDULE pdrt2_v_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt32_h FROM pdrt32.generator
        GENERATOR pdrt32
        PARAMS transpose=true
        SCHEDULE pdrt32_h_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt32_v FROM pdrt32.generator
        GENERATOR pdrt32
        PARAMS transpose=false
        SCHEDULE pdrt32_v_auto_schedule_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt2_bar_detector FROM pdrt2_bar_detector.generator
        GENERATOR pdrt2_bar_detector
        SCHEDULE pdrt2_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt32_bar_detector FROM pdrt32_bar_detector.generator
        GENERATOR pdrt32_bar_detector
        SCHEDULE pdrt32_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(ps_bar_detector FROM ps_bar_detector.generator
        GENERATOR ps_bar_detector
        SCHEDULE ps_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(ps_threshold_jet FROM ps_threshold_jet.generator
        GENERATOR ps_threshold_jet
        SCHEDULE ps_threshold_jet_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt2_threshold_jet FROM pdrt2_threshold_jet.generator
        GENERATOR pdrt2_threshold_jet
        SCHEDULE pdrt2_threshold_jet_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(pdrt32_threshold_jet FROM pdrt32_threshold_jet.generator
        GENERATOR pdrt32_threshold_jet
        SCHEDULE pdrt32_threshold_jet_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_bar_detector_0 FROM mdd_bar_detector.generator
        GENERATOR mdd_bar_detector
        PARAMS stage=1
        SCHEDULE mdd_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_bar_detector_1 FROM mdd_bar_detector.generator
        GENERATOR mdd_bar_detector
        PARAMS stage=2
        SCHEDULE mdd_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_bar_detector_2 FROM mdd_bar_detector.generator
        GENERATOR mdd_bar_detector
        PARAMS stage=3
        SCHEDULE mdd_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_bar_detector_3 FROM mdd_bar_detector.generator
        GENERATOR mdd_bar_detector
        PARAMS stage=4
        SCHEDULE mdd_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(mdd_bar_detector_4 FROM mdd_bar_detector.generator
        GENERATOR mdd_bar_detector
        PARAMS stage=5
        SCHEDULE mdd_bar_detector_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(unpool_0 FROM unpool.generator
        GENERATOR unpool
        PARAMS stage=4 autoscheduler.parallelism=16
        SCHEDULE unpool_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(unpool_1 FROM unpool.generator
        GENERATOR unpool
        PARAMS stage=3 autoscheduler.parallelism=16
        SCHEDULE unpool_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(unpool_2 FROM unpool.generator
        GENERATOR unpool
        PARAMS stage=2 autoscheduler.parallelism=16
        SCHEDULE unpool_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(unpool_3 FROM unpool.generator
        GENERATOR unpool
        PARAMS stage=1 autoscheduler.parallelism=16
        SCHEDULE unpool_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(convolutions_0 FROM convolutions.generator
        GENERATOR convolutions
        PARAMS stage=1
        SCHEDULE convolutions_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(convolutions_1 FROM convolutions.generator
        GENERATOR convolutions
        PARAMS stage=2
        SCHEDULE convolutions_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(convolutions_2 FROM convolutions.generator
        GENERATOR convolutions
        PARAMS stage=3
        SCHEDULE convolutions_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(convolutions_3 FROM convolutions.generator
        GENERATOR convolutions
        PARAMS stage=4
        SCHEDULE convolutions_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})

add_halide_library(argmaxth FROM argmaxth.generator
        GENERATOR argmaxth
        SCHEDULE argmaxth_SCHEDULE
        AUTOSCHEDULER Halide::${autoscheduler_name})


add_library(barcode_segmentation_lib SHARED
        api.cpp
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

add_executable(barcode_segmentation_host
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

target_link_libraries(barcode_segmentation_lib
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

target_link_libraries(barcode_segmentation_host
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
        pdrt2_bar_detector
        pdrt32_bar_detector
        mdd_drt_h
        mdd_drt_v
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

target_compile_definitions(barcode_segmentation_host PUBLIC INPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../inputs/")
target_compile_definitions(barcode_segmentation_host PUBLIC OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../outputs/")

