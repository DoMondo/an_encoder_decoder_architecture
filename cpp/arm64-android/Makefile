
HALIDE_HOST_BIN_DIR := ${Halide_DIR}/lib
HALIDE_INCLUDE_DIR := ${Halide_DIR}/include
HALIDE_TOOLS_DIR := ${Halide_DIR}/share/Halide/tools
DEVICE_PATH:=/data/local/tmp

CPP_FLAGS:= \
	-I $(LIB_PNG_LOCATION) \
	-I $(LIB_PNG_LOCATION)/build \
	-L$(LIB_JPEG_LOCATION)/build \
	-I $(LIB_JPEG_LOCATION) \
	-I $(LIB_JPEG_LOCATION)/build/ \
	-I $(HALIDE_TOOLS_DIR) \
	-I $(HALIDE_INCLUDE_DIR) \
	-std=c++17  \
	-Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers \
	-Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -Wall -O3 \
	-lm -llog -landroid -latomic -lEGL -fPIE -pie -lz \
    -static-libstdc++ -D__ANDROID__ -ldl

EXE_NAME := barcode-segmentation-arm

default: run_android

BINARY_DEPS := ${BUILD_DIR}/argmaxth.a
BINARY_DEPS += ${BUILD_DIR}/ps_drt_h.a
BINARY_DEPS += ${BUILD_DIR}/ps_drt_v.a
BINARY_DEPS += ${BUILD_DIR}/pdrt2_h.a
BINARY_DEPS += ${BUILD_DIR}/pdrt2_v.a
BINARY_DEPS += ${BUILD_DIR}/pdrt32_h.a
BINARY_DEPS += ${BUILD_DIR}/pdrt32_v.a
BINARY_DEPS += ${BUILD_DIR}/mdd_drt_h.a
BINARY_DEPS += ${BUILD_DIR}/mdd_drt_v.a
BINARY_DEPS += ${BUILD_DIR}/pdrt2_bar_detector.a
BINARY_DEPS += ${BUILD_DIR}/pdrt32_bar_detector.a
BINARY_DEPS += ${BUILD_DIR}/ps_bar_detector.a
BINARY_DEPS += ${BUILD_DIR}/ps_threshold_jet.a
BINARY_DEPS += ${BUILD_DIR}/pdrt2_threshold_jet.a
BINARY_DEPS += ${BUILD_DIR}/pdrt32_threshold_jet.a
BINARY_DEPS += ${BUILD_DIR}/mdd_bar_detector_0.a
BINARY_DEPS += ${BUILD_DIR}/mdd_bar_detector_1.a
BINARY_DEPS += ${BUILD_DIR}/mdd_bar_detector_2.a
BINARY_DEPS += ${BUILD_DIR}/mdd_bar_detector_3.a
BINARY_DEPS += ${BUILD_DIR}/mdd_bar_detector_4.a
BINARY_DEPS += ${BUILD_DIR}/unpool_0.a
BINARY_DEPS += ${BUILD_DIR}/unpool_1.a
BINARY_DEPS += ${BUILD_DIR}/unpool_2.a
BINARY_DEPS += ${BUILD_DIR}/unpool_3.a
BINARY_DEPS += ${BUILD_DIR}/convolutions_0.a
BINARY_DEPS += ${BUILD_DIR}/convolutions_1.a
BINARY_DEPS += ${BUILD_DIR}/convolutions_2.a
BINARY_DEPS += ${BUILD_DIR}/convolutions_3.a

CPP_DEPS := main.cpp
CPP_DEPS += ../common/image_utils.cpp
CPP_DEPS += ../common/multiscale_domain_detector_drt.cpp
CPP_DEPS += ../common/partial_drt2.cpp
CPP_DEPS += ../common/partial_drt32.cpp
CPP_DEPS += ../common/partial_strided_drt.cpp

${BUILD_DIR}/${EXE_NAME}: ${BINARY_DEPS} ${CPP_DEPS} Makefile
	@echo generating $@
	${CPP} ${CPP_DEPS} \
 		${BINARY_DEPS} \
 		${LIB_PNG_LOCATION}/build/libpng.a \
 		${LIB_JPEG_LOCATION}/build/libjpeg.a \
 		-g \
		${CPP_FLAGS} \
		-I ${BUILD_DIR} \
		-L${BUILD_DIR} \
		-o $@

run_android: ${BUILD_DIR}/${EXE_NAME} Makefile
	adb push ${BUILD_DIR}/${EXE_NAME} ${DEVICE_PATH}
	adb push "../../examples/cluttered.jpg" ${DEVICE_PATH}
	adb shell "cd ${DEVICE_PATH} && ./${EXE_NAME}"
	adb pull ${DEVICE_PATH}/output_image_mdd.png ../outputs/
	adb pull ${DEVICE_PATH}/output_image_ps.png ../outputs/
	adb pull ${DEVICE_PATH}/output_image_pdrt2.png ../outputs/
	adb pull ${DEVICE_PATH}/output_image_pdrt32.png ../outputs/

PARALLELISM := 8

GEN_ARTIFACTS := schedule,static_library,registration,c_header
AUTOSCHEDULER_GEN_OPTIONS := autoscheduler=Adams2019 autoscheduler.parallelism=${PARALLELISM}

${BUILD_DIR}/ps_drt_h.a: ${BUILD_DIR}/ps_drt_${TARGET}.generator
	@echo generating $@
	@$< -g ps_drt \
	   -f ps_drt_h \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=true

${BUILD_DIR}/ps_drt_v.a: ${BUILD_DIR}/ps_drt_${TARGET}.generator
	@echo generating $@
	@$< -g ps_drt \
	   -f ps_drt_v \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=false

${BUILD_DIR}/mdd_drt_h.a: ${BUILD_DIR}/mdd_drt_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_drt \
	   -f mdd_drt_h \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=true

${BUILD_DIR}/mdd_drt_v.a: ${BUILD_DIR}/mdd_drt_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_drt \
	   -f mdd_drt_v \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=false

${BUILD_DIR}/pdrt2_h.a: ${BUILD_DIR}/pdrt2_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt2 \
	   -f pdrt2_h \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=true

${BUILD_DIR}/pdrt2_v.a: ${BUILD_DIR}/pdrt2_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt2 \
	   -f pdrt2_v \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=false

${BUILD_DIR}/pdrt32_h.a: ${BUILD_DIR}/pdrt32_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt32 \
	   -f pdrt32_h \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=true

${BUILD_DIR}/pdrt32_v.a: ${BUILD_DIR}/pdrt32_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt32 \
	   -f pdrt32_v \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} transpose=false

${BUILD_DIR}/pdrt2_bar_detector.a: ${BUILD_DIR}/pdrt2_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt2_bar_detector \
	   -f pdrt2_bar_detector \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/pdrt32_bar_detector.a: ${BUILD_DIR}/pdrt32_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt32_bar_detector \
	   -f pdrt32_bar_detector \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/ps_bar_detector.a: ${BUILD_DIR}/ps_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g ps_bar_detector \
	   -f ps_bar_detector \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/ps_threshold_jet.a: ${BUILD_DIR}/ps_threshold_jet_${TARGET}.generator
	@echo generating $@
	@$< -g ps_threshold_jet \
	   -f ps_threshold_jet \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/pdrt2_threshold_jet.a: ${BUILD_DIR}/pdrt2_threshold_jet_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt2_threshold_jet \
	   -f pdrt2_threshold_jet \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/pdrt32_threshold_jet.a: ${BUILD_DIR}/pdrt32_threshold_jet_${TARGET}.generator
	@echo generating $@
	@$< -g pdrt32_threshold_jet \
	   -f pdrt32_threshold_jet \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

${BUILD_DIR}/mdd_bar_detector_0.a: ${BUILD_DIR}/mdd_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_bar_detector \
	   -f mdd_bar_detector_0 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=1

${BUILD_DIR}/mdd_bar_detector_1.a: ${BUILD_DIR}/mdd_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_bar_detector \
	   -f mdd_bar_detector_1 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=2

${BUILD_DIR}/mdd_bar_detector_2.a: ${BUILD_DIR}/mdd_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_bar_detector \
	   -f mdd_bar_detector_2 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=3

${BUILD_DIR}/mdd_bar_detector_3.a: ${BUILD_DIR}/mdd_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_bar_detector \
	   -f mdd_bar_detector_3 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=4

${BUILD_DIR}/mdd_bar_detector_4.a: ${BUILD_DIR}/mdd_bar_detector_${TARGET}.generator
	@echo generating $@
	@$< -g mdd_bar_detector \
	   -f mdd_bar_detector_4 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=5

${BUILD_DIR}/unpool_0.a: ${BUILD_DIR}/unpool_${TARGET}.generator
	@echo generating $@
	@$< -g unpool \
	   -f unpool_0 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=4

${BUILD_DIR}/unpool_1.a: ${BUILD_DIR}/unpool_${TARGET}.generator
	@echo generating $@
	@$< -g unpool \
	   -f unpool_1 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=3

${BUILD_DIR}/unpool_2.a: ${BUILD_DIR}/unpool_${TARGET}.generator
	@echo generating $@
	@$< -g unpool \
	   -f unpool_2 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=2

${BUILD_DIR}/unpool_3.a: ${BUILD_DIR}/unpool_${TARGET}.generator
	@echo generating $@
	@$< -g unpool \
	   -f unpool_3 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=1


${BUILD_DIR}/convolutions_0.a: ${BUILD_DIR}/convolutions_${TARGET}.generator
	@echo generating $@
	@$< -g convolutions \
	   -f convolutions_0 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=1

${BUILD_DIR}/convolutions_1.a: ${BUILD_DIR}/convolutions_${TARGET}.generator
	@echo generating $@
	@$< -g convolutions \
	   -f convolutions_1 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=2

${BUILD_DIR}/convolutions_2.a: ${BUILD_DIR}/convolutions_${TARGET}.generator
	@echo generating $@
	@$< -g convolutions \
	   -f convolutions_2 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=3

${BUILD_DIR}/convolutions_3.a: ${BUILD_DIR}/convolutions_${TARGET}.generator
	@echo generating $@
	@$< -g convolutions \
	   -f convolutions_3 \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS} stage=4

${BUILD_DIR}/argmaxth.a: ${BUILD_DIR}/argmaxth_${TARGET}.generator
	@echo generating $@
	@$< -g argmaxth \
	   -f argmaxth \
	   -o ${BUILD_DIR} \
	   -e ${GEN_ARTIFACTS} \
	   -p ${HALIDE_HOST_BIN_DIR}/libautoschedule_adams2019.so \
	   target=${TARGET} ${AUTOSCHEDULER_GEN_OPTIONS}

