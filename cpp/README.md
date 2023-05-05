# C++/Halide implementation


This version in this directory allows the benchmarking and accelerated execution in multiple targets (desktop and mobile). It contains the four algorithms that are compared in the paper:

 * PDRT 2
 * PDRT 32
 * PS DRT
 * MDD DRT

## Setup
Download with `Halide 15.0.0` from the [releases](https://github.com/halide/Halide/releases/tag/v15.0.0) page. And put extract it in a directory of your choice `$HALIDE_DIR`.

## Host (x86 desktop CPU)
### Building
These instructions are for building the executable on the desktop CPU for benchmarking and checking results.

```shell
mkdir build
cd build
export HALIDE_DIR=<YOUR_HALIDE_INSTALLATION_PATH>
cmake -DHalide_DIR=$HALIDE_DIR/lib/cmake/Halide ..
make barcode_segmentation_host
```

### Running 

```shell
cd host
./barcode_segmentation_host
```

This will output the benchmarked times of the four algorithms and save the outputs in `outputs`.

## Building a dynamic library for Python
For convenience, the algorithms can be compiled into a dynamic library that can be called from python. For this run:
```shell
mkdir build
cd build
export HALIDE_DIR=<YOUR_HALIDE_INSTALLATION_PATH>
cmake -DHalide_DIR=$HALIDE_DIR/lib/cmake/Halide ..
make barcode_segmentation_lib
```


## Android (CPU)
These instructions are for building the executable on an Android CPU for benchmarking and checking results.

### Setup

You'll need ADB available in your path. An Android toolchain is also needed. You can use the one that comes within the Android NDK that can be installed by the SDK manager of Android Studio. It is usually located in `~/Android/Sdk/ndk/<ndk_version>`. I'll refer to this path as `<NDK_PATH>`.

#### Build libpng
```shell
git clone https://github.com/glennrp/libpng
cd libpng
mkdir build 
cd build
export NDK_PATH=<YOUR_NDK_INSTALLATION_PATH>
cmake \
   -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
   -DANDROID_ABI=arm64-v8a\
   ..
make
```

#### Build libjpeg
```shell
git clone https://github.com/libjpeg-turbo/libjpeg-turbo
cd libjpeg-turbo
mkdir build 
cd build
export NDK_PATH=<YOUR_NDK_INSTALLATION_PATH>
cmake \
   -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
   -DANDROID_ABI=arm64-v8a\
   ..
make 
```

### Building

```shell
# Omit these three lines if already configured previously for host
mkdir build
cd build
export HALIDE_DIR=<YOUR_HALIDE_INSTALLATION_PATH>
cmake -DHalide_DIR=$HALIDE_DIR/lib/cmake/Halide ..

export NDK_PATH=<YOUR_NDK_INSTALLATION_PATH>
export LIBPNG=<YOUR_LIBPNG_INSTALLATION_PATH>
export LIBJPEG=<YOUR_LIBJPEG_INSTALLATION_PATH>

cd arm64-android
CPP=$NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++ \
    LIB_PNG_LOCATION=$LIBPNG \
    LIB_JPEG_LOCATION=$LIBJPEG \
    make build_and_run_android_executable 
```
This will output the benchmarked times of the four algorithms and save the output in `outputs`.
