import argparse
import ctypes
import glob
import math
import os
import cv2
import numpy as np
from natsort import natsort
import multiscale_domain_detector_drt
import partial_drt
import partial_strided_drt


def get_file_list(path: str):
    extensions = ['jpg', 'png', 'jpeg']
    files = []
    for extension in extensions:
        files.extend(natsort.natsorted(glob.glob(f'{path}/*.{extension}')))
    return files


def run_python_implementation(files: [str]):
    print('Running python implementation')
    for file in files:
        image_name = os.path.basename(file).split('.')[0]
        print(f'Processing {image_name}')
        in_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if in_img.shape[0] != 1024:
            print(f'Warning: input image {image_name} is not 1024x1024. Resizing.')
            in_img = cv2.resize(in_img, (1024, 1024))
        in_img = np.double(in_img)
        in_img -= in_img.min()
        in_img /= in_img.max()
        in_img = np.uint8(255 * in_img)
        intensity_a, slopes_a = partial_strided_drt.run(in_img.copy(), 2, 32)
        intensity_b, slopes_b = multiscale_domain_detector_drt.run(in_img)
        intensity_c, slopes_c = partial_drt.run(in_img, 2)
        intensity_d, slopes_d = partial_drt.run(in_img, 32)

        ps_result = threshold(intensity_a, slopes_a, 0.1832)
        ps_result[0:5, :, :] = 0
        ps_result[:, 0:5, :] = 0
        ps_result[-5:, :, :] = 0
        ps_result[:, -5:, :] = 0

        pdrt2_result = threshold(intensity_c, slopes_c, 0.029)
        pdrt2_result[0, :, :] = 0
        pdrt2_result[:, 0, :] = 0
        pdrt2_result[-1, :, :] = 0
        pdrt2_result[:, -1, :] = 0

        pdrt32_result = threshold(intensity_d, slopes_d, 0.25)
        pdrt32_result[0, :, :] = 0
        pdrt32_result[:, 0, :] = 0
        pdrt32_result[-1, :, :] = 0
        pdrt32_result[:, -1, :] = 0

        mddrt = threshold(intensity_b, slopes_b, 12, normalize=False)

        os.makedirs('out', exist_ok=True)
        cv2.imwrite(f'out/python_{image_name}_pdrt2.png', pdrt2_result)
        cv2.imwrite(f'out/python_{image_name}_pdrt32.png', pdrt32_result)
        cv2.imwrite(f'out/python_{image_name}_ps.png', ps_result)
        cv2.imwrite(f'out/python_{image_name}_mdd.png', mddrt)


def run_halide_implementation(files: [str]):
    print('Running halide implementation')
    libname = '../cpp/build/host/libbarcode_segmentation_lib.so'
    try:
        clib = ctypes.CDLL(libname)
    except OSError as e:
        print(f'The library was not found. Check the README.md for instructions. {e}')

    run_mdd_drt = clib.run_mdd_drt
    run_ps_drt = clib.run_ps_drt
    run_pdrt2 = clib.run_pdrt2
    run_pdrt32 = clib.run_pdrt32

    run_mdd_drt.restype = ctypes.POINTER(ctypes.c_uint8 * 512 * 512 * 3)
    run_mdd_drt.argtypes = [np.ctypeslib.ndpointer(dtype=np.dtype('uint8'), shape=(1024, 1024)),
                            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    run_ps_drt.restype = ctypes.POINTER(ctypes.c_uint8 * 497 * 497 * 3)
    run_ps_drt.argtypes = [np.ctypeslib.ndpointer(dtype=np.dtype('uint8'), shape=(1024, 1024))]

    run_pdrt2.restype = ctypes.POINTER(ctypes.c_uint8 * 512 * 512 * 3)
    run_pdrt2.argtypes = [np.ctypeslib.ndpointer(dtype=np.dtype('uint8'), shape=(1024, 1024))]

    run_pdrt32.restype = ctypes.POINTER(ctypes.c_uint8 * 32 * 32 * 3)
    run_pdrt32.argtypes = [np.ctypeslib.ndpointer(dtype=np.dtype('uint8'), shape=(1024, 1024))]

    for file in files:
        image_name = os.path.basename(file).split('.')[0]
        print(f'Processing {image_name}')
        in_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if in_img.shape[0] != 1024:
            print(f'Warning: input image {image_name} is not 1024x1024. Resizing.')
            in_img = cv2.resize(in_img, (1024, 1024))
        in_img = np.double(in_img)
        in_img -= in_img.min()
        in_img /= in_img.max()
        in_img = np.uint8(255 * in_img)
        mdd_weights = [0.05, 0.527, 0.33, 0.76, 0.84, 0.84, 1.16, 3.47]
        mdd_threshold = 1
        mdd_result = run_mdd_drt(in_img, *mdd_weights, mdd_threshold)
        mdd_result = np.array(mdd_result.contents)
        mdd_result = np.moveaxis(mdd_result, 0, 2)
        mdd_result = cv2.cvtColor(mdd_result, cv2.COLOR_RGB2BGR)

        ps_result = run_ps_drt(in_img)
        ps_result = np.array(ps_result.contents)
        ps_result = np.moveaxis(ps_result, 0, 2)
        ps_result[0:5, :, :] = 0
        ps_result[:, 0:5, :] = 0
        ps_result[-5:, :, :] = 0
        ps_result[:, -5:, :] = 0
        ps_result = cv2.cvtColor(ps_result, cv2.COLOR_RGB2BGR)
        border_size = (512 - 497) // 2 + 1
        ps_result = cv2.copyMakeBorder(ps_result, border_size, border_size, border_size, border_size,
                                       borderType=cv2.BORDER_CONSTANT)

        pdrt2_result = run_pdrt2(in_img)
        pdrt2_result = np.array(pdrt2_result.contents)
        pdrt2_result = np.moveaxis(pdrt2_result, 0, 2)
        pdrt2_result[0, :, :] = 0
        pdrt2_result[:, 0, :] = 0
        pdrt2_result[-1, :, :] = 0
        pdrt2_result[:, -1, :] = 0
        pdrt2_result = cv2.cvtColor(pdrt2_result, cv2.COLOR_RGB2BGR)

        pdrt32_result = run_pdrt32(in_img)
        pdrt32_result = np.array(pdrt32_result.contents)
        pdrt32_result = np.moveaxis(pdrt32_result, 0, 2)
        pdrt32_result[0, :, :] = 0
        pdrt32_result[:, 0, :] = 0
        pdrt32_result[-1, :, :] = 0
        pdrt32_result[:, -1, :] = 0
        pdrt32_result = cv2.cvtColor(pdrt32_result, cv2.COLOR_RGB2BGR)
        pdrt32_result = cv2.resize(pdrt32_result, (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)

        os.makedirs('out', exist_ok=True)
        cv2.imwrite(f'out/halide_{image_name}_mdd.png', mdd_result)
        cv2.imwrite(f'out/halide_{image_name}_ps.png', ps_result)
        cv2.imwrite(f'out/halide_{image_name}_pdrt2.png', pdrt2_result)
        cv2.imwrite(f'out/halide_{image_name}_pdrt32.png', pdrt32_result)


def threshold(intensities, angles, threshold, normalize=True):
    if threshold:
        if normalize:
            intensities = intensities.astype(float)
            intensities /= intensities.max()
        intensities[intensities > threshold] = 1
        intensities[intensities != 1] = 0
    angle = angles.copy()
    angle += math.pi / 4
    angle /= math.pi
    rgb = cv2.applyColorMap(np.uint8(angle * 255), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    intensity = intensities.copy().astype(float)
    rgb = rgb.astype(float)
    rgb[:, :, 0] *= intensity
    rgb[:, :, 1] *= intensity
    rgb[:, :, 2] *= intensity
    rgb = np.uint8(rgb)
    return rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='An encoder-decoder architecture within a classical signal processing framework for real-time '
                    'barcode segmentation.')
    parser.add_argument('--path', type=str, default='../examples/')
    parser.add_argument('-a', '--use-halide', default=False, action='store_true')
    args = parser.parse_args()
    files = get_file_list(args.path)
    if not args.use_halide:
        run_python_implementation(files)
    else:
        run_halide_implementation(files)
