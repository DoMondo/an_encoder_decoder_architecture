"""
Implementation of the PS DRT (partial strided DRT based detector) algorithm
"""

import numpy as np
from numba import njit
import math


@njit
def __size_partial_pdrt(input_height: int, stage: int, tile_size: int, stride: int) -> [int, int]:
    """ Computes the sizes of the output of a stage of the partial DRT with a tile_size and a stride

    Args:
        input_height (int): Size N of the input
        stage (int): Stage number
        stride (int): amount of overlap between tiles
        tile_size (int): size of the tile. It determines the number of stages that the partial DRT runs.
    Returns:
        n_squares (int): number of tiles
        slope_size (int): number of slopes (angles)

    """
    stage_size = 2 ** stage
    n_squares = math.floor((input_height - min(stage_size, tile_size)) / min(stage_size, stride)) + 1
    slope_size = 2 * stage_size - 1
    return n_squares, slope_size


@njit
def __pdrt_partial(input_data: np.ndarray, stride: int, tile_size: int) -> np.ndarray:
    """ Partial DRT with stride (overlap) and tile_size

    Args:
        img (np.ndarray): input image
        stride (int): amount of overlap between tiles
        tile_size (int): size of the tile. It determines the number of stages that the partial DRT runs.
    Returns:
        output[tile_size_bits] (np.ndarray): last computed stage
    """
    stride_bits = int(np.log2(stride))
    input_height, input_width = input_data.shape
    input_height_bits = int(np.floor(np.log2(input_height)))
    tile_size_bits = int(np.log2(tile_size))
    output = []
    for stage in range(tile_size_bits + 1):  # memory allocation
        n_squares, slope_size = __size_partial_pdrt(input_height, stage, tile_size, stride)
        output.append(np.zeros((n_squares, slope_size, input_width), dtype=np.int16))
    for i in range(input_height):
        for j in range(input_width):
            output[0][i, 0, j] = input_data[i, j]
    for stage in range(tile_size_bits):
        current_stage_size = 2 ** stage  # M
        out_stage_size = 2 ** (stage + 1)  # M in stage m plus 1 Mp1
        out_n_squares, out_slope_size = __size_partial_pdrt(input_height, stage + 1, tile_size, stride)
        for out_y_square in range(out_n_squares):
            for unsigned_slope in range(out_slope_size):  # slope as unsigned index
                slope = unsigned_slope - out_stage_size + 1  # slopes positives and negatives
                ab_s = abs(slope)
                s_sign = 1
                if slope < 0:
                    s_sign = -1  # sign of slope
                s2 = ab_s // 2  # floor (half the absolute slope)
                rs = ab_s - 2 * s2  # remainder of absolute slope
                slopeM = current_stage_size - 1 + s2 * s_sign  # slopes of previous segments
                incIndB = s_sign * (s2 + rs)  # displacement among segments
                for writeIdx in range(input_width):
                    readIdx = writeIdx
                    A = 0
                    B = 0
                    if 0 <= readIdx < input_width:
                        if stage < stride_bits:
                            A = output[stage][out_y_square * 2, slopeM, readIdx]
                        else:
                            A = output[stage][out_y_square, slopeM, readIdx]
                    if 0 <= readIdx + incIndB < input_width:
                        if stage < stride_bits:
                            B = output[stage][out_y_square * 2 + 1, slopeM, readIdx + incIndB]
                        else:
                            B = output[stage][out_y_square + stage - stride_bits + 1, slopeM, readIdx + incIndB]
                    output[stage + 1][out_y_square, unsigned_slope, writeIdx] = A + B
    return output[tile_size_bits]


@njit
def __bar_detector(vertical_pidrt: np.ndarray, horizontal_pidrt: np.ndarray, stride: int, tile_size: int) -> [np.ndarray, np.ndarray]:
    """ Detector part for the PS DRT algorithm

    Args:
        vertical_pdrt (np.ndarray): computed partial DRT of the input (n_squares, n_slopes, N)
        vertical_pdrt (np.ndarray): computed partial DRT of the transposed input (n_squares, n_slopes, N)
        stride (int): amount of overlap used
        tile_size (int): size of each square
    Returns:
        intensity (np.ndarray): Intensity of the detections for each square (n_squares x n_squares)
        angle (np.ndarray): Detected slope for each square (n_squares x n_squares)
    """
    intensity = np.zeros((vertical_pidrt.shape[0], horizontal_pidrt.shape[0]))
    max_slopes = np.zeros((vertical_pidrt.shape[0], horizontal_pidrt.shape[0]))
    for y_square in range(tile_size // stride // 2, vertical_pidrt.shape[0] - tile_size // stride // 2):
        for x_square in range(tile_size // stride // 2, horizontal_pidrt.shape[0] - tile_size // stride // 2):
            x_central = x_square * stride + tile_size // 2
            y_central = y_square * stride + tile_size // 2
            max_val = -np.inf
            max_slope = -tile_size + 1
            is_horizontal = True
            for slope in range(-tile_size + 1, tile_size):
                start_h = y_central - tile_size // 2 - slope // 2
                end_h = y_central + tile_size // 2 - slope // 2
                values_h = horizontal_pidrt[x_square, slope + tile_size - 1, start_h:end_h]
                start_v = x_central - tile_size // 2 + slope // 2
                end_v = x_central + tile_size // 2 + slope // 2
                values_v = vertical_pidrt[y_square, - slope + tile_size - 1, start_v:end_v]
                v1 = np.abs(np.diff(values_h)).sum()
                v2 = np.abs(np.diff(values_v)).sum()
                v = np.abs(v1 - v2)
                if v > max_val:
                    max_val = v
                    max_slope = slope
                    is_horizontal = v1 < v2
            intensity[y_square, x_square] = max_val
            if is_horizontal:
                max_slopes[y_square, x_square] = math.pi / 2 + math.atan2(max_slope, tile_size - 1)
            else:
                max_slopes[y_square, x_square] = -math.atan2(-max_slope, tile_size - 1)
    return intensity, max_slopes


def run(img: np.ndarray, stride: int, tile_size: int) -> [np.ndarray, np.ndarray]:
    """ Runs the PS DRT algorithm

    Args:
        img (np.ndarray): input image. Must be 1024x1024, 8 bit grayscale.
        stride (int): amount of overlap between tiles
        tile_size (int): size of the tile. It determines the number of stages that the partial DRT runs.

    Returns:
        intensity (np.ndarray): Intensity of the detections for each square (n_squares x n_squares)
        angle (np.ndarray): Detected slope for each square (n_squares x n_squares)
    """
    v_panel = __pdrt_partial(img, stride, tile_size)
    h_panel = __pdrt_partial(img.transpose(), stride, tile_size)
    return __bar_detector(v_panel, h_panel, stride, tile_size)
