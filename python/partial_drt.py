"""
Implementation of the PDRT (partial discrete Radon transform barcode detector) algorithm. First proposed in
"A local real-time bar detector based on the multiscale Radon transform" https://doi.org/10.1117/12.2617411.
"""

import numpy as np
import math
from numba import njit


@njit
def __size_partial_drt(n_bits: int, stage_bits: int) -> [int, int]:
    """ Computes the sizes of the output of a stage of the partial DRT with a tile_size and a stride

    Args:
        n_bits (int): log2 of the size N of the input
        stage_bits (int): Stage number
    Returns:
        n_squares (int): number of tiles
        slope_size (int): number of slopes (angles)
    """
    size = 2 ** n_bits
    stage_size = 2 ** stage_bits
    n_squares = (size // stage_size)
    n_slopes = 2 * stage_size - 1
    return n_squares, n_slopes


@njit
def __partial_drt(img: np.ndarray, tile_size: int) -> np.ndarray:
    """ Partial DRT executing the number of stages up to reaching a tile_size

    Args:
        img (np.ndarray): input image
        tile_size (int): size of the tile. It determines the number of stages that the partial DRT runs.
    Returns:
        fm[n_stages] (np.ndarray): last computed stage
    """
    size = img.shape[0]
    size_bits = int(np.log2(size))
    n_stages = int(np.log2(tile_size))
    fm = []

    # Memory allocation
    for stage_idx in range(n_stages + 1):
        n_squares, slope_range = __size_partial_drt(size_bits, stage_idx)
        fm.append(np.zeros((n_squares, slope_range, n_squares * 2 ** stage_idx), dtype=np.int16))

    # Copy input to f0
    for i in range(size):
        for j in range(size):
            fm[0][i][0][j] = img[i, j]

    # Run stages
    for stage_idx in range(n_stages):
        m = 2 ** stage_idx
        mp1 = 2 ** (stage_idx + 1)
        n_squares_mp1, slope_range_mp1 = __size_partial_drt(size_bits, stage_idx + 1)
        for y_square_mp1 in range(n_squares_mp1):
            for _slope in range(slope_range_mp1):  # slope as unsigned index
                slope = _slope - mp1 + 1  # slopes positives and negatives
                ab_s = abs(slope)
                s_sign = 1
                if slope < 0:
                    s_sign = -1  # sign of slope
                s2 = ab_s // 2  # floor (half the absolute slope)
                rs = ab_s - 2 * s2  # remainder of absolute slope
                slope_m = m - 1 + s2 * s_sign  # slopes of previous segments
                inc_ind_b = s_sign * (s2 + rs)  # displacement among segments
                for write_idx in range(size):
                    read_idx = write_idx
                    a = 0
                    b = 0
                    if 0 <= read_idx < size:
                        a = fm[stage_idx][y_square_mp1 * 2, slope_m, read_idx]
                    if 0 <= read_idx + inc_ind_b < size:
                        b = fm[stage_idx][y_square_mp1 * 2 + 1, slope_m, read_idx + inc_ind_b]
                    fm[stage_idx + 1][y_square_mp1, _slope, write_idx] = a + b
    return fm[n_stages]


@njit
def __bar_detector(vertical_pdrt: np.ndarray, horizontal_pdrt: np.ndarray, tile_size: int) -> [np.ndarray, np.ndarray]:
    """ Detector part for the Partial DRT algorithm

    Args:
        vertical_pdrt (np.ndarray): computed partial DRT of the input (n_squares, n_slopes, N)
        vertical_pdrt (np.ndarray): computed partial DRT of the transposed input (n_squares, n_slopes, N)
        tile_size (int): size of each square
    Returns:
        intensity (np.ndarray): Intensity of the detections for each square (n_squares x n_squares)
        angle (np.ndarray): Detected slope for each square (n_squares x n_squares)
    """
    intensity: np.ndarray = np.zeros((vertical_pdrt.shape[0], horizontal_pdrt.shape[0]))
    max_slopes: np.ndarray = np.zeros((vertical_pdrt.shape[0], horizontal_pdrt.shape[0]))
    for y_square in range(vertical_pdrt.shape[0]):
        for x_square in range(horizontal_pdrt.shape[0]):
            x_central = x_square * tile_size + tile_size // 2
            y_central = y_square * tile_size + tile_size // 2
            max_val = -np.inf
            max_slope = -tile_size + 1
            is_horizontal = True
            for slope in range(-tile_size + 1, tile_size):
                start_h = y_central - tile_size // 2 - slope // 2
                end_h = y_central + tile_size // 2 - slope // 2
                values_h = horizontal_pdrt[x_square, slope + tile_size - 1, start_h:end_h]
                start_v = x_central - tile_size // 2 + slope // 2
                end_v = x_central + tile_size // 2 + slope // 2
                values_v = vertical_pdrt[y_square, - slope + tile_size - 1, start_v:end_v]
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


def run(img: np.ndarray, tile_size: int):
    v_panel = __partial_drt(img, tile_size)
    h_panel = __partial_drt(img.transpose(), tile_size)
    return __bar_detector(v_panel, h_panel, tile_size)
