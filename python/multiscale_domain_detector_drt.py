"""
Implementation of the MDD DRT (multiscale domain DRT based detector)
"""

import cv2
import numpy as np
from numba import njit
import math


@njit
def __size_pdrt(input_height: int, stage: int, tile_size: int, stride: int) -> [int, int]:
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
def __pdrt(input_data, stride, tile_size):
    """ Partial DRT with stride (overlap) and tile_size and returning all the intermediate stages

    Args:
        img (np.ndarray): input image
        stride (int): amount of overlap between tiles
        tile_size (int): size of the tile. It determines the number of stages that the partial DRT runs.
    Returns:
        output ([np.ndarray]): all computed stages
    """
    stride_bits = int(np.log2(stride))
    input_height, input_width = input_data.shape
    input_height_bits = int(np.floor(np.log2(input_height)))
    tile_size_bits = int(np.log2(tile_size))
    output = []
    for stage in range(tile_size_bits + 1):  # memory allocation
        n_squares, slope_size = __size_pdrt(input_height, stage, tile_size, stride)
        output.append(np.zeros((n_squares, slope_size, input_width), dtype=np.float))
    for i in range(input_height):
        for j in range(input_width):
            output[0][i, 0, j] = input_data[i, j]
    for stage in range(tile_size_bits):
        current_stage_size = 2 ** stage  # M
        out_stage_size = 2 ** (stage + 1)  # M in stage m plus 1 Mp1
        out_n_squares, out_slope_size = __size_pdrt(input_height, stage + 1, tile_size, stride)
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
    return output[1:]


@njit
def __mdd_bar_detector(vertical_pidrt: np.ndarray, horizontal_pidrt: np.ndarray) -> [np.ndarray]:
    """ Create the activation maps for the CNN like encoder, from partial DRTs

    Args:
        vertical_pdrt (np.ndarray): all computed stages of the partial DRT of the input (n_squares, n_slopes, N)
        vertical_pdrt (np.ndarray): all computed stages of the partial DRT of the transposed input (n_squares, n_slopes, N)
    Returns:
        activations (np.ndarray): Activations for all the angles for a stage (n_squares, n_squares, n_slopes)
    """
    stride = vertical_pidrt.shape[2] // vertical_pidrt.shape[0]
    tile_size = stride
    n_slopes = horizontal_pidrt.shape[1]
    activations = np.zeros((vertical_pidrt.shape[0], horizontal_pidrt.shape[0], n_slopes * 2), dtype=np.float)
    for y_square in range(tile_size // stride // 2, vertical_pidrt.shape[0] - tile_size // stride // 2):
        for x_square in range(tile_size // stride // 2, horizontal_pidrt.shape[0] - tile_size // stride // 2):
            x_central = x_square * stride + tile_size // 2
            y_central = y_square * stride + tile_size // 2
            for slope in range(n_slopes):
                signed_slope = slope - tile_size + 1
                start_h = y_central - tile_size // 2 - signed_slope // 2
                end_h = y_central + tile_size // 2 - signed_slope // 2
                values_h = horizontal_pidrt[x_square, signed_slope + tile_size - 1, start_h:end_h]
                start_v = x_central - tile_size // 2 + signed_slope // 2
                end_v = x_central + tile_size // 2 + signed_slope // 2
                values_v = vertical_pidrt[y_square, - signed_slope + tile_size - 1, start_v:end_v]
                if len(values_v) == len(values_h):
                    v1 = np.abs(np.diff(values_h)).sum()
                    v2 = np.abs(np.diff(values_v)).sum()
                    activations[y_square, x_square, slope] = v1 - v2
                    activations[y_square, x_square, slope + n_slopes] = v2 - v1
    activations[0, :, :] = 0
    activations[-1, :, :] = 0
    activations[:, 0, :] = 0
    activations[:, -1, :] = 0
    return activations


@njit
def __mdd_unpool(coarse_activations: np.ndarray, fine_activations: np.ndarray):
    """ Implementation of the up-sampling operation for the CNN like decoder which propagates activations of a lower
    dimension size to a higher one, by using the indices of the previous stage of the encoder.

    Args:
        coarse_activations (np.ndarray): Coarse activations of size (n_squares, n_squares, n_angles_coarse)
        fine_activations (np.ndarray): Fine activations of size (n_squares * 2, n_squares * 2, n_angles_fine)
    Returns:
        new_fine_activations (np.ndarray): New activations of size (n_squares * 2, n_squares * 2, output_slopes)
    """
    output_slopes = coarse_activations.shape[2]
    if output_slopes > 32:
        output_slopes = fine_activations.shape[2]
    new_fine_activations = np.zeros((fine_activations.shape[0], fine_activations.shape[1], output_slopes),
                                    dtype=np.float)
    max_slopes_indices = np.argmax(coarse_activations, axis=2)
    slope_ratio = coarse_activations.shape[2] // fine_activations.shape[2]
    new_activations_slope_ratio = coarse_activations.shape[2] / output_slopes
    for i in range(coarse_activations.shape[0]):
        for j in range(coarse_activations.shape[1]):
            max_slope = max_slopes_indices[i, j]
            fine_activations_coarser_slope = (max_slope // slope_ratio) % fine_activations.shape[2]
            # Find argmax of the fine activation
            max_val = 0
            max_idx_i = -1
            max_idx_j = -1
            for ii in range(2):
                for jj in range(2):
                    val = fine_activations[i * 2 + ii, j * 2 + jj, fine_activations_coarser_slope]
                    if val > max_val:
                        max_val = val
                        max_idx_i = ii
                        max_idx_j = jj
            new_fine_activations[
                i * 2 + max_idx_i, j * 2 + max_idx_j, round(max_slope / new_activations_slope_ratio) % output_slopes] = \
                coarse_activations[i, j, max_slope]
    return new_fine_activations


@njit
def __add_original_activations(new_activations: np.ndarray, original_activations: np.ndarray, w_original: float,
                               w_new: float):
    """ Implementation of the addition operation for the CNN like decoder which adds back the original activations of
    the encoder.

    Args:
        new_activations (np.ndarray): New activations produced by the unpool operation
        original_activations (np.ndarray): Activations coming from the encoder
        w_original (float): weight applied to the activations from the encoder
        w_new (float): weight applied to the activations from the unpool
    Returns:
        new_fine_activations (np.ndarray): New activations of size (n_squares * 2, n_squares * 2, output_slopes)
    """
    slope_ratio = new_activations.shape[2] // original_activations.shape[2]
    for slope in range(new_activations.shape[2]):
        fine_activations_coarser_slope = (slope // slope_ratio) % original_activations.shape[2]
        new_activations[:, :, slope] = new_activations[:, :, slope] * w_new + \
                                       original_activations[:, :, fine_activations_coarser_slope] * w_original
    return new_activations


def __apply_convolutions(activations) -> np.ndarray:
    """ Implementation of the low pass filtering operation for the CNN like decoder.

    Args:
        activations (np.ndarray): New activations produced by the __add_original_activations operation
    Returns:
        filtered_new_activations (np.ndarray): Filtered activations
    """
    # Apply filtering in x, y
    for i in range(activations.shape[2]):
        activations[:, :, i] = cv2.boxFilter(activations[:, :, i], -1, (3, 3))
    for i in range(activations.shape[2]):
        activations[:, :, i] = cv2.boxFilter(activations[:, :, i], -1, (3, 3))
    # Apply filtering in z as well
    filtered_new_activations = np.zeros_like(activations)
    for slope in range(activations.shape[2]):
        a = activations[:, :, (slope - 1) % activations.shape[2]]
        b = activations[:, :, slope]
        c = activations[:, :, (slope + 1) % activations.shape[2]]
        filtered_new_activations[:, :, slope] = a / 4 + b / 2 + c / 4
    return filtered_new_activations


def __mdd_encoder(input_img: np.ndarray) -> [np.ndarray]:
    """ First half of the MDD DRT algorithm. Computes two partial DRTs and runs the bar_detector for each computed
    stage.

    Args:
        input_img (np.ndarray): input image (1024, 1024), 8 bits
    Returns:
        activations ([np.ndarray]): A set of activations per stage
    """
    stride = 32
    tile_size = 32
    vertical_panel = __pdrt(input_img, stride, tile_size)
    horizontal_panel = __pdrt(input_img.transpose(), stride, tile_size)
    activations = []
    for i in range(len(vertical_panel)):
        v = vertical_panel[i]
        h = horizontal_panel[i]
        activation_maps = __mdd_bar_detector(v, h)
        activations.append(activation_maps)
    return activations


def __mdd_decoder(activations) -> [np.ndarray]:
    """ Second half of the MDD DRT algorithm. Sequentially up-samples the activations produced in the encoder using
    unpool, additions and low pass filtering.

    Args:
        activations ([np.ndarray]): A set of activations per stage
    Returns:
        new_activations ([np.ndarray]): A set of activations per stage
    """
    new_activations = activations.copy()
    weights_original_activations = [0.05, 0.527, 0.33, 0.76]
    weights_new_activations_activations = [0.84, 0.84, 1.16, 3.47]
    for idx in range(len(activations) - 1, 0, -1):
        _new_activations = __mdd_unpool(new_activations[idx], new_activations[idx - 1])
        _new_activations = __add_original_activations(_new_activations, new_activations[idx - 1],
                                                      weights_original_activations[len(activations) - 1 - idx],
                                                      weights_new_activations_activations[len(activations) - 1 - idx])
        _new_activations = __apply_convolutions(_new_activations)
        new_activations[idx - 1] = _new_activations
    return new_activations


def run(img: np.ndarray) -> [np.ndarray, np.ndarray]:
    """ Runs the MDD DRT algorithm

    Args:
        img (np.ndarray): input image. Must be 1024x1024, 8 bit grayscale.

    Returns:
        intensity (np.ndarray): Intensity of the detections for each square (n_squares x n_squares)
        angle (np.ndarray): Detected angle for each square (n_squares x n_squares)
    """
    activations = __mdd_encoder(img)
    new_activations = __mdd_decoder(activations)
    intensities = np.max(new_activations[0], axis=2)
    angles = np.argmax(new_activations[0], axis=2)
    angles = np.double(angles) / new_activations[0].shape[2] * math.pi
    angles -= math.pi / 4
    return intensities, angles
