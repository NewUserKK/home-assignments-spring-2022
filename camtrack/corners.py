#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

from typing import List

import click
import cv2
import numpy as np
import pims
from numba import jit
from sklearn.neighbors import KDTree

import utils
from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


CORNER_BLOCK_SIZE = 9
CORNER_QUALITY_LEVEL = 0.01
CORNER_MIN_DISTANCE_PX = 10
MAX_CORNERS = 2 ** 31 - 1

CORNERS_UPDATE_FREQUENCY_FRAMES = 1

PYRAMID_MIN_SIZE_THRESHOLD_PERCENT = 0.25

OPTICAL_FLOW_BLOCK_SIZE = 15
OPTICAL_FLOW_PARAMS = dict(
    winSize=(OPTICAL_FLOW_BLOCK_SIZE, OPTICAL_FLOW_BLOCK_SIZE),
    maxLevel=5,
)


def __log(*args, **kwargs):
    print(*args, **kwargs)
    input()


def _to_frame_corners(corners: np.ndarray, ids: np.array = None):
    if ids is None:
        ids = np.array(list(range(len(corners))))

    return FrameCorners(
        ids=ids,
        points=corners,
        sizes=np.full(corners.shape[0], CORNER_BLOCK_SIZE)
    )


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image = frame_sequence[0]
    corners = _get_corners_for_frame(image)
    frame_corners = _to_frame_corners(corners)
    builder.set_corners_at_frame(0, frame_corners)

    prev_image = utils.to_cv_8u(image)
    prev_corners = frame_corners
    prev_ids = frame_corners.ids
    for frame_index, image in enumerate(frame_sequence[1:], 1):
        image = utils.to_cv_8u(image)

        prev_points = prev_corners.points

        (optical_flow, is_good, _) = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_image,
            nextImg=image,
            prevPts=prev_points,
            nextPts=None,
            **OPTICAL_FLOW_PARAMS
        )

        (back_optical_flow, _, _) = cv2.calcOpticalFlowPyrLK(
            prevImg=image,
            nextImg=prev_image,
            prevPts=optical_flow,
            nextPts=None,
            **OPTICAL_FLOW_PARAMS
        )

        diff = abs(prev_points - back_optical_flow).reshape(-1, 2).max(-1)
        is_good = diff < 1

        optical_flow = optical_flow.reshape(-1, 2)
        points = []
        ids = []
        mask = np.full_like(image, dtype=bool, fill_value=True)
        for point_index, point in enumerate(optical_flow):
            if is_good[point_index]:
                ids.append(prev_ids[point_index])
                points.append(point)
                _update_mask(mask, point)

        if frame_index % CORNERS_UPDATE_FREQUENCY_FRAMES == 0:
            new_points = _get_corners_for_frame(image)
            new_points_filtered = []
            for point in new_points:
                (x, y) = np.int32(point)
                if mask[y, x]:
                    new_points_filtered.append(point)

            points.extend(new_points_filtered)

            start_index = ids[-1][0] + 1
            end_index = start_index + len(new_points_filtered) - 1
            ids.extend([[x] for x in range(start_index, end_index)])

        points = np.array(points)
        ids = np.array(ids)

        new_corners = _to_frame_corners(points, ids)
        builder.set_corners_at_frame(frame_index, new_corners)

        prev_image = image
        prev_ids = ids
        prev_corners = new_corners


def _get_corners_for_frame(frame: np.array, use_pyramid=True) -> np.ndarray:
    """
    Find corners for frame using pyramids.

    :param frame: ndarray of shape (height, width)
    :param use_pyramid: whether to use pyramids
    :return: ndarray of shape (-1, 2) with corners
    """
    all_corners: np.ndarray

    if use_pyramid:
        pyramid = _build_pyramid_for_frame(frame)

        raw_corners = []
        pyramid_size = len(pyramid)

        for layer in range(pyramid_size):
            new_corners = _pyramid_find_corners_for_layer(pyramid, layer)
            raw_corners.extend(new_corners)

        raw_corners = np.array(raw_corners)
        filtered_corners = _filter_close_corners(
            raw_corners,
            CORNER_MIN_DISTANCE_PX
        )

        all_corners = filtered_corners

    else:
        all_corners = _get_corners_for_single_frame(frame)

    return all_corners


def _get_corners_for_single_frame(frame: np.array, mask: np.array = None) -> np.array:
    """
    Detect corners on the frame using opencv.

    :param frame: ndarray of shape (height, width)
    :param mask: mask as in [cv2.goodFeaturesToTrack]
    :return: ndarray of shape (-1, 2) with corners
    """
    block_size = CORNER_BLOCK_SIZE

    prepared_frame = _preprocess_image(frame)

    corners = cv2.goodFeaturesToTrack(
        image=prepared_frame,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY_LEVEL,
        minDistance=block_size + block_size // 2,
        blockSize=block_size,
        mask=mask
    )

    return corners.reshape(-1, 2)


def _update_mask(mask: np.ndarray, point: np.ndarray):
    """
    Updates mask with zeros around given point.
    Is used to discard old corners that are already being tracked.

    Mutates mask in-place.

    :param mask: array of shape (height, width)
    :param point: array of shape (1, 2) representing (x, y) coordinates on image.
    """
    (x, y) = np.int32(point)
    (h, w) = mask.shape
    y_from = utils.coerce_in(y - CORNER_MIN_DISTANCE_PX, 0, h)
    y_to = utils.coerce_in(y + CORNER_MIN_DISTANCE_PX, 0, h) + 1
    x_from = utils.coerce_in(x - CORNER_MIN_DISTANCE_PX, 0, w)
    x_to = utils.coerce_in(x + CORNER_MIN_DISTANCE_PX, 0, w) + 1
    mask[y_from:y_to, x_from:x_to] = 0


def _preprocess_image(frame: np.array) -> np.array:
    """
    Preprocess image for better tracking.

    :param frame: ndarray of shape (height, width)
    :return: processed image of shape (height, width)
    """
    prepared_frame = utils.smooth(frame, ksize=7)
    prepared_frame = utils.sharpen(prepared_frame)

    return prepared_frame


def _build_pyramid_for_frame(frame: np.array) -> List[np.array]:
    """
    Build pyramid for the frame.

    :param frame: ndarray of shape (height, width)
    :return: list of scaled frames from the smallest image to largest.
    """

    (height, width) = np.shape(frame)
    size_threshold = int(
        min(height, width) * PYRAMID_MIN_SIZE_THRESHOLD_PERCENT
    )

    def frame_is_large_enough(cur_frame: np.array):
        (cur_frame_h, cur_frame_w) = np.shape(cur_frame)
        return cur_frame_h > size_threshold and cur_frame_w > size_threshold

    out = [frame]

    diminished_frame = _diminish_frame_size(frame)
    while frame_is_large_enough(diminished_frame):
        out.append(diminished_frame)
        diminished_frame = _diminish_frame_size(diminished_frame)

    return list(reversed(out))


def _pyramid_find_corners_for_layer(
        pyramid: List[np.array],
        layer_index: int
) -> np.ndarray:
    """
    Detect and scale back corners for given pyramid layer.

    :param pyramid: pyramid as returned in [_build_pyramid_for_frame]
    :param layer_index: layer index
    :return: ndarray of shape (-1, 2) with points
    """
    pyramid_size = len(pyramid)
    corners = _get_corners_for_single_frame(pyramid[layer_index])
    corners = _rescale_corners(corners, pyramid_size, layer_index)
    return corners


def _filter_close_corners(corners: np.ndarray, radius: int) -> np.ndarray:
    """
    Filter out close corners with given radius.

    :param corners: array of points with shape (-1, 2)
    :param radius: threshold in which corners will be joint
    :return: filtered corners, ndarray of shape (-1, 2)
    """
    used = np.ones(len(corners), dtype=bool)

    kd_tree = KDTree(corners, metric='manhattan')
    neighbours = kd_tree.query_radius(corners, radius)

    result = []

    for group_index, indices_group in enumerate(neighbours):
        if not used[group_index]:
            continue

        for point_index in indices_group:
            used[point_index] = True

        most_accurate_point_index = indices_group.max()
        result.append(corners[most_accurate_point_index])

    return np.array(result)


def _diminish_frame_size(frame: np.array) -> np.array:
    """
    Scale one step down image for pyramid.

    :param frame: ndarray of shape (height, width)
    :return: ndarray with scaled down image of shape (height // 2, width // 2)
    """
    return cv2.pyrDown(frame)


def _rescale_corners(corners: np.ndarray, pyramid_size: int, layer: int) -> np.ndarray:
    """
    Rescale corners in pyramid to the original image coordinates.
    Lower layer is smaller image.

    :param corners: ndarray of shape (-1, 2)
    :param pyramid_size: size of pyramid
    :param layer: layer index
    :return: ndarray of scaled corners with shape (-1, 2)
    """
    coef = _pyramid_coef(pyramid_size, layer)
    return np.float32(corners * coef)


@jit
def _pyramid_coef(pyramid_size: int, layer: int) -> int:
    """
    Return scaling coefficient of the pyramid layer.
    Lower layer is smaller image.

    :param pyramid_size: size of the pyramid
    :param layer: layer index
    :return: scaling coefficient of the pyramid
    """
    return 2 ** (pyramid_size - layer - 1)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
