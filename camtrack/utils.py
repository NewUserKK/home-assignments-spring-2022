from typing import Iterable

import cv2
import numpy as np
from numba import jit


def map_l(f, sequence): return list(map(f, sequence))


def group_by(iterable: Iterable, key_getter, result_mapping=lambda x: x):
    result = {}
    for item in iterable:
        key = key_getter(item)
        result.setdefault(key, []).append(result_mapping(item))

    return result


@jit(nopython=True)
def coerce_in(n: int, start: int, end: int):
    if n < start:
        return start
    elif n >= end:
        return end - 1
    else:
        return n


@jit(nopython=True)
def manhattan_distance(vec1: Iterable, vec2: Iterable):
    distance = 0
    for (x1, x2) in zip(vec1, vec2):
        distance += abs(x1 - x2)

    return distance


def sharpen(img: np.array, times=1) -> np.array:
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    sharpened = img
    for i in range(times):
        sharpened = cv2.filter2D(sharpened, -1, kernel)

    return sharpened


def smooth(img: np.array, ksize=3) -> np.array:
    return cv2.GaussianBlur(
        src=img,
        ksize=(ksize, ksize),
        sigmaX=0
    )


def to_cv_8u(img: np.array) -> np.array:
    return np.array(np.round(img * 255.), np.uint8)
