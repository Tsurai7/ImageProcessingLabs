import cv2
import numpy as np
from typing import Dict, Any


def rgb_to_grayscale_lib(img_rgb: np.ndarray) -> np.ndarray:
    """RGB -> GRAY с использованием OpenCV (c последующей конвертацией)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def gaussian_blur_lib(img_gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.5) -> np.ndarray:
    k = (int(kernel_size), int(kernel_size))
    return cv2.GaussianBlur(img_gray, k, sigmaX=float(sigma))


def sobel_edge_detection_lib(img_gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edges = cv2.convertScaleAbs(mag)
    return edges


def binary_threshold_lib(img: np.ndarray, threshold: int = 30) -> np.ndarray:
    _, binary = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)
    return binary


def run_pipeline_library(img_rgb: np.ndarray,
                         kernel_size: int = 5,
                         sigma: float = 1.5,
                         threshold: int = 30) -> Dict[str, Any]:
    """Полный конвейер (B-версия, OpenCV): серый -> гаусс -> собель -> бинаризация."""
    gray = rgb_to_grayscale_lib(img_rgb)
    blurred = gaussian_blur_lib(gray, kernel_size=kernel_size, sigma=sigma)
    edges = sobel_edge_detection_lib(blurred)
    binary = binary_threshold_lib(edges, threshold=threshold)
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "binary": binary,
    }
