import os
from utils import save_step
from typing import List, Tuple
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np

def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    windows = sliding_window_view(padded, (kh, kw))
    return np.sum(windows * kernel, axis=(2, 3))


def binary_erosion_fast(mask: np.ndarray, size: int = 3) -> np.ndarray:
    kernel = (size, size)
    pad = size // 2
    padded = np.pad(mask.astype(np.uint8) * 255, pad, mode='edge')
    windows = sliding_window_view(padded, kernel)
    eroded = (np.min(windows, axis=(2, 3)) == 255)
    return eroded


def binary_dilation_fast(mask: np.ndarray, size: int = 3) -> np.ndarray:
    kernel = (size, size)
    pad = size // 2
    padded = np.pad(mask.astype(np.uint8) * 255, pad, mode='edge')
    windows = sliding_window_view(padded, kernel)
    dilated = (np.max(windows, axis=(2, 3)) > 0)
    return dilated


def otsu_threshold(img: np.ndarray) -> int:
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    hist = np.bincount(img8.ravel(), minlength=256).astype(float)
    total = img8.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    var_max = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold


def gaussian_kernel(size: int = 5, sigma: float = 1.5) -> np.ndarray:
    assert size % 2 == 1 and size >= 3
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return k / np.sum(k)


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = convolve2d_fast(gray, sx)
    gy = convolve2d_fast(gray, sy)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    m = np.max(edges)
    return (edges / m * 255).astype(np.uint8) if m > 0 else edges.astype(np.uint8)


def find_top_components(mask: np.ndarray, top_n: int = 6, connectivity8: bool = True) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, np.ndarray]] = []
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] if connectivity8 else [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not visited[i, j]:
                comp = np.zeros((h, w), dtype=bool)
                stack = [(i, j)]
                size = 0
                while stack:
                    y, x = stack.pop()
                    if visited[y, x]:
                        continue
                    visited[y, x] = True
                    if mask[y, x]:
                        comp[y, x] = True
                        size += 1
                        for dy, dx in dirs:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                                stack.append((ny, nx))
                if size > 0:
                    comps.append((size, comp))
    comps.sort(key=lambda x: x[0], reverse=True)
    out = np.zeros((h, w), dtype=bool)
    for size, comp in comps[:top_n]:
        out |= comp
    return out


def compute_blue_ratio(image: np.ndarray) -> np.ndarray:
    denom = image[:, :, 0].astype(float) + image[:, :, 1].astype(float) + image[:, :, 2].astype(float) + 1e-5
    blue_ratio = image[:, :, 2].astype(float) / denom
    return (blue_ratio * 255.0)


def apply_open_close(mask: np.ndarray, open_size: int = 3, close_size: int = 7) -> np.ndarray:
    out = binary_erosion_fast(mask, size=open_size)
    out = binary_dilation_fast(out, size=open_size)
    out = binary_dilation_fast(out, size=close_size)
    out = binary_erosion_fast(out, size=close_size)
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return mask


def overlay_edges(rgb: np.ndarray, mask: np.ndarray, edge_threshold: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.dot(rgb.astype(float), [0.299, 0.587, 0.114])
    edges = sobel_edges(gray * mask)
    edges_high = edges > edge_threshold
    result = rgb.copy()
    result[edges_high] = [255, 0, 0]
    return result, edges_high


def proprietary_impl(image: np.ndarray, gauss_size: int = 5, sigma: float = 1.5, outdir: str = "."):
    blue_ratio = compute_blue_ratio(image)

    k = gaussian_kernel(gauss_size, sigma)
    blurred = convolve2d_fast(blue_ratio, k)
    save_step(blurred, "01_blurred", "A", outdir)

    thr = otsu_threshold(blurred)
    mask = blurred > thr
    save_step(mask, "02_binarized", "A", outdir)

    mask = apply_open_close(mask, open_size=3, close_size=7)
    save_step(mask, "03_morphology", "A", outdir)

    mask = find_top_components(mask, top_n=6)
    save_step(mask, "04_top_components", "A", outdir)

    mask = fill_holes(mask)
    save_step(mask, "05_filled", "A", outdir)

    masked_rgb = image.copy()
    masked_rgb[~mask] = 0
    save_step(masked_rgb, "06_masked_image", "A", outdir)

    result, edges_high = overlay_edges(masked_rgb, mask, edge_threshold=30)
    save_step(edges_high, "07_edges", "A", outdir)
    save_step(result, "08_result", "A", outdir)
    return result, mask.astype(bool), edges_high