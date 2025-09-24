import os
from typing import List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_step(img: np.ndarray, name: str, variant: str, base_dir: str) -> None:
    import matplotlib.pyplot as plt
    folder = os.path.join(base_dir, f"steps_{variant}")
    ensure_dir(folder)
    path = os.path.join(folder, f"{name}.png")
    if img.dtype == bool:
        plt.imsave(path, img.astype(np.uint8) * 255, cmap="gray")
    elif img.ndim == 2:
        plt.imsave(path, img.astype(np.uint8), cmap="gray")
    else:
        plt.imsave(path, img.astype(np.uint8))


def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same', boundary='symm')
    except Exception:
        from numpy.lib.stride_tricks import sliding_window_view
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        windows = sliding_window_view(padded, (kh, kw))
        return np.sum(windows * kernel, axis=(2, 3))


def binary_erosion_fast(mask: np.ndarray, size: int = 3) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    kernel = (size, size)
    pad = size // 2
    padded = np.pad(mask.astype(np.uint8) * 255, pad, mode='edge')
    windows = sliding_window_view(padded, kernel)
    eroded = (np.min(windows, axis=(2, 3)) == 255)
    return eroded


def binary_dilation_fast(mask: np.ndarray, size: int = 3) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
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


def variant_a(image: np.ndarray, gauss_size: int = 5, sigma: float = 1.5, outdir: str = "."):
    denom = image[:, :, 0].astype(float) + image[:, :, 1].astype(float) + image[:, :, 2].astype(float) + 1e-5
    blue_ratio = image[:, :, 2].astype(float) / denom
    blue_ratio *= 255.0

    k = gaussian_kernel(gauss_size, sigma)
    blurred = convolve2d_fast(blue_ratio, k)
    save_step(blurred, "01_blurred", "A", outdir)

    thr = otsu_threshold(blurred)
    mask = blurred > thr
    save_step(mask, "02_binarized", "A", outdir)

    mask = binary_erosion_fast(mask, size=3)
    mask = binary_dilation_fast(mask, size=3)
    save_step(mask, "03_opening", "A", outdir)

    mask = binary_dilation_fast(mask, size=7)
    mask = binary_erosion_fast(mask, size=7)
    save_step(mask, "04_closing", "A", outdir)

    mask = find_top_components(mask, top_n=6)
    save_step(mask, "05_top_components", "A", outdir)

    try:
        from scipy.ndimage import binary_fill_holes
        mask = binary_fill_holes(mask)
    except Exception:
        pass
    save_step(mask, "06_filled", "A", outdir)

    masked_rgb = image.copy()
    masked_rgb[~mask] = 0
    save_step(masked_rgb, "07_masked_image", "A", outdir)

    gray = np.dot(image.astype(float), [0.299, 0.587, 0.114])
    edges = sobel_edges(gray * mask)
    edges_high = edges > 30
    save_step(edges_high, "08_edges", "A", outdir)

    result = masked_rgb.copy()
    result[edges_high] = [255, 0, 0]
    save_step(result, "09_result", "A", outdir)
    return result, mask.astype(bool), edges_high