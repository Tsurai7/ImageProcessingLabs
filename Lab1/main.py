import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from proprietary import variant_a
from withLib import variant_b


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_step(img: np.ndarray, name: str, variant: str, base_dir: str) -> None:
    folder = os.path.join(base_dir, f"impl_{variant}")
    ensure_dir(folder)
    path = os.path.join(folder, f"{name}.png")
    if img.dtype == bool:
        plt.imsave(path, img.astype(np.uint8) * 255, cmap="gray")
    elif img.ndim == 2:
        plt.imsave(path, img.astype(np.uint8), cmap="gray")
    else:
        plt.imsave(path, img.astype(np.uint8))


# ---------- Fast/optimized convolution primitives ----------

def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same', boundary='symm')
    except Exception:
        # NumPy vectorized fallback using sliding window
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


# ---------- Core helpers ----------

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


# ---------- Variant A (author, optimized) ----------

def variant_a(image: np.ndarray, gauss_size: int = 5, sigma: float = 1.5, outdir: str = ".") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Blue ratio for robust separation
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

    # Fill holes (fast boolean trick: apply dilation to inverse then invert back) – use scipy if available
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


# ---------- Variant B (library/scipy) ----------

def variant_b(image: np.ndarray, sigma: float = 1.5, outdir: str = ".") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label, binary_fill_holes

    denom = image[:, :, 0].astype(float) + image[:, :, 1].astype(float) + image[:, :, 2].astype(float) + 1e-5
    blue_ratio = image[:, :, 2].astype(float) / denom
    blue_ratio *= 255.0

    blurred = gaussian_filter(blue_ratio, sigma=sigma)
    save_step(blurred, "01_blurred", "B", outdir)

    thr = otsu_threshold(blurred)
    mask = blurred > thr
    save_step(mask, "02_binarized", "B", outdir)

    mask = binary_opening(mask, structure=np.ones((3, 3)))
    mask = binary_closing(mask, structure=np.ones((7, 7)))
    save_step(mask, "03_morphology", "B", outdir)

    labeled, num = label(mask)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        order = np.argsort(sizes[1:])[::-1] + 1
        keep_labels = order[:6]
        mask = np.isin(labeled, keep_labels)
    save_step(mask, "04_top_components", "B", outdir)

    mask = binary_fill_holes(mask)
    save_step(mask, "05_filled", "B", outdir)

    masked_rgb = image.copy()
    masked_rgb[~mask] = 0
    save_step(masked_rgb, "06_masked_image", "B", outdir)

    gray = np.dot(image.astype(float), [0.299, 0.587, 0.114])
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    from scipy.ndimage import convolve
    gx = convolve(gray * mask, sx)
    gy = convolve(gray * mask, sy)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    m = np.max(edges)
    edges = (edges / m * 255).astype(np.uint8) if m > 0 else edges.astype(np.uint8)
    edges_high = edges > 30
    save_step(edges_high, "07_edges", "B", outdir)

    result = masked_rgb.copy()
    result[edges_high] = [255, 0, 0]
    save_step(result, "08_result", "B", outdir)
    return result, mask.astype(bool), edges_high


# ---------- I/O helpers ----------

def load_image_any(path: str) -> np.ndarray:
    img = plt.imread(path)
    if img.dtype in (np.float32, np.float64) and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def find_first_image(path_or_dir: str, patterns: str = "*.png;*.jpg;*.jpeg;*.bmp") -> Optional[str]:
    p = Path(path_or_dir)
    if p.is_file():
        return str(p)
    pats = [s.strip() for s in patterns.split(';') if s.strip()]
    if p.is_dir():
        for patt in pats:
            found = sorted(p.rglob(patt))
            if found:
                return str(found[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized Variant A/B with steps saving and benchmarking")
    parser.add_argument('--input', '-i', type=str, default='Lab1/dataset', help='Путь к файлу или каталогу')
    parser.add_argument('--pattern', type=str, default='*.png;*.jpg;*.jpeg;*.bmp', help='Шаблоны поиска (через ;)')
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--kernel', type=int, default=5)
    parser.add_argument('--outdir', '-o', type=str, default='outputs', help='Каталог для сохранения изображений')
    args = parser.parse_args()

    ensure_dir(args.outdir)
    path = find_first_image(args.input, args.pattern)
    if not path:
        raise FileNotFoundError("Не найдено изображений по заданному пути/шаблонам")

    image = load_image_any(path)

    t0 = time.time()
    result_a, mask_a, edges_a = variant_a(image, gauss_size=args.kernel, sigma=args.sigma, outdir=args.outdir)
    ta = time.time() - t0

    t0 = time.time()
    result_b, mask_b, edges_b = variant_b(image, sigma=args.sigma, outdir=args.outdir)
    tb = time.time() - t0

    diff = np.abs(result_a.astype(float) - result_b.astype(float)).mean()
    print(f'Mean absolute difference between A and B results: {diff:.3f}')


if __name__ == '__main__':
    main()