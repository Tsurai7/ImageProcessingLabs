import os
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


def scki_kit_impl(image: np.ndarray, sigma: float = 1.5, outdir: str = "."):
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label, binary_fill_holes, convolve

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