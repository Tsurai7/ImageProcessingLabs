import numpy as np
from typing import Dict, Any


def rgb_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Преобразует RGB в оттенки серого: Y = 0.299R + 0.587G + 0.114B.

    Ожидается вход shape (H, W, 3) либо (H, W, 4); альфа-канал игнорируется.
    Возвращает uint8 массив shape (H, W).
    """
    grayscale_float = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])
    return np.clip(grayscale_float, 0, 255).astype(np.uint8)


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Двумерная свёртка с отражающим паддингом. Результат той же формы."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = float(np.sum(region * kernel))
    return output


def _convolve2d_same_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Быстрая свёртка через scipy.signal.convolve2d (если доступна) или векторизованная версия."""
    try:
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same', boundary='symm')
    except ImportError:
        # Векторизованная версия без двойного цикла
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Создаем окна для всех позиций сразу
        H, W = image.shape
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
        # Применяем свёртку ко всем окнам одновременно
        output = np.sum(windows * kernel, axis=(2, 3))
        return output


def gaussian_blur(img_gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, fast: bool = True) -> np.ndarray:
    """Размытие по Гауссу через свёртку, возвращает uint8.

    Параметры:
    - kernel_size: нечётное число >= 3
    - sigma: стандартное отклонение Гаусса
    - fast: использовать быструю свёртку (scipy или векторизованную)
    """
    assert kernel_size % 2 == 1 and kernel_size >= 3, "kernel_size должен быть нечётным и >= 3"
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    kernel = kernel / np.sum(kernel)
    
    if fast:
        blurred = _convolve2d_same_fast(img_gray.astype(float), kernel)
    else:
        blurred = _convolve2d_same(img_gray.astype(float), kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def sobel_edge_detection(img_gray: np.ndarray, fast: bool = True) -> np.ndarray:
    """Выделение контуров оператором Собеля. Возвращает uint8 магнитуду."""
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=float)
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=float)
    
    if fast:
        gx = _convolve2d_same_fast(img_gray.astype(float), kernel_x)
        gy = _convolve2d_same_fast(img_gray.astype(float), kernel_y)
    else:
        gx = _convolve2d_same(img_gray.astype(float), kernel_x)
        gy = _convolve2d_same(img_gray.astype(float), kernel_y)
    
    magnitude = np.abs(gx) + np.abs(gy)
    denom = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    magnitude = (magnitude / denom) * 255.0
    return np.clip(magnitude, 0, 255).astype(np.uint8)


def binary_threshold(img_gray_or_edges: np.ndarray, threshold: int = 50) -> np.ndarray:
    """Пороговая бинаризация: > threshold => 255, иначе 0. Возвращает uint8."""
    mask = (img_gray_or_edges.astype(np.uint16) > int(threshold))
    return (mask.astype(np.uint8) * 255)


def run_pipeline_proprietary(img_rgb: np.ndarray,
                             kernel_size: int = 5,
                             sigma: float = 1.5,
                             threshold: int = 30,
                             fast: bool = True) -> Dict[str, Any]:
    """Полный конвейер (А-версия): серый -> гаусс -> собель -> бинаризация."""
    gray = rgb_to_grayscale(img_rgb)
    blurred = gaussian_blur(gray, kernel_size=kernel_size, sigma=sigma, fast=fast)
    edges = sobel_edge_detection(blurred, fast=fast)
    binary = binary_threshold(edges, threshold=threshold)
    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "binary": binary,
    }

