import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from proprietary import proprietary_impl
from withLib import scki_kit_impl


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_image(path: str) -> np.ndarray:
    img = plt.imread(path)
    if img.dtype in (np.float32, np.float64) and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def resolve_input_path(path_or_dir: str) -> str:
    p = Path(path_or_dir)
    if p.is_file():
        return str(p)
    if p.is_dir():
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for ext in exts:
            found = sorted(p.rglob(ext))
            if found:
                return str(found[0])
    raise FileNotFoundError("Не найдено входного изображения")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A (proprietary) and B (library) pipelines")
    parser.add_argument('--input', '-i', type=str, default='Lab1/dataset', help='Путь к файлу или папке')
    parser.add_argument('--outdir', '-o', type=str, default='outputs', help='Куда сохранять результаты')
    parser.add_argument('--sigma', type=float, default=1.5)
    parser.add_argument('--kernel', type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    img_path = resolve_input_path(args.input)
    image = load_image(img_path)

    t0 = time.time()
    result_a, _, _ = proprietary_impl(image, gauss_size=args.kernel, sigma=args.sigma, outdir=args.outdir)
    ta = time.time() - t0

    t0 = time.time()
    result_b, _, _ = scki_kit_impl(image, sigma=args.sigma, outdir=args.outdir)
    tb = time.time() - t0

    diff = np.abs(result_a.astype(float) - result_b.astype(float)).mean()
    print(f'A: {ta:.2f}s  B: {tb:.2f}s  MAE: {diff:.3f}')


if __name__ == '__main__':
    main()