import argparse
import os
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from proprietary import run_pipeline_proprietary
from withLib import run_pipeline_library


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image_gray(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_synthetic_noisy_shape(size: Tuple[int, int] = (256, 256),
                                   shape: str = 'circle',
                                   noise_level: float = 25.0) -> np.ndarray:
    """Генерирует RGB изображение с фигурами и шумом."""
    w, h = size[0], size[1]
    img = Image.new('RGB', (w, h), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)

    # Фоновые градиенты/тени (простые полосы)
    for y in range(0, h, 16):
        shade = int(235 - (y / h) * 30)
        draw.rectangle([(0, y), (w, min(h, y + 8))], fill=(shade, shade, shade))

    # Основная фигура
    cx, cy = w // 2, h // 2
    size_min = min(w, h)
    r = size_min // 4
    if shape == 'circle':
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 0, 0), width=4)
    elif shape == 'square':
        draw.rectangle([(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 0, 0), width=4)
    else:  # diamond
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline=(0, 0, 0), width=4)

    # Добавим внутреннее содержимое (перекрестие)
    draw.line([(cx - r // 2, cy), (cx + r // 2, cy)], fill=(0, 0, 0), width=2)
    draw.line([(cx, cy - r // 2), (cx, cy + r // 2)], fill=(0, 0, 0), width=2)

    # Преобразуем в массив и добавим гауссов шум
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0.0, noise_level, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return noisy


def main() -> None:
    parser = argparse.ArgumentParser(description='Lab1: Обработка изображений (А: авторская, B: OpenCV)')
    parser.add_argument('--input', '-i', type=str, default='Lab1/dataset', help='Путь к входному файлу ИЛИ каталогу с изображениями')
    parser.add_argument('--pattern', type=str, default='*.png;*.jpg;*.jpeg;*.bmp', help='Шаблоны поиска изображений (через ;) если указан каталог')
    parser.add_argument('--outdir', '-o', type=str, default='outputs', help='Каталог для сохранения результатов')
    parser.add_argument('--kernel', type=int, default=5, help='Размер ядра Гаусса (нечетное)')
    parser.add_argument('--sigma', type=float, default=1.5, help='Sigma для Гаусса')
    parser.add_argument('--thr', type=int, default=30, help='Порог бинаризации')
    parser.add_argument('--shape', type=str, choices=['circle', 'square', 'diamond'], default='circle', help='Фигура для синтетики')

    args = parser.parse_args()
    ensure_dir(args.outdir)

    # Определение входного файла: поддержка директории и паттернов
    selected_file: Optional[Path] = None
    input_path = Path(args.input)
    if input_path.is_dir():
        patterns = [p.strip() for p in args.pattern.split(';') if p.strip()]
        for patt in patterns:
            matches = sorted(input_path.rglob(patt))
            if matches:
                selected_file = matches[0]
                break
    elif input_path.is_file():
        selected_file = input_path

    # Загрузка или генерация
    if selected_file and selected_file.exists():
        img_rgb = load_image_rgb(str(selected_file))
    else:
        img_rgb = generate_synthetic_noisy_shape(shape=args.shape)
        synth_path = os.path.join(args.outdir, 'synthetic_input.png')
        Image.fromarray(img_rgb).save(synth_path)

    # Вариант А (авторский)
    prop = run_pipeline_proprietary(img_rgb, kernel_size=args.kernel, sigma=args.sigma, threshold=args.thr)
    save_image_gray(prop['gray'], os.path.join(args.outdir, 'A_gray.png'))
    save_image_gray(prop['blurred'], os.path.join(args.outdir, 'A_blur.png'))
    save_image_gray(prop['edges'], os.path.join(args.outdir, 'A_edges.png'))
    save_image_gray(prop['binary'], os.path.join(args.outdir, 'A_binary.png'))

    # Вариант B (библиотека OpenCV)
    lib = run_pipeline_library(img_rgb, kernel_size=args.kernel, sigma=args.sigma, threshold=args.thr)
    save_image_gray(lib['gray'], os.path.join(args.outdir, 'B_gray.png'))
    save_image_gray(lib['blurred'], os.path.join(args.outdir, 'B_blur.png'))
    save_image_gray(lib['edges'], os.path.join(args.outdir, 'B_edges.png'))
    save_image_gray(lib['binary'], os.path.join(args.outdir, 'B_binary.png'))

    # Сохранить исходник
    Image.fromarray(img_rgb).save(os.path.join(args.outdir, 'input_rgb.png'))

    print('Готово. Результаты сохранены в:', os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()


