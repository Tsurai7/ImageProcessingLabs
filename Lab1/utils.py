import os
import numpy as np

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_step(img: np.ndarray, name: str, variant: str, base_dir: str) -> None:
    import matplotlib.pyplot as plt
    folder = os.path.join(base_dir, f"variant_{variant}")
    ensure_dir(folder)
    path = os.path.join(folder, f"{name}.png")
    if img.dtype == bool:
        plt.imsave(path, img.astype(np.uint8) * 255, cmap="gray")
    elif img.ndim == 2:
        plt.imsave(path, img.astype(np.uint8), cmap="gray")
    else:
        plt.imsave(path, img.astype(np.uint8))