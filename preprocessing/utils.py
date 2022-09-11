import os
import numpy as np
from PIL import Image


def check_path_exist(path: str):
    if not os.path.exists(path):
        print(f"Path {path} isn't exist, creating now!")
        os.makedirs(path)


def get_total_files(path: str) -> int:
    total_files = 0
    for _, _, files in os.walk(path):
        total_files += len(files)
    return total_files


def discard_image(img: Image.Image, sprite: bool = False) -> bool:
    img: np.ndarray = np.asarray(img)

    # Check is RGBA image and  fully transparent
    if len(img.shape) == 3 and img.shape[2] == 4 and (img[:, :, 3] == 0).all():
        return True
    # Check all pixel has same color
    if (img[0, 0] == img[:, :]).all():
        return True
    return False
