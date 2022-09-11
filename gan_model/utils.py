import os
from glob import glob
from math import log2
from typing import Union


def check_path_exist(path: str):
    if not os.path.exists(path):
        print(f'Path {path} is not exist, creating now!')
        os.makedirs(path)


def check_aspect_ratio(width: int, height: int) -> str:
    if width == height:
        return '1:1'
    if width / 4 == height / 3:
        return '4:3'
    return str(width/height)


def determine_network_depth(width: int, height: int) -> int:
    aspect_ratio = check_aspect_ratio(width, height)

    if aspect_ratio == '4:3':
        min_depth = 3  # for block 5,4 10,8 and 20,15
        extra_depth = log2(width//20)

        return int(min_depth + extra_depth)
    else:  # assume 1:1 aspect ratio
        depth = int(log2(width)) - 1
        return depth


def get_ckpt(name, ckpt_name) -> Union[str, None]:
    ckpt_path = None

    if ckpt_name is not None:
        if os.path.exists(f'./{name}/checkpoints/{ckpt_name}'):
            return f'./{name}/checkpoints/{ckpt_name}'
        else:
            print('Argument ckpt_name specified, but not found! Fallback to latest ckpt file if exists.')

    files = glob(f'./{name}/checkpoints/*ckpt')
    if len(files) > 0:
        ckpt_path = max(files, key=os.path.getctime)

    return ckpt_path
