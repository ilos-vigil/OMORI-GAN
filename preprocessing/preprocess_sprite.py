import os
import argparse
import utils
import numpy as np
from glob import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://github.com/python-pillow/Pillow/issues/1510
BASE_SIZE: int = 32  # both width and height
SELECTED_GLOB: list = [
    'characters/*',
    'pictures/FA_thank_you.png',
    # 'tilesets/*',  # Has few human sprite, but mixed with various size of non-human sprite
    # 'pictures/ROCK*'
    # 'pictures/SNAKE*'
]
# These list only consider 32x32 sprite and some 32x64, 64x32 and 64x64 sprite
CHARACTER_NAME: list = [
    # main cast
    'omori', 'player', 'mari', 'something', 'aubrey', 'kel', 'hero', 'basil',
    # dw multiple
    'dw_sh',
    'heart', 'mannequin', 'masked',
    'mermaid', 'reef', 'slimegirls',
    # dw specific
    'biscuit', 'boss', 'duck', 'faceguy', 'gator',
    'headbutt', 'jawsum', 'kite_kid', 'marina', 'pinkbeard',
    'pirate', 'pluto', 'rosa', 'sbf', 'seacow', 'van',
    # fa multiple
    'fa_climb', 'fa_cooking', 'fa_en', 'fa_fighting', 'fa_j',
    'fa_memory', 'fa_rai', 'fa_religious', 'fa_sitting', 'fa_thank_you',
    'average', 'ayee', 'bebe', 'church', 'fruitseller', 'old',
    'pharm_strict', 'pizzapeople', 'primadonas', 'recyc', 'secretlake'
    # fa specific
    'angel', 'artist', 'bowen', 'brandi', 'brent', 'buttscratch', 'candice',
    'candypink', 'cashier', 'cesar', 'charlie', 'cris', 'curtsey', 'dustine',
    'gino', 'grandma', 'groundskeeper', 'hobo', 'hooligans', 'jesse', 'joy',
    'karen', 'katie',  'kim', 'maverick', 'meatguymovers', 'michael',
    'mondo', 'polly', 'pretty_boy', 'priest', 'residents', 'sally', 'sarah',
    'sean', 'tim', 'tucker', 'vance'
    # mixed
    'breaktime', 'npclist', 'party',
    'boy', 'dad', 'girl', 'mom',
    # 'npc'
]
EXCLUDE_NAME: list = [
    # dw non-humanoid/mixed data
    'bluegirl', 'bunnies', 'dw_en', 'dw_sprm', 'dungeon', 'ghostparty', 'humphrey',
    'mole', 'birds', 'crow', 'ems',  'mewo', 'snaley', 'tvgirl', 'whaley', 'budgirl',
    'gibs', 'cages', 'masked',
    # fa non-humanoid/mixed data
    'en_faraway', 'dog', 'cat', 'pet', 'aubrey_bat', 'kel_dig', 'kevin',
    # not human being
    'bathhub', 'bolb', 'bridge', 'bulb', 'cut', 'door', 'dw_64',
    'dw_en_pf', 'fishing', 'gate', 'jukebox', 'manonfire', 'mirror', 'object',
    'parasol', 'playerhouse', 'raft', 'sink', 'sleep', 'statue', 'switch',
    'tumor', 'wave_of_hands', 'barrel'
]


def get_image_path(args):
    image_path = []
    for g in SELECTED_GLOB:
        image_path.extend(
            glob(f'{args.src}/{g}')
        )
    print(f'Total files: {len(image_path)}')

    filtered_image_path = []
    for ip in image_path:
        image_filename = os.path.basename(ip).lower()
        if (any([name in image_filename for name in CHARACTER_NAME]) \
            and not any([name in image_filename for name in EXCLUDE_NAME]) ):
            filtered_image_path.append(ip)

    print(f'Total files after filter: {len(filtered_image_path)}')
    return filtered_image_path


def save_image(img, image_path, args, idx=None):
    image_filename = os.path.basename(image_path)
    if idx is not None:
        sep_index = image_filename.rindex('.')
        image_filename = f'{image_filename[:sep_index]}_{idx}{image_filename[sep_index:]}'

    processed_image_path = os.path.join(
        args.dst, image_filename
    )

    if args.debug:
        print(f'Saving procssed image to {processed_image_path}')

    img_mode = 'RGBA'
    if len(img.shape) == 2:
        img_mode = 'L'  # grayscale
    elif img.shape[2] == 3:
        img_mode = 'RGB'

    img = Image.fromarray(img, mode=img_mode)
    img.save(processed_image_path)


def check_all_seperator(img: np.ndarray, type: str, gap: int) -> bool:
    color = 'RGBA'
    if len(img.shape) == 2:
        color = 'L'  # grayscale
    elif img.shape[2] == 3:
        color = 'RGB'

    if type == 'width':
        gap_left, gap_right = [], []
        for w in range(gap, img.shape[1]+1, gap):
            if color == 'RGBA':  # check all pixel transparent
                gap_left.append(
                    (img[:, w-gap, 3] == 0).all()
                )
                gap_right.append(
                    (img[:, w-1, 3] == 0).all()
                )
            else:
                gap_left.append(
                    (img[:, w-gap] == img[0, w-gap]).all()
                )
                gap_right.append(
                    (img[:, w-1] == img[0, w-1]).all()
                )
        if (sum(gap_left) > len(gap_left) * args.split_threshold \
            or sum(gap_right) > len(gap_right) * args.split_threshold):
            return True

    elif type == 'height':
        gap_top, gap_bottom = [], []
        for h in range(gap, img.shape[0]+1, gap):
            if color == 'RGBA':  # check all pixel transparent
                gap_top.append(
                    (img[h-gap, :, 3] == 0).all()
                )
                gap_bottom.append(
                    (img[h-1, :, 3] == 0).all()
                )
            else:
                gap_top.append(
                    (img[h-gap, :] == img[h-gap, 0]).all()
                )
                gap_bottom.append(
                    (img[h-1, :] == img[h-1, 0]).all()
                )
        if (sum(gap_top) > len(gap_top) * args.split_threshold \
            or sum(gap_bottom) > len(gap_bottom) * args.split_threshold):
            return True

    return False


def get_image_seperator(img: np.ndarray, args) -> tuple:
    height, width = img.shape[:2]
    width_sep, height_sep = None, None

    # Check if has 32px width seperator
    if width == BASE_SIZE:
        width_sep = BASE_SIZE
    else:
        if check_all_seperator(img, 'width', BASE_SIZE):
            width_sep = BASE_SIZE

    # Check if has 64px width seperator
    if width_sep is None:
        if width == BASE_SIZE * 2:
            width_sep = BASE_SIZE * 2
        elif check_all_seperator(img, 'width', BASE_SIZE * 2):
            width_sep = BASE_SIZE * 2

    # Check if has 32px height seperator
    if height == BASE_SIZE:
        height_sep = BASE_SIZE
    else:
        if check_all_seperator(img, 'height', BASE_SIZE):
            height_sep = BASE_SIZE

    # Check if has 64 px height seperator
    if height_sep is None:
        if height == BASE_SIZE * 2:
            height_sep = BASE_SIZE * 2
        elif check_all_seperator(img, 'height', BASE_SIZE * 2):
            height_sep = BASE_SIZE * 2

    return width_sep, height_sep


def convert_to_zero(img: np.ndarray, args) -> np.ndarray:
    # Only for RGBA image, [?, ?, ?, 0] -> [0, 0, 0, 0]
    if len(img.shape) == 3 and img.shape[2] == 4:
        condition = img[:, :, 3] == 0
        condition = np.stack(
            [condition, condition, condition, condition],
            axis=2
        )

        color = args.replace_transparent
        new_data = np.empty(img.shape, dtype=np.uint8)
        new_data.fill(color)

        new_img = np.where(condition, new_data, img)
        return new_img
    else:
        return img


def filter_image(image_path, args):
    # 1. load img
    try:
        img = Image.open(image_path)
        img = np.asarray(img)
        height, width = img.shape[:2]
    except Exception as ex:
        print(f'Error at {image_path}: {ex}')
        return 0

    # 2. check image size
    if width % BASE_SIZE == 0 and height % BASE_SIZE == 0:
        # 3. get total column and row
        width_sep, height_sep = get_image_seperator(img, args)
        if width_sep is None or height_sep is None:
            return 0
        if args.try_crop_extra and width_sep == BASE_SIZE * 2 and height_sep == BASE_SIZE * 2:
            return 0
        if not args.try_crop_extra and not (width_sep == BASE_SIZE and height_sep == BASE_SIZE):
            return 0
        column, row = width // width_sep, height // height_sep
        if args.debug:
            print(f'Image at {image_path} may contain {column * row} sprite with resolution {width_sep}x{height_sep}')

        # 4. Convert to zero
        if args.replace_transparent is not None:
            img = convert_to_zero(img, args)

        # 5. split image
        imgs = []
        for c in range(column):
            for r in range(row):
                # some false positive
                if not args.only_front or (args.only_front and r % 4 == 0):
                    x_start = c * width_sep
                    x_end = (c+1) * width_sep
                    y_start = r * height_sep
                    y_end = (r+1) * height_sep

                    img_part = img[y_start:y_end, x_start:x_end]
                    imgs.append(img_part)

        # 6. extra crop for 32x64, 64x32 and 64x64 sprite
        if args.try_crop_extra and (width_sep == 64 or height_sep == 64):
            pass

        # 7. save
        for idx, img in enumerate(imgs):
            if not utils.discard_image(img, sprite=True):
                save_image(img, image_path, args, idx)

        return 1
    else:
        if args.debug:
            print(f'Image at {image_path} has non 32x32, 32x64, 64x32 or 64x64 sprite')
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess sprite')
    parser.add_argument('--src', default='../_dataset/OMORI/www/img')
    parser.add_argument('--dst', default='../_dataset/processed_sprite')
    parser.add_argument('--split_threshold', default=0.74)
    parser.add_argument('--try_crop_extra', default=False)
    parser.add_argument('--replace_transparent', type=int, default=None)
    parser.add_argument('--only_front', action='store_true')
    parser.add_argument('--debug', default=False)

    args = parser.parse_args()
    args.src = os.path.abspath(args.src)
    args.dst = os.path.abspath(args.dst)
    utils.check_path_exist(args.dst)
    print(args)

    print(f"Sprite source: {args.src}")
    print(f"Sprite destination: {args.dst}")

    image_path = get_image_path(args)
    total_processed = 0
    for idx, ip in enumerate(image_path):
        if args.debug:
            print(f'({idx+1}/{len(image_path)}) Check image {ip}')
        total_processed += filter_image(ip, args)

    print(f'Total processed file: {total_processed}')
    total_files = utils.get_total_files(args.dst)
    print(f'Total sprite: {total_files}')
