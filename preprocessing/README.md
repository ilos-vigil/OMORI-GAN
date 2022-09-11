# Preprocessing

## File description

* `preprocess_sprite.py` is used to preprocess sprite
* `get_hash_file.py` is used to create CSV file which contain filename and it's SHA256 hash. Useful for logging/documentation.
* `utils.py` contains some shared function used to process dataset.

## How to use

First of all, you need to decrypt image file from OMORI game. This tutorial use `mvdecoder` which written by kin-kun. However you also could use [OneLoader](https://github.com/rphsoftware/OneLoader) or [playtest](https://mods.one/mod/playtest) if you want to decrypt all game data. Take note both alternative require you to open the game.

```
python mv_decoder.py /path/to/OMORI/directory ../_dataset
```

Second step is filter and preprocess sprite from directory. Here are few additional parameter you could use,
* `--replace_transparent` to replace transparent pixel `(R, G, B, A=0)` with certain black/white color within range 0-255.
* `--only_front` only to use sprite with front direction. Take note this script assume 1st, 5th, ... row used to store sprite with front direction, so expect few false positive on processed sprite.

```
python preprocess_sprite.py \
   --src ../_dataset/OMORI/www/img \
   --dst ../_dataset/processed_sprite \
   --replace_transparent 127 \
   --only_front
```

Optionally you can create CSV file which contain filename and it's hash from processed sprite. This script depends on `sha256sum` executable, so this script won't run on Windows or Mac OS X by default.

```
python get_file_hash.py \
   --path ../_dataset/processed_sprite
   --out_file hash_sprite.csv
```
