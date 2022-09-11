# GAN Model

This directory contain DCGAN implementation using PyTorch Lightning.

## File description

* `cli.py` main entry of the code.
* `dataset.py` used to load and transform images. 
* `diffaug.py` is Differentiable Augmentation with extra horizontal flip augmentation.
* `model.py` contain DCGAN Generator and Discriminator module.
* `pl_fun.py` contain function to train GAN model, generate iamge and convert Generator to ONNX format.
* `utils.py`

## How to use

You can run `python cli.py -h` to see all available parameter to train model, generate image from trained model or convert Generator to ONNX format. Here are few basic example.

```sh
python cli.py \
  --train \
  --name SPRITE_RUN_1 \
  --image_path /path/to/image/directory \
  --im_size 32
```

```sh
python cli.py \
  --generate \
  --name SPRITE_RUN_1 \
  --total_image 1000
```

```sh
python cli.py \
  --to_onnx \
  --name SPRITE_RUN_1 \
  --ckpt_name checkpoint_filename_with_best_performance.ckpt
```

You also can install Aim if you perform to track training on your training runs. It's useful when you wish to perform hyperparameter search.

```sh
# install Aim
pip install aim==3.13.1
# train while enabling Aim loggin
python cli.py \
  --train \
  --use-aim \
  --name SPRITE_RUN_1 \
  --image_path /path/to/image/directory \
  --im_size 32
# Run Aim on directory where you run cli.py
aim up
```

## Run history

Directory `run_history` contain details of some training run of GAN training. Here's explanation of file and directory.

* `run_XX.sh` is script is used to train GAN model.
* `OMORI_SPRITE_AIM_XX_LONG` contains model weight and sample of generated image during training.
* `hash_sprite.csv` contain 1128 filename and filename hash of processed sprite which used to train GAN model.
* `fid_report.ods` contains FID evaluation based on 1128 train image and 10000 generated image. `pytorch-fid==0.2.1` is used to obtain FID result on 2 different layer.

```sh
# generate 10000 image
python cli.py \
    --generate --name OMORI_SPRITE_AIM_25_LONG \
    --ckpt_name GAN_epoch=9999_g_loss=2.7776_d_loss=0.2691.ckpt \
    --total_image 10000 \
    --output_path /path/to/generated/image/directory
# get FID from 2 different layer
pytorch-fid --device cuda /path/to/generated/image/directory
pytorch-fid --device cuda --dims 768 /path/to/generated/image/directory
```
