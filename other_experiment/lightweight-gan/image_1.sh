#!/bin/bash
# This script assume you already activate virtual environment and on directory 2_lightweight_gan

lightweight_gan \
    --name OMORI_GAN --data ../_dataset/processed_img \
    --amp --save_every 250 \
    --dual-contrast-loss \
    --batch-size 8 \
    --gradient-accumulate-every 4 \
    --num-train-steps 150000 \
    --image_size 256 \
    --attn_res_layers=[]
