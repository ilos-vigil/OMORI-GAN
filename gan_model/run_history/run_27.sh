python cli.py \
    --train \
    --name OMORI_SPRITE_AIM_27_LONG \
    --image_path ../_dataset/processed_sprite_humanoid_front --im_size 32 \
    --aug_level 2 --aug_type color translation cutout hflip \
    --n_z 16 --n_gf 64 --n_df 64 \
    --g_conv_type 1 --g_upscale_type 0 --d_norm_type 0 \
    --use_aim --batch_size 32 --n_epochs 2000 \
    --smoothing_value 0.05 0 --loss_type 0 \
    --g_lr 0.0001 --d_lr 0.0003 \
    --save_freq 25 \