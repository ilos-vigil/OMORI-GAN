lightweight_gan \
    --name OMORI_SPRITE4 --data ../_dataset/processed_sprite \
    --amp --save_every 100 --evaluate_every 100 --transparent \
    --batch-size 40 \
    --gradient-accumulate-every 1 \
    --num-train-steps 60000 \
    --image_size 32 \
    --attn_res_layers=[] \
    --aug-types [cutout,color] \
    --dual_contrast_loss