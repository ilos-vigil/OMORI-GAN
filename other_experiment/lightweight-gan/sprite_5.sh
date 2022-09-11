lightweight_gan \
    --name OMORI_SPRITE5 --data ../_dataset/processed_sprite \
    --amp --save_every 100 --evaluate_every 100 \
    --batch-size 32 \
    --gradient-accumulate-every 1 \
    --num-train-steps 60000 \
    --image_size 32 \
    --attn_res_layers=[] \
    --aug-types [translation,cutout,color]