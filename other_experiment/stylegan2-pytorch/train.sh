stylegan2_pytorch \
    --data ../_dataset/processed_sprite \
    --name OMORI_SPRITE \
    --aug-prob 0.6 \
    --image-size 32 \
    --save-every 500 \
    --evaluate-every 250 \
    --batch-size 32 \
    --gradient-accumulate-every 1
