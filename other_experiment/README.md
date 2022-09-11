# Other experiment

This directory contain short report of training OMORI sprite with [lightweight-gan](https://github.com/lucidrains/lightweight-gan) and [stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch). Short experiment shows that,

* lightweight-gan generate diverse sprite, although with lots of noise
* stylegan2-pytorch prone to mode collapse during training

Directory `lightweight-gan` and `stylegan2-pytorch` show small part of experiment result.

## Miscellaneous

> Few tips

* Accumulating gradient with parameter `--gradient-accumulate-every` could significantly reduce training speed. Try to increase `--batch-size` first.
* Pay attention to your storage free space. Frequent model saving could take lots of space.
* Use fast storage to prevent bottleneck.

> Install required tools

```
pip install lightweight_gan==1.0.0 stylegan2_pytorch==1.8.9
```

> Generate video progress of training where the sample is long horizontal image

```sh
ffmpeg \
-framerate 10 -pattern_type glob -i '*[!ema].jpg' \
-framerate 10 -pattern_type glob -i '*[!ema].jpg' \
-framerate 10 -pattern_type glob -i '*-ema.jpg' \
-framerate 10 -pattern_type glob -i '*-ema.jpg' \
-filter_complex \
"[0]crop=1033:260:0:0[in0]; \
 [1]crop=1033:260:1033:0[in1]; \
 [2]crop=1033:260:0:0[in2]; \
 [3]crop=1033:260:1033:0[in3]; \
 [in0][in1][in2][in3]vstack=inputs=4[v]" \
-map "[v]" \
-c:v libx265 \
-crf 30 \
out.mp4
```

```sh
# Get 4 input, where input 0-1 are default image and input 2-3 are EMA image with 10 FPS
-framerate 10 -pattern_type glob -i '*[!ema].jpg'
-framerate 10 -pattern_type glob -i '*[!ema].jpg'
-framerate 10 -pattern_type glob -i '*-ema.jpg'
-framerate 10 -pattern_type glob -i '*-ema.jpg'
# Parameter, crop=width:height:start_x:start_y
# Crop input N and store it to "inN"
[0]crop=1033:260:0:0[in0]
[1]crop=1033:260:1033:0[in1]
[2]crop=1033:260:0:0[in2]
[3]crop=1033:260:1033:0[in3]
# Combine 4 cropped input with vertical stack to "v"
[in0][in1][in2][in3]vstack=inputs=4[v]
# Map video
-map "[v]"
# Use H.265 codec
-c:v libx265
# Video quality, where 28 is the H.265 default
-crf 30
```

> Generate video progress of training

```sh
ffmpeg \
-framerate 10 -pattern_type glob -i '*[!ema][!interp].jpg' \
-framerate 10 -pattern_type glob -i '*-ema.jpg' \
-filter_complex "[0][1]hstack=inputs=2[v]" \
-map "[v]" \
-c:v libx265 \
-crf 30 \
out.mp4
```

> Create 1 big image from 4 small image

```sh
# 4 img to one
ffmpeg -i 1.jpg -i 2.jpg -i 3.jpg -i 4.jpg -filter_complex "[0]pad=ih+4:iw+4[i0];[2]pad=iw+4[i2];[i0][1][i2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map [v] -y o.jpg
```

> Create 1 big image from 4 small image with black padding

```sh
# 4 img to one
ffmpeg -i 1.jpg -i 2.jpg -i 3.jpg -i 4.jpg -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" output.jpg
```
