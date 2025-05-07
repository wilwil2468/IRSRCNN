
# [Pytorch] Super-Resolution CNN

Implementation of SRCNN model in **Image Super-Resolution using Deep Convolutional Network** paper with Pytorch.
Trained with FLIR ADAS dataset to focus on upscaling thermal images.

Adam with optimize tuned hyperparameters is used instead of SGD + Momentum. 

3 models in the paper were implemented, SRCNN-915, SRCNN-935, SRCNN-955. But out of consideration for accuracy, only SRCNN-955 will be used. 


# Branches:

full-adas: Full FLIR ADAS dataset is used and experimental algorithm is also used (using LMDB to precompute the dataset and accelerate the training process, and trains based on precomputed patches, enables the full use of the dataset on lower end computers but currently shows worse result)

default: Scaled down (randomly selected, all picture has different properties) FLIR ADAS dataset, and uses the original training approach.


## Contents
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [Evaluate](#evaluate)
- [References](#references)


## Train
Run this command to begin the training:
```
python train.py  --steps=300000                    \
                 --architecture="955"       \
                 --batch_size=128           \
                 --save-best-only=1        \
                 --save-every=1000          \
                 --save-log=1               \
                 --ckpt-dir="checkpoint/x2" 
```
- **--save-best-only**: if it's equal to **0**, model weights will be saved every **save-every** steps.
- **--save-log**: if it's equal to **1**, **train loss, train metric, validation loss, validation metric** will be saved every **save-every** steps.


**NOTE**: if you want to re-train a new model, you should delete all files in sub-directories in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

Download pretrained models here:
- [SRCNN-915.pt](checkpoint/SRCNN915/SRCNN-915.pt)
- [SRCNN-935.pt](checkpoint/SRCNN935/SRCNN-935.pt)
- [SRCNN-955.pt](checkpoint/SRCNN955/SRCNN-955.pt)


## Test
Teledyne's FLIR ADAS dataset is used for training. After Training, models can be tested with scale factors **x2, x3, x4**, the result is calculated by computing average PSNR of all images.

Testing against **Set5** dataset
```
python test.py --scale=4 --architecture=955 --ckpt-path="default"
```

Testing against **FLIR ADAS** dataset 
```
python test_ir.py --scale=4 --architecture=955 --ckpt-path="default"
```

- **--ckpt-path="default"** means you are using default model path, aka **checkpoint/SRCNN{architecture}/SRCNN-{architecture}.pt**. If you want to use your trained model, you can pass yours to **--ckpt-path**.

## Demo 
After Training, you can test models with this command, the result is the **sr.png**.
```
python demo.py --image-path="dataset/test3.png" \
               --architecture="955"             \
               --ckpt-path="default"            \
               --scale=4
```
- **--ckpt-path** is the same as in [Test](#test)

## Evaluate

Models are evaluated against Set5 and FLIR ADAS dataset by PSNR:

<div align="center">

|        Model       |  Set5 ×2 |  Set5 ×3 |  Set5 ×4 |  ADAS ×2 |  ADAS ×3 |  ADAS ×4 |
| :----------------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Original SRCNN‑955 |  36.7996 |  34.2977 |  32.1393 | 38.13365 |  37.2313 | 36.24281 |
| Train1\* SRCNN‑955 | 23.09706 | 23.20932 | 22.96412 | 37.89486 | 37.20422 | 35.98730 |
| Train2\* SRCNN‑955 | 29.00672 | 29.56570 | 28.57095 | 37.89199 | 37.24147 | 35.99952 |
| Train3\* SRCNN‑955 | 29.42715 | 30.00482 | 28.93902 | 37.92979 | 37.24948 | 36.01410 |

</div>

*Train1 = Trained from start exclusively on FLIR ADAS dataset, 300.000 steps

*Train2 = Finetuned from original training checkpoint (300.000 steps original + 131.000 steps finetuning) on FLIR ADAS dataset.

*Train3 = Finetuned from original training checkpoint (300.000 steps original + 70.000 steps finetuning) on FLIR ADAS dataset.

## References
- Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/abs/1501.00092
- SRCNN Matlab code: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
- T91: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- FLIR ADAS: https://www.flir.com/oem/adas/adas-dataset-form/
- Forked from: https://github.com/Nhat-Thanh/SRCNN-Pytorch
