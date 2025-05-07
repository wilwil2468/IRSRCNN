
# [Pytorch] Super-Resolution CNN

Implementation of SRCNN model in **Image Super-Resolution using Deep Convolutional Network** paper with Pytorch.
Nhat-Thanh/SRCNN-Pytorch implementation trained on Teledyne's FLIR ADAS dataset.


## Contents
- [Train](#train)
- [Test](#test)
- [Demo](#demo)
- [Evaluate](#evaluate)
- [References](#references)


## Train
You run this command to begin the training:
```
python train.py  --steps=300000                    \
                 --architecture="915"       \
                 --batch_size=128           \
                 --save-best-only=0         \
                 --save-every=1000          \
                 --save-log=0               \
                 --ckpt-dir="checkpoint/x2" 
```
- **--save-best-only**: if it's equal to **0**, model weights will be saved every **save-every** steps.
- **--save-log**: if it's equal to **1**, **train loss, train metric, validation loss, validation metric** will be saved every **save-every** steps.


**NOTE**: if you want to re-train a new model, you should delete all files in sub-directories in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

You can get original models here:
- [SRCNN-915.pt](checkpoint/SRCNN915/SRCNN-915.pt)
- [SRCNN-935.pt](checkpoint/SRCNN935/SRCNN-935.pt)
- [SRCNN-955.pt](checkpoint/SRCNN955/SRCNN-955.pt)


## Test
FLIR ADAS dataset is used for training. After Training, you can test models with scale factors **x2, x3, x4**, the result is calculated by compute average PSNR of all images.

Below is to test the model against the Set5 RGB dataset 
```
python test.py --scale=2 --architecture=955 --ckpt-path="default"
```

Below is to test the model against the FLIR ADAS validation dataset
```
python test_ir.py --scale=2 --architecture=955 --ckpt-path="default"
```

- **--ckpt-path="default"** means you are using default model path, aka **checkpoint/SRCNN{architecture}/SRCNN-{architecture}.pt**. If you want to use your trained model, you can pass yours to **--ckpt-path**.


## Demo 
After Training, you can test models with this command, the result is **sr.png** in the main folder.
```
python demo.py --image-path="dataset/test4.png" \
               --architecture="955"             \
               --ckpt-path="default"            \
               --scale=2
```
- **--ckpt-path** is the same as in [Test](#test)

## Evaluate

Out of the three models (SRCNN-915, SRCNN-935, SRCNN-955) provided in the original implementation, only SRCNN-955 is used in consideration of accuracy.

<div align="center">

|        Model        | Set5 x2 | Set5 x3 | Set5 x4 |  ADAS x2  |  ADAS x3  |  ADAS x4  |
|:-------------------:|:-------:|:-------:|:-------:|:---------:|:---------:|:---------:|
| Original SRCNN-955	| 36.7996 | 34.2977 | 32.1393 |	38.13365  | 37.2313   | 36.24281  |
| Train1* SRCNN-955	  |23.097055|23.209316|22.964123| 37.894855 | 37.204224 | 35.9873   |
| Train2* SRCNN-955	  |29.006723|29.565699|28.57095 | 37.891994 | 37.241474 | 35.999523 |
| Train3* SRCNN-955	  |29.427145|30.004822|28.939016|	37.92979  | 37.24983  | 36.014103 |

*Train1 = Trained from start exclusively on FLIR ADAS dataset, 300.000 steps

*Train2 = Finetuned from original training checkpoint (300.000 steps original + 131.000 steps finetuning) on FLIR ADAS dataset.

*Train3 = Finetuned from original training checkpoint (300.000 steps original + 70.000 steps finetuning) on FLIR ADAS dataset.

</div>

## References
- Image Super-Resolution Using Deep Convolutional Networks: https://arxiv.org/abs/1501.00092
- SRCNN Matlab code: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
- T91: http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_training_datasets.zip
- Set5: https://filebox.ece.vt.edu/~jbhuang/project/selfexsr/Set5_SR.zip
- FLIR ADAS dataset: https://www.flir.com/oem/adas/adas-dataset-form/
- Forked from: https://github.com/Nhat-Thanh/SRCNN-Pytorch
