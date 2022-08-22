# ATLAS-MVSNet

## About

We present ATLAS-MVSNet, an end-to-end deep learning multi-view stereo network for depth map inference from high-resolution images. Distinct from existing works, we introduce a novel module design for neural networks, which we termed Hybrid Attention Block (HAB), that utilizes ATtention LAyerS (ATLAS).
While many state-of-the-art methods need multiple high-end GPUs in the
training phase, we are able to train our network on a single
consumer grade GPU (11GB VRAM).

<b>ATLAS-MVSNet is able to evaluate images of up to 14 MP (4536 × 3024 pixel).</b>

<img src="images/network.png">

If you find this project useful for your research, please cite:
```
@InProceedings{weilharter2022atlas,
    author = {Weilharter, Rafael and Fraundorfer, Friedrich},
    title = {ATLAS-MVSNet: Attention Layers for Feature Extraction and Cost Volume Regularization in Multi-View Stereo},
    booktitle = {2022 26th International Conference on Pattern Recognition (ICPR)},
    year = {2022},
    organization={IEEE}
}
```

## How To Use

### Requirements

* Nvidia GPU with 11GB or more VRAM
* CUDA 10.1+
* python 3.6+
* pytorch 1.4+
* opencv 3.4.2+

### Datasets
Pre-processed datasets can be downloaded on the github page of [MVSNet](https://github.com/YoYo000/MVSNet) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS).
Our repository provides 3 dataloaders for [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), [BlendedMVS](https://github.com/YoYo000/BlendedMVS) and [Tanks and Temples (TaT)](https://www.tanksandtemples.org/), respectively.

### Training
Run the command `python train.py -h` to get information about the usage. An example can be found in `train_dtu.sh` (set the correct paths to the training data).

### Testing
Run the command `python test.py -h` to get information about the usage. An example can be found in `test_dtu.sh` (set the correct paths to the testing data).
You can use your own trained weights or use the weights provided in `checkpoints/atlas_blended.ckpt`. These are the weights obtained by training on the DTU dataset and then finetuning on BlendedMVS as described in our paper and evaluated on the TaT benchmark.

## DEMO VIDEO
https://youtu.be/ZLKGKlTloAI

## Performance

### DTU
| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.278     | 0.377      | 0.327        |

### TaT (F-score)
| Mean  | Family | Francis | Horse | LH    | M60   | Panther | PG    | Train |
|-------|--------|---------|-------|-------|-------|---------|-------|-------|
| 60.71 | 77.62  | 61.94   | 49.55 | 61.63 | 60.04 | 58.69   | 63.58 | 52.59 |


## Acknowledgement

Parts of the code are adapted from [HSM](https://github.com/gengshan-y/high-res-stereo), [MVSNet](https://github.com/YoYo000/MVSNet) and [Stand-Alone-Self-Attention](https://github.com/leaderj1001/Stand-Alone-Self-Attention).
