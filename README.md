# kaggle-pku-autonomous-driving

## Approach
My approach is based on [CenterNet](https://github.com/xingyizhou/CenterNet).
I reimplemented CenterNet almost from scratch with reference to [the author's implementation](https://github.com/xingyizhou/CenterNet).

### Heads
- heatmap[1]
- xy offset[2]
- z (depth)[1]
- pose[6]: cos(yaw), sin(yaw) cos(pitch), sin(pitch), cos(roll), sin(roll)
- wh[2]: It's not used for prediction, but PublicLB was improved by learning this as an auxiliary task.

Heatmap's loss is Focal Loss, and the others are L1Loss. The weight of wh loss is 0.05. Mask regions of mask images are ignored when calculating loss.

### Network architecture
- [ResNet18 (pretrained ImageNet)](https://github.com/Cadene/pretrained-models.pytorch) + FPN (channels: 256->256->256)
- [DLA34 (pretrained KITTI 3DOP)](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md) + FPN (channels: 256->128->64)
- Input size: 2560 x 2048
- Output size: 640 x 512

Increasing the input size is very effective, mAP was improved dramatically.
I tried deeper networks (ResNet34, 50) but not worked.

### Augmentation
- HFlip (p=0.5): Flip images horizontally and `yaw *= -1, roll *= -1`.
- RandomShift (p=0.5, limit=0.1): Shift images and positions (x, y).
- RandomScale (p=0.5, limit=0.1): Scale images and positions (x, y, z).
- RandomHueSaturationValue (p=0.5, hue_limit=20)
- RandomBrightness (p=0.5, limit=0.2)
- RandomContrast (p=0.5, limit=0.2)

### Training
- Optimizer: RAdam
- LR scheduler: CosineAnnealingLR (lr=1e-3 -> 1e-5)
- 50epochs
- 5-folds cv
- Batch size: 4

### Post Processing
- Remove mask regions from predictions by multiplying heatmap by masks.
- NMS (distance threshold: 0.1): I'm not sure how effective this is...
- Find duplicate images with imagehash and ensemble them. PublicLB was slighly improved.
- Score threshold: 0.3

### Ensemble
Ensemble each fold models and two models (ResNet18, DLA34) by averaging the raw output maps.

### Score summary
model | val mAP | PublicLB|
------|:-------:|:----------:
ResNet18 + FPN (resnet18_fpn_12222) | 0.2572243059006951  | 0.118
DLA34 + FPN (dla34_ddd_3dop_123008) | 0.2681900383192367  | 0.118
Ensemble (ensemble_010414)          | -                   | 0.121
