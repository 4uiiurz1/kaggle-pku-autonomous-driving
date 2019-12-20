## Note
- 6DoF (yaw, pitch, roll, x, y, z)を推定。
- ~~x, yはマスク画像から得られる画像座標とz(depth)から計算できるはずなので、zだけ推定すればいいはず。~~ →誤り。
- zの推定はcenternetの3d pose estimationで使われているEigenの手法を参考にすればいいかも。z = 1 / sigmoid(output) − 1で推定する。ロスはオリジナルのgtとのL1 Loss。→直接回帰した方が精度良い。
- 640 x 512
- 960 x 768
- 1280 x 1024
- 姿勢の学習方法
  - 直接回帰
  - いくつかのビンにわけて各ビン内で回帰
  - クォータニオンとの回帰
- 解像度が大きいモデルはdepthの予測精度が悪化している。(恐らく公式のmetricの並進誤差がより厳しいっぽい)
  - ミス。
- NonLocalBlockを向き推定のheadに追加。
  - 計算が重すぎて無理。
- アンサンブル
  - NMS
  - Soft NMS
  - Weighted Points Fusion
  - decode前のmapを加重平均するのもありかもしれない → とりあえず複数foldのアンサンブルはこれでやる
- 重複・類似画像がテストデータにあるかどうか調べる。
  - 重複画像のアンサンブル結果をそれぞれの画像のpredにする。

## TODO
- [ ] wh用いて車両crop→それぞれ回帰
- [ ] hflip TTA
- [ ] pseudo labeling

## Experiments
### NMS
- resnet18_fpn_120123
- fold1
- w/o mask

NMS  | mAP
-----|---
None | 0.19142559036561727
0.05 | 0.21370899204220017
0.1  | 0.21371086361037867
0.5  | 0.2136521156341741
1.0  | 0.21339207211238534

### 1280 x 1024
- resnet18_fpn
- 1fold
- val score_th=0.1
- test score_th=0.6
- nms (th=0.1) (testのみ)
- w/o mask

model | val mAP | PublicLB
------|---------|----------
120708 (w/o gn+ws, w/o wh, det=DepthL1) | 0.21807085313661206
120713 (w/o gn+ws, w/o wh, det=L1) | 0.23062173249021703 | 0.070
120720 (w/o gn+ws, w/o wh, det=L1) | 0.23069422381934448 | 0.073
120814 (w/o gn+ws, w/ wh, det=L1) | 0.23285714122775697 | 0.078
120823 (w/o gn+ws, w/ wh, det=L1, lhalf) | 0.2357547338344435 | 0.070

### Epochs
- resnet18_fpn
- 5fold
- nms (th=0.1)
- ensemble each fold preds by merging output maps
- test score_th=0.3
- w/o mask
- 1280 x 1024
- w/o gn+ws

model | val mAP | PublicLB
------|---------|----------
120823 (30epochs) | 0.23875358249433876 | 0.095
121123 (50epochs) | 0.2456167416237179 | 0.100

### Various architecture
- 5fold
- ensemble each fold preds by merging output maps
- test score_th=0.3
- 1280 x 1024
- w/o gn+ws

#### w/o mask
model | val mAP | PublicLB
------|---------|----------
resnet18_fpn_121123 | 0.2456167416237179  | 0.100
resnet34_fpn_121307 | 0.24720156734031282 | 0.098

#### w/ mask
model | val mAP | PublicLB
------|---------|----------
resnet18_fpn_121123 | 0.24946095932243853 | 0.109
resnet50_v1d_fpn_121512 | 0.25247039883369304 | 0.100
resnet34_v1b_fpn_121721 | 0.24874580307912306 |

#### gn+ws
- resnet18_fpn
- 1fold
- val score_th=0.1
- test score_th=0.6
- nms (th=0.1) (testのみ)
- w/o mask

model | val mAP | PublicLB
------|---------|----------
121123 (bn) | 0.24137380667425745 |
122001 (gn+ws) | 0.23614911826902013 | 0.082
