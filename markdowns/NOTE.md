## Note
- 6DoF (yaw, pitch, roll, x, y, z)を推定。
- ~~x, yはマスク画像から得られる画像座標とz(depth)から計算できるはずなので、zだけ推定すればいいはず。~~ →誤り。
- zの推定はcenternetの3d pose estimationで使われているEigenの手法を参考にすればいいかも。z = 1 / sigmoid(output) − 1で推定する。ロスはオリジナルのgtとのL1 Loss。→直接回帰した方が精度良い。
- 640 x 512
- 960 x 768
- 1280 x 1024
- 2560 x 2048
- 3360 x 2688
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
- DCNの効果は確認できていない。
  - decoderのp3~p4, p4に入れてみたけどlossの下がり方はベースラインよりも良くない。
- ネットワークを大きくしても効果なし。
  - 他の人も同じのよう。
  - 入力画像サイズの方が重要か？
```sh
python train.py --arch dla34_ddd_3dop --input_w 2560 --input_h 2048 --head_conv 64 --num_filters 256,128,64 --wh_weight 0.05
```
- ws, gnがうまくいかないのはbackboneはbnだからとか？
- pseudo labelingの結果がvalの精度は高くてtestの精度はいまいち。
  - train + testで学習させると予測精度の低いtestの出力をモデルがそのまま学習してそう
- pose_trainではhflipは精度下がる。
  - detectionでもposeのlossには悪影響及ぼしてたのかも...

## TODO
### 実装
- [ ] wh用いて車両crop→それぞれ回帰
- [x] hflip TTA
- [ ] pseudo labeling
- [x] ensemble

### 実験
- [ ] より大きい画像サイズ(2560 x 2048)
  - whの重み変えた方がいいかも

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
resnet18_fpn_121123     | 0.24946095932243853 | 0.109
resnet50_v1d_fpn_121512 | 0.25247039883369304 | 0.110
resnet34_v1b_fpn_121721 | 0.24874580307912306 |
dla34_ddd_3dop_122006   | 0.2542148697800283  | 0.117
resnet18_fpn_122223     | 0.2572243059006951  | 0.118
dla34_ddd_3dop_123008   | 0.2681900383192367  | 0.118

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

#### hflip tta
- resnet50_v1d_fpn_121512

model | val mAP | PublicLB
------|---------|----------
baseline | 0.25247039883369304 | 0.114
hflip    | 0.2529201030598733  | 0.098

- resnet18_fpn_122223

model | val mAP | PublicLB
------|---------|----------
baseline | 0.2572243059006951 | 0.118
hflip    | 0.25485486558962805  |
hflip (w/o 1- reg)| 0.2549927494751977   |   |
hflip (w/o trig) | 0.2537672048466255  |   |   |
hflip (w/o reg, depth, trig)| 0.22222510066549236  |   |   |
hflip (0.9:0.1) | 0.25769850325224364  |   |   |

#### 2560 x 2048
- resnet18_fpn
- 1fold
- num_filters
  - 256
  - 128
  - 64
- head_conv
  - 64

model | val mAP | PublicLB
------|---------|----------
121123 (1280x1024) | 0.24946095932243853 | 0.087
122207 (wh_weight=0.1) | 0.25570035514025424 | 0.091 |  
122223 (wh_weight=**0.05**) | **0.25947901273784046** | **0.109** |
122319 (wh_weight=0.025) | 0.2542011669483893  | 0.089 |

#### input size
- resnet18_fpn
- 1fold
- num_filters
  - 256
  - 128
  - 64
- head_conv
  - 64

model | val mAP | PublicLB
------|---------|----------
121123 (1280x1024) | 0.24946095932243853 | 0.087
122223 (2560x2048) | 0.25947901273784046 | 0.109 |
122319 (3360x2688, wh=0.038) | 0.2552452199269424  | 0.102 |

#### dcn
- resnet18_fpn
- 1fold
- num_filters
  - 256
  - 128
  - 64
- head_conv
  - 64
- wh_weight: 0.05

model | val mAP | PublicLB
------|---------|----------  
122223 (w/o dcn) | 0.25947901273784046 | 0.109 |
122607 (dcn@p5) | 0.25446012488178965 |   |

#### pseudo labeling
- 0.2746132815008275
- 0.2535992165861809

#### dla34
- 0.2566528431787339

#### pose
- detection model: resnet18_fpn_122223

model | val mAP | PublicLB
------|---------|----------
w/o pose | 0.2572243059006951  | 0.118
resnet18_011006 (wh * 1.0) | 0.257399093175697 |
resnet18_011006 (wh * 1.1) | 0.25740901101286534 | 0.118
resnet18_011006 (wh * 1.2) | 0.25737551183546703 |
se_resnext50_32x4d_011118  | 0.25973812992359474 |

##### fold 1
- detection model: resnet18_fpn_122223

model | val mAP | PublicLB
------|---------|----------
baseline (w/o pose model) | 0.25947901273784046 | 0.109 |
resnet18_011006 (trig, L1) | 0.2596  |
se_resnext50_32x4d_011118 (trig, L1) | 0.25973812992359474 |
resnet18_011212 (eular, MSE) | 0.25979223079859665 |
resnet18_011221 (eular, L1) | 0.2597922785295369  |

map: 0.25973812992359474
