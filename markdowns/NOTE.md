## Note
- 6DoF (yaw, pitch, roll, x, y, z)を推定。
- x, yはマスク画像から得られる画像座標とz(depth)から計算できるはずなので、zだけ推定すればいいはず。
- zの推定はcenternetの3d pose estimationで使われているEigenの手法を参考にすればいいかも。z = 1 / sigmoid(output) − 1で推定する。ロスはオリジナルのgtとのL1 Loss。
- 640 x 512
- 960 x 768
- 姿勢の学習方法
  - 直接回帰
  - いくつかのビンにわけて各ビン内で回帰
  - クォータニオンとの回帰
- 解像度が大きいモデルはdepthの予測精度が悪化している。(恐らく公式のmetricの並進誤差がより厳しいっぽい)
  - ミス。
- NonLocalBlockを向き推定のheadに追加。
  - 計算が重すぎて無理。

## Experiments
### NMS
- resnet18_fpn_120123
- fold1

NMS  | mAP
-----|---
None | 0.19142559036561727  
0.1  | 0.21371086361037867
0.5  | 0.2136521156341741
1.0  | 0.21339207211238534

### 1280 x 1024
- resnet18_fpn
- fold1
- val score_th=0.1
- test score_th=0.6

model | val mAP | PublicLB
------|---------|----------
120708 (w/o gn+ws, w/o wh, det=DepthL1) | 0.21807085313661206
120713 (w/o gn+ws, w/o wh, det=L1) | 0.23062173249021703 | 0.070
120720 (w/ gn+ws, w/o wh, det=L1)  | 0.23069422381934448 | 0.073
120814 (w/ gn+ws, w/ wh, det=L1)  | 0.23285714122775697 | 0.078
