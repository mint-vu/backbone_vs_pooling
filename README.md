## Pooling Methods
### Non-learning based
* sum, mean, generalized mean
* max, n-max pooling, [recycled max](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Why_Discard_if_You_Can_Recycle_A_Recycling_Max_Pooling_CVPR_2022_paper.html)
* covariance pooling
* [FSpool](https://github.com/Cyanogenoid/fspool)
* Janossy pooling
* OT-based: SWE, WE, [Regularized optimal transport pooling (ROTP)](https://arxiv.org/pdf/2212.06339.pdf)
### Learning based (Attention based)
* PMA (Set Transformer)
* [Trainable OT embedding](https://openreview.net/forum?id=ZK6vTvb84s)
## Backbones
### Point cloud classification
* Identity
* MLP (see [PointNet](https://arxiv.org/pdf/1612.00593.pdf))
* ISAB (induced set attention block, see [Set Transformer](https://arxiv.org/pdf/1810.00825.pdf))
### Graph classification
* GCN (graph convolutional network)
* GAN (graph attention network)
* GIN (graph isomorphism network)
### Image classification
* ViT (vision transformer)
* CNNs (i.e. ResNet)
## On the complexity measures for Neural Nets
* FLOPs and FLOPS (https://github.com/mint-vu/backbone_vs_pooling/issues/2, https://github.com/mint-vu/backbone_vs_pooling/issues/3)
* [Rademacher, Golowich's Method, Barlett's Method](https://github.com/mint-vu/backbone_vs_pooling/issues/1)
* [Size-Independent Sample Complexity of Neural Networks](https://arxiv.org/abs/1712.06541) + [Their Talk](https://www.youtube.com/watch?v=3nhavy2sUEA)
* [All-layer Margin](https://openreview.net/pdf?id=HJe_yR4Fwr)
## Some useful references: 

* [Deep Learning on Sets](https://fabianfuchsml.github.io/learningonsets/#fn:limitations_result)

