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
[General overview](https://arxiv.org/pdf/2206.08016.pdf)
### Point cloud classification
* Identity
* MLP (see [PointNet](https://arxiv.org/pdf/1612.00593.pdf), [PointNet++](https://arxiv.org/pdf/1706.02413.pdf))
* ISAB (induced set attention block, see [Set Transformer](https://arxiv.org/pdf/1810.00825.pdf))
* Convolution-based: [DG-CNN](https://arxiv.org/pdf/1801.07829.pdf), [RS-CNN](https://arxiv.org/pdf/1904.07601.pdf), [InterpConv](https://arxiv.org/pdf/1908.04512.pdf)
* PCT ([Point Cloud Transformer](https://arxiv.org/pdf/2012.09688.pdf))
* [SimpleView](https://arxiv.org/pdf/2106.05304.pdf)
* Others: [DensePoint](https://arxiv.org/pdf/1909.03669.pdf), [KCNet](https://arxiv.org/pdf/1712.06760.pdf), [KPConv](https://arxiv.org/pdf/1904.08889.pdf), [GSNet](https://arxiv.org/pdf/1912.10644.pdf)
### Graph classification
* GCN (graph convolutional network)
* GAN (graph attention network)
* GIN (graph isomorphism network)
### Image classification
* ViT (vision transformer) (see [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf))
* CNNs (i.e. ResNet) (see [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))
### Object detection
* [DetNet](https://arxiv.org/pdf/1804.06215.pdf)
## On the complexity measures for Neural Nets
* FLOPs and FLOPS (https://github.com/mint-vu/backbone_vs_pooling/issues/2, https://github.com/mint-vu/backbone_vs_pooling/issues/3)
* [Rademacher, Golowich's Method, Barlett's Method](https://github.com/mint-vu/backbone_vs_pooling/issues/1)
* [Size-Independent Sample Complexity of Neural Networks](https://arxiv.org/abs/1712.06541) + [Their Talk](https://www.youtube.com/watch?v=3nhavy2sUEA)
* [All-layer Margin](https://openreview.net/pdf?id=HJe_yR4Fwr)
## Some useful references: 

* [Deep Learning on Sets](https://fabianfuchsml.github.io/learningonsets/#fn:limitations_result)

## TO-DOs for 2/2:
* Eva: Adding other poolings type
* Abi: Editing code + sending papers to Robert
* Robert: Backbone types.
* Ashkan: Janossy Pooling, PoinNet, PointNet++, RS-CNN, InterpConv, DensePoint, SimpleView, Transformer-based Models, SnapNet, SplatNet, SegCloud + Editing Interpolation's code +Dataset overall search

