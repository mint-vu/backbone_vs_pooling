### List of backbones needing further investigation:

* [3DmFV](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8394990&tag=1)
Notes: [github](https://github.com/sitzikbs/3DmFV-Net), is permutation invariant (so can't be used? need confirmation on this), takes point cloud and converts to 3DmFV representation, trained using backpropagation and softmax cross-enropy loss with batch normalization after every layer, dropout after each fully connected layer. 4.6 M params
* [A Graph-CNN for 3D Point Cloud Classification](https://ieeexplore.ieee.org/abstract/document/8462291)
Notes: [github](https://github.com/leondelee/PointGCN) pointGCN, is permutation invariant, 2 fast localized graph convolutional layers + point cloud data specific designed pooling layer using global pooling or multi-res pooling
* [Adaptive Hierarchical Down-Sampling for Point Cloud Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nezhadarya_Adaptive_Hierarchical_Down-Sampling_for_Point_Cloud_Classification_CVPR_2020_paper.pdf)
Notes: can't find code for this
* [PointGuard](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_PointGuard_Provably_Robust_3D_Point_Cloud_Classification_CVPR_2021_paper.pdf)
Notes: can't find code for this
* [SpiderCNN](https://arxiv.org/pdf/1803.11527.pdf)
Notes: [github](https://github.com/xyf513/SpiderCNN) permutation invariant, can directly process 3D point clouds with parameterized convolutional filters, obtains 87.7% accuracy with only 32 points
* [PointCNN](https://arxiv.org/pdf/1801.07791.pdf)
Notes: "Invariance vs. Equivariance. A line of pioneering work aiming at achieving equivariance has been proposed to address the information loss problem of pooling in achieving invariance [16, 40]. The X -transformations in our formulation, ideally, are capable of realizing equivariance, and are demonstrated to be effective in practice. We also found similarity between PointCNN and Spatial Transformer Networks [20], in the sense that both of them provided a mechanism to “transform” input into latent canonical forms for being further processed, with no explicit loss or constraint in enforcing the canonicalization. In practice, it turns out that the networks find their ways to leverage the mechanism for learning better. In PointCNN, the X -transformation is supposed to serve for both weighting and permutation, thus is modelled as a general matrix. This is different than that in [8], where a permutation matrix is the desired output, and is approximated by a doubly stochastic matrix."
* [PCNN](https://dl.acm.org/doi/pdf/10.1145/3197517.3201301)
Notes: [github](https://github.com/pencoa/PCNN) Permutation equivariant, two operators: extension and restriction, mapping point cloud functions to volumetric functions and vise-versa
* [PointConv](https://arxiv.org/abs/1811.07246)
Notes: [github](https://github.com/DylanWusee/pointconv) Permutation invariant, allows deep convolutional networks to be built directly on 3D point clouds
* [Pointwise Rotation-Invariant Network with Adaptive Sampling and 3D Spherical Voxel Convolution](https://arxiv.org/pdf/1811.09361.pdf)
Notes: [github](https://github.com/qq456cvb/PRIN), rotation invariant, network that takes any input point cloud and leverages Density Aware Adaptive Sampling (DAAS) to
construct signals on spherical voxels. Spherical Voxel Convolution (SVC) and Point Re-sampling follow to extract pointwise rotation-invariant features.
* [DensePoint](https://arxiv.org/pdf/1909.03669.pdf)
Notes: [github](https://github.com/Yochengliu/DensePoint), permutation invariant, extends regular grid CNN to irregular point configuration by generalizing a convolution operator and achieves efficient inductive learning
of local patterns
