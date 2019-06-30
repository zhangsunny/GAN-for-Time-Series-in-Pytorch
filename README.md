# GAN For Time Series In Pytorch

尝试使用GAN实现生成时间序列，经过测试发现，基于RNN的GAN很容易过拟合，难以训练

而基于conv的GAN则可以较好地生成时间序列，且很多稳定GAN训练的技术，例如权重裁剪、

梯度惩罚，谱规范化技术都无法应用于RNN，所以还是选择不带有RNN的网络

至于，Transformer能否起效，尚未测试，直觉上应该是可以的

GAN，RNNGAN，D2GAN2文件夹下都是基于RNN的尝试，不适用；

WGAN，DCGAN,D2GAN,BiGAN下是基于卷积的尝试，有点作用