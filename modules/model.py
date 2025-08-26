"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GeometryEstimator(nn.Module):
	def __init__(self, input_dim=64, patch_size=8):
		super(GeometryEstimator, self).__init__()
		self.fine_matcher = nn.Sequential(
			nn.Linear(128, 512),  # 输入为128维（拼接后的两个64维向量）
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace=True),

			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace=True),

			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace=True),

			nn.Linear(512, 512),
			nn.BatchNorm1d(512, affine=False),
			nn.ReLU(inplace=True),

			# 最后一层输出8个值，代表四个角点的偏移量 (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
			nn.Linear(512, 8),
			# nn.Tanh()
		)

	def forward(self, patch1, patch2):
		x = torch.cat((patch1, patch2), dim=-1)
		h_flat = self.fine_matcher(x)  # 输出 [batch_size, 8]
		h = torch.cat([h_flat, torch.ones(h_flat.shape[0], 1, device=h_flat.device)], dim=1)  # 补 1
		h = h.view(-1, 3, 3)  # 重塑为 [batch_size, 3, 3]
		return h

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)


class MultiScaleConv(nn.Module):
    """
    多尺度卷积模块
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.softmax = nn.Softmax(dim=1)  # 用于加权融合

    def forward(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        # 将不同卷积结果按通道融合并加权
        out = torch.stack([c1, c3, c5], dim=1)  # 在新维度堆叠
        weights = self.softmax(out)  # 计算权重
        return torch.sum(weights * out, dim=1)  # 权重加权求和


class WaveletTransform(nn.Module):
    """
    小波变换模块
    """
    def __init__(self, scale=2):
        super(WaveletTransform, self).__init__()
        self.scale = scale

    def forward(self, x):
        # 高频特征提取模拟
        high_freq = F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)
        # 恢复到原尺寸
        high_freq = F.interpolate(high_freq, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return high_freq


class WTGCModule(nn.Module):
    """
    核心模块：结合上下文建模和小波变换
    """
    def __init__(self, in_channels, out_channels):
        super(WTGCModule, self).__init__()
        self.context_modeling = MultiScaleConv(in_channels, out_channels)
        self.wavelet_transform1 = WaveletTransform(scale=2)
        self.wavelet_transform2 = WaveletTransform(scale=4)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 上下文建模
        context_features = self.context_modeling(x)
        # 小波高频特征提取
        high_freq1 = self.wavelet_transform1(x)
        high_freq2 = self.wavelet_transform2(x)
        high_freq = high_freq1 + high_freq2  # 融合高频特征
        # 残差连接
        out = self.conv1x1(context_features) + high_freq + self.residual(x)
        return out





class XFeatModel(nn.Module):
	"""
	   Implementation of architecture described in 
	   "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	"""

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(1)

		self.MSPGC_module = WTGCModule(in_channels=64, out_channels=64)
		########### ⬇️ CNN Backbone & Heads ⬇️ ###########

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

		self.block1 = nn.Sequential(
										BasicLayer( 1,  4, stride=1),
										BasicLayer( 4,  8, stride=2),
										BasicLayer( 8,  8, stride=1),
										BasicLayer( 8, 24, stride=2),
									)

		self.block2 = nn.Sequential(
										BasicLayer(24, 24, stride=1),
										BasicLayer(24, 24, stride=1),
									 )

		self.block3 = nn.Sequential(
										BasicLayer(24, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, 1, padding=0),
									 )
		self.block4 = nn.Sequential(
										BasicLayer(64, 64, stride=2),
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 )

		self.block5 = nn.Sequential(
										BasicLayer( 64, 128, stride=2),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128, 128, stride=1),
										BasicLayer(128,  64, 1, padding=0),
									 )

		self.block_fusion =  nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0)
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)


		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)

		self.geometry_estimator = GeometryEstimator(input_dim=64, patch_size=8)


	def _unfold2d(self, x, ws = 2):
		"""
			Unfolds tensor in 2D with desired ws (window size) and concat the channels
		"""
		B, C, H, W = x.shape
		x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
			.reshape(B, C, H//ws, W//ws, ws**2)
		return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


	def forward(self, x):
		"""
			input:
				x -> torch.Tensor(B, C, H, W) grayscale or rgb images
			return:
				feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
				keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
				heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

		"""
		#dont backprop through normalization
		with torch.no_grad():
			x = x.mean(dim=1, keepdim = True)
			x = self.norm(x)

		#main backbone
		x1 = self.block1(x)
		x2 = self.block2(x1 + self.skip1(x))
		x3 = self.block3(x2)
		x4 = self.block4(x3)
		x5 = self.block5(x4)

		#pyramid fusion
		x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
		x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')

		x_combined = x3 + x4 + x5

		# 后续融合
		feats = self.block_fusion(x_combined)
		enhanced_feats = self.MSPGC_module(feats)
		#heads
		heatmap = self.heatmap_head(enhanced_feats) # Reliability map
		keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

		return enhanced_feats,keypoints,heatmap
