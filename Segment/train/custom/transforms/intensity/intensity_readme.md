# RandGaussianNoise

- `数据增强：图像随机添加随机高斯噪音`

参数: 
- prob:  prob=0.5, 表示依概率0.5添加高斯噪音
- mean: 默认为0.0
- std：默认为0.1， 表示噪音方差范围为(0, 0.1),
输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `gaussian = RandGaussianNoise(prob=0.5)`
- `img = gaussian(img)`

# RandRicianNoise

- `数据增强：图像随机添加随机Rician噪音，常用于MRI图像数据增强`

参数: 
- prob:  prob=0.5, 表示依概率0.5添加随机噪音
- mean: 默认为0.0
- std：默认为1.0， 表示噪音方差范围为(0, 1.0),
输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `rician = RandRicianNoise(prob=0.5)`
- `img = rician(img)`

# ShiftIntensity

- `数据增强：增加图像像素值，即图像变亮`

参数: 
- offset:  图像亮度值

用例：
- `shift = ShiftIntensity(offset=0.1)`
- `img = shift(img)`

# RandShiftIntensity

- `图像亮度：随机增加随机图像像素值`

参数: 
- prob:  prob=0.5, 表示依概率0.5随机进行亮度增强
- offsets:  图像亮度偏移量范围

用例：
- `shift = RandShiftIntensity(offsets=(-0.2, 0.2), prob=0.5)`
- `img = shift(img)`

# StdShiftIntensity

- `图像亮度：根据图像自身像素值方差和相关量度进行图像数据增强，即v = v + factor * std(v)`

参数: 
- factors:  给定量度

用例：
- `shift = StdShiftIntensity(factors=0.1)`
- `img = shift(img)`

# RandStdShiftIntensity

- `图像亮度：根据图像自身像素值方差和相关量度进行随机图像数据增强，即v = v + factor * std(v)`

参数: 
- prob:  prob=0.5, 表示依概率0.5随机进行亮度增强
- factors:  (-0.1, 0.1), 表示给定量度范围

用例：
- `shift = RandStdShiftIntensity(factors=(-0.1, 0.1), prob=0.5)`
- `img = shift(img)`

# RandScaleIntensity

- `图像亮度：随机线性增强随机图像像素值，即v = v * (1 + factor)`

参数: 
- prob:  prob=0.5, 表示依概率0.5随机进行亮度增强
- factors:  (-0.1, 0.1), 表示给定量度范围

用例：
- `scale = RandScaleIntensity(factors=(-0.1, 0.1), prob=0.5)`
- `img = scale(img)`

# RandBiasField

- `随机偏差域增强，常用于MR图像`

参数: 
- prob:  prob=0.5, 表示依概率0.5随机进行亮度增强
- degree \ coeff_range:默认参数值即可

用例：
- `bias = RandBiasField(prob=0.5)`
- `img = bias(img)`

# NormalizeIntensity
- `根据给定均值和方差进行图像归一化，`
参数: 
- subtrahend:  给定减值，默认None， 其值为图像本身均值
- divisor: 给定除值，默认None， 其值为图像本身方差

用例：
- `normalize = NormalizeIntensity()`
- `img = normalize(img)`

# ThresholdIntensity
- `图像阈值剪切，类似torch.clip or np.clip`

# ScaleIntensityRange
- `将图像像素值scale至固定范围，类似torch.clip or np.clip`
用例：
- `normalise = ScaleIntensityRange(a_min=(window_level-windom_width/2), a_max=(window_level-windom_width/2), b_min=0.0, b_max=1.0,clip=True)`
- `normalise = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0,clip=True)`
- `img = normalise(img)`


# RandAdjustContrast

- `数据增强：随机调整图像对比度，即x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

参数: 
- prob:  prob=0.5, 表示依概率0.5随机进行对比度增强
- gamma:  (0.5, 1.5), 表示给定像素对比度指数范围

用例：
- `adjust = RandAdjustContrast(prob=0.5, gamma=(0.5, 1.5))`
- `img = scale(img)`

# ScaleIntensityRangePercentiles
- `根据图像像素值分布，将图像像素值scale至固定区间`

# MaskIntensity

- `对图像进行掩码处理`

参数: 
- mask_data: 需与输入原始图像保持shape一致
- select_fn: 对mask_data进行处理的功能函数，例 select_fn = threshold_at_one,  
            `def threshold_at_one(x):`
            `    # threshold at 1`
            `    return x > 1`
用例：
- `mask = MaskIntensity(mask_data=mask_data, select_fn=threshold_at_one)`
- `img = mask(img)`


# SavitzkyGolaySmooth

- `对图像进行S-G滤波操作，主要通过窗宽内多项式拟合方法过滤到高频信息，暂时不补充其信息`

# RandGaussianSmooth

- `对图像进行随机高斯平滑，去除高斯噪点`

参数: 
- sigma_x: 第一空间维度的随机高斯核方差
- sigma_y: 第二空间维度的随机高斯核方差
- sigma_z: 第三空间维度的随机高斯核方差
- prob:  prob=0.5, 表示依概率0.5随机进行高斯光滑

用例：
- `smooth = RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y= (0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.1)`
- `img = smooth(img)`

# RandGaussianSharpen

- `数据增强：通过高斯模糊滤波函数对图像进行随机高斯锐化`

参数: 
- sigma1_x: (0.5, 1.0), 第一空间维度的第一随机高斯核方差范围
- sigma1_y: (0.5, 1.0), 第二空间维度的第一随机高斯核方差范围
- sigma1_z: (0.5, 1.0),第三空间维度的第一随机高斯核方差范围
- sigma2_x: 0.5, 第一空间维度的随机高斯核方差
- sigma2_y: 0.5, 第二空间维度的随机高斯核方差
- sigma2_z: 0.5, 第三空间维度的随机高斯核方差
- alpha：(10.0, 30.0), 高斯锐化中的两个高斯核相关权重系数
- prob:  prob=0.5, 表示依概率0.5随机进行高斯锐化

用例：
- `sharpen = RandGaussianSharpen(prob=0.5)`
- `img = sharpen(img)`

# RandHistogramShift

- `对图像随机增加非线性成分`

参数: 
- num_control_points: 10, 非线性点控制点树
- prob:  prob=0.5
用例：
- `hist = RandHistogramShift(prob=0.5)`
- `img = hist(img)`

# RandGibbsNoise

- `对图像随机增加Gibbs噪音`

用例：
- `gibbs = RandGibbsNoise(prob=0.5)`
- `img = gibbs(img)`

# RandCoarseDropout

- `对图像随机丢弃粗糙区域，并进行填充`

参数: 
- holes: 随机丢弃区域数量
- spatial_size：随机丢弃区域尺寸
- fill_value: 丢弃区域填充值，默认为None，随机选取图像像素值
- prob:  prob=0.5
用例：
- `coarse = RandCoarseDropout(holes=4, spatial_size=(16,16,16), prob=0.5)`
- `img = hist(img)`

# RandCoarseShuffle

- `对图像随机丢弃区域进行填充，并随机打乱`

参数: 
- holes: 随机丢弃区域数量
- spatial_size：随机丢弃区域尺寸
- fill_value: 丢弃区域填充值，默认为None，随机选取图像像素值
- prob:  prob=0.5
用例：
- `coarse = RandCoarseDropout(holes=4, spatial_size=(16,16,16), prob=0.5)`
- `img = hist(img)`


# HistogramNormalize

- `对图像进行直方归一化，默认为0~255`

用例：
- `histnormal = HistogramNormalize()`
- `img = histnormal(img)`