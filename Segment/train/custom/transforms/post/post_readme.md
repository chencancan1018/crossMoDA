# Activations

- `使用激活函数处理输出特征`

参数：
- sigmoid: 布尔类型，是否使用sigmoid激活函数
- softmax: 布尔类型，是否使用softmax激活函数
- other: 布尔类型，是否使用tanh激活函数

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过激活函数处理后的图像特征
用例：
- ``activation = Activations(sigmoid=True)``
- ``activation(img)``

# AsDiscrete

- `将模型转换为离散值`

参数：
- argmax: 布尔类型，是否使用argmax函数
- to_onehot: 将输入数据转化为N类的one-hot值
- threshold: 按照阈值将输入转化为0或1
- rounding: 将输入数据四舍五入

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 处理为离散值后的图像
用例：
- ``asdiscrete = AsDiscrete(sigmoid=True)``
- ``asdiscrete(img)``

# KeepLargestConnectedComponent

- `只保留图像中最大的连通域`

参数:
- applied_labels: 连通量分析标签
- is_onehot: 是否按one-hot对待输入数据
- independent: 是否将applied_labels看做前景标签的联合
- connectivit:  将一个像素/体素视为邻域的最大正交跳数

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 只保留最大连通域的图像
用例：
- ``KLCC = KeepLargestConnectedComponent(applied_labels=[0,1], is_onehot=True)``
- ``KLCC(img)``

# LabelFilter

- `过滤掉特定标签，仅查看部分标签`

参数:
- applied_labels: 要过滤的标签值

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 过滤掉标签值之后的图像
用例：
- ``labelfilter = LabelFilter(applied_labels=[1, 2])``
- ``labelfilter(img)``

# FillHoles

- `填补图像中的洞，可以用来删除段内的伪影`

参数:
- kernel_type: 需要填充的标签。 默认为None，即为所有标签填充空洞。
- connectivity: 将一个像素/体素视为邻域的最大正交跳数。 

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 填充孔洞之后的图像
用例：
- ``fillholes = FillHoles(applied_labels=[1, 2], connectivity=2)``
- ``fillholes(img)`

# LabelToContour

- `利用拉普拉斯核函数返回仅由0和1组成的二值输入图像的轮廓`
参数:
- kernel_type: 默认为"Laplace"

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- output: 从图像中提取到的轮廓信息
用例：
- ``labeltocontour = LabelToContour()``
- ``labeltocontour(img)`

# MeanEnsemble

- `对输入数据执行平均集成`

参数:
- weights: list, 各个类别的权重

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- output: 转换后的图像
用例：
- ``meanensemble = MeanEnsemble(weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]])``
- ``meanensemble(img)`

# VoteEnsemble

- `对输入数据执行集成投票`

参数:
- num_classes: 类别的数量

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- output: 转换后的图像
用例：
- ``voteensemble = VoteEnsemble(num_classes=2)``
- ``voteensemble(img)`

# ProbNMS

- `通过迭代选择概率最高的坐标，然后移动它及其周围的值，对概率图执行基于概率的非最大抑制(NMS)`

参数:
- spatial_dims: 图像的维度
- sigma: 标准差
- prob_threshold: 概率的阈值
- box_size: bounding box的大小

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- outputs: 极大值抑制后的图像
用例：
- ``probnms = ProbNMS(spatial_dims=2, sigma=0, prob_threshold=0.5, box_size=48)``
- ``probnms(img)`