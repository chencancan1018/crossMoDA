# RandSmoothFieldAdjustContrast: 利用随机平滑场，随机调整输入图像的对比度
# RandSmoothFieldAdjustIntensity: 利用随机平滑场，随机调整输入图像的强度。
# RandSmoothDeform: 使用随机平滑场和Pytorch的grid_sample变形图像。

# RandSmoothFieldAdjustContrast
参数:
- spatial_size: 输入数组的尺寸
- rand_size: 随机场开始的大小
- pad: 随机场边缘要填充1的像素/体素数
- mode: 调整场大小的插值方式
- align_corners: 是否在边缘对齐，如果为True，则对齐
- prob: 应用随机场变换的概率
- gamma: (最小，最大)范围的对比度倍增器
- device: pytorch设备

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- out: 经过平滑场调整对比度后的图像
用例：
- ``RSFAC = RandSmoothFieldAdjustContrast(spatial_size=[64, 64], rand_size=[16], pad=0, prob=0.5, gamma=(0.2, 1.2))``
- ``RSFAC(img)``

# RandSmoothFieldAdjustIntensity
参数：
- spatial_size: 输入数组的尺寸
- rand_size: 随机场开始的大小
- pad: 随机场边缘要填充1的像素/体素数
- mode: 调整场大小的插值方式
- align_corners: 是否在边缘对齐，如果为True，则对齐
- prob: 应用随机场变换的概率
- gamma: (最小，最大)范围的强度倍增器
- device: pytorch设备

输入:
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- out: 经过平滑场调整强度后的图像
用例：
- ``RSFAI = RandSmoothFieldAdjustIntensity(spatial_size=[64, 64], rand_size=[16], pad=0, prob=0.5, gamma=(0.2, 1.2))``
- ``RSFAI(img)``

# RandSmoothDeform
参数:
- spatial_size: 输入数组的尺寸
- rand_size: 随机场开始的大小
- pad: 随机场边缘要填充0的像素/体素数
- field_mode: 调整场大小的插值方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- prob: 应用随机场变换的概率
- def_range: 图像大小分数的变形范围值，单个最小/最大值或最小/最大对
- grid_dtype: 从场计算的变形网格的类型
- grid_mode: 网格的插值方式
- grid_padding_mode: 使用变形网格对输入进行采样的填充模式
- grid_align_corners: 是否在网格边缘对齐，如果为True，对齐
- device: pytorch设备

输入:
- img: 3D图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过平滑场变形后的图像
用例：
- ``RSD = RandSmoothDeform(spatial_size=[64, 64], rand_size=[16], pad=0, field_mode='linear', align_corners=True,prob=0.5)``
- ``RSD(img)``

