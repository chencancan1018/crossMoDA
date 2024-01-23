# AffineTransform

- `应用仿射变换` 

参数：
- spatial_size: 输出图像维度
- normalized: 是否使用归一化操作
- mode: {``"bilinear"``, ``"nearest"``}，双线性插值或最近邻插值法
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}，填充的方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- reverse_indexing: 是否反转图像和坐标的空间索引

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
- theta:仿射变换矩阵
输出：
- img: 经过仿射变换后的图像
用例：
- ``affine = AffineTransform(spatial_size=[64,64], normalized=True, mode='bilinear',padding_mode='zeros')``
- ``affine(img, theta)``

# SpatialResample

- `从方向/间距重采样输入图像` 

参数：
- mode: {``"bilinear"``, ``"nearest"``}，插值方式
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}，填充的方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- dtype: 要重采样的数据类型

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
- src_affine: 源仿射矩阵
- dst_affine: 目标仿射矩阵
输出：
- output_data: 重采样后的图像
- dst_affine: 目标仿射矩阵
用例：
- ``spatial = SpatialResample(mode='bilinear',padding_mode='zeros')``
- ``spatial(img, src_affine, dst_affine)``

# Spacing

- ` 重采样输入图像到指定的维度` 

参数：
- pixdim: 输出的维度
- diagonal: 是否对输入重新采样，使其具有对角仿射矩阵、
- mode: {``"bilinear"``, ``"nearest"``}，插值方式
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}，填充的方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- dtype: 要重采样的数据类型
- image_only: 返回值是否仅为图像

输入：
- data_array: 图像, 格式为numpy.ndarray 或者 torch.Tensor
- affine: 仿射矩阵
输出：
- output_data: 重采样后的图像
- affine: 原始的仿射矩阵
- new_affine: 重采样后的仿射矩阵
用例：
- ``spacing = Spacing(pixdim=[64,64], diagonal=True,mode='bilinear',padding_mode='zeros')``
- ``spacing(img, affine)``

# Orientation

- `基于'axcodes'，更改输入图像的方向`

参数：
- axcodes: 空间ND输入方向的N个元素序列
- as_closest_canonical: 是否以最近邻标准轴格式加载图像
- labels: 可选，None或序列(2，)序列(2，)序列是输出轴(开始，结束)的标签
- image_only: 返回值是否仅为图像

输入：
- data_array: 图像, 格式为numpy.ndarray 或者 torch.Tensor
- affine: 仿射矩阵
输出：
- output_data: 重采样后的图像
- affine: 原始的仿射矩阵
- new_affine: 重采样后的仿射矩阵
用例：
- ``orientation = Orientation()``
- ``orientation(img, affine)`

# Flip

- `在给定的空间轴上翻转`

参数：
- spatial_axis: 指定翻转所围绕的轴

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- output_data: 翻转后的图像
用例：
- ``flip = Flip()``
- ``flip(img)`

# Resize

- `将输入图像的大小缩放为给定的大小`

参数：
- spatial_size: 输出图像维度
- size_mode: 指定为"all"或"longest"，如果是"all"，则对所有空间均使用'spatial_size'，如果是"longest"，则重新缩放图像，使最长的边等于指定的'spatial_size'  
- mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}，插值方式
- align_corners: 是否在边缘对齐，如果为True，对齐
输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过缩放后的图像
用例：
- ``resize = Resize(spatial_size=[64,64], size_mode='all', mode='bilinear')``
- ``resize(img)``

# Rotate

- `根据给定的角度旋转输入图像`

参数：
- angle: 指定角度
- keep_size: 输入输出的维度是否保持一致
- mode: {``"bilinear"``, ``"nearest"``}，插值方式
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}，填充方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- dtype: 要处理的数据类型
输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过旋转后的图像
用例：
- ``rotate = Rotate(angle=90, keep_size=False, mode='bilinear')``
- ``rotate(img)``

# Zoom

- `缩放图像`

参数：
- zoom: 缩放率
- mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}，插值方式
- padding_mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}，填充方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- keep_size: 输入输出的维度是否保持一致
- kwargs: 其他需要输入到方法中的参数

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过缩放后的图像
用例：
- ``zoom = Zoom(zoom=0.5, mode='bilinear', keep_size=False)``
- ``zoom(img)``

# Rotate90

- `旋转90度`

参数：
- k: 旋转90度的次数
- spatial_axes: 2个int数字, 定义旋转的轴

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过旋转后的图像
用例：
- ``rotate = Rotate90(k=2)``
- ``rotate(img)``

# RandRotate90


- `随机旋转90度`

参数：
- prob: 旋转的概率
- max_k: 旋转90度的最大次数
- spatial_axes: 2个int数字, 定义旋转的轴

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过旋转后的图像
用例：
- ``rotate = RandRotate90(prob=0.2, max_k=2)``
- ``rotate(img)``

# RandRotate

- `随机旋转`

参数：
- range_x: 绕x轴旋转的角度范围
- range_y: 绕y轴旋转的角度范围
- range_z: 绕z轴旋转的角度范围
- prob: 旋转的概率
- keep_size: 输入输出的维度是否保持一致
- mode: {``"bilinear"``, ``"nearest"``}, 插值方式
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}, 填充方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- dtype: 要处理的数据类型

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过旋转后的图像
用例：
- ``rotate = RandRotate(range_x=[30,60], range_y=[45,90], range_z=[30,60], prob=0.2, keep_size=False)``
- ``rotate(img)``

# RandFlip

- `随机翻转`

参数：
- prob: 翻转的概率
- spatial_axis: 翻转所绕的轴

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过翻转后的图像
用例：
- ``randflip = RandFlip(prob=0.2, spatial_axis=(0,1))``
- ``randflip(img)``

# RandAxisFlip

- `随机选择一个轴，绕其翻转`

参数：
- prob: 翻转的概率

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过翻转后的图像
用例：
- ``randflip = RandAxisFlip(prob=0.2)``
- ``randflip(img)``

# RandZoom

- `随机缩放`

参数：
- prob: 缩放的概率
- min_zoom: 最小的缩放率
- max_zoom: 最大的缩放率
- mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}，插值方式
- padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}，填充方式
- align_corners: 是否在边缘对齐，如果为True，对齐
- keep_size: 输入输出的维度是否保持一致
- kwargs: 其他需要输入到方法中的参数

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过随机缩放后的图像
用例：
- ``randzoom = RandZoom(prob=0.2, min_zoom=0.2, max_zoom=0.9, keep_size=False)``
- ``randzoom(img)``

# Resample

- `图像重采样`

参数：
- mode: {``"bilinear"``, ``"nearest"``}, 插值方式
- padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}, 填充方式
- norm_coords: 是否将坐标归一化
- device: 处理tensor的设备
- dtype: 要处理的数据类型

输入：
- img: 图像, 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 经过重采样后的图像
用例：
- ``resample = Resample(mode='bilinear', norm_coords=True)``
- ``resample(img)``
