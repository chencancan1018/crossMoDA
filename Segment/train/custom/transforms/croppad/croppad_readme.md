# SpatialPad

- ``以给定size为目标的图像pading方法``

参数: 
- spatial_size: padding之后的图像大小
- method: ``"symmetric"``, ``"end"``
- mode: 若图像为numpy array， 可选``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``；
        若图像为torch tensor， 可选``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)， 尺寸为spatial_size的padding后图像
用例：
- ``spatialpad = SpatialPad(spatial_size=(64,64,64))``
- ``img = spatialpad(img)``


# BorderPad

- ``图像边缘插入给定size数据的pading方法``

参数: 
- spatial_border: 图像所有维度的边缘padding大小
- mode: 若图像为numpy array， 可选``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``；
        若图像为torch tensor， 可选``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``
输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)， 边缘插入指定尺寸数据之后的图像
用例：
- ``borderpad = BorderPad(spatial_border=(4,4,4))``
- ``img = borderpad(img)``


# DivisiblePad

- ``图像padding方法，使得图像可被整数倍均匀分割``

参数: 
- k: 图像均匀分割的倍数
- method: ``"symmetric"``, ``"end"``
- mode: 若图像为numpy array， 可选``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``；
        若图像为torch tensor， 可选``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)， 尺寸为spatial_size的padding后图像
用例：
- `divisiblepad = DivisiblePad(k=(4,4,4))`
- `img = divisiblepad(img)`


# SpatialCrop

- `图像空间剪切`

参数: 
- roi_center: 剪切所得ROI的中心坐标，不包括channel维度
- roi_size: 剪切所得ROI的尺寸，不包括channel维度
- roi_start: 剪切所得ROI的起始坐标，不包括channel维度
- roi_end: 剪切所得ROI的终止坐标，不包括channel维度
- roi_slices: 图像剪切指定的slice，例如，(slice(1, 78, None),slice(2, 76, None),slice(1, 77, None)),分别为图像第一、二、三维度的起始位置，通常默认为None

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `spatialcrop = SpatialCrop(roi_center=(o1,o2,o3), roi_size=(64,64,64)), 或 spatialcrop = SpatialCrop(roi_start=(s1,s2,s3), roi_end=(e1,e2,e3))`
- ` img = spatialcrop(img)`

# CenterSpatialCrop

- `以图像中心点为crop 中心，给定roi size的图像剪切方法`

参数: 
- roi_size: 剪切所得ROI的尺寸，不包括channel维度

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `centerspatialcrop = CenterSpatialCrop(roi_size=(64,64,64))`
- `img = centerspatialcrop(img)`

# CenterScaleCrop

- `以图像中心点为crop 中心，给定roi scale的图像剪切方法`

参数: 
- roi_scale: 所得roi区域与原图之间的固定比例，不包括channel维度

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
-``centerscalecrop = CenterScaleCrop(roi_scale=(0.5,0.5,0.5))`
- `img = centerscalecrop(img)`

# RandSpatialCrop

- ``给定roi size的图像随机剪切方法，用于模型训练过程中图像随机采样``

参数: 
- roi_size: 随机采样所得ROI的尺寸，不包括channel维度

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `randspatialcrop = RandSpatialCrop(roi_size=(64,64,64))`
- `img = randspatialcrop(img)`

# RandScaleCrop

- `给定roi scale的图像随机剪切方法，用于模型训练过程中图像随机采样`

参数: 
- roi_scale: 所得roi区域与原图之间的固定比例，不包括channel维度

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `randscalecrop = RandScaleCrop(roi_scale=(0.5,0.5,0.5))`
- `img = randscalecrop(img)`

# RandSpatialCropSamples

- `给定roi size的多样本图像随机剪切方法，用于模型训练过程中图像随机采样`

参数: 
- roi_size: 随机采样所得ROI的尺寸，不包括channel维度
- num_samples: 随机采样数量

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `randspatialcropsamples = RandSpatialCropSamples(roi_size=(64,64,64), num_samples=16)`
- `img_list = randspatialcropsamples(img)`

# CropForeground

- `满足给定条件的图像前景区域剪切`

参数: 
- select_fn: 选取图像前景区域的功能函数，例 select_fn = threshold_at_one， 
            `def threshold_at_one(x):`
            `    # threshold at 1`
            `    return x > 1`
- margin: 距离图像边缘的裕度（像素数量）
- return_coords: 是否返回boundingbox的起始终止坐标
- k_divisible：图像剪切所得ROI是否能被k完整划分

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `crop_fg = CropForeground(select_fn=threshold_at_one, margin=0, return_coords=False,k_divisible=1) `
- `fg = crop_fg(img)`
- `crop_fg = CropForeground(select_fn=threshold_at_one, margin=0, return_coords=True,k_divisible=1) `
- `fg, bbox_start, bbox_end = crop_fg(img)`

# RandWeightedCrop

- `依据给定的加权map, 其shape与img相同，对图像进行多样本随机采样`

参数: 
- spatial_size: 采样所得图像size，不包括channel size
- num_samples: 采样数量
- weight_map: 加权图，与需采样原图的shape一致，默认None

输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
- weight_map: 加权图，与需采样原图的shape一致，默认None
输出：img list, 采样所得多样本图像列表
用例：
- `randweightedcrop = RandWeightedCrop(spatial_size=(64,64,64), num_samples=16,weight_map=None) `
- `patches_list = randweightedcrop(img=img, weight_map=weight_map)`


# RandCropByPosNegLabel

- `依据给定的正负样本比例和总样本数量，对图像进行多样本随机采样`

参数: 
- spatial_size: 采样所得图像size，不包括channel size
- label: img图像的标注数据，即mask
- num_samples: 采样所得样本数量
- pos \ neg: 正负样本所占比例
- image: 原始输入有效区域图像
- image_threshold: 图像阈值，用于去除图像空白区域，如病理图像中的非组织区域，默认为image_threshold=0.0


输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
- label: img图像的标注数据，即mask
- image: 原始输入有效区域图像,若无有效指定采样区域，可设image=img
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `randcrop_pn = RandCropByPosNegLabel(spatial_size=(64,64,64), pos=0.5, neg=0.5, num_samples=16, image_threshold=0.0) `
- `patches_list = randcrop_pn(img=img, label=mask, image=img)`

# RandCropByLabelClasses

- `依据给定的正负样本比例和总样本数量，对图像进行多样本随机采样`

参数: 
- spatial_size: 采样所得图像size，不包括channel size
- label: img图像的标注数据，即mask
- num_samples: 采样所得样本数量
- num_classes: 类别数量
- ratios: 各类别采样所得样本比例，默认None，即各类别比例为1:1:1:...
- image: 原始输入有效区域图像
- image_threshold: 图像阈值，用于去除图像空白区域，如病理图像中的非组织区域，默认为image_threshold=0.0


输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
- label: img图像的标注数据，即mask;若num_classes=None,必须为one-hot label
- image: 原始输入有效区域图像,若无有效指定采样区域，可设image=img
输出：
- img: 图像(channel first, channel维度不执行操作)
用例：
- `randcrop_label = RandCropByLabelClasses(spatial_size=(64,64,64), ratios=None, num_classes=num_classes,num_samples=16, img_threshold=0.0) `
- `patches_list = randcrop_label(img=img, label=mask, image=img)`

# BoundingRect

- `计算给定条件的图像前景区域boundingbox`

参数: 
- select_fn: 选取图像前景区域的功能函数，例 select_fn = threshold_at_one， 
            `def threshold_at_one(x):`
            `    # threshold at 1`
            `    return x > 1`
输入: 
- img: 图像(channel first, channel维度不执行操作), 格式为numpy.ndarray 或者 torch.Tensor
输出：
- bbox: 图像前景区域
用例：
- `boundingrect = BoundingRect(select_fn=threshold_at_one) `
- `bbox = boundingrect(img)`
