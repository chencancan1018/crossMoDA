# Getting Started
Follow the instructions to run this example

## 下载代码

    git clone ssh://git@code.infervision.com:2022/ccancan/iarsegmentation.git

## python版本

- python3.7

## 配置环境

- starship运行环境

    python3.7

- 安装numpy

    $ python3.7 -m pip install numpy==1.20.1 -i https://repos.infervision.com/repository/pypi/simple

- 安装starship和starship.umtf

    $ python3.7 -m pip install starship==0.4.5 starship.umtf==0.4.3 -i https://repos.infervision.com/repository/pypi/simple

- 查看当前starship版本，确保当前版本为0.4.5及以上版本：运行命令
    $ python3.7 -m pip list | grep starship

- 安装 pytorch(适配本机cuda版本，支持torch >= 1.7.0)

    若 cuda 版本为10.2:

    $ python3.7 -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
 
    or
 
    $ python3.7 -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

    其他cuda版本((若cuda版本=11.X, 可兼容11.0版本pytorch)),请参考 https://pytorch.org/get-started/previous-versions/  


- 安装配置其他包, 如skimage, scikit-image等

    $ python3.7 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  XXX

- 另：starship 设定 torch==1.4.0, 而此项目所需torch >= 1.7.0, 配置环境过程中会报错误，关于starship和torch的版本冲突（无需处理），先安装starship，再安装torch和torchvision即可

## 数据下载和模型训练

- 详细实现步骤见./tools/README.md

## 文件结构说明

```
├── train
│    └── config
│        ├── train_config.py
│        ├── test_config.py
│    └── apis
│        ├── runner
│            ├── __init__.py
│            ├── epoch_based_runner.py
│        ├── __init__.py
│        ├── trainner.py
│        ├── tester.py
│    └── tools
│        ├── __init__.py
│        ├── train.py
│        ├── test.py
│        ├── dist_train.sh
│        ├── dist_test.sh
│        ├── generate_dataset.py
│        ├── save_torchscript.py
│        ├── README.md
│    └── custom
│        ├── datastet
│            └── __init__.py
│            ├── dataset.py
│        ├── model
│            ├── backbones
|                 |── ResUnet.py
|                 |── ResNet3D.py
|                 |── OctaveUNet.py
|                 |── SKUNet.py
|                 |── SPUNet.py
|                 |── SPConvUnet.py
|                 |── vit.py
|                 |── unetr.py
│            ├── heads
|                 |── segsofthead.py
│            ├── networks
|                 |── segnetwork.py
│            └── __init__.py
│        ├── transforms
│            └── __init__.py
│            ├── croppad
|                 |── array.py
|                 |── dictionary.py
|                 |── croppad_readme.md
│            ├── ...
│        ├── utils
│            └── __init__.py
│            ├── resample_image.py
│            ├── utils.py
│    └── __init__.py
│    └── requirements.txt
│    └── README.md
├── README.md
```

- ./tools/train.py: 训练代码入口，需要注意的是，在train.py里import custom 和import apis，训练相关需要注册的模块可以直接放入到custom或apis文件夹下面，会自动进行注册; 一般来说，训练相关的代码务必放入到custom或apis文件夹下面!!!<br>

- ./custom/dataset/dataset.py: dataset类，需要@DATASETS.register_module进行注册方可被识别<br>

- ./custom/dataset/generate_dataset.py: 从原始数据生成输入到模型的数据

- ./custom/model/augmentation3d.py: 数据扩增方法, 需要@PIPELINES.register_module进行注册方可被识别<br>

- ./custom/model/backbones/ResUnet.py: 模型backbone，需要@BACKBONES.register_module进行注册方可被识别<br>

- ./custom/model/heads/segsofthead.py: 模型head文件，需要@HEADS.register_module进行注册方可被识别<br>

- ./custom/model/networks/segnetwork.py: 整个网络部分，训练阶段构建模型，forward方法输出loss的dict, 通过@NETWORKS.register_module进行注册

- ./config/train_config.py: 配置文件

- ./tools/dist_train.sh: 分布式训练的运行脚本
