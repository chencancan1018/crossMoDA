# 下载数据集
    $ rsync -av ccancan@172.30.0.56:/volume1/先进研究院/public_data/MICCAI2021_FLARE21/TrainImage ./checkpoints/
    $ rsync -av ccancan@172.30.0.56:/volume1/先进研究院/public_data/MICCAI2021_FLARE21/TrainMask ./checkpoints/
- Remark: 可使用个人nas账户下载数据

## ResUnet

# step 1 准备数据集
    $ python3.7 generate_dataset_resunet.py

# step 2 模型训练

# 分布式训练
    对于单机分布式训练, nproc_per_node相当于使用的GPU数量, CUDA_VISIBLE_DEVICES决定使用哪几个GPU,一般来说,CUDA_VISIBLE_DEVICES数量和nproc_per_node数量保持一致,这样刚好每个训练进程在独立的某块GPU卡上

# 使用 train.py
    $ CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 -m torch.distributed.launch  --nproc_per_node=4 train.py --config ../config/train_config_resunet.py --launcher pytorch

# 使用 dist_train.sh
    $ sh dist_train.sh -d 0,1,2,3 -g 4 -c ../config/train_config_resunet.py 

# coarse-to-fine
    设计文件dataset 和 example中main.py即可
- coarse seg: 示例 train_config_resunet.py 
- fine seg: 示例 train_config_sunetr.py 

# step 3 模型测试

# 分布式测试

# 使用 test.py
    $ CUDA_VISIBLE_DEVICES=0 python3.7 -m torch.distributed.launch  --nproc_per_node=1 test.py --config ../config/test_config.py --launcher pytorch

# 使用 dist_test.sh
    $ sh dist_test.sh -d 0 -g 1 -c ../config/test_config.py 


# step 4 模型转换成静态图pt文件(与starship等组件分离,便于后续打包)
    $ python3.7 save_torchscript.py


# 当前模型和参数在FLARE2021平台测试结果

- AttentionUnet， avg dice， 0.8171

- DeepLab， avg dice， 0.7658

- ResUnet， avg dice， 0.8326

- SwinUNETR， avg dice， 0.7744

- UNETR， avg dice， 0.7021

请使用者根据项目需求自行选择模型和调整模型参数量
