
trainner = dict(type='Trainner_', runner_config=dict(type='EpochBasedRunner'))

in_ch = 1
patch_size = [40, 256, 256] # for coarse- and fine-seg


aux_loss = True
model = dict(
    type='SegNetwork',
    backbone=dict(type='ResUnet', in_ch=in_ch, channels=32, blocks=3, is_aux=aux_loss),
    apply_sync_batchnorm=False,
    head=dict(
        type='SegSoftHead',
        in_channels=32,
        classes=4,
        is_aux=aux_loss,
        patch_size=patch_size,
    ),
)
train_cfg = None
test_cfg = None

import numpy as np

data = dict(
    imgs_per_gpu=20,
    workers_per_gpu=15,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader',
        source_batch_size=2,
        source_thread_count=1,
        source_prefetch_count=1,
    ),
    train=dict(
        type='CropSegSampleDataset', # for coarse segmentation
        dst_list_file='./checkpoints/predata_fakeT2_round3/train_0.lst',
        data_root='./checkpoints/predata_fakeT2_round3/',
        patch_size=patch_size,
        sample_frequent=6,
        pipelines=[
            dict(
                type="MonaiElasticDictTransform",
                aug_parameters={
                    "prob": 0.5,
                    "patch_size": patch_size,
                    "roi_scale": 1.0,
                    "max_roi_scale": 1.0,
                    "rot_range_x": (-np.pi/6, np.pi/6),
                    "rot_range_y": (-np.pi/6, np.pi/6),
                    "rot_range_z": (-np.pi/6, np.pi/6),
                    "rot_90": False,
                    "flip": True,
                    "bright_bias": (-0.3, 0.3),
                    "bright_weight": (-0.3, 0.3),
                    "translate_x": (-5.0, 5.0),
                    "translate_y": (-5.0, 5.0),
                    "translate_z": (-5.0, 5.0),
                    "scale_x": (-0.1, 0.1),
                    "scale_y": (-0.1, 0.1),
                    "scale_z": (-0.1, 0.1),
                    "elastic_sigma_range": (3, 5),  # x,y,z
                    "elastic_magnitude_range": (100, 200),
                },
            )
        ],
    ),
    val=dict(
        type='CropSegSampleDataset', # for coarse segmentation
        dst_list_file='./checkpoints/predata_fakeT2_round3/val_0.lst',
        data_root='./checkpoints/predata_fakeT2_round3/',
        patch_size=patch_size,
        sample_frequent=1,
        pipelines=[],
    ),
)

# optimizer: 构建优化器,继承至pytorch
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# lr_config: 构建训练过程中学习率调整方式
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, min_lr=23e-6)

# checkpoint 文件设置, interval=1表示每epoch存储一次
checkpoint_config = dict(interval=5)

# log 文件设置, interval=5,表示每5个iteration进行一次存储和打印
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

# torch.backends.cudnn.benchmark
cudnn_benchmark = False

# 推荐使用分布式训练, gpus=4表示使用gpu的数量
gpus = 2
find_unused_parameters = True
total_epochs = 200
autoscale_lr = None # 是否使用根据batch_size自动调整学习率
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None
fp16 = dict(loss_scale=512.)

load_from = './checkpoints/results/crop_fakeT2_epoch90_round3_0618/crop_round3_0615_epoch_105.pth'
work_dir = './checkpoints/results/crop_fakeT2_epoch90_round3_0618'

evaluate = False
workflow = [('train', 1), ('val', 1)]
# workflow = [('train', 1)]