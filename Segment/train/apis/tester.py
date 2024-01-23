import copy
import os
import os.path as osp
import random
import time
import json
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
# try:
#     from mmcv.cnn.utils import revert_sync_batchnorm
# except:
#     from runner import revert_sync_batchnorm
from mmcv.runner import init_dist, wrap_fp16_model, load_checkpoint, get_dist_info

from starship.umtf.common.dataset import build_dataloader, build_dataset
from starship.umtf.common.model import build_network
from starship.umtf.service.component.utils import (
    SSDataParallel,
    SSDistributedDataParallel,
    multi_gpu_test, single_gpu_test
)
from starship.utils.logging import get_logger

class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]
    if hasattr(mmcv, 'ops'):
        module_checklist.append(mmcv.ops.SyncBatchNorm)
    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Tester(object):

    def __init__(self, cfg):
        self.cfg = cfg
        
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = '../'
        if cfg.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)

        self.distributed = distributed
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_logger(cfg.get('task_name', ''), log_file=log_file, log_level=cfg.log_level)
        self.timestamp = timestamp

        logger.info('Distributed training: {}'.format(distributed))
        logger.info('Config:\n{}'.format(cfg.text))

        cfg.seed = cfg.get('seed', None)
        if cfg.get('seed') is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(cfg.seed, cfg.deterministic))
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)
        self.logger = logger
        
        
        self.dataset = build_dataset(self.cfg.data.test)
        self.dataloader = build_dataloader(
            dataset=self.dataset,
            imgs_per_gpu=self.cfg.data.imgs_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            num_gpus=self.cfg.gpus,
            dataloader_cfg=self.cfg.data.get('dataloader', None),
            dist=self.distributed,
            shuffle=self.cfg.data.shuffle if self.cfg.data.get('shuffle', None) is not None else True,
            seed=self.cfg.seed,
            drop_last=self.cfg.data.drop_last if self.cfg.data.get('drop_last', None) is not None else False,
        )
        self.suffix = None
        self.fp16_cfg = self.cfg.get('fp16', None)



    def _build_model(self):
        self.cfg.test_cfg = self.cfg.get('test_cfg', None)
        model = build_network(self.cfg.model, test_cfg=self.cfg.test_cfg)
        return model

    def run(self):
        load_from = self.cfg.load_from
        if osp.isdir(load_from):
            _pths = sorted([osp.join(load_from, p) for p in os.listdir(load_from) if p.endswith('.pth')])
        elif osp.isfile(load_from) and load_from.endswith('.pth'):
            _pths = [load_from]
        else:
            raise ValueError(' No exists .pth_file under current path: {}'.format(load_from))
        
        for _pth in _pths:
            self.logger.info('load checkpoint from %s', _pth)
            self.suffix = osp.splitext(osp.basename(_pth))[0]
            model = self._build_model()

            # fp16 setting
            fp16_cfg = self.cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)

            checkpoint = load_checkpoint(model, _pth, map_location='cpu', logger=self.logger)
            self._test(
                model, 
                self.cfg,
                self.dataloader, 
                distributed=self.distributed
                )

    def _test(self, model, cfg, dataloader, distributed=False):

        if distributed:
            results = self._dist_test(model, cfg, dataloader)
        else:
            results = self._not_dist_test(model, cfg, dataloader)
        rank, _ = get_dist_info()
        if rank == 0:
            print('\n')
            metric = dataloader.dataset.evaluate(results, logger=self.logger, **cfg.get('evaluation', {}))
            print('current metric: ', metric)
            metric_dict = dict(config=cfg, metric=metric)
            pkl_file = osp.join(cfg.work_dir, f'test_metric_{self.suffix}.pkl')
            mmcv.dump(metric_dict,pkl_file)
            tmpdir = osp.join(cfg.work_dir, '.eval_hook')
            if osp.exists(tmpdir):
                import shutil
                shutil.rmtree(tmpdir)
    
    def _dist_test(self, model, cfg, dataloader):

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = SSDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )
        if self.fp16_cfg is not None:
            with autocast():
                results = multi_gpu_test(
                    model,
                    dataloader,
                    tmpdir=osp.join(cfg.work_dir, '.eval_hook'),
                    gpu_collect=cfg.get('evaluation', {}),
                    )
        else:
            results = multi_gpu_test(
                model,
                dataloader,
                tmpdir=osp.join(cfg.work_dir, '.eval_hook'),
                gpu_collect=cfg.get('evaluation', {}),
                )
        return results

    def _not_dist_test(self, model, cfg, dataloader):
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        model = SSDataParallel(model, device_ids=cfg.gpu_ids).cuda()
        if self.fp16_cfg is not None:
            with autocast():
                results = single_gpu_test(
                    model,
                    dataloader,
                    show=False,
                    )
        else:
            results = single_gpu_test(
                model,
                dataloader,
                show=False,
                )
        return results