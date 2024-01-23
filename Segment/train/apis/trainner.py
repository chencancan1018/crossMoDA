# TODO: 暂时将mmdetection复制过来，后面需要仔细思考拆分

import copy
import os.path as osp
import random
import time
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from mmcv.runner import DistSamplerSeedHook, init_dist

from starship.umtf.common.dataset import build_dataloader, build_dataset
from starship.umtf.common.model import build_network
from starship.umtf.common.optimizer import build_optimizer
from starship.umtf.common.trainer import TRAINNERS, build_runner
from starship.umtf.common.utils import DistOptimizerHook, collect_env
from starship.umtf.service.component.utils import (
    DistEvalHook,
    EvalHook,
    SSDataParallel,
    SSDistributedDataParallel,
    get_batchsize_from_dict,
)
from starship.utils.logging import get_logger
from .optimizer.optimizer import Fp16OptimizerHook


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


@TRAINNERS.register_module
class Trainner_(object):

    def __init__(self, cfg, runner_config):
        self.cfg = cfg
        self.runner_config = runner_config

        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = '../'
        if cfg.get('autoscale_lr', None) is not None and cfg.get('autoscale_lr'):
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8
        if cfg.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(cfg.launcher, **cfg.dist_params)
        if cfg.get('optimizer_config', None) is None:
            cfg.optimizer_config = {}

        self.distributed = distributed

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_logger(cfg.get('task_name', ''), log_file=log_file, log_level=cfg.log_level)
        self.timestamp = timestamp

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v)) for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('Config:\n{}'.format(cfg.text))

        # set random seeds
        cfg.seed = cfg.get('seed', None)
        if cfg.get('seed') is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(cfg.seed, cfg.deterministic))
            set_random_seed(cfg.seed, deterministic=cfg.deterministic)
        meta['seed'] = cfg.seed
        self.meta = meta

        # 构造模型和datasets
        self.model = self._build_model()
        self.datasets = self._build_datasets(self.cfg.data.train)

        if len(self.cfg.workflow) == 2:
            val_dataset = copy.deepcopy(self.cfg.data.val)
            if self.cfg.data.train.get('pipeline', None) is not None:
                val_dataset.pipeline = self.cfg.data.train.get('pipeline')
            self.datasets.append(self._build_datasets(val_dataset)[0])
        # add an attribute for visualization convenience
        self.model.CLASSES = self.datasets[0].CLASSES if hasattr(self.datasets[0], 'CLASSES') else None

        # fp16 setting
        self.fp16_cfg = cfg.get('fp16', None)

    def _build_model(self):
        self.cfg.train_cfg = self.cfg.get('train_cfg', None)
        self.cfg.test_cfg = self.cfg.get('test_cfg', None)
        model = build_network(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        return model

    def _build_datasets(self, data_cfg):
        return [build_dataset(data_cfg)]

    def _build_optimizer(self, model, optimizer):
        return build_optimizer(model, optimizer)

    def _builder_runner(self, model, _batch_processor, optimizer, work_dir, logger, meta):
        return build_runner(
            self.runner_config,
            model=model,
            batch_processor=_batch_processor,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta
        )

    def _build_dataloader(self, dataset, dist):
        dataloader_cfg = self.cfg.data.get('dataloader', None)
        return build_dataloader(
            dataset=dataset,
            imgs_per_gpu=self.cfg.data.imgs_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            num_gpus=self.cfg.gpus,
            dataloader_cfg=dataloader_cfg,
            dist=dist,
            shuffle=self.cfg.data.shuffle if self.cfg.data.get('shuffle', None) is not None else True,
            seed=self.cfg.seed,
            drop_last=self.cfg.data.drop_last if self.cfg.data.get('drop_last', None) is not None else False,
        )

    def _dist_train(self, model, dataset, cfg, evaluate=False, logger=None, timestamp=None, meta=None):
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [(self._build_dataloader(ds, dist=True)) for ds in dataset]
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        if hasattr(model, 'train_step') or hasattr(model, 'val_step'):
            self._batch_processor = None
        model = SSDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )

        # build runner
        optimizer = self._build_optimizer(model, cfg.optimizer)
        runner = self._builder_runner(model, self._batch_processor, optimizer, cfg.work_dir, logger=logger, meta=meta)
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = self.fp16_cfg
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg)
        else:
            optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

        # register hooks
        runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)
        runner.register_hook(DistSamplerSeedHook())
        # register eval hooks
        if evaluate:
            test_dataset = build_dataset(cfg.data.test)
            dataloader_cfg = cfg.data.get('dataloader', None)
            test_dataloader = build_dataloader(
                test_dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=True,
                shuffle=False,
                dataloader_cfg=dataloader_cfg
            )
            eval_cfg = cfg.get('evaluation', {})
            runner.register_hook(DistEvalHook(test_dataloader, **eval_cfg))

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    def _non_dist_train(self, model, dataset, cfg, evaluate=False, logger=None, timestamp=None, meta=None):
        # prepare data loaders
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [(self._build_dataloader(ds, dist=False)) for ds in dataset]
        # put model on gpus
        if hasattr(model, 'train_step') or hasattr(model, 'val_step'):
            self._batch_processor = None
        model = SSDataParallel(model, device_ids=range(cfg.gpus)).cuda()

        # build runner
        optimizer = self._build_optimizer(model, cfg.optimizer)
        runner = self._builder_runner(model, self._batch_processor, optimizer, cfg.work_dir, logger=logger, meta=meta)
        # an ugly walkaround to make the .log and .log.json filenames the same
        runner.timestamp = timestamp
        # fp16 setting
        fp16_cfg = self.fp16_cfg
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=False)
        else:
            optimizer_config = cfg.optimizer_config
        runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)

        if evaluate:
            test_dataset = build_dataset(cfg.data.test)
            dataloader_cfg = cfg.data.get('dataloader', None)
            test_dataloader = build_dataloader(
                test_dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=True,
                shuffle=False,
                dataloader_cfg=dataloader_cfg
            )
            eval_cfg = cfg.get('evaluation', {})
            runner.register_hook(EvalHook(test_dataloader, **eval_cfg))

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    def _train(self, model, dataset, cfg, distributed=False, evaluate=False, timestamp=None, meta=None):
        logger = get_logger(name=cfg.get('task_name', ''), log_level=cfg.log_level)

        # start training
        if distributed:
            self._dist_train(model, dataset, cfg, evaluate=evaluate, logger=logger, timestamp=timestamp, meta=meta)
        else:
            self._non_dist_train(model, dataset, cfg, evaluate=evaluate, logger=logger, timestamp=timestamp, meta=meta)

    def _batch_processor(self, model, data):
        """Process a data batch.

        This method is required as an argument of Runner, which defines how to
        process a data batch and obtain proper outputs.

        Args:
            model (nn.Module): A PyTorch model.
            data (dict): The data batch in a dict.

        Returns:
            dict: A dict containing losses and log vars.
        """
        if self.fp16_cfg is not None:
            with autocast():
                losses = model(**data)
        else:
            losses = model(**data)
        loss, log_vars = self._parse_losses(losses)

        num_samples = get_batchsize_from_dict(data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def run(self):
        self._train(
            self.model,
            self.datasets,
            self.cfg,
            distributed=self.distributed,
            evaluate=self.cfg.evaluate,
            timestamp=self.timestamp,
            meta=self.meta
        )
