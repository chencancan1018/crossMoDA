import copy

import torch
import torch.nn as nn
from itertools import chain
from _collections import defaultdict
from mmcv.runner import OptimizerHook
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner import wrap_fp16_model

from starship.umtf.common.utils.dist_utils import allreduce_grads


try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass

if (TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION)[:2] >= digit_version('1.6')):

    class Fp16OptimizerHook(OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of GradScalar.
                Defaults to 512. For Pytorch >= 1.6, mmcv uses official
                implementation of GradScaler. If you use a dict version of
                loss_scale to create GradScaler, please refer to:
                https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                for the parameters.

        Examples:
            >>> loss_scale = dict(
            ...     init_scale=65536.0,
            ...     growth_factor=2.0,
            ...     backoff_factor=0.5,
            ...     growth_interval=2000
            ... )
            >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
        """

        def __init__(self,
                     grad_clip=None,
                     coalesce=True,
                     bucket_size_mb=-1,
                     loss_scale=512.,
                     distributed=True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.distributed = distributed
            self._scale_update_param = None
            if loss_scale == 'dynamic':
                self.loss_scaler = GradScaler()
            elif isinstance(loss_scale, float):
                self._scale_update_param = loss_scale
                self.loss_scaler = GradScaler(init_scale=loss_scale)
            elif isinstance(loss_scale, dict):
                self.loss_scaler = GradScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                 f'"dynamic", got {loss_scale}')

        def before_run(self, runner):
            """Preparing steps before Mixed Precision Training."""
            # # wrap model mode to fp16
            # wrap_fp16_model(runner.model)
            # resume from state dict
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net, fp32_weights):
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(
                            fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net, fp32_weights):
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                              fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer to
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients.
            3. Unscale the optimizer’s gradient tensors.
            4. Call optimizer.step() and update scale factor.
            5. Save loss_scaler state_dict for resume purpose.
            """
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer)
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

else:

    class Fp16OptimizerHook(OptimizerHook):
        """FP16 optimizer hook.

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            loss_scale (float): Scale factor multiplied with loss.
        """

        def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1, loss_scale=512., distributed=True):
            self.grad_clip = grad_clip
            self.coalesce = coalesce
            self.bucket_size_mb = bucket_size_mb
            self.loss_scale = loss_scale
            self.distributed = distributed
            if loss_scale == 'dynamic':
                self.loss_scaler = LossScaler(mode='dynamic')
            elif isinstance(loss_scale, float):
                self.loss_scaler = LossScaler(init_scale=loss_scale, mode='static')
            elif isinstance(loss_scale, dict):
                self.loss_scaler = LossScaler(**loss_scale)
            else:
                raise ValueError('loss_scale must be of type float, dict, or '
                                    f'"dynamic", bug got {loss_scale}')

        def before_run(self, runner):
            # keep a copy of fp32 weights
            old_groups = runner.optimizer.param_groups
            runner.optimizer.param_groups = copy.deepcopy(runner.optimizer.param_groups)
            state = defaultdict(dict)
            p_map = {
                old_p: p
                for old_p, p in zip(
                    chain(*(g['params'] for g in old_groups)),
                    chain(*(g['params']
                            for g in runner.optimizer.param_groups)))
            }
            for k, v in runner.optimizer.state.items():
                state[p_map[k]] = v
            runner.optimizer.state = state
            # convert model to fp16
            wrap_fp16_model(runner.model)
            if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
                scaler_state_dict = runner.meta['fp16']['loss_scaler']
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net, fp32_weights):
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net, fp32_weights):
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner):
            # clear grads of last iteration
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()
            # copy fp16 grads in the model to fp32 params in the optimizer
            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow: 
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update({'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                # update fp32 params
                runner.optimizer.step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            self.loss_scaler.update_scale(has_overflow)
            if has_overflow:
                runner.logger.warning('Check overflow, downscale loss scale '
                                        f'to {self.loss_scaler.cur_scale}')

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


class LossScaler:
    """Class that manages loss scaling in mixed precision training which
    supports both dynamic or static mode.

    The implementation refers to
    https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py.
    Indirectly, by supplying ``mode='dynamic'`` for dynamic loss scaling.
    It's important to understand how :class:`LossScaler` operates.
    Loss scaling is designed to combat the problem of underflowing
    gradients encountered at long times when training fp16 networks.
    Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.
    If overflowing gradients are encountered, :class:`FP16_Optimizer` then
    skips the update step for this particular iteration/minibatch,
    and :class:`LossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients
    detected,:class:`LossScaler` increases the loss scale once more.
    In this way :class:`LossScaler` attempts to "ride the edge" of always
    using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float): Initial loss scale value, default: 2**32.
        scale_factor (float): Factor used when adjusting the loss scale.
            Default: 2.
        mode (str): Loss scaling mode. 'dynamic' or 'static'
        scale_window (int): Number of consecutive iterations without an
            overflow to wait before increasing the loss scale. Default: 1000.
    """

    def __init__(self,
                    init_scale=2**32,
                    mode='dynamic',
                    scale_factor=2.,
                    scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        assert mode in ('dynamic',
                        'static'), 'mode can only be dynamic or static'
        self.mode = mode
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, params):
        """Check if params contain overflow."""
        if self.mode != 'dynamic':
            return False
        for p in params:
            if p.grad is not None and LossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        """Check if params contain NaN."""
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') \
                    or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        """update the current loss scale value when overflow happens."""
        if self.mode != 'dynamic':
            return
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % \
                    self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    def state_dict(self):
        """Returns the state of the scaler as a :class:`dict`."""
        return dict(
            cur_scale=self.cur_scale,
            cur_iter=self.cur_iter,
            mode=self.mode,
            last_overflow_iter=self.last_overflow_iter,
            scale_factor=self.scale_factor,
            scale_window=self.scale_window)

    def load_state_dict(self, state_dict):
        """Loads the loss_scaler state dict.

        Args:
            state_dict (dict): scaler state.
        """
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        self.mode = state_dict['mode']
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']

    @property
    def loss_scale(self):
        return self.cur_scale
