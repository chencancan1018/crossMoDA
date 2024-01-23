from mmcv.runner.base_runner import BaseRunner
# Template: 

# @RUNNERS.register_module
class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self):
        # TODO: 训练实现接口
        pass

    def val(self, data_loader, **kwargs):
        # TODO: 在线验证实现接口
        pass

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        # TODO: start训练/验证流程
        pass

    def save_checkpoint(self):
        # TODO: 模型保存
        pass

