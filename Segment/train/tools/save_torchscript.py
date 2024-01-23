"""生成torchscript很难直接从Config构建的模型进行转换，需要剥离出组件."""

import argparse
import os
import sys
import torch
from mmcv import Config
from starship.umtf.common.model import build_network

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import custom  # noqa: F401


def load_model(config, model_path):
    seg_net = build_network(config.model)
    checkpoint = torch.load(model_path)
    seg_net.load_state_dict(checkpoint['state_dict'])
    seg_net.eval()
    return seg_net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/models/resunet.pth')
    parser.add_argument('--config_path', type=str, default='./checkpoints/models/train_config_resunet.py')
    parser.add_argument('--output_path', type=str, default='./checkpoints/models')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    config = Config.fromfile(args.config_path)
    config.model['apply_sync_batchnorm'] = False
    model = load_model(config, args.model_path)
    res_unet_jit = torch.jit.script(model)
    res_unet_jit.save(os.path.join(args.output_path, 'resunet.pt'))
