import warnings

warnings.filterwarnings('ignore')
import argparse
import os

from mmcv import Config
from starship.umtf.common.trainer import build_trainner

import sys
sys.path.append("..")
import custom
import apis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', type=str, default='none')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    for arg in vars(args):
        cfg[arg] = getattr(args, arg)
    if cfg['launcher'] != 'none':
        cfg['find_unused_parameters'] = True
    t = build_trainner(cfg)
    t.run()
