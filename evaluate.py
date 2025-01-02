import sys
sys.path.append('thirdparty/vidar/')
import os

import torch
from evaluation_utils.r3d3_argparser import argparser
from evaluation_utils.eval_dataset import Evaluator


if __name__ == '__main__':
    import torch

    print("CUDA version:", torch.version.cuda)
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))

    os.environ['DIST_MODE'] = 'gpu' if torch.cuda.is_available() else 'cpu'
    args = argparser()

    wrapper = Evaluator(args)
    wrapper.eval_datasets()
