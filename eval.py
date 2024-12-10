#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import argparse
import lightning.pytorch as pl
from glob import glob
from omegaconf import OmegaConf
from loguru import logger

from owdfa.sl import SLModel
from owdfa.memory.lmh_memory import HybridMemory
from owdfa.datasets import create_dataloader

import warnings
warnings.filterwarnings("ignore")


def main():
    # set configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='./configs/eval.yaml')
    parser.add_argument('--exam_id', type=str, default='lmh-DFA-no14.6-tri')
    parser.add_argument('--ckpt_path', type=str, default='/data5/liuminghui/lmh/NewIdea/DeepfakeAttribution/model/')
    parser.add_argument('--output_log', type=str, default='eval.log')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--distributed', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    local_config = OmegaConf.load(args.config)
    for k, v in local_config.items():
        setattr(args, k, v)

    os.environ['TORCH_HOME'] = args.torch_home
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # search the checkpoint file according EXAM ID
    if args.exam_id:
        model_root = os.path.join(args.ckpt_path, args.exam_id)
        model_path = glob(f'{model_root}/ckpts/*.ckpt')[0]
    exam_dir = os.path.dirname(os.path.dirname(model_path))

    # add log file
    if len(args.output_log) > 0:
        logger.add(f'{exam_dir}/{args.output_log}', level="INFO")

    # load dataset
    train_dataloader, train_items, train_sampler = create_dataloader(args, split='train')
    test_dataloader, test_items, test_sampler = create_dataloader(args, split='test')

    memory = HybridMemory(momentum=args.train.memory.momentum).cuda()
    method = SLModel(args, memory,
                     train_sampler, test_sampler,)

    trainer = pl.Trainer(default_root_dir=exam_dir)
    trainer.validate(method, test_dataloader, model_path)


if __name__ == '__main__':
    main()
