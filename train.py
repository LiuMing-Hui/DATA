#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from loguru import logger
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from owdfa.datasets import create_dataloader
from owdfa.utils import get_parameters, init_workspace, setup
from owdfa.sl import SLModel
from owdfa.memory.lmh_memory import HybridMemory

import better_exceptions
better_exceptions.hook()

import setproctitle


args = get_parameters(False,'configs/train.yaml')
project_name = args.project_name
setproctitle.setproctitle(project_name)
args = init_workspace(args, project_name) #LMH_CHANGE
logger.add(f'{args.exam_dir}/{project_name}.log', level="INFO")
logger.info(OmegaConf.to_yaml(args))


def main():
    # Init setup
    setup(args)

    # Create dataloader
    train_dataloader, train_items, train_sampler = create_dataloader(args, split='train')
    test_dataloader, test_items, test_sampler = create_dataloader(args, split='test')

    all_train_dataloader, all_train_items, all_train_sampler = create_dataloader(args, split='train', all_sample=True)

    # Resume from checkpoint
    checkpoint_dir = os.path.join(args.exam_dir, 'ckpts') if args.exam_dir else None

    # 在所有epoch之前：建立memory
    memory = HybridMemory(momentum=args.train.memory.momentum).cuda()

    model = SLModel(args, memory,
                    train_sampler, test_sampler,
                    train_dataloader, test_dataloader, all_train_dataloader,
                    train_items, test_items,
                    )


    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        min_epochs=1,
        max_epochs=args.train.epochs,
        default_root_dir=args.exam_dir,
        callbacks=[ModelCheckpoint(
            dirpath=checkpoint_dir,
            verbose=True,
            monitor='novel_acc',
            mode='max',
            save_on_train_epoch_end=True,
        )],
        num_sanity_val_steps=0,  # 1
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
