from loguru import logger
import torch.utils.data as data

from owdfa.datasets.dfa import DFA
from owdfa.datasets.utils import create_data_transforms, MultilabelBalancedRandomSampler


def create_dataloader(args, split, all_sample=False, loader='torch'):
    if args.dataset.loader == 'torch':
        return create_torch_dataloader(args, split, all_sample)
    else:
        logger.error(f'Unknown loader: {args.dataset.loader}')


def create_torch_dataloader(args, split, all_sample):
    num_workers = args.num_workers if 'num_workers' in args else 8
    balance_sample = args.balance_sample if 'balance_sample' in args else False

    batch_size = getattr(args, split).batch_size

    transform = create_data_transforms(args.transform, split)
    kwargs = getattr(args.dataset, args.dataset.name)
    dataset = eval(args.dataset.name)(split=split, transform=transform, **kwargs)
    items = dataset.get_items()

    sampler = None

    if balance_sample and split == 'train':
        if all_sample:
            sampler = None
        else:
            sampler = MultilabelBalancedRandomSampler(dataset)

    shuffle = False  # 打乱找不到index

    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 pin_memory=True)

    return dataloader, items, sampler
