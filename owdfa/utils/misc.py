#!/usr/bin/env python
# coding: utf-8
import os
import random
import shutil
import warnings
import wandb
import numpy as np
import torch
import torch.distributed as dist

from sympy.matrices import Matrix, GramSchmidt
import torch.nn.functional as F


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(args):
    os.environ['TORCH_HOME'] = args.torch_home
    warnings.filterwarnings("ignore")
    set_seed(args)


def init_workspace(args, project_name):
    args.exam_dir = "lmh_log/" + project_name
    if os.path.exists(args.exam_dir):
        shutil.rmtree(args.exam_dir)
    os.makedirs(args.exam_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exam_dir, 'ckpts'), exist_ok=True)
    return args


def save_test_results(img_paths, y_preds, y_trues, filename='results.log'):
    assert len(y_trues) == len(y_preds) == len(img_paths)

    with open(filename, 'w') as f:
        for i in range(len(img_paths)):
            print(img_paths[i], end=' ', file=f)
            print(y_preds[i], file=f)
            print(y_trues[i], end=' ', file=f)


def gather_tensor(inp, world_size=None, dist_=True, to_numpy=False):
    """Gather tensor in the distributed setting.

    Args:
        inp (torch.tensor): 
            Input torch tensor to gather.
        world_size (int, optional): 
            Dist world size. Defaults to None. If None, world_size = dist.get_world_size().
        dist_ (bool, optional):
            Whether to use all_gather method to gather all the tensors. Defaults to True.
        to_numpy (bool, optional): 
            Whether to return numpy array. Defaults to False.

    Returns:
        (torch.tensor || numpy.ndarray): Returned tensor or numpy array.
    """
    inp = torch.stack(inp)
    if dist_:
        if world_size is None:
            world_size = dist.get_world_size()
        gather_inp = [torch.ones_like(inp) for _ in range(world_size)]
        dist.all_gather(gather_inp, inp)
        gather_inp = torch.cat(gather_inp)
    else:
        gather_inp = inp

    if to_numpy:
        gather_inp = gather_inp.cpu().numpy()

    return gather_inp


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


def update_meter(meter, value, size, is_dist=False):
    if is_dist:
        meter.update(reduce_tensor(value.data).item(), size)
    else:
        meter.update(value.item(), size)
    return meter


def modification(idxes_2, x2, unlab_dict):
    pse_labels = []
    for u_idx in idxes_2:
        pse_label = unlab_dict[u_idx.item()][0][0]
        pse_labels.append(pse_label)

    pse_labels = torch.LongTensor(pse_labels).cuda()
    return pse_labels



def orthogonal_tensor(x):
    x = x.cpu()
    m, n = x.size()
    x_np = x.t().numpy()
    matrix = [Matrix(col) for col in x_np.T]
    gram = GramSchmidt(matrix)
    ort_list = []
    for i in range(m):
        vector = []
        for j in range(n):
            vector.append(float(gram[i][j]))
        ort_list.append(vector)
    ort_list = np.mat(ort_list)
    ort_list = torch.from_numpy(ort_list.astype(np.float32)).cuda()
    ort_list = F.normalize(ort_list, dim=1)
    return ort_list