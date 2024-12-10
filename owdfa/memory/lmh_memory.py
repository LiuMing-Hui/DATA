from __future__ import print_function, absolute_import
import collections
import torch
from loguru import logger
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from torch import nn, autograd
import numpy as np
import copy


from owdfa.memory.lmh_all_features import extract_labeled_features, extract_unlabeled_features
from owdfa.losses import entropy
from owdfa.utils.faiss_rerank import compute_jaccard_distance


def init_memory(model, memory, all_train_dataloader):
    print("==> Extract labeled features")
    model.eval()

    lab_features, lab_labels, lab_idxes = extract_labeled_features(model, all_train_dataloader)

    lab_fea_dict = collections.defaultdict(list)

    for idx in sorted(lab_idxes):   # 依据所有path找标签和特征
        label = lab_labels[idx]     # tensor会出问题
        lab_fea_dict[label].append(lab_features[idx].unsqueeze(0))

    lab_centers = []
    for key in sorted(lab_fea_dict.keys()):
        lab_centers.append(torch.cat(lab_fea_dict[key], 0).mean(0))

    lab_centers = torch.stack(lab_centers, 0)

    memory.features = lab_centers.cuda()

    del lab_centers, lab_fea_dict, label, idx, key,\
        lab_features, lab_labels, lab_idxes

    model.train()

    return memory.features



def use_memory(model, memory, all_train_dataloader,
               args, epoch):
    print("==> Extract unlabeled features")
    model.eval()

    unlab_features, unlab_targets, unlab_tags, unlab_indexes = extract_unlabeled_features(model,
                                                                                          all_train_dataloader,
                                                                                          args,
                                                                                          epoch)
    memory_features = copy.deepcopy(memory.features.detach())

    cluster_labels  = DBSCAN(eps=0.03, min_samples=6, n_jobs=-1).fit_predict(unlab_features.cpu())
    cluster_numbers = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    cluster_centers = []
    for idx in range(cluster_numbers):
        center = torch.mean(unlab_features[cluster_labels == idx], dim=0)
        cluster_centers.append(center)
    cluster_centers = torch.stack(cluster_centers, dim=0).cuda()

    # 计算相似度
    sims = torch.cosine_similarity(cluster_centers.unsqueeze(1), memory_features.unsqueeze(0), dim=2)

    max_sims, pse_labels = sims.max(1)

    mask = torch.zeros_like(sims)
    for i, label in enumerate(pse_labels):
        mask[i][label] = 1

    known_labels   = []
    unknown_labels = []
    unlab_labels   = torch.from_numpy(copy.deepcopy(cluster_labels))

    known_cluster_idx = set(torch.mul(sims, mask).max(0)[1]                   # 行标签（簇索引）
                            [torch.mul(sims, mask).max(0)[0] != 0]            # 最大值 （去掉全0列）
                            .tolist())
    for k_idx in known_cluster_idx:
        pse_label = pse_labels[k_idx]
        known_labels.append([k_idx, pse_label.item()])                        # [簇标签，伪标签]
    for known_label in known_labels:
        unlab_labels[cluster_labels == known_label[0]] = known_label[1]
        memory_features[known_label[1]] = cluster_centers[known_label[0]]     # 更新已存储memory

    unknown_cluster_idx = list(set(range(cluster_numbers)) - known_cluster_idx)
    for i, u_idx in enumerate(unknown_cluster_idx):
        pse_label = memory_features.size(0)
        unknown_labels.append([u_idx, pse_label])
        memory_features = torch.cat((memory_features, cluster_centers[u_idx].unsqueeze(0)), dim=0)
    memory.features = memory_features
    for unknown_label in unknown_labels:
        unlab_labels[cluster_labels == unknown_label[0]] = unknown_label[1]

    logger.info("==> There are {} clusters, {} known clusters, {} features, {} outliers",
                cluster_numbers, len(known_labels),
                len(unlab_labels), len(unlab_labels[unlab_labels == -1]))
    unlab_dict = collections.defaultdict(list)
    for unlab_index, unlab_label, unlab_tag in zip(unlab_indexes, unlab_labels, unlab_tags):
        unlab_dict[unlab_index.item()].append([unlab_label.item(), unlab_tag.item()])

    del unlab_features, unlab_targets, \
        memory_features, \
        cluster_numbers, cluster_labels, cluster_centers, \
        sims, max_sims, pse_labels, mask,  \
        known_labels, known_cluster_idx, unknown_labels, unknown_cluster_idx, \
        unlab_labels, unlab_indexes, unlab_tags,\

    model.train()
    return unlab_dict



class HM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        labeled_inputs  = inputs[indexes != -1]
        labeled_indexes = indexes[indexes != -1]

        ctx.features = features  # (,2048)
        ctx.momentum = momentum
        ctx.save_for_backward(labeled_inputs, labeled_indexes)  # 有标签数据：存储更新memory
        return inputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # logger.info(f'==> Start backward')
        input0, indexes = ctx.saved_tensors

        grad_input0 = None
        if ctx.needs_input_grad[0]:
            grad_input0 = grad_outputs  # 维度变换

        # momentum update
        for x, y in zip(input0, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_input0, None, None, None  # lmh_change



def hm(input0, indexes, features, momentum):
    return HM.apply(input0, indexes, features, torch.Tensor([momentum]).to(input0.device))



class HybridMemory(nn.Module):
    def __init__(self, momentum=None):  # momentum=0.2
        super(HybridMemory, self).__init__()

        self.momentum = momentum
        self.register_buffer('features', torch.zeros(8, 20))              # 模型的常数

    def forward(self, method_soft, label):
        method_soft = hm(method_soft, label, self.features, self.momentum)

        # Compute pairwise distance
        # n = self.features.size(0)
        # dist = torch.pow(self.features, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, self.features, self.features.t())
        # dist = dist.clamp(min=1e-12).sqrt()
        # dist = F.sigmoid(dist)
        #
        # sims = torch.mm(self.features, self.features.t())

        return method_soft









