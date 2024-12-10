#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import numpy as np


def margin_loss(x, target):
    m = -0.2
    s = 1
    weight = None

    index = torch.zeros_like(x, dtype=torch.uint8)
    index.scatter_(1, target.data.view(-1, 1), 1)   # 将1填充到index中，对标签进行onehot编码
    x_m = x - m * s
    output = torch.where(index, x_m, x)             # 按index取x_m或x:正确概率更大
    return F.cross_entropy(output, target, weight=weight)


def entropy(x):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    min = 1e-8
    x_  = torch.clamp(x, min=min)
    en  = x_ * torch.log(x_)

    if len(en.size()) == 2:
        return - en.sum(dim=1).mean()
    elif len(en.size()) == 1:
        return - en.sum()



def contrastive_loss(u_probs, label2,
                     k_probs, label1,
                     memory_dists, memory_sims, t_len):
    # 有标签
    k = label1.size(0)
    k_sims = torch.mm(k_probs, k_probs.t())
    mask_k = label1.expand(k, k).eq(label1.expand(k, k).t())  # 逐元素比较
    known_pos_pairs = []
    # known_neg_pairs = []
    for i in range(k):
        known_pos_pairs.append(k_sims[i][mask_k[i]].min())
        # known_neg_pairs.append(k_sims[i][mask_k[i] == 0].max())
    known_pos_pairs = torch.stack(known_pos_pairs, dim=0)
    # known_neg_pairs = torch.stack(known_neg_pairs, dim=0)
    ones_k = torch.ones_like(known_pos_pairs)
    loss_k = F.binary_cross_entropy(known_pos_pairs, ones_k)  # pos>neg


    # 无标签
    u_sims = torch.mm(u_probs, u_probs.t())
    _, max_idxes = torch.topk(input=u_sims, k=2, dim=1)
    unknown_pairs  = []
    unknown_weight = []
    unknown_target = []
    for i in range(max_idxes.size(0)):
        neighbor_idx = max_idxes[i][1].item()
        neighbor_lab = label2[neighbor_idx]
        anchor_idx = i
        anchor_lab = label2[anchor_idx]
        if anchor_lab != -1 and neighbor_lab != -1 and anchor_lab != neighbor_lab:
            max_sim = u_sims[anchor_idx][neighbor_idx]
            unknown_pairs.append(max_sim)
            unknown_weight.append(memory_sims[anchor_lab][neighbor_lab]/memory_sims.sum())
            unknown_target.append(torch.tensor(1.).cuda())
        else:
            max_sim = u_sims[anchor_idx][neighbor_idx]
            unknown_pairs.append(max_sim)
            unknown_weight.append(torch.tensor(1.).cuda())
            unknown_target.append(torch.tensor(1.).cuda())

    unknown_pairs  = torch.stack(unknown_pairs)
    unknown_weight = torch.stack(unknown_weight)
    unknown_target = torch.stack(unknown_target)

    loss_u = F.binary_cross_entropy(unknown_pairs, unknown_target, unknown_weight)

    return loss_u + loss_k  # loss_u +



def csp_loss(method, method_soft):
    rand_prob = F.gumbel_softmax(method.detach(), tau=1, hard=False)
    target_idx = torch.argmax(rand_prob, dim=1)
    max_prob_pl = method_soft.gather(1, target_idx.view(-1, 1)).squeeze()
    loss = (F.cross_entropy(method, rand_prob, reduction='none') * max_prob_pl).mean()
    return loss



def glv_loss(probs, labels, feature, t_len, batch_size, memory_features):

    feat = feature[0]
    feat_p = feature[1]
    labeled_len = t_len//2
    total_len = t_len

    pos_pairs, mask_1 = _get_pair(labels[:t_len//2], feat, feat_p, labeled_len, total_len)
    tar_prob = probs[pos_pairs, :]
    if len(tar_prob) < batch_size:
        return torch.zeros(1).cuda(), mask_1

    # tar_sim = torch.bmm(probs.view(batch_size, 1, -1),
    #                     tar_prob.view(batch_size, -1, 1)).squeeze()

    # 计算近邻与聚类中心相似度
    cen_features = []
    for label, prob in zip(labels, probs):
        if label == -1:
            cen_feature = prob
        else:
            cen_feature = memory_features[label]
        cen_features.append(cen_feature)
    cen_features = torch.stack(cen_features, 0)

    cen_sim = torch.bmm(cen_features.view(batch_size, 1, -1),
                        tar_prob.view(batch_size, -1, 1)).squeeze()

    tar_ones = torch.ones_like(cen_sim)
    return F.binary_cross_entropy(cen_sim, tar_ones), mask_1


def _get_pair(target, f_g, f_p, labeled_len, total_len):
    pos_pairs = []
    target_np = target.cpu().numpy()
    # label part
    for ind in range(labeled_len):
        target_i = target_np[ind]
        idxs = np.where(target_np == target_i)[0]
        if len(idxs) == 1:
            pos_pairs.append(idxs[0])
        else:
            selec_idx = np.random.choice(idxs, 1)
            while selec_idx == ind:
                selec_idx = np.random.choice(idxs, 1)
            pos_pairs.append(int(selec_idx))
    # unlabel part
    # 1. global feature
    feat_detach = f_g.detach()
    feat_g_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
    global_cosine_dist = torch.mm(feat_g_norm, feat_g_norm.t())
    global_sim = global_cosine_dist
    # 2. local feature
    part_feat = f_p.detach()
    part_norm = torch.norm(part_feat, 2, 1)
    feat_p_norm = part_feat / torch.norm(part_feat, 2, 1, keepdim=True)
    part_cosine_dist = (torch.bmm(feat_p_norm.permute(2, 0, 1), feat_p_norm.permute(2, 1, 0))  # s
                        .permute(1, 2, 0))
    # 3. partial mask
    part_norm = part_norm / torch.norm(part_norm, 2, 1, keepdim=True)  # w
    a = part_norm.repeat(total_len, 1, 1)
    part_sim = (part_cosine_dist * part_norm.repeat(total_len, 1, 1).permute(1, 0, 2)).sum(dim=2)

    # Global similarity & Part filter
    _, pos_idx = torch.topk(global_sim[labeled_len:, :], 2, dim=1)
    vals, _ = torch.topk(part_sim[labeled_len:, :], 2, dim=1)
    choose_k = 2  # this parameter should be fine-tuned with different task
    max_pos = torch.topk(part_sim[:, pos_idx[:, 1]], choose_k, dim=0)[0][choose_k-1]

    mask_1 = (vals[:, 1] - max_pos).ge(0).float()  # >=
    mask_0 = (vals[:, 1] - max_pos).lt(0).float()  # <
    pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
    pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
    pos_idx = (pos_idx_1 + pos_idx_0).flatten().tolist()
    pos_pairs.extend(pos_idx)
    return pos_pairs, mask_1


def align_loss(ge, sp, oth):
    n = ge.size(0)

    dist_n = torch.pow(ge, 2).sum(dim=1, keepdim=True) + torch.pow(sp, 2).sum(dim=1, keepdim=True) - 2*(ge*sp)
    dist_p = torch.pow(ge, 2).sum(dim=1, keepdim=True) + torch.pow(oth, 2).sum(dim=1, keepdim=True) - 2*(ge*oth)

    dist_n = dist_n.clamp(min=1e-12).sqrt()  # for numerical stability
    dist_p = dist_p.clamp(min=1e-12).sqrt()

    # Compute ranking hinge loss
    y = torch.ones_like(dist_n)
    loss = F.margin_ranking_loss(dist_n, dist_p, y, 20)

    return loss