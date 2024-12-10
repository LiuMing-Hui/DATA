from __future__ import print_function, absolute_import
from collections import OrderedDict
import torch
from owdfa.utils import visualize
import torch.nn.functional as F


def extract_cnn_feature(model, inputs):
    inputs = inputs.cuda()
    method, _, _, _, _ = model(inputs)
    outputs = method.detach()
    return outputs

def extract_labeled_features(model, data_loader):
    with torch.no_grad():

        lab_features = OrderedDict()  # 有序字典
        labels = OrderedDict()
        all_lab_idxes = []

        for batch_index, batch in enumerate(data_loader):

            tags = batch['tag']
            images = batch['image']
            targets = batch['target']
            idxes = batch['idx']

            lab_imgs = images[tags == 1]
            lab_targets = targets[tags == 1]
            lab_idxes = idxes[tags == 1]

            if len(lab_imgs) != 0:
                lab_outputs = extract_cnn_feature(model, lab_imgs)
                lab_outputs = F.softmax(lab_outputs, dim=1)

                for lab_idx, lab_output, lab_label in zip(lab_idxes, lab_outputs, lab_targets):  # 建立特征与地址的联系
                    lab_features[lab_idx.item()] = lab_output
                    labels[lab_idx.item()] = lab_label.item()
                    all_lab_idxes.append(lab_idx.item())

    return lab_features, labels, all_lab_idxes




def extract_unlabeled_features(model, data_loader, args, epoch):
    with torch.no_grad():

        unlab_features = []
        unlab_targets = []
        unlab_tags = []
        unlab_indexes = []

        # all_features = []
        # all_targets = []
        # all_tags = []

        for batch_index, batch in enumerate(data_loader):

            tags = batch['tag']
            images = batch['image']
            targets = batch['target']  # 不能用，仅作参考
            indexes = batch['idx']

            outputs = extract_cnn_feature(model, images)
            outputs = F.softmax(outputs, dim=1)

            unlab_tar = targets[tags == 2]
            unlab_tag = tags[tags == 2]
            unlab_idx = indexes[tags == 2]
            unlab_out = outputs[tags == 2]

            unlab_features.extend(unlab_out)
            unlab_targets.extend(unlab_tar)
            unlab_tags.extend(unlab_tag)
            unlab_indexes.extend(unlab_idx)

            # all_features.extend(outputs)
            # all_targets.extend(targets)
            # all_tags.extend(tags)

        unlab_features = torch.stack(unlab_features, 0)
        unlab_targets = torch.stack(unlab_targets, 0)
        unlab_tags = torch.stack(unlab_tags, 0)
        unlab_indexes = torch.stack(unlab_indexes, 0)

        # all_features = torch.stack(all_features, 0).cpu().numpy()
        # all_targets = torch.stack(all_targets, 0).cpu().numpy()
        # all_tags = torch.stack(all_tags, 0).cpu().numpy()

        # visualize(all_tags, all_targets, all_features, args, epoch, 'train_img')  # 无返回值

    return unlab_features, unlab_targets, unlab_tags, unlab_indexes


