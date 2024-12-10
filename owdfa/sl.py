import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from loguru import logger
from timm.models import resume_checkpoint

from torch.optim import *
from torch.optim.lr_scheduler import *

from owdfa.utils import gather_tensor, val_stat, visualize
from owdfa.base_net import BaseClassifier
from owdfa.utils import AverageMeter, update_meter, modification
from owdfa.losses import entropy, margin_loss, align_loss, glv_loss
from owdfa.memory.lmh_memory import init_memory, use_memory



best_novel_acc = 0
best_epoch = 0

class SLModel(pl.LightningModule):
    def __init__(self,
                 args, memory=None,
                 train_sampler=None, test_sampler=None,
                 train_dataloader=None, test_dataloader=None, all_train_dataloader=None,
                 train_items=None, test_items=None,
                 **kwargs):

        super().__init__()
        self.args = args
        self.memory = memory

        self.encoder = BaseClassifier(**args.model.params)
        self.fc = self.encoder.fc2

        if args.model.resume is not None:
            resume_checkpoint(self.encoder, args.model.resume)

        self.all_train_dataloader = all_train_dataloader
        self.train_sampler = train_sampler
        self.num_label_classes = self.train_sampler.num_label_classes
        self.batch_size = args.train.batch_size

        self.disentangle = args.model.params.disentangle

    def configure_optimizers(self):
        optimizer = Adam(self.encoder.parameters(), **self.args.optimizer.params)
        scheduler = StepLR(optimizer, **self.args.scheduler.params)
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        self.train_losses = {loss: AverageMeter(loss, ':.2f') for loss in self.get_loss_names()}  # 管理自定义变量的更新

        # 初始化memory，存储有标签类原型(prob)
        if self.current_epoch == self.args.train.memory.init_epoch:
            self.memory.features = init_memory(self.encoder, self.memory, self.all_train_dataloader)
            self.train_sampler.__iter__()

        # 填充新类原型(prob)
        if self.current_epoch > self.args.train.memory.use_epoch:
            self.unlab_dict = use_memory(self.encoder, self.memory, self.all_train_dataloader,
                                         self.args, self.current_epoch)
            self.train_sampler.__iter__()

    def training_step(self, batch, batch_idx):
        tags    = batch['tag']
        images  = batch['image']
        targets = batch['target']
        idxes   = batch['idx']
        types = batch['face_type']

        label_1 = targets[tags == 1]
        idxes_2 = idxes[tags == 2]
        types_1 = types[tags == 1]

        x1 = images[tags == 1]
        x2 = images[tags == 2]

        if self.current_epoch > self.args.train.memory.use_epoch:
            label_2 = modification(idxes_2, x2, self.unlab_dict)  # x2=(x2_3, x2_2)
        else:
            label_2 = torch.full_like(label_1, -1)
        label = torch.cat((label_1, label_2), dim=0)

        t_len = len(label)

        x = torch.cat((x1, x2), dim=0)

        method, face_type, feature, orth_fake, types_2 = self.encoder(x, types_1)  # feature=(g,p)

        pse_types = torch.cat((types_1, types_2), dim=0)

        loss_map = self.loss(t_len=t_len,
                             method=method, face_type=face_type, feature=feature, orth_fake=orth_fake,
                             label=label, pse_types=pse_types,
                             memory_features=self.memory.features,
                             disentangle=self.disentangle)

        loss = loss_map['total']

        for key, value in loss_map.items():
            update_meter(self.train_losses[key], value, self.args.train.batch_size, self.args.distributed)
        for ls in self.train_losses.values():
            self.log(ls.name, ls.avg, on_step=True, prog_bar=False)

        if batch_idx % self.args.train.log_time == 0:
            results = {format(key, '^6'): format(value.avg, '<.3f') for key, value in self.train_losses.items()}
            logger.info(results)

        return loss

    def get_loss_names(self):
        loss_name = ['total', 'ce', 'en', 'glv', 'align']  #'memory_loss_u'
        return loss_name

    def loss(self, **kwargs):
        loss_map = {}

        t_len   = kwargs['t_len']
        method  = kwargs['method']
        feature = kwargs['feature']
        label   = kwargs['label']
        pse_types = kwargs['pse_types']
        face_type = kwargs['face_type']
        fake_orth = kwargs['orth_fake']
        disentangle = kwargs['disentangle']
        memory_features = kwargs['memory_features']

        method_soft = F.softmax(method, dim=1)

        method_soft = self.memory(method_soft, label)

        # Cross Entropy loss
        loss_ce = margin_loss(method[:t_len//2], label[:t_len//2])  # + margin_loss(face_type, pse_types)
        loss_map['ce'] = loss_ce

        # Global Local Voting loss
        loss_glv, mask = glv_loss(method_soft, label, feature, t_len, self.batch_size, memory_features)
        loss_map['glv'] = loss_glv

        # Align loss
        if disentangle:
            fake_features_ge = feature[2][pse_types == 0]
            fake_features_sp = feature[0][pse_types == 0]
            num = fake_features_ge.size(0)
            fake_orthogonal = fake_orth.repeat(num, 1)
            loss_align = align_loss(fake_features_ge, fake_features_sp, fake_orthogonal)
        else:
            loss_align = torch.tensor(0.)
        loss_map['align'] = loss_align

        # En loss
        loss_en = entropy(torch.mean(method_soft, 0))
        loss_map['en'] = loss_en

        # Total loss
        total_loss = loss_ce + 0.5*loss_glv + 0.05*loss_align - loss_en
        loss_map['total'] = total_loss

        return loss_map


    def on_validation_epoch_start(self):
        self.val_step_outputs = {'tags': [], 'preds': [], 'label': [], 'conf': [], 'outputs': [],}

    def validation_step(self, batch, batch_idx):
        tags = batch['tag']
        images = batch['image']
        targets = batch['target']

        image = images[tags != 0]
        target = targets[tags != 0]
        tag = tags[tags != 0]

        output, _, _, _, _ = self.encoder(image)
        prob = F.softmax(output, dim=1)
        conf, pred = prob.max(1)

        self.val_step_outputs['tags'].extend(tag)
        self.val_step_outputs['preds'].extend(pred)
        self.val_step_outputs['label'].extend(target)
        self.val_step_outputs['conf'].extend(conf)
        self.val_step_outputs['outputs'].extend(output)
        return pred

    def on_validation_epoch_end(self):

        global best_novel_acc
        global best_epoch

        y_tags = gather_tensor(self.val_step_outputs['tags'], dist_=self.args.distributed, to_numpy=True).astype(int)
        y_pred = gather_tensor(self.val_step_outputs['preds'], dist_=self.args.distributed, to_numpy=True).astype(int)
        y_label = gather_tensor(self.val_step_outputs['label'], dist_=self.args.distributed, to_numpy=True).astype(int)
        y_conf = gather_tensor(self.val_step_outputs['conf'], dist_=self.args.distributed, to_numpy=True)
        y_outputs = gather_tensor(self.val_step_outputs['outputs'], dist_=self.args.distributed, to_numpy=True)

        results, best_acc, best_epo = val_stat(y_tags, y_pred, y_label, y_conf, best_novel_acc, best_epoch, self.current_epoch)

        if self.args.vis:
            visualize(y_tags, y_label, y_outputs, self.args, self.current_epoch, 'test_img')

        best_novel_acc = best_acc
        best_epoch = best_epo

        self.log_dict(results, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        print('\n')

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        feature, _ = self.encoder.forward_features(images)
        feature = feature.detach().cpu().numpy()

        return {
            'feature': feature,
            'tag': batch['tag'],
            'target': batch['target'],
            'img_path': batch['img_path'],
        }
