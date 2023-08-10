import copy
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2

import time

class Augment(object):

    def __init__(self, cfg):
        self.num_aug = cfg['num_aug']
        self.cutmix = CutMix(cfg['cutmix'])

    def __call__(self, batch):
        batch_img = batch['image']
        batch_dai = batch['daily']
        batch_gen = batch['gender']
        batch_emb = batch['embel']

        for i in range(self.num_aug):
            aug_img, aug_dai, aug_gen, aug_emb = self.cutmix(batch)
            batch_img = torch.cat([batch_img, aug_img])
            batch_dai = torch.cat([batch_dai, aug_dai])
            batch_gen = torch.cat([batch_gen, aug_gen])
            batch_emb = torch.cat([batch_emb, aug_emb])

        batch['image'] = batch_img
        batch['daily'] = batch_dai
        batch['gender'] = batch_gen
        batch['embel'] = batch_emb

        return batch
          
class CutMix(object):
    def __init__(self, cfg):
        self.beta = cfg['beta']

    def __call__(self, batch):
        src_image = batch['image']
        src_daily = batch['daily']
        src_gender = batch['gender']
        src_embel = batch['embel']

        lam = np.random.beta(self.beta, self.beta)
        rand_idx = torch.randperm(src_image.shape[0])
        bbx1, bby1, bbx2, bby2, lam = crop_bbox(src_image.shape[-2:], lam)

        tgt_image = src_image.clone()
        tgt_image[:, :, bby1:bby2, bbx1:bbx2] = src_image[rand_idx, :, bby1:bby2, bbx1:bbx2]
        tgt_daily = src_daily * (1 - lam) + src_daily[rand_idx] * lam
        tgt_gender = src_gender * (1 - lam) +  src_gender[rand_idx] * lam
        tgt_embel = src_embel * (1 - lam) + src_embel[rand_idx] * lam

        return tgt_image, tgt_daily, tgt_gender, tgt_embel

def crop_bbox(size, lam):
    h, w = size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = np.int(h * cut_ratio)
    cut_w = np.int(w * cut_ratio)
    bby1 = np.random.randint(h - cut_h); bby2 = bby1 + cut_h
    bbx1 = np.random.randint(w - cut_w); bbx2 = bbx1 + cut_w
    lam = (cut_h * cut_w) / (h * w)
    return bbx1, bby1, bbx2, bby2, lam

    