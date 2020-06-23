from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2

import torch
import numpy as np
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def get_ra_value(hm, wh, ind, ra_scale):
    ys = ind / hm.shape[3]
    xs = ind % hm.shape[3]
    half_ra_w = (wh*ra_scale/2)[:,:,0:1].type(torch.IntTensor)
    half_ra_h = (wh*ra_scale/2)[:,:,1:2].type(torch.IntTensor)
    ras = np.array([])
    for i in range(ind.shape[1]):
      min_ra_y = ys[0][i].item() - half_ra_h[0][i][0].item()
      min_ra_y = max(0, min(min_ra_y, hm.shape[2]))

      max_ra_y = ys[0][i].item() + half_ra_h[0][i][0].item()
      max_ra_y = max(0, min(max_ra_y, hm.shape[2])) + 1

      min_ra_x = xs[0][i].item() - half_ra_w[0][i][0].item()
      min_ra_x = max(0, min(min_ra_x, hm.shape[3]))

      max_ra_x = xs[0][i].item() + half_ra_w[0][i][0].item()
      max_ra_x = max(0, min(max_ra_x, hm.shape[3])) + 1
      ra = hm[:, :, min_ra_y:max_ra_y, min_ra_x:max_ra_x]
      mean = ra.mean().item()
      ras = np.append(ras, mean)
    return torch.from_numpy(ras[None,:,None]).type(torch.cuda.FloatTensor)

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)
