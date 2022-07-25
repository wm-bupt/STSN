from __future__ import print_function, absolute_import
from collections import OrderedDict
import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import errno

def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
    lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
    if i_iter % 2000 == 0:
        print('learning rate: %f'%lr)
    for opt in opts:
        opt.param_groups[0]['lr'] = lr
        if len(opt.param_groups) > 1:
            opt.param_groups[1]['lr'] = lr * 10


def save_models(model_dict, prefix='./'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key+'.pth'))


def load_models(model_dict, prefix='./'):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(os.path.join(prefix, key+'.pth')))


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


class CheckpointManager(object):
    def __init__(self, logs_dir='./logs', **modules):
        self.logs_dir = logs_dir
        self.modules = modules

    def save(self, epoch, fpath=None, **modules):
        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            if isinstance(module, nn.DataParallel):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1

        fpath = osp.join(self.logs_dir, "checkpoint-epoch%d.pth.tar"%epoch) if fpath is None else fpath
        save_checkpoint(ckpt, fpath)

    def load(self, ckpt):
        for name, module in self.modules.items():
            missing_keys, unexpected_keys = module.load_state_dict(ckpt.get(name, {}), strict=False)
            print("Loading %s... \n"
                  "missing keys %s \n"
                  "unexpected keys %s \n" % (name, missing_keys, unexpected_keys))
        return ckpt["epoch"]
