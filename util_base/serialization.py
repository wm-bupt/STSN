from __future__ import print_function, absolute_import
import json
import os.path as osp
import shutil

import torch
import torch.nn as nn
from torch.nn import Parameter
from osutils import mkdir_if_missing


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))



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
