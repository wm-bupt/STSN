import sys, torch, argparse, os, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn
import os.path as osp

import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

import util.loader.pre_process as prep
from util.loader.data_list import ImageList
from util.loss import VGGLoss, VGGLoss_for_trans, DANN
from util.eval import image_classification_test
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models, CheckpointManager
from model.model import SharedEncoder, PrivateEncoder, PrivateDecoder, Discriminator, DomainClassifier


# Hyper-parameters
CUDA_DIVICE_ID = '0'

parser = argparse.ArgumentParser(description='STSN')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='./log', help='the path to where you save plots and logs.')
parser.add_argument('--gen_img_dir', type=str, default='./generated_imgs', help='the path to where you save transformed images.')
parser.add_argument('--src_tr', type=str, default="./data/handprint_train.txt", help="source train dataset path list")
parser.add_argument('--src_te', type=str, default="./data/handprint_test.txt", help="source test dataset path list")
parser.add_argument('--tgt_tr', type=str, default="./data/scan_train.txt", help="target train dataset path list")
parser.add_argument('--tgt_te', type=str, default="./data/scan_test.txt", help="target test dataset path list")
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--num_steps', type=int, default=250000, help='max number of training step.')
parser.add_argument('--learning_rate_enc', type=float, default=2.5e-4, help='learning rate of encoder.')
parser.add_argument('--learning_rate_d', type=float, default=1e-4, help='learning rate of feature level discriminator.')
parser.add_argument('--learning_rate_rec', type=float, default=1e-3, help='learning rate of generator.')
parser.add_argument('--learning_rate_dis', type=float, default=1e-4, help='learning rate of image level discriminator.')
parser.add_argument('--private_code_size', type=str, default=8, help='the dimension of texture features.')
parser.add_argument('--num_classes', type=int, default=241, help='the number of classes of oracle characters.')
parser.add_argument('--power', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
parser.add_argument('--gpu', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.gen_img_dir):
    os.makedirs(args.gen_img_dir)

if args.dump_logs == True:
	old_output = sys.stdout
	sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

# Setup Augmentations
prep_train = prep.image_train(resize_size=256, crop_size=224, alexnet=False, transform_our=0)
prep_target_train = prep.image_train(resize_size=224)
prep_test = prep.image_test(resize_size=256, crop_size=224, alexnet=False, transform_our=0)
prep_target_test = prep.image_train(resize_size=224)

# ==== DataLoader ====
source_train_set = ImageList(open(args.src_tr).readlines(), transform=prep_train)
source_train_loader = torch_data.DataLoader(source_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

target_train_set = ImageList(open(args.tgt_tr).readlines(), transform=prep_train)
target_train_loader = torch_data.DataLoader(target_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

source_test_set = ImageList(open(args.src_te).readlines(), transform=prep_test)
source_test_loader = torch_data.DataLoader(source_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

target_test_set = ImageList(open(args.tgt_te).readlines(), transform=prep_test)
target_test_loader = torch_data.DataLoader(target_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

len_train_source = len(source_train_loader)
len_train_target = len(target_train_loader)

# Setup Model
print ('building models ...')
enc_shared = SharedEncoder(resnet_name="ResNet18").cuda()
shared_code_channels = enc_shared.output_num()
dis_f      = DomainClassifier(enc_shared.output_num(), 1024).cuda()
enc_s      = PrivateEncoder(64, args.private_code_size).cuda()
enc_t      = PrivateEncoder(64, args.private_code_size).cuda()
dec_s      = PrivateDecoder(shared_code_channels, args.private_code_size).cuda()
dec_t      = dec_s
dis_s2t    = Discriminator().cuda()
dis_t2s    = Discriminator().cuda()


enc_shared_opt = optim.SGD(enc_shared.optim_parameters(args.learning_rate_enc), lr=args.learning_rate_enc, momentum=0.9, weight_decay=args.weight_decay)
dis_f_opt = optim.Adam(dis_f.parameters(), lr=args.learning_rate_d, betas=(0.9, 0.99))

enc_s_opt = optim.Adam(enc_s.parameters(), lr=args.learning_rate_rec, betas=(0.5, 0.999))
enc_t_opt = optim.Adam(enc_t.parameters(), lr=args.learning_rate_rec, betas=(0.5, 0.999))
dec_s_opt = optim.Adam(dec_s.parameters(), lr=args.learning_rate_rec, betas=(0.5, 0.999))
dec_t_opt = optim.Adam(dec_t.parameters(), lr=args.learning_rate_rec, betas=(0.5, 0.999))
dis_s2t_opt = optim.Adam(dis_s2t.parameters(), lr=args.learning_rate_dis, betas=(0.5, 0.999))
dis_t2s_opt = optim.Adam(dis_t2s.parameters(), lr=args.learning_rate_dis, betas=(0.5, 0.999))

manager = CheckpointManager(logs_dir=args.gen_img_dir, enc_shared = enc_shared, enc_s = enc_s, enc_t = enc_t, dec_s = dec_s)

enc_opt_list  = []
dclf_opt_list = []
rec_opt_list  = []
dis_opt_list  = []

# Optimizer list for quickly adjusting learning rate
enc_opt_list.append(enc_shared_opt)
dclf_opt_list.append(dis_f_opt)
rec_opt_list.append(enc_s_opt)
rec_opt_list.append(enc_t_opt)
rec_opt_list.append(dec_s_opt)
rec_opt_list.append(dec_t_opt)
dis_opt_list.append(dis_s2t_opt)
dis_opt_list.append(dis_t2s_opt)

cudnn.enabled   = True
cudnn.benchmark = True

mse_loss = nn.MSELoss(size_average=True).cuda()
bce_loss = nn.BCEWithLogitsLoss().cuda()
cls_loss  = nn.CrossEntropyLoss().cuda()
VGG_loss = VGGLoss()
VGG_loss_for_trans = VGGLoss_for_trans()


dis_f.train()
enc_shared.train()
enc_s.train()
enc_t.train()
dec_s.train()
dec_t.train()
dis_s2t.train()
dis_t2s.train()

true_label, fake_label, best_acc = 1, 0, 0.0
train_D_t2s = train_transfer = train_cross_entorpy = train_rec = train_transform = train_per = 0.0
for i_iter in range(args.num_steps):
    sys.stdout.flush()

    enc_shared.train()
    adjust_learning_rate(enc_opt_list , base_lr=args.learning_rate_enc, i_iter=i_iter, max_iter=args.num_steps, power=args.power)
    adjust_learning_rate(dclf_opt_list, base_lr=args.learning_rate_d  , i_iter=i_iter, max_iter=args.num_steps, power=args.power)
    adjust_learning_rate(rec_opt_list , base_lr=args.learning_rate_rec, i_iter=i_iter, max_iter=args.num_steps, power=args.power)
    adjust_learning_rate(dis_opt_list , base_lr=args.learning_rate_dis, i_iter=i_iter, max_iter=args.num_steps, power=args.power)

    # ==== sample data ====
    if i_iter % len_train_source == 0:
        iter_source = iter(source_train_loader)
    if i_iter % len_train_target == 0:
        iter_target = iter(target_train_loader)
    source_data, source_label = iter_source.next()
    target_data, target_label = iter_target.next()

    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()
    tlabelv = Variable(target_label)

    # forwarding
    low_s, code_s_common, s_pred1, s_pred2 = enc_shared(sdatav)
    low_t, code_t_common, t_pred1, t_pred2  = enc_shared(tdatav)
    code_s_private    = enc_s(low_s)
    code_t_private    = enc_t(low_t)

    rec_s   = dec_s(code_s_common, code_s_private, 0)
    rec_t   = dec_t(code_t_common, code_t_private, 1)
    rec_t2s = dec_s(code_t_common, code_s_private, 0)
    rec_s2t = dec_t(code_s_common, code_t_private, 1)

    for p in dis_f.parameters():
        p.requires_grad = True
    for p in dis_s2t.parameters():
        p.requires_grad = True
    for p in dis_t2s.parameters():
        p.requires_grad = True

    # train feature level discriminator
    prob_dis_f_real = dis_f(s_pred1.detach())
    prob_dis_f_fake = dis_f(t_pred1.detach())
    loss_dis_f = bce_loss(prob_dis_f_real, Variable(torch.FloatTensor(prob_dis_f_real.data.size()).fill_(true_label)).cuda()).cuda() \
                     + bce_loss(prob_dis_f_fake, Variable(torch.FloatTensor(prob_dis_f_fake.data.size()).fill_(fake_label)).cuda()).cuda()
    if i_iter%1 == 0:
        dis_f_opt.zero_grad()
        loss_dis_f.backward()
        dis_f_opt.step()

    # train image level discriminator -> LSGAN
    # ===== dis_s2t =====
    if i_iter%5 == 0:
        prob_dis_s2t_real1 = dis_s2t(tdatav)
        prob_dis_s2t_fake1 = dis_s2t(rec_s2t.detach())
        loss_d_s2t = 0.5* mse_loss(prob_dis_s2t_real1, Variable(torch.FloatTensor(prob_dis_s2t_real1.data.size()).fill_(true_label).cuda())).cuda() \
                   + 0.5* mse_loss(prob_dis_s2t_fake1, Variable(torch.FloatTensor(prob_dis_s2t_fake1.data.size()).fill_(fake_label).cuda())).cuda()
        dis_s2t_opt.zero_grad()
        loss_d_s2t.backward()
        dis_s2t_opt.step()

        train_D_t2s += loss_d_s2t.item()

    # ===== dis_t2s =====
    if i_iter%5 == 0:
        prob_dis_t2s_real1 = dis_t2s(sdatav)
        prob_dis_t2s_fake1 = dis_t2s(rec_t2s.detach())
        loss_d_t2s = 0.5* mse_loss(prob_dis_t2s_real1, Variable(torch.FloatTensor(prob_dis_t2s_real1.data.size()).fill_(true_label).cuda())).cuda() \
                   + 0.5* mse_loss(prob_dis_t2s_fake1, Variable(torch.FloatTensor(prob_dis_t2s_fake1.data.size()).fill_(fake_label).cuda())).cuda()
        dis_t2s_opt.zero_grad()
        loss_d_t2s.backward()
        dis_t2s_opt.step()

        train_D_t2s += loss_d_t2s.item()

    for p in dis_f.parameters():
        p.requires_grad = False
    for p in dis_s2t.parameters():
        p.requires_grad = False
    for p in dis_t2s.parameters():
        p.requires_grad = False

    # ==== reconstruction and perpetual loss ====
    loss_rec_s = VGG_loss(rec_s, sdatav)
    loss_rec_t = VGG_loss(rec_t, tdatav)
    loss_rec_self = loss_rec_s + loss_rec_t

    loss_per_s2t = VGG_loss_for_trans(rec_s2t, sdatav, tdatav, weights=[0, 0, 0, 1.0/4, 1.0])
    loss_per_t2s = VGG_loss_for_trans(rec_t2s, tdatav, sdatav, weights=[0, 0, 0, 1.0/4, 1.0])
    loss_per_tran = loss_per_s2t + loss_per_t2s

    # ==== feature adversarial loss ====
    prob_dis_f_fake2 = dis_f(t_pred1)
    loss_feat_similarity = bce_loss(prob_dis_f_fake2, Variable(torch.FloatTensor(prob_dis_f_fake2.data.size()).fill_(true_label)).cuda())

    # ==== image adversarial loss ====
    prob_dis_s2t_fake2 = dis_s2t(rec_s2t)
    loss_gen_s2t = mse_loss(prob_dis_s2t_fake2, Variable(torch.FloatTensor(prob_dis_s2t_fake2.data.size()).fill_(true_label)).cuda()) \

    prob_dis_t2s_fake2 = dis_t2s(rec_t2s)
    loss_gen_t2s = mse_loss(prob_dis_t2s_fake2, Variable(torch.FloatTensor(prob_dis_t2s_fake2.data.size()).fill_(true_label)).cuda()) \

    loss_image_transform = loss_gen_s2t + loss_gen_t2s

    # ==== classification loss ====
    loss_sim_cls = cls_loss(s_pred2, slabelv)

    # ==== transformed classification====
    if i_iter >= 0:
        _, _, _, s2t_pred2 = enc_shared(rec_s2t.detach())
        loss_sim_cls += cls_loss(s2t_pred2, slabelv)


    total_loss = \
              1.0 * loss_sim_cls \
            + 1.0 * loss_feat_similarity \
            + 0.5 * loss_rec_self \
            + 0.01* loss_image_transform \
            + 0.05 * loss_per_tran \

    enc_shared_opt.zero_grad()
    enc_s_opt.zero_grad()
    enc_t_opt.zero_grad()
    dec_s_opt.zero_grad()

    total_loss.backward()

    enc_shared_opt.step()
    enc_s_opt.step()
    enc_t_opt.step()
    dec_s_opt.step()

    train_transfer += 1.0 * loss_feat_similarity.item()
    train_cross_entorpy += 1.0 * loss_sim_cls.item()
    train_rec += 0.5 * loss_rec_self.item()
    train_transform += 0.01* loss_image_transform.item()
    train_per += 0.05 * loss_per_tran.item()

    if i_iter % 30 == 0:
        print(
            "Iter {:05d}, Cls: {:.4f}; Rec_vgg: {:.4f}; Per_vgg: {:.4f}; Img_tran_G: {:.4f}; Img_tran_D: {:.4f}; Share_fea_D: {:.4f}".format(
                i_iter, train_cross_entorpy / float(30), train_rec / float(30), train_per / float(30),
                train_transform / float(30), train_D_t2s / float(6), train_transfer / float(30)))
        train_D_t2s = train_transfer = train_cross_entorpy = train_rec = train_transform = train_per =  0.0


    if i_iter%500 == 0 :
        imgs_s = torch.cat(((sdatav[:2,[2, 1, 0],:,:].cpu()+1)/2, (rec_s[:2,[2, 1, 0],:,:].cpu()+1)/2, (rec_s2t[:2,[2, 1, 0],:,:].cpu()+1)/2, ), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=args.batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s*255,0,255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1,2,0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))

        imgs_t = torch.cat(((tdatav[:2,[2, 1, 0],:,:].cpu()+1)/2, (rec_t[:2,[2, 1, 0],:,:].cpu()+1)/2, (rec_t2s[:2,[2, 1, 0],:,:].cpu()+1)/2, ), 0)
        imgs_t = vutils.make_grid(imgs_t.data, nrow=args.batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_t = np.clip(imgs_t*255,0,255).astype(np.uint8)
        imgs_t = imgs_t.transpose(1,2,0)
        imgs_t = Image.fromarray(imgs_t)
        filename = '%05d_target.jpg' % i_iter
        imgs_t.save(os.path.join(args.gen_img_dir, filename))

    if i_iter % 300 == 0:
        enc_shared.eval()
        print ('evaluating models when %d iter...'%i_iter)
        temp_acc = image_classification_test(source_test_loader, enc_shared)
        print('test_source_acc:%.4f' % (temp_acc))

        temp_acc = image_classification_test(target_test_loader, enc_shared)
        if temp_acc > best_acc:
            best_acc = temp_acc
            manager.save(epoch=i_iter, fpath=osp.join(args.gen_img_dir, 'checkpoint_max.pth.tar'))
            print('save checkpoint of iteration {:d} when acc1 = {:4.1%}'.format(i_iter, best_acc))
        print('test_target_acc:%.4f' % (temp_acc))
        print('max_acc: {:4.1%}'.format(best_acc))

    if i_iter % 5000 == 0 and i_iter != 0:
        manager.save(epoch=i_iter)
