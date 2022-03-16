import argparse, os, torch, random, pdb, math, sys
import os.path as osp
from torch.backends import cudnn
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
sys.path.append(osp.join(osp.dirname(__file__), 'util_base'))
import network, loss, lr_schedule, data_list_oracle
import pre_process as prep
from data_list_oracle import ImageList
from serialization import load_checkpoint, CheckpointManager

def image_classification_test(loader, model, data_type):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader[data_type])
        for i in range(len(loader[data_type])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source_train"]["batch_size"]
    test_bs = data_config["source_test"]["batch_size"]
    dsets["source_train"] = ImageList(open(data_config["source_train"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target_train"] = ImageList(open(data_config["target_train"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target_train"] = DataLoader(dsets["target_train"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["source_test"] = ImageList(open(data_config["source_test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)
    dsets["target_test"] = ImageList(open(data_config["target_test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["transfer"]:
        if config["loss"]["random"]:
            random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
            ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
        else:
            random_layer = None
            if config['method']  == 'DANN':
                ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
            else:
                ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)

        if config["loss"]["random"]:
            random_layer.cuda()
        ad_net = ad_net.cuda()
        parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    else:
        parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(config['gpu'])
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_network = nn.DataParallel(base_network).to(device)
    if config["transfer"]:
        ad_net = nn.DataParallel(ad_net).to(device)
        manager = CheckpointManager(logs_dir=config["output_path"], model=base_network, ad_net=ad_net)
    else:
        manager = CheckpointManager(logs_dir=config["output_path"], model=base_network)


    ## train
    len_train_source = len(dset_loaders["source_train"])
    len_train_target = len(dset_loaders["target_train"])
    train_cross_loss = train_transfer_loss = train_total_loss = 0.0

    best_acc = 0.0
    print("start training ...")
    for i in range(config["num_iterations"]):
        if i % 300 == 0:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network, data_type = 'source_test')
            print('source_acc:%.4f' % (temp_acc))

            temp_acc = image_classification_test(dset_loaders, base_network, data_type = 'target_test')
            print('target_acc:%.4f' % (temp_acc))

            if temp_acc > best_acc:
                best_acc = temp_acc
                manager.save(epoch=i, fpath=osp.join(config["output_path"], 'checkpoint_max.pth.tar'))
                print('save checkpoint of iteration {:d} when acc1 = {:4.1%}'.format(i, best_acc))
            print('max_target_acc: {:4.1%}'.format(best_acc))
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            # print(log_str)
        if i % 10000 == 0 and i != 0:
            manager.save(epoch=i)


        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        if config["transfer"]:
            ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source_train"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target_train"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        if config["transfer"]:
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            if config['method'] == 'CDAN+E':
                entropy = loss.Entropy(softmax_out)
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
            elif config['method']  == 'CDAN':
                transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
            elif config['method']  == 'DANN':
                transfer_loss = loss.DANN(features, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')

            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        else:
            total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        if config["transfer"]:
            train_transfer_loss += transfer_loss.item()
        train_total_loss += total_loss.item()
        if i % config["test_interval"]  == 0:
            print(
            "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f}; Average Training Loss: {:.4f}".format(
                i, train_cross_loss / float(config["test_interval"]), train_transfer_loss / float(config["test_interval"]),
                          train_total_loss / float(config["test_interval"])))
            train_cross_loss = train_transfer_loss = train_total_loss = 0.0
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet18', help="network")
    parser.add_argument('--src_tr', type=str, default="./data/handprint_train.txt", help="The source dataset path list")
    parser.add_argument('--src_te', type=str, default="./data/handprint_test.txt", help="The target dataset path list")
    parser.add_argument('--tgt_tr', type=str, default="./data/scan_train.txt", help="The source dataset path list")
    parser.add_argument('--tgt_te', type=str, default="./data/scan_test.txt", help="The target dataset path list")

    parser.add_argument('--test_interval', type=int, default=50, help="interval of two continuous test phase")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--transfer', type=int, default=1, choices=[0,1], help="whether use DA")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 50004 #100004
    config["test_interval"] = args.test_interval
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {'params':{"resize_size":256, "crop_size":224}}
    config["loss"] = {"trade_off":1.0}

    config["network"] = {"name":network.ResNetFc, "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["data"] = {"source_train":{"list_path":args.src_tr, "batch_size":36}, \
                      "target_train":{"list_path":args.tgt_tr, "batch_size":36}, \
                      "target_test":{"list_path":args.tgt_te, "batch_size":128}, \
                      "source_test":{"list_path":args.src_te, "batch_size":128}}

    # config["optimizer"]["lr_param"]["lr"] = args.lr # optimal parameters
    config["network"]["params"]["class_num"] = 241
    config["transfer"] = args.transfer
    config["out_file"].write(str(config))
    config["out_file"].flush()
    print(str(config))
    train(config)
