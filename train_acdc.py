import argparse
import logging
import os
import random
import numpy as np
import torch
import Config as config
import torch.backends.cudnn as cudnn
from networks.FSCANet import FSCANet
import logging
from trainer import trainer_datasets


parser = argparse.ArgumentParser()

parser.add_argument('--train_root_path', type=str,
                    default='../ACDC_preprocess/train_npz', help='train root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='lists/lists_ACDC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='train_log channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'ACDC': {
            'train_root_path': '../ACDC_preprocess/train_npz',
            'list_dir': 'lists/lists_ACDC',
            'num_classes': 4,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.train_root_path = dataset_config[dataset_name]['train_root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False
    args.exp = 'ee++_' + dataset_name + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'ee++')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed != 1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = config.get_CTranS_config()
    device = torch.device("cuda:2")
    net = FSCANet(config_vit, n_channels=1, n_classes=args.num_classes, img_size=args.img_size).to(device)
    names = []
    for n, v in net.named_parameters():
         names.append(n)
    print(names)
    #print(net)
    input = torch.randn(1, 1, 224, 224).to(device)
    from thop import profile
    flops, params = profile(net, inputs=(input,))
    print('flops:{}(G)'.format(flops / (1000 ** 3)))
    print('params:{}(M)'.format(params / (1000 ** 2)))
    # pretrained_UNet_model_path = "./pre_model/epoch_149_1234.pth"
    # pretrained_UNet = torch.load(pretrained_UNet_model_path, map_location='cuda:1')
    # # pretrained_UNet = pretrained_UNet['state_dict']
    # if 'state_dict' not in pretrained_UNet:
    #     pretrained_UNet = pretrained_UNet
    # else:
    #     pretrained_UNet = pretrained_UNet['state_dict']
    # model2_dict = net.state_dict()
    # state_dict = {k: v for k, v in pretrained_UNet.items() if k in model2_dict.keys()}
    # print(state_dict.keys())
    # model2_dict.update(state_dict)
    # net.load_state_dict(model2_dict)

    trainer = {'ACDC': trainer_datasets}
    trainer[dataset_name](args, net, snapshot_path)