import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import Config as config
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.FSCANet import FSCANet
from datasets.dataset_acdc import ACDC_dataset
from utils import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../ACDC_preprocess/test_vol_h5', help='root dir for validation volume data')  # for hippocampus volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Chaos', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Chaos', help='list dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True,action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:1', help='gpu device')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=8)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    all_nii_dict = {}
    count_dict = {}
    count = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing, device=args.device)
    #     if metric_i[0] != (0, 0):
    #             metric_list += np.array(metric_i)
    #             count += 1
    #             logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
    #             i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    #
    #             case_name_key = str(case_name).split('_')[0]
    #             if case_name_key not in all_nii_dict.keys() and case_name_key not in count_dict.keys():
    #                 all_nii_dict.setdefault(case_name_key, np.array(metric_i))
    #                 # print(all_nii_dict[case_name_key])
    #                 count_dict.setdefault(case_name_key, 1)
    #                 # print(count_dict[case_name_key])
    #             else:
    #                 old_metric = all_nii_dict[case_name_key]
    #                 all_nii_dict[case_name_key] = old_metric + np.array(metric_i)
    #                 # print(all_nii_dict[case_name_key])
    #                 old_count = int(count_dict[case_name_key])
    #                 count_dict[case_name_key] = old_count + 1
    #                 # print(count_dict[case_name_key])
    #
    # metric_list = metric_list / count
    #
    # for i, j in all_nii_dict.items():
    #     count = int(count_dict[i])
    #     j = j / count
    #     logging.info('nii_name %s mean_dice %f mean_hd95 %f' % (i, j[0][0], j[0][1]))
    #
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # return "Testing Finished!"
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



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

    dataset_config = {
        'Chaos': {
            'Dataset': ACDC_dataset,
            'volume_path': '../ACDC_preprocess/test_vol_h5',
            'list_dir': './lists/lists_Chaos',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = False

    # name the same snapshot defined in train script!
    args.exp = 'ee++_' + dataset_name + str(args.img_size)
    snapshot_path = "model/{}/{}".format(args.exp, 'ee++')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = config.get_CTranS_config()
    device = torch.device("cuda:1")
    net = FSCANet(config_vit, n_channels=1, n_classes=args.num_classes,img_size=args.img_size).to(device)

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)

    else:
        test_save_path = None
    inference(args, net, test_save_path)


