import argparse
import logging
import os
import random
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume, calculate_metric_percase
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts


def trainer_datasets(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # synapse_dataset 这里只用了对数据的处理以及训练时的读入，可适用于所有输入格式为npz的数据集

    from datasets.dataset_acdc import ACDC_dataset, RandomGenerator
    db_train = ACDC_dataset(base_dir=args.train_root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)
    lr_ = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    iterator = tqdm(range(max_epoch), ncols=70)
    f_loss_all = open('picture/ACDC/loss_all.csv', "w", encoding='utf-8')
    f_loss_ce = open('picture/ACDC/loss_ce.csv', "w", encoding='utf-8')
    f_loss_dice = open('picture/ACDC/loss_dice.csv', "w", encoding='utf-8')

    csv_writer_1 = csv.writer(f_loss_all)
    csv_writer_2 = csv.writer(f_loss_ce)
    csv_writer_3 = csv.writer(f_loss_dice)
    '''csv_writer_1.writerow(["epoch", "loss_all数值"])
    csv_writer_2.writerow(["epoch", "loss_ce数值"])
    csv_writer_3.writerow(["epoch", "loss_dice数值"])'''

    for epoch_num in iterator:
        loss_ce_all = 0.0
        loss_dice_all = 0.0
        loss_all = 0.0
        loss_dice = 0.0
        num = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            outputs = model(image_batch).to(device)
            #outputs = torch.sigmoid(outputs)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            if args.num_classes == 2:
                loss_dice = dice_loss(outputs, label_batch, softmax=False)
            elif args.num_classes > 2:
                loss_dice = dice_loss(outputs, label_batch, softmax=True)  # 多分类问题
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            #metric_list += np.array(metric_i)
            loss_all = loss_all+loss
            loss_ce_all = loss_ce_all+loss_ce
            loss_dice_all = loss_dice_all+loss_dice

            num = num+1
            iter_num = iter_num + 1
            #writer.add_scalar('info/lr', lr_, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                if len(outputs.shape) == 2:
                    outputs = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
                else:
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        print("第%d个epoch的学习率：%f" % (epoch_num, optimizer.param_groups[0]['lr']))
        # lr_list.append(optimizer.param_groups[0]['lr'])
        lr_.step()
        csv_writer_1.writerow([epoch_num, float(loss_all/num)])
        csv_writer_2.writerow([epoch_num, float(loss_ce_all/num)])
        csv_writer_3.writerow([epoch_num, float(loss_dice_all/num)])
        writer.add_scalar('info/total_loss', loss_all/num, epoch_num)
        writer.add_scalar('info/loss_ce', loss_ce_all/num, epoch_num)
        writer.add_scalar('info/loss_dice', loss_dice_all/num, epoch_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    f_loss_all.close()
    f_loss_ce.close()
    f_loss_dice.close()
    writer.close()
    return "Training Finished!"