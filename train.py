#coding:gbk

import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

from dataset import Batch_Balanced_Dataset, label_num, AlignCollate, hierarchical_dataset
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open('./saved_models/{}/log_dataset.txt'.format(opt.experiment_name), 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """部分参数初始化"""
    learning_rate = 1e-4
    label2num, num2label = label_num('all_labels.txt')
    num_classes = len(label2num)
    print('训练类别数：{}'.format(num_classes))
    print('训练集标签列表：\n{}'.format(num2label.values()))
    print('-' * 80)

    class VGGNet(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(VGGNet, self).__init__()
            net = models.vgg16(pretrained=True)
            net.classifier = nn.Sequential()
            self.features = net
            self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 128),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    #--------------------训练过程---------------------------------
    model = VGGNet()
    if torch.cuda.is_available():
        model.cuda()
    params = [{'params': md.parameters()} for md in model.children()
              if md in [model.classifier]]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    Loss_list = []
    Accuracy_list = []


    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print('continue to train, start_iter: {}'.format(start_iter))
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter
    num2label = opt.num2label
    while(True):
        # train part
        # training-----------------------------
        image_tensors, labels = train_dataset.get_batch()
        batch_x = image_tensors.to(device)
        #labels = [num2label[x] for x in labels]#将汉字转换回标签
        batch_y = torch.from_numpy(np.asarray(labels, dtype=np.int8)).to(device)
        train_loss = 0.
        train_acc = 0.

        out = model(batch_x)
        loss = loss_func(out, batch_y.long())
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 0.5e+2 == 0:
            print('Step{}:'.format(i + 1))
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                labels)), train_acc / (len(labels))))
        # save model per 1e+5 iter.
        if (i + 1) % 5e+2 == 0:
            torch.save(
                model.state_dict(), './saved_models/{}/iter_{}.pth'.format(opt.experiment_name, i+1))

        if i == opt.num_iter:
            torch.save(
                model.state_dict(), './saved_models/{}/iter_{}.pth'.format(opt.experiment_name, i+1))
            print('end the training')
            break
        i += 1
        
        # evaluation--------------------------------
        if i % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            # for log
            model.eval()
            eval_loss = 0.
            eval_acc = 0.
            length_of_data = 0
            for image_tensors, labels in valid_loader:
                batch_x = image_tensors.to(device)
                batch_y = torch.from_numpy(np.asarray(labels, dtype=np.int8)).to(device)
                length_of_data += len(labels)
                #batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
                out = model(batch_x)
                loss = loss_func(out, batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                num_correct = (pred == batch_y).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (length_of_data), eval_acc / (length_of_data)))
                
            Loss_list.append(eval_loss / (len(labels)))
            Accuracy_list.append(100 * eval_acc / (len(labels)))
        
    x1 = np.arange(0, 100).reshape(1,-1)
    x2 = np.arange(0, 100).reshape(1,-1)
    y1 = np.array(Accuracy_list).reshape(1,-1)
    y2 = np.array(Loss_list).reshape(1,-1)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")
    sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='experiment01', help='Where to store logs and models')
    parser.add_argument('--train_data', help='path to training dataset', default='Lmdb_trainset')
    parser.add_argument('--valid_data', help='path to validation dataset', default='Lmdb_valset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)#windows下只能为0
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')#batch_size大小
    parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=50, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    #parser.add_argument('--select_data', type=str, default='MJ-ST',
    #                    help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    #parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
    #                    help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=5, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=128, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', default=True, action='store_true', help='use rgb input')#使用灰度图
    #parser.add_argument('--character', type=str,
    #                    default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    label2num = {}#标签转换为编号
    num2label = {}#编号转换回标签
    with open('all_labels.txt', 'r', encoding='gbk') as data:
        datalist = data.readlines()
    nSamples = len(datalist)
    for i in range(nSamples):
        label, CN = datalist[i].strip('\n').split('\t')
        label2num[label] = CN
        num2label[CN] = label

    label_char = ''.join(list(num2label.keys()))#读取所有中文标签
    parser.add_argument('--character', type=str, default=label_char, help='character label')
    #parser.add_argument('--character', type=str,
    #                    default='0123456789abcdefghijklmnopqrstuvwxyz%&', help='character label')
    parser.add_argument('--sensitive', default=False, action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', default='False', action='store_true', help='for data_filtering_off mode')#除去其他标签

    opt = parser.parse_args()
    opt.label2num = label2num
    opt.num2label = num2label

    if not opt.experiment_name:
        #opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += '-Seed{}'.format(opt.manualSeed)
        # print(opt.experiment_name)

    os.makedirs('./saved_models/{}'.format(opt.experiment_name), exist_ok=True)
        
    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
