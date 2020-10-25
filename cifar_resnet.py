import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from models.resnet_cifar import resnet16_cifar, resnet20_cifar, spresnet15_cifar, spresnet18_cifar, spresnet21_cifar
from train_utils import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=160, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
parser.add_argument('-rn', '--run-num', default='1', type=int, help='number of experiments(default 1)')
parser.add_argument('--pool', default='regular', type=str, help='pooling method', choices=['regular', 'hartley', 'cosine'])
parser.add_argument('-nw', '--num-workers', default=0, type=int, help='number of workers')

if __name__=='__main__':
    global args, device
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_prec = np.zeros(args.run_num + 1) # the last one is average acc
    run_time = np.zeros(args.run_num)
    for i in range(1, args.run_num+1):
        # Model building
        print('=>Run %i Building model...' % i)

        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !
        if args.pool == 'regular':
            model = resnet20_cifar()
        else:
            model = spresnet15_cifar(s_pool=args.pool)

        print('#Parameters: %d' % count_params(model))
        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = os.path.join('result/cifar/resnet', args.pool)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        model_type = 2

        model = model.to(device)
        if device == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint "{}"'.format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec[i-1] = checkpoint['best_prec']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # Data loading and preprocessing
        # CIFAR10
        if args.cifar_type == 10:
            print('=> loading cifar10 data...')
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

            train_dataset = torchvision.datasets.CIFAR10(
                root='./data/cifar', 
                train=True, 
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            test_dataset = torchvision.datasets.CIFAR10(
                root='./data/cifar',
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        # CIFAR100
        else:
            print('=> loading cifar100 data...')
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

            train_dataset = torchvision.datasets.CIFAR100(
                root='./data/cifar',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

            test_dataset = torchvision.datasets.CIFAR100(
                root='./data/cifar',
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args.num_workers)

        time_start = time.time()
        if args.evaluate:
            validate(testloader, model, criterion, args.print_freq)
        else:

            train_logger = []
            val_logger = []    
            for epoch in range(args.start_epoch, args.epochs):
                adjust_learning_rate(optimizer, epoch, model_type, args.lr)

                # train for one epoch
                epoch_logger = train(trainloader, model, criterion, optimizer, epoch, args.print_freq)
                train_logger = np.append(train_logger, epoch_logger)

                # evaluate on test set
                prec, epoch_logger = validate(testloader, model, criterion, args.print_freq)
                val_logger = np.append(val_logger, epoch_logger)

                # remember best precision and save checkpoint
                is_best = prec > best_prec[i-1]
                best_prec[i-1] = max(prec,best_prec[i-1])
                time_epoch_end = time.time()
                print('Best Prec: {:.3f}%, Time: {:.1f}'.format(best_prec[i-1], time_epoch_end-time_start))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec': best_prec[i-1],
                    'optimizer': optimizer.state_dict(),
                }, is_best, fdir)
            np.savetxt(os.path.join(fdir, 'TrainLog'+str(i)+'.out'), train_logger, delimiter=',')
            np.savetxt(os.path.join(fdir, 'ValLog'+str(i)+'.out'), val_logger, delimiter=',') 
            np.savetxt(os.path.join(fdir, 'bestprec.out'), best_prec, delimiter=',')
        time_end = time.time()
        run_time[i-1] = time_end - time_start
    avg_best_prec = np.mean(best_prec[:args.run_num])
    best_prec[args.run_num] = avg_best_prec
    np.savetxt(os.path.join(fdir, 'bestprec.out'), best_prec, delimiter=',') 
    avg_time = np.mean(run_time)
    print("Average best prediction acc: %.3f, running time: %.1f" % (avg_best_prec, avg_time) )              