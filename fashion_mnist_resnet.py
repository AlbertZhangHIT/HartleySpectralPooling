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
from models.resnet_mnist import resnet16_mnist, resnet20_mnist, spresnet15_mnist, spresnet21_mnist
from train_utils import *

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
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

        if args.pool == 'regular':
            model = resnet20_mnist()
        else:
            model = spresnet21_mnist(s_pool=args.pool)
        print('#Parameters: %d' % count_params(model))
        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = os.path.join('result/fashion_mnist/resnet', args.pool)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type     
        model_type = 3
            
        model = model.to(device)
        if device == 'cuda':
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), args.lr)

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


        print('=> loading Fashion-MNIST data...')

        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data/fashion-mnist', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data/fashion-mnist',
            train=False,
            download=True,
            transform=transforms.ToTensor())
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