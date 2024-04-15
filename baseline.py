# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
import platform
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import PolynomialLR
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import functional as F
from utils.utils import get_model, get_optimizer, get_transforms, get_dataset, print_header, get_unique_filename
from optimizers import *

from torch.optim.swa_utils import AveragedModel

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpu', default='0', type=str,help='GPU number')
parser.add_argument('--log', action='store_false', default=True, help='save log')
parser.add_argument('--batch_size', default=128, type=int,help='minibatch size')
parser.add_argument('--num_worker', default=8, type=int, help='number of workers')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epoch', default=200, type=int,help='epoch')
parser.add_argument('--lr', default=0.1, type=float,help='learning rate')
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--weight_decay', default=5e-3, type=float)
parser.add_argument('--optimizer',default='SGD',type=str,help='optimizer')
parser.add_argument('--model',default='ResNet18',type=str,help='model')
parser.add_argument('--milestones', default=[60, 120, 160], nargs='*', type=int, help='milestones of scheduler')
parser.add_argument('--dataset',default='CIFAR10',type=str)
parser.add_argument('--lr_decay',default=0.2,type=float)
parser.add_argument('--lr_type',default='MultiStepLR',type=str,help='learning rate scheduler')
parser.add_argument('--eta_min',default=0.00,type=float,help='minimum lerning rate')
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--warmup_start_lr', default=0.01, type=float)
parser.add_argument('--power', default=1.0, type=float)
parser.add_argument('--train_policy', default='policy1', type=str, help='train policy')
parser.add_argument('--test_policy', default='no_policy', type=str, help='test policy')
# for Averaging
parser.add_argument('--start_averaged', default=160, type=int)
# save model
parser.add_argument('--saving-folder', default='checkpoints/', type=str, help='choose saving name')
parser.add_argument('--save_model', action='store_true', default=False, help='save model')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

today = datetime.now(timezone(timedelta(hours=+9))).strftime("%Y-%m-%d")
now = datetime.now(timezone(timedelta(hours=+9))).strftime("%H%M")

# 出力先を標準出力orファイルで選択
if args.log is True:
    log_dir = f"./logs/{args.dataset}/{args.model}/{today}/{args.optimizer}"
    os.makedirs(log_dir, exist_ok=True)
    logpath = log_dir+f"/{now}-{args.epoch}-{args.lr_type}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}.log"
    logpath = get_unique_filename(logpath)
    sys.stdout = open(logpath, "w") # 表示内容の出力をファイルに変更

print(' '.join(sys.argv))
print('date:', today, 'time:', now)
print('python:', platform.python_version(), ' torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('GPU count',torch.cuda.device_count())
print('epoch:', args.epoch)
print('start_averaged:',args.start_averaged)
print('model:',args.model)
print('dataset:',args.dataset)
print('batch_size:', args.batch_size)
print('optimizer:',args.optimizer)
print('learning_rate:', args.lr)
print('lr_decay:',args.lr_decay)
print('lr_type:',args.lr_type)
print('milestones:', args.milestones)
print('weight_decay:',args.weight_decay)
print('momentum:', args.momentum)
print('eta_min:',args.eta_min)
print('train_policy:',args.train_policy)
print('test_policy:',args.test_policy)

use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

#define dataset and transform
transform_train, transform_test = get_transforms(args.dataset, args.train_policy, args.test_policy)
print('train_transform:', transform_train)
print('test_transform:', transform_test)

trainset, testset = get_dataset(args.dataset, transform_train, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_worker, pin_memory=True)

num_class = len(trainset.classes)

#define model
print('==> Building model..')
model = get_model(args.model, num_class, args.dataset)
model = torch.nn.DataParallel(model)
averaged_model = AveragedModel(model)
if use_cuda:
    torch.cuda.empty_cache()
    model.to(device)
    averaged_model.to(device)
    cudnn.benchmark = True
print('==> Finish model')

#lr decay milestones
optimizer = get_optimizer(args, model)
if args.lr_type == 'MultiStepLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, milestones=args.milestones, gamma=args.lr_decay)
elif args.lr_type == 'CosineAnnealingLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, T_max = args.epoch, eta_min = args.eta_min)
elif args.lr_type == 'WarmupPolynomialLR':
    scheduler = WarmupPolynomialLR(optimizer, total_iters=args.epoch, warmup_epochs=args.warmup_epochs, 
                                    warmup_start_lr=args.warmup_start_lr, power=args.power)
elif args.lr_type == "LinearWarmupCosineAnnealingLR":
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, 
                                              max_epochs=args.epoch, warmup_start_lr=args.warmup_start_lr, eta_min=args.eta_min)

criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    torch.cuda.synchronize()
    time_ep = time.time()
    model.train()
    averaged_model.train()
    train_loss = 0.0
    correct = 0   
    total = 0

    for batch_idx, (input, target) in enumerate(trainloader):
        input_size = input.size()[0]
        input, target = input.to(device), target.to(device)
        
        output = model(input)
        total += input_size
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update Averaged model
        if epoch >= args.start_averaged:
            averaged_model.update_parameters(model)

        with torch.no_grad():
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += pred.eq(target).sum()

    if epoch >= args.start_averaged:
        torch.optim.swa_utils.update_bn(trainloader, averaged_model, optimizer.param_groups[0]["params"][0].device)
    
    torch.cuda.synchronize()   
    time_ep = time.time() - time_ep

    return train_loss/batch_idx, 100*correct/total, time_ep

def test(epoch, model):
    model.eval()
    averaged_model.eval()
    total = 0

    test_loss = 0.0
    test_correct = 0
    ave_test_loss = 0.0
    ave_test_correct = 0.0
    
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input_size = input.size()[0]
            input, target = input.to(device), target.to(device)   
            # normal test
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            test_correct += pred.eq(target).sum()
            # averaged test
            if epoch >= args.start_averaged:
                ave_output = averaged_model(input)
                loss = criterion(ave_output, target)
                ave_test_loss += loss.item()
                _, pred = torch.max(ave_output, 1)
                ave_test_correct += pred.eq(target).sum()
            total += input_size
    return test_loss/batch_idx, 100*test_correct/total, ave_test_loss/batch_idx, 100*ave_test_correct/total

total_time = 0
for epoch in range(args.start_epoch, args.epoch):
    if epoch == 0:
        print_header()
    train_loss, train_acc, time_ep = train(epoch)
    total_time += time_ep
    if not args.lr_type == "fixed":
        scheduler.step()
    lr_ = optimizer.param_groups[0]['lr']
    test_loss, test_acc, ave_loss, ave_acc = test(epoch, model)

    print(f"{epoch+1:^10d} | {lr_:^10.4f} | {time_ep:^10.3f} | {train_loss:^10.4f} | {train_acc:^9.2f}% | "\
      f"{test_loss:^10.4f} | {test_acc:^9.2f}% | {ave_loss:^10.4f} | {ave_acc:^9.2f}%")

if args.save_model:
    archive_base = f'{args.saving_folder}{args.dataset}/{args.model}/{args.optimizer}/Base'
    os.makedirs(archive_base, exist_ok=True)
    archive_base = f'{archive_base}/{args.epoch}-{args.lr_type}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}.pkl'
    archive_base = get_unique_filename(archive_base)
    torch.save(model.state_dict(), archive_base)
    print('Base model save to', archive_base)

    archive_ave = f'{args.saving_folder}{args.dataset}/{args.model}/{args.optimizer}/Averaged'
    os.makedirs(archive_ave, exist_ok=True)
    archive_ave = f'{archive_ave}/{args.epoch}-{args.lr_type}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}_ave.pkl'
    archive_ave = get_unique_filename(archive_ave)
    torch.save(model.state_dict(), archive_ave)
    print('Averaged model save to', archive_ave)

print(f'Last accurasy: {test_acc:.2f}')
print(f'Last averaged accurasy: {ave_acc:.2f}')
print(f'Total {total_time//3600:.0f}:{total_time%3600//60:02.0f}:{total_time%3600%60:02.0f}')
print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
# ファイルを閉じて標準出力を元に戻す
if args.log is True:
    sys.stdout.close()
    sys.stdout = sys.__stdout__