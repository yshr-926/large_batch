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
from torch.utils.data.sampler import SequentialSampler
from utils.utils import MyBatchSampler, get_model, get_optimizer, get_transforms, get_dataset, print_header, get_unique_filename
from utils.utils_sam import enable_running_stats, disable_running_stats
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
parser.add_argument('--optimizer',default='SAM_L2',type=str,help='optimizer')
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
# for SAM
parser.add_argument('--rho', default=0.05, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--adaptive', default=False, action='store_true')
# for Averaging
parser.add_argument('--start_averaged', default=160, type=int)
# save model
parser.add_argument('--saving-folder', default='checkpoints/', type=str, help='choose saving name')
parser.add_argument('--save_model', action='store_true', default=False, help='save model')
# for TTA
parser.add_argument('--tta_num', default=5, type=int, help='number of TTA')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

today = datetime.now(timezone(timedelta(hours=+9))).strftime("%Y-%m-%d")
now = datetime.now(timezone(timedelta(hours=+9))).strftime("%H%M")

# 出力先を標準出力orファイルで選択
if args.log is True:
    log_dir = f"./logs/{args.dataset}/{args.model}/{today}/{args.optimizer}"
    os.makedirs(log_dir, exist_ok=True)
    logpath = log_dir+f"/{now}-{args.epoch}-{args.lr_type}-eta{args.lr}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}.log"
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
print('warmup_epochs', args.warmup_epochs)
print('warmup_start_lr', args.warmup_start_lr)
print('power', args.power)
print("rho", args.rho)
print("label_smoothing", args.label_smoothing)
print("adaptive", args.adaptive)
print('train_policy:',args.train_policy)
print('test_policy:',args.test_policy)
print('tta_num:', args.tta_num)

tta_num = args.tta_num
use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

#define dataset and transform
transform_train, transform_test, transform_tta = get_transforms(args.dataset, args.train_policy, args.test_policy)
print('train_transform:', transform_train)
print('test_transform:', transform_test)
print('tta_transform:', transform_tta)

trainset, testset, ttaset = get_dataset(args.dataset, transform_train, transform_test, transform_tta)

# train_sampler = RandomSampler(trainset)
test_sampler = SequentialSampler(ttaset)
# train_batch_sampler = MyBatchSampler(sampler=train_sampler, batch_size=args.batch_size, aug_size=1)
test_batch_sampler = MyBatchSampler(sampler=test_sampler, batch_size=100, aug_size=tta_num)

# trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=args.num_worker, pin_memory=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_worker, pin_memory=True)
ttaloader = torch.utils.data.DataLoader(ttaset, batch_sampler=test_batch_sampler, num_workers=args.num_worker, pin_memory=True)

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

# Optimizer
optimizer = SAM_L2(model.parameters(), base_optimizer=optim.SGD, rho=args.rho, adaptive=args.adaptive,
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.lr_type == 'MultiStepLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, milestones=args.milestones, gamma=args.lr_decay)
elif args.lr_type == 'CosineAnnealingLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, T_max = args.epoch, eta_min = args.eta_min)
elif args.lr_type == 'WarmupPolynomialLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, warmup_epochs=5, total_epochs=args.epoch, warmup_start_lr=0, eta_min=args.eta_min)
elif args.lr_type == 'LinearWarmupCosineAnnealingLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, warmup_epochs=5, max_epochs=args.epoch, eta_min=args.eta_min)

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
        
        # first forward-backward pass
        enable_running_stats(model)
        output = model(input)
        total += input_size
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        disable_running_stats(model)
        criterion(model(input), target).backward()
        optimizer.second_step(zero_grad=True)

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

def tta(epoch, model):
    model.eval()
    averaged_model.eval()
    tta_loss = 0.0
    tta_correct = 0
    total = 0

    # test_loss = 0.0
    # test_correct = 0
    ave_tta_loss = 0.0
    ave_tta_correct = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(ttaloader):
            input_size = int(inputs.size()[0] / tta_num)
            output_sum = torch.zeros(input_size, num_class, requires_grad=False).to(device)
            ave_output_sum = torch.zeros(input_size, num_class, requires_grad=False).to(device)
            if use_cuda:
                inputs, target = inputs.to(device), targets.split(input_size)[0].to(device)    

            for input in inputs.split(input_size):
                output_sum += model(input)
                if epoch >= args.start_averaged:
                    ave_output_sum += averaged_model(input)

            output = output_sum / tta_num
            ave_output = ave_output_sum / tta_num

            loss = criterion(output, target)
            
            tta_loss += loss.item()
            _, pred = torch.max(output, 1)
            tta_correct += pred.eq(target).sum()
            total += input_size

            # single prediction
            # predict = model(input)
            # loss = criterion(predict, target)
            # test_loss += loss.item()
            # _, pred = torch.max(predict, 1)
            # test_correct += pred.eq(target).sum()

            # averaged test
            if epoch >= args.start_averaged:
                # ave_predict = averaged_model(input)
                loss = criterion(ave_output, target)
                ave_tta_loss += loss.item()
                _, pred = torch.max(ave_output, 1)
                ave_tta_correct += pred.eq(target).sum()

    return tta_loss/batch_idx, 100*tta_correct/total, ave_tta_loss/batch_idx, 100*ave_tta_correct/total

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

tta_loss, tta_acc, tta_ave_loss, tta_ave_acc = tta(epoch, model)

print(f'Last test accurasy: {test_acc:.2f}')
print(f'Last averaged accurasy: {ave_acc:.2f}')
print(f'TTA accurasy: {tta_acc:.2f}')
print(f'TTA averaged accurasy: {tta_ave_acc:.2f}')
print(f'Total {total_time//3600:.0f}:{total_time%3600//60:02.0f}:{total_time%3600%60:02.0f}')
print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

if args.save_model:
    archive_base = f'{args.saving_folder}{args.dataset}/{args.model}/{args.optimizer}/Base'
    os.makedirs(archive_base, exist_ok=True)
    archive_base = f'{archive_base}/{args.epoch}-{args.lr_type}-eta{args.lr}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}.pkl'
    archive_base = get_unique_filename(archive_base)
    torch.save(model.state_dict(), archive_base)
    print('Base model save to', archive_base)

    archive_ave = f'{args.saving_folder}{args.dataset}/{args.model}/{args.optimizer}/Averaged'
    os.makedirs(archive_ave, exist_ok=True)
    archive_ave = f'{archive_ave}/{args.epoch}-{args.lr_type}-eta{args.lr}-eta_min{args.eta_min}-m{args.momentum}-wd{args.weight_decay:.0e}-aug{args.train_policy}_ave.pkl'
    archive_ave = get_unique_filename(archive_ave)
    torch.save(averaged_model.state_dict(), archive_ave)
    print('Averaged model save to', archive_ave)

# ファイルを閉じて標準出力を元に戻す
if args.log is True:
    sys.stdout.close()
    sys.stdout = sys.__stdout__