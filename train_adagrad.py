# -*- coding: utf-8 -*-
import argparse
import sys
import os
import time
from datetime import datetime, timedelta, timezone
import datetime
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import PolynomialLR
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn import functional as F
from utils.utils import get_model, get_transforms, get_dataset

from optimizers import *
        
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_worker', default=8, type=int, help='number of workers')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer',default='LARS',type=str,help='SGD/LARS/Lamb')
parser.add_argument('--model',default='ResNet18',type=str)
parser.add_argument('--dataset',default='CIFAR10',type=str)
parser.add_argument('--lr_type',default='MultiStepLR',type=str)
# parser.add_argument('--milestones', default=[60, 120, 160], nargs='*', type=int, help='milestones of scheduler')
parser.add_argument('--lr_decay',default=0.2,type=float)
parser.add_argument('--eta_min',default=0.00,type=float)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--warmup_start_lr', default=0.01, type=float)
parser.add_argument('--power', default=1.0, type=float)


args = parser.parse_args()

# 出力先を標準出力orファイルで選択
log = True
if log is True:
    today = datetime.date.today()
    log_dir = './logs/{}/{}/{}'.format(args.dataset, args.model, today)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now = datetime.datetime.now(timezone(timedelta(hours=+9))).strftime("%H%M")
    logpath = log_dir+'/inv-{}-{}-{}.txt'.format(now, args.dataset, args.model)
    sys.stdout = open(logpath, "w") # 表示内容の出力をファイルに変更

print(' '.join(sys.argv))
print('GPU count',torch.cuda.device_count())
print('epoch:', args.epoch)
print('model:',args.model)
print('dataset:',args.dataset)
print('batch_size:', args.batch_size)
print('optimizer:',args.optimizer)
print('learning_rate:', args.lr)
print('lr_type:',args.lr_type)
# print('milestones:', args.milestones)
print('lr_decay:',args.lr_decay)
print('eta_min:',args.eta_min)
print('weight_decay:',args.weight_decay)
print('momentum:', args.momentum)
print('warmup_epochs', args.warmup_epochs)
print('warmup_start_lr', args.warmup_start_lr)
print('power', args.power)

use_cuda = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

#define dataset and transform
transform_train, transform_test = get_transforms(args.model, args.dataset)
trainset, testset = get_dataset(args.dataset, transform_train, transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_worker)

num_class = len(trainset.classes)

#define model
print('==> Building model..')
model = get_model(args.model, num_class, args.dataset)
if use_cuda:
    model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True    

# Optimizer
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'LARS':
    optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'Lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'Adaggrad':
    optimizer = Adaggrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'Adainverse':
    optimizer = Adainverse(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'Adainverse2':
    optimizer = Adainverse2(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'Adasam':
    optimizer = Adasam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Scheduler
if args.lr_type == 'MultiStepLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, milestones=[60, 120, 160], gamma=args.lr_decay)
elif args.lr_type == 'CosineAnnealingLR':
    scheduler = eval(args.lr_type)(optimizer=optimizer, T_max = args.epoch, eta_min = args.eta_min)
elif args.lr_type == 'WarmupPolynomialLR':
    scheduler = WarmupPolynomialLR(optimizer, total_iters=args.epoch, warmup_epochs=args.warmup_epochs, 
                                    warmup_start_lr=args.warmup_start_lr, power=args.power)
elif args.lr_type == "LinearWarmupCosineAnnealingLR":
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epoch)

criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    torch.cuda.synchronize()
    time_ep = time.time()
    model.train()
    train_loss = 0.0
    correct = 0.0    
    total = 0

    for batch_idx, (input, target) in enumerate(trainloader):
        input_size = input.size()[0]
        input, target = input.to(device), target.to(device)
        
        output = model(input)
        _, pred = torch.max(output, 1)
        correct += pred.eq(target).sum()
        total += input_size
        loss = criterion(output, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()   
    time_ep = time.time() - time_ep

    return train_loss/batch_idx, 100*correct/total, time_ep

def test(epoch, model):
    model.eval()
    total = 0

    test_loss = 0.0
    test_correct = 0

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testloader):
            input_size = input.size()[0]
            input, target = input.to(device), target.to(device)   
            output = model(input)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            test_correct += pred.eq(target).sum()
            total += input_size

    return test_loss/batch_idx, 100*test_correct/total

total_time = 0
for epoch in range(args.start_epoch, args.epoch):
    train_loss, train_acc, time_ep = train(epoch)
    total_time += time_ep
    if not args.lr_type == "fixed":
        scheduler.step()
    test_loss, test_acc = test(epoch, model)
    print("| epoch : {} | lr : {:.7f} | train_loss : {:.5f} | train_acc : {:.2f} | test_loss : {:.5f} | test_acc : {:.2f}| time : {:.3f}" \
          .format(epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc, time_ep))

print('Total {:.0f}:{:.0f}:{:.0f}'.format(total_time//3600, total_time%3600//60, total_time%3600%60))

# ファイルを閉じて標準出力を元に戻す
sys.stdout.close()
sys.stdout = sys.__stdout__