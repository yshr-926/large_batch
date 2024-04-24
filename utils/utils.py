import os
import models
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.batch_cutout import batch_Cutout
from utils.cutout_sam import Cutout_Sam
from optimizers import *
import copy

class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return index, data, label

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes

class MyBatchSampler(BatchSampler):
  def __init__(self, sampler, batch_size, aug_size):
    self.sampler = sampler
    self.batch_size = batch_size
    self.aug_size = aug_size
  def __iter__(self):
    batch = [0] * self.batch_size
    idx_in_batch = 0
    for idx in self.sampler:
      batch[idx_in_batch] = idx
      idx_in_batch += 1
      if idx_in_batch == self.batch_size:
        yield batch*self.aug_size
        idx_in_batch = 0
        batch = [0] * self.batch_size
    if idx_in_batch > 0:
        yield batch[:idx_in_batch]*self.aug_size

def get_model(model, num_class, dataset):
    if model.startswith('ResNet'):
        resnet_number = int(model[6:])
        assert resnet_number in [18, 34, 50, 101, 152], f"Invalid ResNet model number: {resnet_number}"
        return models.__dict__[model](num_classes=num_class, dataset=dataset)
    elif model == 'WideResNet28x10':
        return models.__dict__['WRN28_10'](num_classes=num_class)
    elif model == 'PyramidNet':
        return models.__dict__['Pyramid'](num_classes=num_class)
    elif model == 'ShakePyramidNet':
        return models.__dict__['ShakePyramidNet272_200'](num_classes=num_class)
    elif model == 'ShakeWideResNet':
        return models.__dict__[model](depth=28, width_factor=10, dropout=0.0, in_channels=3, num_classes=num_class)
    elif model == 'ShakeShake26x32':
        return models.__dict__['ShakeShake26_2x32d'](num_classes=num_class)
    elif model == 'ShakeShake26x64':
        return models.__dict__['ShakeShake26_2x64d'](num_classes=num_class)
    elif model == 'ShakeShake26x96':
        return models.__dict__['ShakeShake26_2x96d'](num_classes=num_class)

def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'LARS':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adainverse':
        optimizer = Adainverse(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_transforms(dataset, train_policy, test_policy):
    if dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # batch_Cutout(n_holes=1, length=16),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif dataset == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC), #original is 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop(64, padding=8), #original is 224
            transforms.RandomHorizontalFlip(),
            batch_Cutout(n_holes=1, length=16), #original length = 16 #64
        ])
        transform_test = transforms.Compose([
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC), #original is 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif dataset == 'CIFAR100' or dataset == 'CIFAR10':
        #define dataset and transform
        for i, policy in enumerate([train_policy, test_policy]):
            transform_list=[]
            if policy == 'policy3':
                transform_list.append(transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10))
            if policy == 'policy1' or policy == 'policy2' or policy == 'policy3':
                transform_list.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
            if dataset == 'CIFAR100':
                transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            elif dataset == 'CIFAR10':
                transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
            if policy == 'policy2' or policy == 'policy3':
                transform_list.append(batch_Cutout(n_holes=1, length=16))
            
            if i == 0:
                transform_train = transforms.Compose(transform_list)
                transform_tta = transforms.Compose(transform_list)
            elif i == 1:
                transform_test = transforms.Compose(transform_list)

    return transform_train, transform_test, transform_tta

def get_sam_transform(dataset):
    train_set = datasets.__dict__[dataset](root='./data', train=True, download=True, transform=transforms.ToTensor())
    data = torch.cat([d[0] for d in DataLoader(train_set)])
    mean, std = data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

    train_transform = transforms.Compose([
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout_Sam()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, test_transform

def get_dataset(dataset, transform_train, transform_test, transform_tta):
    if dataset == 'ImageNet':
        trainset = datasets.ImageNet(root='./dataset/imagenet', split='train', transform=transform_train)
        testset = datasets.ImageNet(root='./dataset/imagenet', split='val', transform=transform_test)
    elif dataset == 'TinyImageNet':
        trainset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_train)
        testset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val', transform=transform_test)
    elif dataset == 'CIFAR10' or 'CIFAR100':
        trainset = datasets.__dict__[dataset](root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.__dict__[dataset](root='./data', train=False, download=False, transform=transform_test)
        ttaset = datasets.__dict__[dataset](root='./data', train=False, download=False, transform=transform_tta)

    return trainset, testset, ttaset

def get_dataset_val(dataset, transform_train, transform_test, transform_tta):
    if dataset == 'ImageNet':
        trainset = datasets.ImageNet(root='../dataset/imagenet', split='val', transform=transform_train)
        testset = datasets.ImageNet(root='../dataset/imagenet', split='val', transform=transform_test)
    elif dataset == 'CIFAR10' or 'CIFAR100':
        trainset = datasets.__dict__[dataset](root='./data', train=True, download=True, transform=None)
        # データセットの分割
        train_size = int(0.8 * len(trainset))
        valid_size = len(trainset) - train_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, valid_size])
        ttaset = copy.deepcopy(valset)
        # データ拡張
        trainset.dataset.transform = transform_train
        valset.dataset.transform = transform_test
        ttaset.dataset.transform = transform_tta

    return trainset, valset, ttaset

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)

def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(filename):
        filename = f"{base}_{i}{ext}"
        i += 1
    return filename

def print_header():
    header = f"{'epoch':^10} | {'lr':^10} | {'time':^10} | {'loss':^10} | {'acc':^10} | {'test_loss':^10} | {'test_acc':^10} | {'ave_loss':^10} | {'ave_acc':^10}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
