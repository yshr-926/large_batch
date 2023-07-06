import torchvision.models
import models
from torch import nn
import torchvision.models
import torchvision.transforms as transforms
from utils.batch_cutout import batch_Cutout
import torchvision.datasets as datasets

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

def get_model(model, num_class, dataset):
    if model == 'ResNet18':
        return models.__dict__[model](num_classes=num_class, dataset=dataset)
    if model == 'ResNet50':
        return models.__dict__[model](num_classes=num_class, dataset=dataset)
    elif model == 'WideResNet28-2':
        return models.__dict__['WRN28_2'](num_classes=num_class)
    elif model == 'WideResNet28-10':
        return models.__dict__['WRN28_10'](num_classes=num_class)
    # elif model == 'PyramidNet':
    #     return models.__dict__['Pyramid'](num_classes=num_class)
    # elif model == 'ShakeResNeXt':
    #     return models.__dict__[model](depth=26, w_base=64, cardinary=4, num_classes=num_class)
    # elif model == 'ShakePyramidNet':
        # return torchvision.models.__dict__[model](depth=110, alpha=270, num_classes=num_class)
    # elif model == 'MobileNetV2':
        # return torchvision.models.__dict__['mobilenetv2'](num_classes=num_class)
    # elif model == 'ShakeWideResNet':
        # return torchvision.models.__dict__[model](depth=28, width_factor=6, dropout=0.0, in_channels=3, num_classes=num_class)
    # elif model == 'EfficientNet':
        # return torchvision.models.__dict__[model].from_name('efficientnet-b7', num_classes=num_class)

def get_transforms(model, dataset):
    if dataset == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            batch_Cutout(n_holes=1, length=16),
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

    elif dataset == 'CIFAR100' or 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in[125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            batch_Cutout(n_holes=1, length=16),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in[125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
    
    return transform_train, transform_test


def get_dataset(dataset, transform_train, transform_test):
    if dataset == 'ImageNet':
        trainset = datasets.ImageNet(root='./dataset/imagenet', split='train', transform=transform_train)
        testset = datasets.ImageNet(root='./dataset/imagenet', split='val', transform=transform_test)
    elif dataset == 'TinyImageNet':
        trainset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/train', transform=transform_train)
        testset = datasets.ImageFolder(root='./dataset/tiny-imagenet-200/val', transform=transform_test)
    elif dataset == 'CIFAR10' or 'CIFAR100':
        trainset = datasets.__dict__[dataset](root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.__dict__[dataset](root='./data', train=False, download=False, transform=transform_test)

    return trainset, testset
