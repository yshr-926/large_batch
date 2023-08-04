# CIFAR10
# dataset=CIFAR10

# CIFAR100
dataset=CIFAR100

# ImageNet
# dataset=ImageNet

# TinyImageNet
# dataset=TinyImageNet

# ResNet18
# model=ResNet18

# ResNet50
model=ResNet50

# WideResNet28-10
# model=WideResNet28-10

# optimizer=LARS
# lr_type=WarmupPolynomialLR
# python baseline.py --epoch 100 --lr 0.1 --batch_size 8192 --model $model --dataset $dataset 

# python train_lars.py --epoch 100 --lr 0.02 --batch_size 512 --model $model --dataset $dataset \
# --lr_type $lr_type --optimizer $optimizer --warmup_start_lr 0.001 --power 2


optimizer=Adasam
# lr_type=fixed
lr_type=CosineAnnealingLR
python train_adagrad.py --epoch 200 --lr 0.1 --model $model --dataset $dataset --optimizer $optimizer --lr_type $lr_type