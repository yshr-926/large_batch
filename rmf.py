import os
import glob
import shutil

# for p in glob.glob('logs/2022-12-12/*.txt', recursive=True):
#         if os.path.isfile(p):
#             os.remove(p)

# for p in glob.glob('save/objects/rm/*.pkl', recursive=True):
#         if os.path.isfile(p):
#             os.remove(p)

# ディレクトリ丸ごと消す
for p in glob.glob('logs/CIFAR100/ResNet50/*/*131*.txt', recursive=True):
    if os.path.isfile(p):
        print(p)
        os.remove(p)

# for p in glob.glob('logs/CIFAR100/ResNet50/2023-06-29/sd-1605-CIFAR100-ResNet50.txt', recursive=True):
#     if os.path.isfile(p):
#         os.remove(p)