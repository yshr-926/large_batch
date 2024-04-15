import os
import glob
import shutil

# ディレクトリ丸ごと消す
shutil.rmtree('logs/TinymageNet')


# ファイル指定して消す
# for p in glob.glob('checkpoints/*', recursive=True):
#     if os.path.isfile(p):
#         print(p)
#         os.remove(p)

# for p in glob.glob('light_logs/CIFAR100/WideResNet28-10/2023-09-16/SAM2/134*.log', recursive=True):
#     if os.path.isfile(p):
#         print(p)
#         os.remove(p)