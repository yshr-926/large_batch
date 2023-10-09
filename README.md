# large_batch

## LARS

Inplementation of [Large batch training of convolutional networks](https://arxiv.org/abs/1708.03888).

## Lamb

Inplementation of [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962).

## Sharpness-Aware Minimization(SAM)

Inplementation of [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412).

## Example

If you want to use the SGD optimizer,
```bash
python baseline.py --epoch 200 --lr 0.1 --model ResNet50 --dataset CIFAR100 --optimizer SGD
```

## Environment

OS: Ubuntu

CPU: AMD EPYC 7642

GPU: NVIDIA A100 80GB

Python: 3.10

PyTorch: 2.0

## References

Yang You, Igor Gitman, Boris Ginsburg. "[Large batch training of convolutional networks.](https://arxiv.org/abs/1708.03888)" *arXiv preprint arXiv:1708.03888*, 2017.

Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh. "[Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.](https://arxiv.org/abs/1904.00962)" *arXiv preprint arXiv:1904.00962*, 2020.

Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. "[Sharpness-Aware Minimization for Efficiently Improving Generalization.](https://arxiv.org/abs/2010.01412)" *arXiv preprint arXiv:2010.01412*, 2021.