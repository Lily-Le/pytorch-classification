#!/bin/bash
# 3219M
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110
# 3221M
python cifar10_loc.py -a resnet_loc --depth 110 --epochs 164 --ratio 1 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-loc

python cifar10_loc.py -a resnet_loc --depth 110 --epochs 164 --ratio 0.5 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-loc

python cifar10_loc.py -a resnet_loc --depth 110 --epochs 164 --ratio 2 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-loc

CUDA_VISIBLE_DEVICES=1 python cifar10_loc.py -a resnet_loc --depth 110 --epochs 164 --ratio 0.2 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110-loc --gpu-id 1