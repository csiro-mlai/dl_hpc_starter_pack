#!/bin/bash

source /scratch2/nic261/environments/simplified_12_08_22/bin/activate

## CIFAR10
python3 main.py --task cifar10 --config baseline --submit 1
