# TL;DR
dlhpcstarter -t cifar10 -c tests/task/cifar10/config/tl_dr.yaml --trial 0 --stages_module tests.task.cifar10.stages --train --test

# https://github.com/aehrc/cxrmate/issues/13
dlhpcstarter -t cifar10 -c tests/task/cifar10/config/tl_dr.yaml --trial 0 --stages_module tests.task.cifar10.stages --test --test-ckpt-path experiments/cifar10/tl_dr/trial_0/last.ckpt