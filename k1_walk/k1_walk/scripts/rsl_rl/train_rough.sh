#!/bin/bash
source ~/.bash_functions
# _labpython -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py --task K1-Walk-Train-fast-sac --num_envs 16394 --headless --distributed  $@
_labpython train.py --task K1-Walk-Train-rough --num_envs 8192 --headless --max_iterations 1500  $@
