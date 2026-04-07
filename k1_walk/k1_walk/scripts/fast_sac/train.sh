#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train-fast-sac --num_envs 8192 --headless --max_iterations 1500 $@
