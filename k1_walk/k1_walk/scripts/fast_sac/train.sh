#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train-fast-sac --num_envs 8196 --headless $@
