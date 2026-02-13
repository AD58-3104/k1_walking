#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train-rsl --num_envs 15000 --headless  $@
