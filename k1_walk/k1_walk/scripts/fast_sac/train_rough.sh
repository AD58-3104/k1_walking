#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train-rough --num_envs 8192 --headless --max_iterations 3000 $@
