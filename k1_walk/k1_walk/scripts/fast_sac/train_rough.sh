#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train-rough --num_envs 4096 --headless --max_iterations 8000 $@
