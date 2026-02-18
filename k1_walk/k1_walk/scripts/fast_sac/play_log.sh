#!/bin/bash
source ~/.bash_functions
_labpython play.py --task K1-Walk-Play-fast-sac --log_data --log_steps 3000 --num_envs 1 --headless --checkpoint $@ 