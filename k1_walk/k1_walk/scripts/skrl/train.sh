#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train --num_envs 15000 --headless agent.trainer.timesteps=25000 $@
