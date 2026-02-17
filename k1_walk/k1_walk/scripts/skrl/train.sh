#!/bin/bash
source ~/.bash_functions
_labpython train.py --task K1-Walk-Train --num_envs 16384 --headless agent.trainer.timesteps=24000 $@
