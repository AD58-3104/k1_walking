#!/bin/bash
source ~/.bash_functions

_labpython train.py --task K1-Walk-Train-fast-sac --max_iterations 2200 --num_envs 4096 --headless "$@" > run.log 2>&1
if [ $? -ne 0 ]; then
    echo "Running failed. You should check the run.log file for details."
else
    _labpython show_final_reward_table.py
fi