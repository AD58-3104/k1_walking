#!/bin/bash
# Optuna最適化スクリプトの実行ラッパー
# 使用方法:
#   bash run_optuna.sh --study-name my_study --n-trials 50
#   bash run_optuna.sh --resume --study-name my_study

source ~/.bash_functions

cd "$(dirname "$0")"

_labpython optuna_optimizer.py "$@"
