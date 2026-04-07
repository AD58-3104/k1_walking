#!/bin/bash
source ~/.bash_functions

# --headless / --video 使用時はDISPLAYをunsetしてNVIDIA EGLレンダリングを使用
# (DISPLAYが設定されているとIsaacSimがMesa GLXに接続してGLXBadFBConfigが発生するため)
for arg in "$@"; do
    if [[ "$arg" == "--headless" || "$arg" == "--video" ]]; then
        unset DISPLAY
        break
    fi
done

_labpython play.py --task K1-Walk-Play-rsl --num_envs 32 --video --video_length 1000 --checkpoint $@
