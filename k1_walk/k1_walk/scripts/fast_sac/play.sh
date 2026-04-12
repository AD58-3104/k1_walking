#!/bin/bash
source ~/.bash_functions

# --headless / --video / streaming 使用時はDISPLAYをunsetしてNVIDIA EGLレンダリングを使用
# (DISPLAYが設定されているとIsaacSimがMesa GLXに接続してGLXBadFBConfigが発生するため)
for arg in "$@"; do
    if [[ "$arg" == "--headless" || "$arg" == "--video" || "$arg" == "--livestream" || "$arg" == "--streaming-mode" || "$arg" == "--streaming_mode" ]]; then
        unset DISPLAY
        export PYOPENGL_PLATFORM=egl
        export __GLX_VENDOR_LIBRARY_NAME=nvidia
        if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]] && [[ "${VK_ICD_FILENAMES:-}" != *"nvidia_icd.json"* ]]; then
            export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
        fi
        if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]] && [[ "${VK_DRIVER_FILES:-}" != *"nvidia_icd.json"* ]]; then
            export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
        fi
        export NODEVICE_SELECT=1
        export VK_LOADER_LAYERS_DISABLE='*MESA*'
        break
    fi
done

_labpython play.py --task K1-Walk-Play-fast-sac --num_envs 32 --checkpoint $@
