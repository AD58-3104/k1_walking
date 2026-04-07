#!/bin/bash
source ~/.bash_functions

forward_args=()
requested_device=""
requested_gpu=""
requires_headless=0
display_gpu=""

# Limit visible GPUs when the caller selects a concrete CUDA device.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            if [[ -n "$2" ]]; then
                requested_device="$2"
                shift 2
                continue
            fi
            forward_args+=("$1")
            shift
            ;;
        --device=*)
            requested_device="${1#*=}"
            shift
            ;;
        --gpu-id)
            if [[ -n "$2" ]]; then
                requested_gpu="$2"
                shift 2
                continue
            fi
            forward_args+=("$1")
            shift
            ;;
        --gpu-id=*)
            requested_gpu="${1#*=}"
            shift
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$requested_gpu" && -z "$requested_device" ]]; then
    requested_device="cuda:${requested_gpu}"
fi

if [[ -n "$requested_device" ]]; then
    forward_args+=(--device "$requested_device")
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    while IFS=',' read -r gpu_index _gpu_name display_active; do
        gpu_index="${gpu_index//[[:space:]]/}"
        display_active="${display_active//[[:space:]]/}"
        if [[ "$display_active" == "Enabled" ]]; then
            display_gpu="$gpu_index"
            break
        fi
    done < <(nvidia-smi --query-gpu=index,name,display_active --format=csv,noheader 2>/dev/null)
fi

if [[ "$requested_device" =~ ^cuda:([0-9]+)$ ]]; then
    requested_gpu="${BASH_REMATCH[1]}"
    if [[ -n "$display_gpu" && "$requested_gpu" != "$display_gpu" ]]; then
        requires_headless=1
    fi
fi

if [[ $requires_headless -eq 1 ]]; then
    has_headless_flag=0
    for arg in "${forward_args[@]}"; do
        if [[ "$arg" == "--headless" || "$arg" == "--video" || "$arg" == "--livestream" || "$arg" == "--streaming-mode" || "$arg" == "--streaming_mode" ]]; then
            has_headless_flag=1
            break
        fi
    done
    if [[ $has_headless_flag -eq 0 ]]; then
        echo "[INFO] GPU ${requested_gpu} is not display-active. Forcing --headless." >&2
        forward_args+=(--headless)
    fi
fi

# --headless / --video / streaming 使用時はDISPLAYをunsetしてNVIDIA EGLレンダリングを使用
for arg in "${forward_args[@]}"; do
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

_labpython play.py --task K1-Walk-Play-rsl --num_envs 1 --checkpoint "${forward_args[@]}"
