# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_or_update_gait_state(
    env: "ManagerBasedRLEnv",
    gait_period: float = 1.0 / 1.2,
    gait_period_randomization_width: float = 0.0,
    randomize_phase: bool = True,
    stand_phase_value: float = torch.pi,
) -> dict[str, torch.Tensor]:
    """Maintain a holosoma-like gait state on the IsaacLab env.

    The stock IsaacLab velocity command used by this task does not own a gait-state
    buffer. To keep the task structure close to holosoma, we maintain per-env gait
    phase buffers here and reinitialize them on reset.
    """
    if not hasattr(env, "_gait_state"):
        env._gait_state = {}

    state = env._gait_state
    num_envs = env.num_envs
    device = env.device
    mean_gait_freq = 1.0 / gait_period

    needs_init = (
        "phase_offset" not in state
        or state["phase_offset"].shape[0] != num_envs
        or state["phase_offset"].device != device
    )

    if needs_init:
        state["phase_offset"] = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        state["phase"] = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        state["gait_freq"] = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        state["phase_dt"] = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
        state["initialized"] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    initialized = state["initialized"]

    if hasattr(env, "reset_buf"):
        reset_ids = torch.nonzero(env.reset_buf > 0, as_tuple=False).flatten()
    else:
        reset_ids = torch.empty(0, dtype=torch.long, device=device)

    init_ids = torch.nonzero(~initialized, as_tuple=False).flatten()
    env_ids = torch.unique(torch.cat([init_ids, reset_ids], dim=0)) if (init_ids.numel() or reset_ids.numel()) else init_ids

    if env_ids.numel() > 0:
        if getattr(env, "is_evaluating", False):
            state["phase_offset"][env_ids, 0] = 0.0
            state["phase_offset"][env_ids, 1] = -torch.pi
            state["gait_freq"][env_ids] = mean_gait_freq
        else:
            if randomize_phase:
                phase0 = torch.empty((env_ids.numel(),), device=device).uniform_(-torch.pi, torch.pi)
                state["phase_offset"][env_ids, 0] = phase0
                state["phase_offset"][env_ids, 1] = torch.fmod(phase0 + 2 * torch.pi, 2 * torch.pi) - torch.pi
            else:
                state["phase_offset"][env_ids, 0] = 0.0
                state["phase_offset"][env_ids, 1] = -torch.pi

            if gait_period_randomization_width > 0.0:
                low = mean_gait_freq - gait_period_randomization_width
                high = mean_gait_freq + gait_period_randomization_width
                state["gait_freq"][env_ids] = torch.empty((env_ids.numel(), 1), device=device).uniform_(low, high)
            else:
                state["gait_freq"][env_ids] = mean_gait_freq

        state["phase_dt"][env_ids] = 2 * torch.pi * env.step_dt * state["gait_freq"][env_ids]
        state["phase"][env_ids] = state["phase_offset"][env_ids]
        state["initialized"][env_ids] = True

    if hasattr(env, "episode_length_buf"):
        episode_length = env.episode_length_buf.unsqueeze(1).float()
    else:
        # ObservationManager initializes term shapes before the env runtime buffers exist.
        episode_length = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)

    phase_tp1 = episode_length * state["phase_dt"] + state["phase_offset"]
    state["phase"] = torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi

    commands = env.command_manager.get_command("base_velocity")
    stand_mask = torch.logical_and(
        torch.linalg.norm(commands[:, :2], dim=1) < 0.01,
        torch.abs(commands[:, 2]) < 0.01,
    )
    if stand_mask.any():
        state["phase"][stand_mask] = torch.full((int(stand_mask.sum().item()), 2), stand_phase_value, device=device)

    return state


def body_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ロボットの胴体の高さを観測として取得する

    Returns:
        torch.Tensor: shape (num_envs, 1) の観測値
    """
    asset = env.scene[SceneEntityCfg("robot").name]
    body_height = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    return body_height

def feet_contact(env: ManagerBasedRLEnv, sensor_cfg_right: SceneEntityCfg,sensor_cfg_left: SceneEntityCfg) -> torch.Tensor:
    """
    足の接地に関するバイナリ情報を観測として取得する(両脚分)

    Returns:
        torch.Tensor: shape (num_envs, 2) の観測値
            - [:, 0]: 右足の接地状態 (0 or 1)
            - [:, 1]: 左足の接地状態 (0 or 1)
    """
    # Penalize feet sliding
    contact_sensor_right: ContactSensor = env.scene.sensors[sensor_cfg_right.name]
    contact_sensor_left: ContactSensor = env.scene.sensors[sensor_cfg_left.name]

    contacts_right = contact_sensor_right.data.net_forces_w_history[:, :, sensor_cfg_right.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contacts_left = contact_sensor_left.data.net_forces_w_history[:, :, sensor_cfg_left.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # contacts_right と contacts_left の形状を確認して適切に処理
    # もし (num_envs, 1) の形状なら squeeze して (num_envs,) にする
    if contacts_right.dim() > 1:
        contacts_right = contacts_right.squeeze(-1)
    if contacts_left.dim() > 1:
        contacts_left = contacts_left.squeeze(-1)

    # (num_envs, 2) の形状で返す
    return torch.stack([contacts_right, contacts_left], dim=-1)



def clock_phase(env: ManagerBasedRLEnv, frequency: float = 1.0) -> torch.Tensor:
    """
    Args:
        env: 環境インスタンス
        frequency: 歩行の周期.一回の歩行に掛かる時間

    Returns:
        torch.Tensor: shape (num_envs, 3) の観測値
            - [:, 0]: sin(2π * (time / frequency))
            - [:, 1]: cos(2π * (time / frequency))
            - [:, 2]: combine_phase
    """
    # 環境の現在時刻を取得（秒単位）
    # episode_length_buf: エピソード開始からの経過ステップ数
    # step_dt: 1ステップの時間（秒）
    # 初期化時にepisode_length_bufが存在しない場合はゼロ配列を使用
    if hasattr(env, 'episode_length_buf'):
        time = env.episode_length_buf.float() * env.step_dt
    else:
        # 初期化時のダミー値（次元確認用）
        time = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    # 位相角度を計算（ラジアン）
    # 2π * f * t で周期的な角度を生成
    phase = 2.0 * torch.pi * (time / frequency)

    # sin と cos のペアを計算して返す
    # これにより位相情報が連続的な2次元空間で表現される
    sin_phase = torch.sin(phase).unsqueeze(-1)  # (num_envs, 1)
    cos_phase = torch.sin(phase + 0.5).unsqueeze(-1)  # (num_envs, 1)
    combine_phase = sin_phase / 2 * torch.sqrt(torch.square(sin_phase) + 0.04) + 0.5  # (num_envs, 1)

    return torch.cat([sin_phase, cos_phase, combine_phase], dim=-1)  # (num_envs, 3)


def phase_time(
    env: ManagerBasedRLEnv,
    gait_period: float = 1.0 / 1.2,
    gait_period_randomization_width: float = 0.1,
    randomize_phase: bool = True,
    stand_phase_value: float = torch.pi,
) -> torch.Tensor:
    """
    Args:
        env: 環境インスタンス
        frequency: 歩行の周期(Hz)

    Returns:
        torch.Tensor: shape (num_envs, 2) の観測値
            - [phase1, phase2]: 位相 (0から2πの範囲)
    """
    state = _get_or_update_gait_state(
        env,
        gait_period=gait_period,
        gait_period_randomization_width=gait_period_randomization_width,
        randomize_phase=randomize_phase,
        stand_phase_value=stand_phase_value,
    )
    return state["phase"]

def sincos_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Args:
        env: 環境インスタンス
        frequency: 歩行の周期.一回の歩行に掛かる時間

    Returns:
        torch.Tensor: shape (num_envs, 4) の観測値
            - [:, 0:2]: sin(phase) for each phase
            - [:, 2:4]: cos(phase) for each phase
    """
    phase = phase_time(env)  # (num_envs, 2)

    sin_phase = torch.sin(phase)  # (num_envs, 2)
    cos_phase = torch.cos(phase)  # (num_envs, 2)

    return torch.cat([sin_phase, cos_phase], dim=-1)  # (num_envs, 4)


def foot_height(env: ManagerBasedRLEnv, foot_cfg_right: SceneEntityCfg, foot_cfg_left: SceneEntityCfg) -> torch.Tensor:
    """
    足の高さを観測として取得する(両脚分)

    Returns:
        torch.Tensor: shape (num_envs, 2) の観測値
            - [:, 0]: 右足の高さ
            - [:, 1]: 左足の高さ
    """
    asset = env.scene[SceneEntityCfg("robot").name]
    foot_height_right = asset.data.body_com_pos_w[:, foot_cfg_right.body_ids, 2]  # (num_envs, 1)
    foot_height_left = asset.data.body_com_pos_w[:, foot_cfg_left.body_ids, 2]  # (num_envs, 1)
    return torch.cat([foot_height_right, foot_height_left], dim=-1)  # (num_envs, 2)
