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
        frequency: 位相の周波数 [Hz]
            - 0.5 Hz = 2秒周期（ゆっくりとした歩行）
            - 1.0 Hz = 1秒周期（通常の歩行）
            - 1.5 Hz = 0.67秒周期（速い歩行）

    Returns:
        torch.Tensor: shape (num_envs, 3) の観測値
            - [:, 0]: sin(2π * frequency * time)
            - [:, 1]: cos(2π * frequency * time)
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
    phase = 2.0 * torch.pi * frequency * time

    # sin と cos のペアを計算して返す
    # これにより位相情報が連続的な2次元空間で表現される
    sin_phase = torch.sin(phase).unsqueeze(-1)  # (num_envs, 1)
    cos_phase = torch.cos(phase).unsqueeze(-1)  # (num_envs, 1)
    combine_phase = sin_phase / 2 * torch.sqrt(torch.square(sin_phase) + 0.04) + 0.5  # (num_envs, 1)

    return torch.cat([sin_phase, cos_phase, combine_phase], dim=-1)  # (num_envs, 3)