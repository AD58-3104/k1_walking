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


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def orientation_potential(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma: float = 0.5 , discount_factor: float = 0.99) -> torch.Tensor:
    """
    Potential-based reward shaping for maintaining upright orientation.

    Returns the difference between current and previous potential values.
    This implements potential-based shaping: reward = Φ(s_t) - Φ(s_{t-1})

    The potential function is: Φ = exp(-(ux^2 + uy^2) / σ)
    where (ux, uy) are the x,y components of the upright vector in world frame.

    Args:
        env: Environment instance
        asset_cfg: Asset configuration (default: robot)
        sigma: Exponential kernel width parameter (default: 0.5)
        discount_factor: Discount factor for potential-based shaping (default: 0.99)
    Returns:
        torch.Tensor: Shaped reward (current_potential - previous_potential)

    Note:
        - Potential-based shaping preserves optimal policy
        - Automatically handles episode resets
        - Higher potential when robot is more upright
    """
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_cfg.name]

    # 現在のpotentialを計算
    # Z軸単位ベクトルをロボット座標系から世界座標系へ変換
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    upright_vector = quat_rotate_inverse(asset.data.root_quat_w, z_axis)

    # Potential function: exp(-(ux^2 + uy^2) / σ)
    # ロボットが完全に直立している時、upright_vector = [0, 0, 1] なので potential = 1
    # ロボットが傾くと、ux, uy が増加し、potential が減少
    current_potential = torch.exp(-(torch.square(upright_vector[:, 0]) + torch.square(upright_vector[:, 1])) / sigma)

    # バッファキー名
    buffer_key = "orientation_potential_prev"

    # カスタムバッファの初期化
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}

    if buffer_key not in env._custom_buffers:
        # 初回は前回のpotentialを現在値と同じにする（差分=0）
        env._custom_buffers[buffer_key] = current_potential.clone()

    prev_potential = env._custom_buffers[buffer_key]

    # 差分を計算（shaped reward）
    # Potential が増加 → 正の報酬（ロボットがより直立）
    # Potential が減少 → 負の報酬（ロボットが傾いた）
    shaped_reward = discount_factor * current_potential - prev_potential

    # リセットされた環境の処理
    # reset_buf > 0 の環境はリセットされたので、報酬を0にする
    reset_mask = env.reset_buf > 0
    shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)

    # 次回のために保存（リセットされた環境は新しいpotentialで初期化）
    env._custom_buffers[buffer_key] = current_potential.clone()

    return shaped_reward

def robot_height_potential(env: ManagerBasedRLEnv, target_height: float = 0.8, sigma: float = 0.5, discount_factor: float = 0.99, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    current_potential = torch.exp(-torch.square(current_height - target_height) / sigma)
    buffer_key = "robot_height_potential_prev"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = current_potential.clone()
    prev_potential = env._custom_buffers[buffer_key]
    shaped_reward = discount_factor * current_potential - prev_potential
    reset_mask = env.reset_buf > 0
    shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)
    env._custom_buffers[buffer_key] = current_potential.clone()
    return shaped_reward



def joint_reqularization_potential(env: ManagerBasedRLEnv, sigma: float = 0.5, discount_factor: float = 0.99) -> torch.Tensor:
    asset = env.scene["robot"]

    # Pitch joints (Hip, Knee, Ankle)
    asset_cfg_left_p = SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Left_Knee_Pitch", "Left_Ankle_Pitch"])
    asset_cfg_right_p = SceneEntityCfg("robot", joint_names=["Right_Hip_Pitch", "Right_Knee_Pitch", "Right_Ankle_Pitch"])
    asset_cfg_right_p.resolve(env.scene)
    asset_cfg_left_p.resolve(env.scene)

    joint_pos_p = (asset.data.joint_pos[:, asset_cfg_right_p.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_p.joint_ids]) \
                   - (asset.data.joint_pos[:, asset_cfg_left_p.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_p.joint_ids])

    # Roll joints (Hip, Ankle)
    asset_cfg_left_r = SceneEntityCfg("robot", joint_names=["Left_Hip_Roll", "Left_Ankle_Roll"])
    asset_cfg_right_r = SceneEntityCfg("robot", joint_names=["Right_Hip_Roll", "Right_Ankle_Roll"])
    asset_cfg_right_r.resolve(env.scene)
    asset_cfg_left_r.resolve(env.scene)
    joint_pos_r = (asset.data.joint_pos[:, asset_cfg_right_r.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_r.joint_ids]) \
                     - (asset.data.joint_pos[:, asset_cfg_left_r.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_r.joint_ids])

    # Yaw joints (Hip)
    asset_cfg_left_y = SceneEntityCfg("robot", joint_names=["Left_Hip_Yaw"])
    asset_cfg_right_y = SceneEntityCfg("robot", joint_names=["Right_Hip_Yaw"])
    asset_cfg_left_y.resolve(env.scene)
    asset_cfg_right_y.resolve(env.scene)
    joint_pos_yr = asset.data.joint_pos[:, asset_cfg_right_y.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_y.joint_ids]
    joint_pos_yl = asset.data.joint_pos[:, asset_cfg_left_y.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_y.joint_ids]

    current_potential = torch.exp(-torch.sum(torch.square(joint_pos_p), dim=1) / sigma) + torch.exp(-torch.sum(torch.square(joint_pos_r), dim=1) / sigma) + \
                            torch.exp(-torch.sum(torch.square(joint_pos_yr), dim=1) / sigma) + torch.exp(-torch.sum(torch.square(joint_pos_yl), dim=1) / sigma)

    buffer_key = "joint_reqularization_potential_prev"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = current_potential.clone()
    prev_potential = env._custom_buffers[buffer_key]
    shaped_reward = discount_factor * current_potential - prev_potential
    reset_mask = env.reset_buf > 0
    shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)
    env._custom_buffers[buffer_key] = current_potential.clone()
    return shaped_reward