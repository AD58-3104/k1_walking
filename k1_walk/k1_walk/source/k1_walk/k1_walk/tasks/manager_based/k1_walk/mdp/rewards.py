# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, euler_xyz_from_quat, wrap_to_pi
from .data_logger import send_data_stream
from .observations import phase_time

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_action_subset_indices(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    joint_name_patterns: list[str],
) -> torch.Tensor:
    """Resolve action-vector indices for joints matching the given regex patterns."""
    cache_key = f"action_subset_indices::{action_term_name}::{'|'.join(joint_name_patterns)}"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if cache_key in env._custom_buffers:
        return env._custom_buffers[cache_key]

    action_term = env.action_manager._terms[action_term_name]
    action_joint_names = action_term._joint_names
    matched_indices = [
        index
        for index, joint_name in enumerate(action_joint_names)
        if any(re.fullmatch(pattern, joint_name) for pattern in joint_name_patterns)
    ]
    if not matched_indices:
        raise ValueError(
            f"No action joints matched patterns {joint_name_patterns} for action term '{action_term_name}'."
        )

    subset_indices = torch.tensor(matched_indices, device=env.device, dtype=torch.long)
    env._custom_buffers[cache_key] = subset_indices
    return subset_indices


def action_rate_l2_subset(
    env: ManagerBasedRLEnv,
    joint_name_patterns: list[str],
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize action-rate L2 only for a subset of action-controlled joints."""
    subset_indices = _resolve_action_subset_indices(env, action_term_name, joint_name_patterns)
    actions = env.action_manager.action[:, subset_indices]
    prev_actions = env.action_manager.prev_action[:, subset_indices]
    return torch.sum(torch.square(actions - prev_actions), dim=1)


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

def orientation_potential(env: ManagerBasedRLEnv, 
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                          sigma: float = 0.5 , 
                          discount_factor: float = 0.99,
                          enable_potential: bool = True,
                          ) -> torch.Tensor:
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
    err_value = torch.square(upright_vector[:, 0]) + torch.square(upright_vector[:, 1])
    current_potential = torch.exp(-err_value / sigma)

    if not enable_potential:
        return err_value

    # send_data_stream({"ux": upright_vector[0, 0],
    #                   "uy": upright_vector[0, 1],
    #                   "ux2": torch.square(upright_vector[0, 0]),
    #                   "uy2": torch.square(upright_vector[0, 1]),
    #                     "potential": current_potential[0],
    #                   })

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

def second_order_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    buffer_key = "prev_prev_action"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = env.action_manager.action.clone()
    prev_prev_action = env._custom_buffers[buffer_key]
    env._custom_buffers[buffer_key] = env.action_manager.prev_action.clone()
    # send_data_stream({"action_rate": torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action),dim=1)[0]})
    return torch.sum(torch.square((env.action_manager.action - 2 * env.action_manager.prev_action + prev_prev_action) / env.step_dt), dim=1)


def foot_ref_height(env: ManagerBasedRLEnv, target_height: float = 0.2,frequency = 1.5 , sigma: float = 0.5, min_height: float = 0.02) -> torch.Tensor:
    
    # 初期化時にepisode_length_bufが存在しない場合はゼロ配列を使用
    if hasattr(env, 'episode_length_buf'):
        time = env.episode_length_buf.float() * env.step_dt
    else:
        # 初期化時のダミー値（次元確認用）
        time = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    # 位相角度を計算（ラジアン）
    # 2π * f * t で周期的な角度を生成
    phase_right = 2.0 * torch.pi * (time / frequency)
    phase_left = 2.0 * torch.pi * (time / (frequency) + 0.5)  # 左足は右足と位相がπずれる
    asset = env.scene["robot"]
    foot_right_height = asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 2]
    foot_left_height = asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 2]
    
    desired_foot_right_height = torch.clamp(target_height * torch.clamp(torch.sin(phase_right),min=0.0),min=min_height)  # 0以下は単に0にする
    desired_foot_left_height = torch.clamp(target_height * torch.clamp(torch.sin(phase_left),min=0.0),min=min_height)

    right_foot_height_error = torch.square(foot_right_height - desired_foot_right_height)
    left_foot_height_error = torch.square(foot_left_height - desired_foot_left_height)

    total_height_error = right_foot_height_error + left_foot_height_error

    shaped_reward = torch.exp(-total_height_error / sigma)

    # send_data_stream({"foot_right_height": foot_right_height[0],
    #                   "foot_left_height": foot_left_height[0],
    #                     "desired_foot_right_height": desired_foot_right_height[0],
    #                     "desired_foot_left_height": desired_foot_left_height[0],
    #                     "reward": shaped_reward[0],
    #                   })

    return shaped_reward

def upper_body_joint_regularization(
            env: ManagerBasedRLEnv, 
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(joint_pos_rel), dim=1)
    regularization_reward = err
    # send_data_stream(
    #     {
    #         "err": err[0],
    #     }
    # )

    return regularization_reward

def joint_reqularization_potential(env: ManagerBasedRLEnv, sigma: float = 0.5, 
                                    discount_factor: float = 0.99,
                                    pitch_slack: list[float] = [1.0, 1.0, 1.0],
                                    roll_slack: list[float] = [1.0, 1.0],
                                    yaw_slack: float = 0.8,
                                    enable_exp_func: bool = True,
                                    enable_potential: bool = True
                                    ) -> torch.Tensor:
    """Regularization potential for joint positions.

    This function encourages:
    - Pitch joints (Hip, Knee, Ankle): Stay close to default positions (NOT left-right symmetry)
    - Roll joints (Hip, Ankle): Left-right symmetry
    - Yaw joints (Hip): Stay close to default positions

    This allows natural asymmetric walking motion while preventing extreme joint angles.

    Args:
        env: Environment instance
        sigma: Exponential kernel width parameter
        discount_factor: Discount factor for potential-based shaping
        pitch_slack: ピッチに掛けるペナルティを緩和するためのスラック変数
                    [Hip_Pitch", Knee_Pitch, Ankle_Pitch] の順番
        roll_slack: ロールに掛けるペナルティを緩和するためのスラック変数
                    [Hip_Roll, Ankle_Roll] の順番
        yaw_slack: ヨーに掛けるペナルティを緩和するためのスラック変数

    Returns:
        torch.Tensor: Shaped reward for joint regularization
    """
    asset = env.scene["robot"]
    # テンソルに変換
    pitch_slack = torch.tensor(pitch_slack, device=env.device)
    roll_slack = torch.tensor(roll_slack, device=env.device)

    # Pitch joints (Hip, Knee, Ankle) - penalize deviation from default, NOT symmetry
    # This allows left and right legs to have different angles during walking
    asset_cfg_left_p = SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch", "Left_Knee_Pitch", "Left_Ankle_Pitch"])
    asset_cfg_right_p = SceneEntityCfg("robot", joint_names=["Right_Hip_Pitch", "Right_Knee_Pitch", "Right_Ankle_Pitch"])
    asset_cfg_right_p.resolve(env.scene)
    asset_cfg_left_p.resolve(env.scene)

    # Penalize each leg's deviation from default independently (not left-right difference)
    joint_pos_left_p = asset.data.joint_pos[:, asset_cfg_left_p.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_p.joint_ids]
    joint_pos_right_p = asset.data.joint_pos[:, asset_cfg_right_p.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_p.joint_ids]

    # Roll joints (Hip, Ankle) - enforce left-right symmetry for balance
    asset_cfg_left_r = SceneEntityCfg("robot", joint_names=["Left_Hip_Roll", "Left_Ankle_Roll"])
    asset_cfg_right_r = SceneEntityCfg("robot", joint_names=["Right_Hip_Roll", "Right_Ankle_Roll"])
    asset_cfg_right_r.resolve(env.scene)
    asset_cfg_left_r.resolve(env.scene)
    joint_pos_r = (asset.data.joint_pos[:, asset_cfg_right_r.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_r.joint_ids]) \
                     - (asset.data.joint_pos[:, asset_cfg_left_r.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_r.joint_ids])

    # Yaw joints (Hip) - penalize deviation from default
    asset_cfg_left_y = SceneEntityCfg("robot", joint_names=["Left_Hip_Yaw"])
    asset_cfg_right_y = SceneEntityCfg("robot", joint_names=["Right_Hip_Yaw"])
    asset_cfg_left_y.resolve(env.scene)
    asset_cfg_right_y.resolve(env.scene)
    joint_pos_yr = asset.data.joint_pos[:, asset_cfg_right_y.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_right_y.joint_ids]
    joint_pos_yl = asset.data.joint_pos[:, asset_cfg_left_y.joint_ids] - asset.data.default_joint_pos[:, asset_cfg_left_y.joint_ids]
    yaw_slack_when_rotate = torch.where(env.command_manager.get_command("base_velocity")[:, :2].norm(dim=1) > 0.1, yaw_slack, 4.0)  # コマンド速度が大きいときはyawのペナルティを緩和する


    # Compute potential:
    # - Pitch: penalize deviation from default for each leg independently
    # - Roll: penalize left-right asymmetry
    # - Yaw: penalize deviation from default
    if enable_exp_func:
        current_potential = torch.exp(-(torch.square(joint_pos_left_p * pitch_slack)) / sigma).sum(dim=1) + \
                                torch.exp(-(torch.square(joint_pos_right_p * pitch_slack)) / sigma).sum(dim=1) + \
                                torch.exp(-(torch.square(joint_pos_r * roll_slack)) / sigma).sum(dim=1) + \
                                torch.exp(-(torch.square(joint_pos_yr * yaw_slack_when_rotate)) / sigma).sum(dim=1) + \
                                torch.exp(-(torch.square(joint_pos_yl * yaw_slack_when_rotate)) / sigma).sum(dim=1)
    else:
        current_potential = (torch.square(joint_pos_left_p  ) * pitch_slack).sum(dim=1) + \
                                (torch.square(joint_pos_right_p) * pitch_slack).sum(dim=1) + \
                                (torch.square(joint_pos_r ) * roll_slack).sum(dim=1) + \
                                (torch.square(joint_pos_yr) * yaw_slack_when_rotate).sum(dim=1) + \
                                (torch.square(joint_pos_yl) * yaw_slack_when_rotate).sum(dim=1)

    # ポテンシャルベースでない場合、そのままを負の報酬として返す
    if not enable_potential:
        return -current_potential

    # send_data_stream({
    #     "joint_pos_left_p": (torch.square(joint_pos_left_p * pitch_slack))[0],
    #     "pos_p": joint_pos_left_p * pitch_slack,
    #     "joint_pos_r": (torch.square(joint_pos_r * roll_slack))[0],
    #     "pos_r": joint_pos_r * roll_slack,
    #     "joint_pos_yl": (torch.square(joint_pos_yl * yaw_slack_when_rotate))[0],
    #     "pos_yl": joint_pos_yl * yaw_slack_when_rotate,
    # })

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


def feet_y_distance(env: ManagerBasedRLEnv, feet_distance_ref = 0.3,sigma = 0.5, discount_factor = 0.99 ) -> torch.Tensor:
    y_vel_com = env.command_manager.get_command("base_velocity")[:, 1]
    y_vel_scale = torch.exp(-torch.abs(y_vel_com) / 0.4)  # コマンドのy方向速度が大きいほど、足のy距離が目標値から乖離することを許容するためのスケーリング
    y_offset = torch.square(get_feet_offset(env, feet_distance_ref)[1])
    y_offset_reward = torch.exp(-y_offset / sigma * y_vel_scale)
    # send_data_stream({"y_offset": y_offset[0]})
    buffer_key = "feet_y_distance_potential_prev"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = y_offset_reward.clone()
    prev_potential = env._custom_buffers[buffer_key]
    shaped_reward = discount_factor * y_offset_reward - prev_potential
    reset_mask = env.reset_buf > 0
    shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)
    env._custom_buffers[buffer_key] = y_offset_reward.clone()
    return shaped_reward

def get_feet_offset(env: ManagerBasedRLEnv, feet_distance_ref = 0.3) -> torch.Tensor:
    """Get the offset between left and right foot in the robot frame.

    This function computes the offset between the left and right foot positions in the robot frame.
    TODO:これ計算あってる？
    """
    asset = env.scene["robot"]
    _,_,base_yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    feet_x_offset = (
        torch.cos(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 0] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 0])
         - torch.sin(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 1] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 1])
    )
    feet_y_offset = (
        -torch.sin(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 0] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 0])
         + torch.cos(base_yaw) * (asset.data.body_pos_w[:, asset.find_bodies("left_foot_link")[0][0], 1] - asset.data.body_pos_w[:, asset.find_bodies("right_foot_link")[0][0], 1])
    )

    feet_y_offset = feet_y_offset - feet_distance_ref
    return feet_x_offset, feet_y_offset

def feet_close_penalty(env: ManagerBasedRLEnv, feet_distance_threshold = 0.15) -> torch.Tensor:
    """Penalize feet being too close.

    This function penalizes the agent for having its feet too close together. The reward is computed as the
    distance between the feet positions.
    """
    _, feet_y_offset = get_feet_offset(env, 0.0) # そのままの値が欲しいのでrefは0にする

    return (feet_y_offset < feet_distance_threshold).float()

def knee_limit_lower(env: ManagerBasedRLEnv, knee_limit_angle: float = 0.0) -> torch.Tensor:
    """Penalize knee joint limit violation.

    This function penalizes the agent for violating the knee joint limits.
    """
    asset = env.scene["robot"]
    asset_cfg_left_knee = SceneEntityCfg("robot", joint_names=["Left_Knee_Pitch"])
    asset_cfg_right_knee = SceneEntityCfg("robot", joint_names=["Right_Knee_Pitch"])
    asset_cfg_left_knee.resolve(env.scene)
    asset_cfg_right_knee.resolve(env.scene)

    left_knee_pos = asset.data.joint_pos[:, asset_cfg_left_knee.joint_ids]
    right_knee_pos = asset.data.joint_pos[:, asset_cfg_right_knee.joint_ids]

    left_knee_violation = torch.clamp(-left_knee_pos - knee_limit_angle, min=0.0)
    right_knee_violation = torch.clamp(-right_knee_pos - knee_limit_angle, min=0.0)

    total_violation = left_knee_violation + right_knee_violation

    return total_violation.squeeze(-1)


def feet_parallel_to_ground(env: ManagerBasedRLEnv, 
                            sigma: float = 0.3,
                            enable_potential: bool = True, 
                            discount_factor: float = 0.99) -> torch.Tensor:
    """Reward feet being parallel to the ground.

    This function rewards the agent for keeping its feet parallel to the ground.
    The reward is computed based on the pitch and roll angles of the feet.
    When the feet are perfectly parallel to the ground, pitch and roll should be close to zero.

    Args:
        env: Environment instance
        sigma: Exponential kernel width parameter (default: 0.3)

    Returns:
        torch.Tensor: Reward value for each environment
    """
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    # Get foot orientations (quaternions) in world frame
    left_foot_quat = asset.data.body_quat_w[:, left_foot_idx, :]
    right_foot_quat = asset.data.body_quat_w[:, right_foot_idx, :]

    # Convert quaternions to euler angles (roll, pitch, yaw)
    left_roll, left_pitch, _ = euler_xyz_from_quat(left_foot_quat)
    right_roll, right_pitch, _ = euler_xyz_from_quat(right_foot_quat)
    left_roll = wrap_to_pi(left_roll)
    right_roll = wrap_to_pi(right_roll)
    left_pitch = wrap_to_pi(left_pitch)
    right_pitch = wrap_to_pi(right_pitch)

    # Compute squared errors for pitch and roll
    # When feet are parallel to ground, both pitch and roll should be ~0
    left_foot_error = torch.square(left_pitch) + torch.square(left_roll)
    right_foot_error = torch.square(right_pitch) + torch.square(right_roll)

    # Total error for both feet
    total_error = left_foot_error + right_foot_error

    current_potential = torch.exp(-total_error / sigma)

    # send_data_stream({
    #     "left_foot_error": left_foot_error[0],
    #     "left_pitch": left_pitch[0],
    #     "left_roll": left_roll[0],
    #     "right_foot_error": right_foot_error[0],
    #     "right_pitch": right_pitch[0],
    #     "right_roll": right_roll[0],
    #     "total_foot_error": total_error[0],
    #     "value": current_potential[0]
    # })


    if enable_potential:
        buffer_key = "feet_parallel_to_ground_potential_prev"
        if not hasattr(env, "_custom_buffers"):
            env._custom_buffers = {}
        if buffer_key not in env._custom_buffers:
            env._custom_buffers[buffer_key] = current_potential.clone()
        prev_potential = env._custom_buffers[buffer_key]
        shaped_reward = discount_factor * current_potential - prev_potential
        reset_mask = env.reset_buf > 0
        shaped_reward = torch.where(reset_mask, torch.zeros_like(shaped_reward), shaped_reward)
        env._custom_buffers[buffer_key] = current_potential.clone()
    else:
        shaped_reward = current_potential

    return shaped_reward



def foot_clearance_ji(env: ManagerBasedRLEnv, target_clearance: float = 0.09) -> torch.Tensor:
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
    right_foot_xy_vel_sq = torch.sqrt(torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :2], dim=1))
    left_foot_xy_vel_sq = torch.sqrt(torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :2], dim=1))

    right_foot_height_err = torch.square(target_clearance - asset.data.body_pos_w[:, right_foot_idx, 2])
    left_foot_height_err = torch.square(target_clearance - asset.data.body_pos_w[:, left_foot_idx, 2])

    right_reward = right_foot_height_err * right_foot_xy_vel_sq
    left_reward = left_foot_height_err * left_foot_xy_vel_sq
    return right_reward + left_reward


def _expected_foot_height_bezier(phi: torch.Tensor, swing_height: float, stance_ratio: float = 0.5) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile.

    Args:
        phi: Gait phase in [-pi, pi].
        swing_height: Peak foot height during swing [m].
        stance_ratio: Fraction of the cycle spent in stance (foot on ground).
                      e.g. 0.6 means 60% stance, 40% swing.
    """

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)  # x ∈ [0, 1]

    # Normalize to swing phase: t ∈ [0, 1] only during swing; clamped to 0 during stance
    t = torch.clamp((x - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)

    up   = cubic_bezier_interpolation(torch.zeros_like(t), torch.full_like(t, swing_height), 2 * t)
    down = cubic_bezier_interpolation(torch.full_like(t, swing_height), torch.zeros_like(t), 2 * t - 1)
    profile = torch.where(t <= 0.5, up, down)

    return torch.where(x <= stance_ratio, torch.zeros_like(x), profile)

def feet_height_bezier(env: ManagerBasedRLEnv, 
                        swing_height: float = 0.09, 
                        sigma: float = 0.08, 
                        stance_ratio: float = 0.5,
                        ground_height: float = 0.02) -> torch.Tensor:
    asset = env.scene["robot"]

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    # Compute foot heights
    right_foot_height = asset.data.body_pos_w[:, right_foot_idx, 2]
    left_foot_height = asset.data.body_pos_w[:, left_foot_idx, 2]

    phase = phase_time(env)
    rz_left = _expected_foot_height_bezier(phase[:, 0], swing_height, stance_ratio)
    rz_right = _expected_foot_height_bezier(phase[:, 1], swing_height, stance_ratio)

    # 歩行コマンドが非常に小さい時は目標高さは0にする（足を上げない歩行も許容する）
    command_lin_vel = env.command_manager.get_command("base_velocity")[:, :2]
    command_speed = torch.norm(command_lin_vel, dim=1)
    rz_left = torch.where(command_speed > 0.1, rz_left, torch.zeros_like(rz_left))
    rz_right = torch.where(command_speed > 0.1, rz_right, torch.zeros_like(rz_right))
    rz_left = torch.clamp(rz_left, min=ground_height)
    rz_right = torch.clamp(rz_right, min=ground_height)

    # send_data_stream({
    #     "knee_torque": asset.data.applied_torque[:, asset.find_joints(".*_Knee_.*")[0]].tolist(),
    #     "hip_torque": asset.data.applied_torque[:, asset.find_joints(".*_Hip_.*")[0]].tolist(),
    #     "ankle_torque": asset.data.applied_torque[:, asset.find_joints(".*_Ankle_.*")[0]].tolist(),
    # })
    # send_data_stream({
    #     "knee_vel": asset.data.joint_vel[:, asset.find_joints(".*_Knee_.*")[0]].tolist(),
    #     "hip_vel": asset.data.joint_vel[:, asset.find_joints(".*_Hip_.*")[0]].tolist(),
    #     "ankle_vel": asset.data.joint_vel[:, asset.find_joints(".*_Ankle_.*")[0]].tolist(),
    # })
    # send_data_stream({
    #     "knee_acc": asset.data.joint_acc[:, asset.find_joints(".*_Knee_.*")[0]].tolist(),
    #     "hip_acc": asset.data.joint_acc[:, asset.find_joints(".*_Hip_.*")[0]].tolist(),
    #     "ankle_acc": asset.data.joint_acc[:, asset.find_joints(".*_Ankle_.*")[0]].tolist(),
    # })

    # Calculate height tracking errors
    error_left = torch.square(left_foot_height - rz_left)
    error_right = torch.square(right_foot_height - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right
    # send_data_stream({
    #                   "phase": phase[0].tolist(),
    #                   "rz_left": rz_left[0],
    #                   "rz_right": rz_right[0],
    #                   "left_foot_height": left_foot_height[0],
    #                   "right_foot_height": right_foot_height[0],
    #                   "error_left": error_left[0],
    #                   "error_right": error_right[0],
    #                   "reward": torch.exp(-total_error / sigma)[0],
    #                   "right_foot_vel": torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :2], dim=1)[0],
    #                     "left_foot_vel": torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :2], dim=1)[0],
    #                   })

    return torch.exp(-total_error / sigma)


def stride_length_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_stride: float = 0.3,
    max_speed: float = 1.0,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for achieving target stride length scaled by velocity command.

    Stride length is measured as the xy-distance (norm) that each foot travels
    between consecutive ground contacts. The target stride scales linearly with
    the commanded velocity: target = max_stride * (command_speed / max_speed).

    Args:
        env: Environment instance
        command_name: Name of the velocity command
        sensor_cfg: Configuration for the contact sensor (should include both feet)
        max_stride: Maximum target stride length at max speed [m]
        max_speed: Maximum expected velocity command [m/s]
        sigma: Exponential kernel width parameter
        asset_cfg: Asset configuration (default: robot)

    Returns:
        torch.Tensor: Reward value for each environment
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Get velocity command and compute speed ratio
    command_vel_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.norm(command_vel_xy, dim=1)
    speed_ratio = torch.clamp(command_speed / max_speed, min=0.0, max=1.0)

    # Scale target stride by speed ratio
    target_stride = max_stride * speed_ratio

    # Get foot body indices
    left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
    right_foot_idx = asset.find_bodies("right_foot_link")[0][0]

    # Get current foot positions (xy only)
    left_foot_pos_xy = asset.data.body_pos_w[:, left_foot_idx, :2]
    right_foot_pos_xy = asset.data.body_pos_w[:, right_foot_idx, :2]

    # Detect first contact (foot just landed)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # sensor_cfg.body_ids should be [left_foot, right_foot] or similar

    # Initialize buffers for storing previous landing positions
    buffer_key_left = "stride_left_foot_last_landing_pos"
    buffer_key_right = "stride_right_foot_last_landing_pos"
    buffer_key_last_landed = "stride_last_landed_foot"  # 0: none, 1: left, 2: right

    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}

    if buffer_key_left not in env._custom_buffers:
        env._custom_buffers[buffer_key_left] = left_foot_pos_xy.clone()
    if buffer_key_right not in env._custom_buffers:
        env._custom_buffers[buffer_key_right] = right_foot_pos_xy.clone()
    if buffer_key_last_landed not in env._custom_buffers:
        env._custom_buffers[buffer_key_last_landed] = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    prev_left_pos = env._custom_buffers[buffer_key_left]
    prev_right_pos = env._custom_buffers[buffer_key_right]
    last_landed_foot = env._custom_buffers[buffer_key_last_landed]

    # Calculate stride length for each foot (xy-norm)
    left_stride = torch.norm(left_foot_pos_xy - prev_left_pos, dim=1)
    right_stride = torch.norm(right_foot_pos_xy - prev_right_pos, dim=1)

    # Compute reward based on how close stride is to target
    # Only give reward when foot just landed (first_contact)
    # first_contact shape: (num_envs, num_feet)
    # Assuming body_ids[0] is left foot, body_ids[1] is right foot
    left_contact = first_contact[:, 0] if first_contact.shape[1] > 0 else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    right_contact = first_contact[:, 1] if first_contact.shape[1] > 1 else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Check for consecutive landing of the same foot (hopping penalty)
    # last_landed_foot: 0=none, 1=left, 2=right
    left_consecutive = (last_landed_foot == 1) & left_contact
    right_consecutive = (last_landed_foot == 2) & right_contact

    # Reward: exponential of negative squared error from target stride
    # Zero reward if the same foot lands consecutively (hopping motion)
    left_reward = torch.exp(-torch.square(left_stride - target_stride) / sigma) * left_contact.float() * (~left_consecutive).float()
    right_reward = torch.exp(-torch.square(right_stride - target_stride) / sigma) * right_contact.float() * (~right_consecutive).float()

    # send_data_stream({
    #     "command_speed": command_speed[0],
    #     "target_stride": target_stride[0],
    #     "left_stride": left_stride[0],
    #     "right_stride": right_stride[0],
    # })
    reward = left_reward + right_reward

    # Update stored positions when foot lands
    env._custom_buffers[buffer_key_left] = torch.where(
        left_contact.unsqueeze(-1), left_foot_pos_xy, prev_left_pos
    )
    env._custom_buffers[buffer_key_right] = torch.where(
        right_contact.unsqueeze(-1), right_foot_pos_xy, prev_right_pos
    )

    # Update last landed foot: 1=left, 2=right (prioritize left if both land simultaneously)
    new_last_landed = torch.where(
        left_contact,
        torch.ones_like(last_landed_foot),
        torch.where(right_contact, torch.full_like(last_landed_foot, 2), last_landed_foot)
    )
    env._custom_buffers[buffer_key_last_landed] = new_last_landed

    # Reset stored positions on environment reset
    reset_mask = env.reset_buf > 0
    env._custom_buffers[buffer_key_left] = torch.where(
        reset_mask.unsqueeze(-1), left_foot_pos_xy, env._custom_buffers[buffer_key_left]
    )
    env._custom_buffers[buffer_key_right] = torch.where(
        reset_mask.unsqueeze(-1), right_foot_pos_xy, env._custom_buffers[buffer_key_right]
    )
    # Reset last landed foot on environment reset
    env._custom_buffers[buffer_key_last_landed] = torch.where(
        reset_mask, torch.zeros_like(last_landed_foot), env._custom_buffers[buffer_key_last_landed]
    )

    # No reward for zero command (standing still)
    reward *= command_speed > 0.1

    return reward

def joint_jerk(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize high joint jerk (third derivative of position).

    This function computes the joint jerk for all joints and penalizes high values to encourage smooth motion.
    """
    asset = env.scene[asset_cfg.name]
    # Compute joint acceleration and jerk
    joint_acc = asset.data.joint_acc
    # Approximate jerk using finite difference: jerk ≈ (acc_t - acc_{t-1}) / dt
    buffer_key = "prev_joint_acc"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = joint_acc.clone()
    prev_joint_acc = env._custom_buffers[buffer_key]
    env._custom_buffers[buffer_key] = joint_acc.clone()
    joint_jerk = (joint_acc - prev_joint_acc) / env.step_dt
    # Penalize high jerk (L2 norm across all joints)
    jerk_penalty = torch.norm(joint_jerk, dim=1)
    return jerk_penalty



def base_jerk(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize high base jerk (third derivative of position).

    This function computes the base jerk for the robot's root body and penalizes high values to encourage smooth motion.
    """
    asset = env.scene[asset_cfg.name]
    # Compute base linear acceleration and jerk
    base_lin_acc = asset.data.root_com_lin_vel_w
    # Approximate jerk using finite difference: jerk ≈ (acc_t - acc_{t-1}) / dt
    buffer_key = "prev_base_lin_acc"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key not in env._custom_buffers:
        env._custom_buffers[buffer_key] = base_lin_acc.clone()
    prev_base_lin_acc = env._custom_buffers[buffer_key]
    env._custom_buffers[buffer_key] = base_lin_acc.clone()
    base_jerk = (base_lin_acc - prev_base_lin_acc) / env.step_dt
    # Penalize high jerk (L2 norm)
    jerk_penalty = torch.norm(base_jerk[:, :2], dim=1)
    return jerk_penalty


def bad_gait_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_air_time: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    悪い歩容にペナルティを掛ける。
    悪い歩容は、片脚での連続接地と指定した時間以下の遊脚期間で定義される

    Args:
        env: Environment instance
        sensor_cfg: Configuration for the contact sensor (should include both feet)
        min_air_time: 最小の遊脚時間。これより短い遊脚はペナルティ対象 [s]
        asset_cfg: Asset configuration (default: robot)

    Returns:
        torch.Tensor: ペナルティ値 (悪い歩容ほど高い値)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 接地検出（first_contact = 今ステップで初めて接地した）
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # last_air_time = 直前の遊脚期間
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # sensor_cfg.body_ids: [left_foot, right_foot] を想定
    left_contact = first_contact[:, 0] if first_contact.shape[1] > 0 else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    right_contact = first_contact[:, 1] if first_contact.shape[1] > 1 else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    left_air_time = last_air_time[:, 0] if last_air_time.shape[1] > 0 else torch.zeros(env.num_envs, device=env.device)
    right_air_time = last_air_time[:, 1] if last_air_time.shape[1] > 1 else torch.zeros(env.num_envs, device=env.device)

    # バッファ初期化: 最後に接地した足を記録 (0: none, 1: left, 2: right)
    buffer_key_last_landed = "bad_gait_last_landed_foot"
    if not hasattr(env, "_custom_buffers"):
        env._custom_buffers = {}
    if buffer_key_last_landed not in env._custom_buffers:
        env._custom_buffers[buffer_key_last_landed] = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    last_landed_foot = env._custom_buffers[buffer_key_last_landed]

    # ペナルティ1: 同じ足が連続で接地（ホッピング）
    left_consecutive = (last_landed_foot == 1) & left_contact
    right_consecutive = (last_landed_foot == 2) & right_contact
    consecutive_penalty = (left_consecutive | right_consecutive).float()

    # ペナルティ2: 遊脚時間が短すぎる
    # 接地した瞬間のみチェック（first_contact時のlast_air_timeを見る）
    left_short_air = (left_air_time < min_air_time) & left_contact & (left_air_time > 0)
    right_short_air = (right_air_time < min_air_time) & right_contact & (right_air_time > 0)
    short_air_penalty = (left_short_air | right_short_air).float()

    # 最後に接地した足を更新
    new_last_landed = torch.where(
        left_contact,
        torch.ones_like(last_landed_foot),
        torch.where(right_contact, torch.full_like(last_landed_foot, 2), last_landed_foot)
    )
    env._custom_buffers[buffer_key_last_landed] = new_last_landed

    # リセット時の処理
    reset_mask = env.reset_buf > 0
    env._custom_buffers[buffer_key_last_landed] = torch.where(
        reset_mask, torch.zeros_like(last_landed_foot), env._custom_buffers[buffer_key_last_landed]
    )

    # 合計ペナルティ
    total_penalty = consecutive_penalty + short_air_penalty

    # コマンド速度が小さい時はペナルティなし（静止時）
    command_vel_xy = env.command_manager.get_command("base_velocity")[:, :2]
    command_speed = torch.norm(command_vel_xy, dim=1)
    total_penalty *= (command_speed > 0.1).float()

    return total_penalty
