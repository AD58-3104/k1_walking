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
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, euler_xyz_from_quat
from .data_logger import send_data_stream
from .observations import phase_time

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
    # send_data_stream({"ux": upright_vector[0, 0],
    #                   "uy": upright_vector[0, 1],
    #                   "ux2": torch.square(upright_vector[0, 0]),
    #                   "uy2": torch.square(upright_vector[0, 1]),
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

def joint_reqularization_potential(env: ManagerBasedRLEnv, sigma: float = 0.5, discount_factor: float = 0.99) -> torch.Tensor:
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

    Returns:
        torch.Tensor: Shaped reward for joint regularization
    """
    asset = env.scene["robot"]

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

    # Compute potential:
    # - Pitch: penalize deviation from default for each leg independently
    # - Roll: penalize left-right asymmetry
    # - Yaw: penalize deviation from default
    current_potential = torch.exp(-(torch.square(joint_pos_left_p)) / sigma).sum(dim=1) + \
                            torch.exp(-(torch.square(joint_pos_right_p)) / sigma).sum(dim=1) + \
                            torch.exp(-(torch.square(joint_pos_r)) / sigma).sum(dim=1) + \
                            torch.exp(-(torch.square(joint_pos_yr)) / sigma).sum(dim=1) + \
                            torch.exp(-(torch.square(joint_pos_yl)) / sigma).sum(dim=1)

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


def feet_parallel_to_ground(env: ManagerBasedRLEnv, sigma: float = 0.3, discount_factor: float = 0.99) -> torch.Tensor:
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

    # Compute squared errors for pitch and roll
    # When feet are parallel to ground, both pitch and roll should be ~0
    left_foot_error = torch.square(left_pitch) + torch.square(left_roll)
    right_foot_error = torch.square(right_pitch) + torch.square(right_roll)

    # Total error for both feet
    total_error = left_foot_error + right_foot_error

    current_potential = torch.exp(-total_error / sigma)

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

def feet_height_bezier(env: ManagerBasedRLEnv, swing_height: float = 0.09, sigma: float = 0.08, stance_ratio: float = 0.6) -> torch.Tensor:
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
    # send_data_stream({"rz_left": rz_left[0],
    #                   "rz_right": rz_right[0],
    #                   "left_foot_height": left_foot_height[0],
    #                   "right_foot_height": right_foot_height[0],
    #                   "error_left": error_left[0],
    #                   "error_right": error_right[0],
    #                   "reward": torch.exp(-total_error / sigma)[0],
    #                   })

    return torch.exp(-total_error / sigma)