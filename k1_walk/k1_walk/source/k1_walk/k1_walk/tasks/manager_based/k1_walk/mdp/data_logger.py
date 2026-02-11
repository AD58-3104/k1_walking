# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data logger for recording robot state, foot trajectories, and command tracking."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List
from pathlib import Path
import pickle

from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RobotDataLogger:
    """Logs robot posture, foot trajectories, and command tracking data every step.

    Records:
    - Robot base position and orientation (quaternion and euler angles)
    - Left and right foot positions
    - Commanded velocities (linear x, y, angular z)
    - Actual velocities (linear x, y, angular z)
    - Tracking error
    """

    def __init__(self, env: ManagerBasedRLEnv, max_steps: int = 10000, env_ids: list = None):
        """Initialize the data logger.

        Args:
            env: The environment instance
            max_steps: Maximum number of steps to record (default: 10000)
            env_ids: List of environment IDs to record (default: [0] - only first env)
        """
        self.env = env
        self.device = env.device
        self.max_steps = max_steps

        # Which environments to record (default: only first one)
        if env_ids is None:
            self.env_ids = [0]
        else:
            self.env_ids = env_ids

        self.num_recorded_envs = len(self.env_ids)

        # Data buffers
        self.data = {
            # Robot base data
            'base_pos': [],  # [step, env, 3]
            'base_quat': [],  # [step, env, 4]
            'base_euler': [],  # [step, env, 3] (roll, pitch, yaw)
            'base_lin_vel': [],  # [step, env, 3]
            'base_ang_vel': [],  # [step, env, 3]

            # Foot positions
            'left_foot_pos': [],  # [step, env, 3]
            'right_foot_pos': [],  # [step, env, 3]

            # Command tracking
            'cmd_lin_vel_x': [],  # [step, env]
            'cmd_lin_vel_y': [],  # [step, env]
            'cmd_ang_vel_z': [],  # [step, env]
            'actual_lin_vel_x': [],  # [step, env]
            'actual_lin_vel_y': [],  # [step, env]
            'actual_ang_vel_z': [],  # [step, env]

            # Tracking error
            'lin_vel_error': [],  # [step, env]
            'ang_vel_error': [],  # [step, env]
        }

        self.step_count = 0

    def record(self, command_name: str = "base_velocity"):
        """Record data for the current step.

        Args:
            command_name: Name of the velocity command (default: "base_velocity")
        """
        if self.step_count >= self.max_steps:
            return True # Indicate that max steps reached

        asset = self.env.scene["robot"]

        # Select only the environments we want to record
        env_indices = torch.tensor(self.env_ids, device=self.device)

        # Robot base data
        base_pos = asset.data.root_pos_w[env_indices].cpu().numpy()
        base_quat = asset.data.root_quat_w[env_indices].cpu().numpy()

        # Convert quaternion to euler angles
        roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w[env_indices])
        base_euler = torch.stack([roll, pitch, yaw], dim=-1).cpu().numpy()

        # Velocities
        base_lin_vel = asset.data.root_lin_vel_w[env_indices].cpu().numpy()
        base_ang_vel = asset.data.root_ang_vel_w[env_indices].cpu().numpy()

        # Foot positions
        left_foot_idx = asset.find_bodies("left_foot_link")[0][0]
        right_foot_idx = asset.find_bodies("right_foot_link")[0][0]
        left_foot_pos = asset.data.body_pos_w[env_indices, left_foot_idx].cpu().numpy()
        right_foot_pos = asset.data.body_pos_w[env_indices, right_foot_idx].cpu().numpy()

        # Commands
        commands = self.env.command_manager.get_command(command_name)[env_indices]
        cmd_lin_vel_x = commands[:, 0].cpu().numpy()
        cmd_lin_vel_y = commands[:, 1].cpu().numpy()
        cmd_ang_vel_z = commands[:, 2].cpu().numpy()

        # Actual velocities in yaw frame (gravity-aligned)
        vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w[env_indices]),
                                      asset.data.root_lin_vel_w[env_indices])
        actual_lin_vel_x = vel_yaw[:, 0].cpu().numpy()
        actual_lin_vel_y = vel_yaw[:, 1].cpu().numpy()
        actual_ang_vel_z = asset.data.root_ang_vel_w[env_indices, 2].cpu().numpy()

        # Tracking error
        lin_vel_error = np.sqrt((cmd_lin_vel_x - actual_lin_vel_x)**2 +
                                (cmd_lin_vel_y - actual_lin_vel_y)**2)
        ang_vel_error = np.abs(cmd_ang_vel_z - actual_ang_vel_z)

        # Store data
        self.data['base_pos'].append(base_pos)
        self.data['base_quat'].append(base_quat)
        self.data['base_euler'].append(base_euler)
        self.data['base_lin_vel'].append(base_lin_vel)
        self.data['base_ang_vel'].append(base_ang_vel)
        self.data['left_foot_pos'].append(left_foot_pos)
        self.data['right_foot_pos'].append(right_foot_pos)
        self.data['cmd_lin_vel_x'].append(cmd_lin_vel_x)
        self.data['cmd_lin_vel_y'].append(cmd_lin_vel_y)
        self.data['cmd_ang_vel_z'].append(cmd_ang_vel_z)
        self.data['actual_lin_vel_x'].append(actual_lin_vel_x)
        self.data['actual_lin_vel_y'].append(actual_lin_vel_y)
        self.data['actual_ang_vel_z'].append(actual_ang_vel_z)
        self.data['lin_vel_error'].append(lin_vel_error)
        self.data['ang_vel_error'].append(ang_vel_error)

        self.step_count += 1
        return False

    def get_data_as_arrays(self) -> Dict[str, np.ndarray]:
        """Convert recorded data to numpy arrays.

        Returns:
            Dictionary with arrays of shape [num_steps, num_envs, ...]
        """
        arrays = {}
        for key, value_list in self.data.items():
            if len(value_list) > 0:
                arrays[key] = np.array(value_list)
            else:
                arrays[key] = np.array([])
        return arrays

    def save(self, filepath: str | Path):
        """Save recorded data to a pickle file.

        Args:
            filepath: Path to save the data
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'data': self.get_data_as_arrays(),
            'env_ids': self.env_ids,
            'num_steps': self.step_count,
            'max_steps': self.max_steps,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[DataLogger] Saved {self.step_count} steps to {filepath}")

    def save_csv(self, output_dir: str | Path):
        """Save recorded data to CSV files (one file per environment).

        Args:
            output_dir: Directory to save CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        arrays = self.get_data_as_arrays()

        for env_idx in range(self.num_recorded_envs):
            env_id = self.env_ids[env_idx]
            filepath = output_dir / f"robot_data_env_{env_id}.csv"

            # Prepare data for CSV
            rows = []
            for step in range(self.step_count):
                row = {
                    'step': step,
                    # Base position
                    'base_pos_x': arrays['base_pos'][step, env_idx, 0],
                    'base_pos_y': arrays['base_pos'][step, env_idx, 1],
                    'base_pos_z': arrays['base_pos'][step, env_idx, 2],
                    # Base orientation (euler)
                    'base_roll': arrays['base_euler'][step, env_idx, 0],
                    'base_pitch': arrays['base_euler'][step, env_idx, 1],
                    'base_yaw': arrays['base_euler'][step, env_idx, 2],
                    # Foot positions
                    'left_foot_x': arrays['left_foot_pos'][step, env_idx, 0],
                    'left_foot_y': arrays['left_foot_pos'][step, env_idx, 1],
                    'left_foot_z': arrays['left_foot_pos'][step, env_idx, 2],
                    'right_foot_x': arrays['right_foot_pos'][step, env_idx, 0],
                    'right_foot_y': arrays['right_foot_pos'][step, env_idx, 1],
                    'right_foot_z': arrays['right_foot_pos'][step, env_idx, 2],
                    # Commands
                    'cmd_lin_vel_x': arrays['cmd_lin_vel_x'][step, env_idx],
                    'cmd_lin_vel_y': arrays['cmd_lin_vel_y'][step, env_idx],
                    'cmd_ang_vel_z': arrays['cmd_ang_vel_z'][step, env_idx],
                    # Actual velocities
                    'actual_lin_vel_x': arrays['actual_lin_vel_x'][step, env_idx],
                    'actual_lin_vel_y': arrays['actual_lin_vel_y'][step, env_idx],
                    'actual_ang_vel_z': arrays['actual_ang_vel_z'][step, env_idx],
                    # Errors
                    'lin_vel_error': arrays['lin_vel_error'][step, env_idx],
                    'ang_vel_error': arrays['ang_vel_error'][step, env_idx],
                }
                rows.append(row)

            # Write CSV
            import csv
            with open(filepath, 'w', newline='') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            print(f"[DataLogger] Saved env {env_id} data to {filepath}")

    def reset(self):
        """Reset all data buffers."""
        for key in self.data.keys():
            self.data[key] = []
        self.step_count = 0
        print("[DataLogger] Data buffers reset")
