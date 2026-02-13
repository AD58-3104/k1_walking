# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data logger for recording robot state, foot trajectories, and command tracking."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Any
from pathlib import Path
import pickle
import socket
import json
import threading
from queue import Queue, Full

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


class DataStreamSender:
    """Singleton class for sending JSON-serialized data to 127.0.0.1:9870 via UDP.

    This class uses a background thread to send data asynchronously, preventing
    blocking of the main simulation loop. Data is queued and sent in the order received.

    UDP is used for low-latency, connectionless transmission. No connection establishment
    is required, making it ideal for real-time data streaming.

    Usage:
        # Get the singleton instance and send data
        DataStreamSender.get_instance().send({"key": "value", "data": [1, 2, 3]})

        # Or use the convenience function
        send_data_stream({"robot_state": state_dict})
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, host: str = "127.0.0.1", port: int = 9870, queue_size: int = 100):
        """Initialize the data sender.

        Args:
            host: Target host address (default: "127.0.0.1")
            port: Target port number (default: 9870)
            queue_size: Maximum number of messages to queue (default: 100)
        """
        if DataStreamSender._instance is not None:
            raise RuntimeError("DataStreamSender is a singleton. Use get_instance() instead.")

        self.host = host
        self.port = port
        self.queue_size = queue_size

        # Message queue for async sending
        self._queue = Queue(maxsize=queue_size)

        # UDP socket (connectionless)
        self._socket = None
        self._enabled = True

        # Statistics
        self._send_count = 0
        self._error_count = 0
        self._last_error = None

        # Initialize UDP socket
        self._init_socket()

        # Start sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._running = True
        self._sender_thread.start()

        print(f"[DataStreamSender] Initialized (UDP). Target: {host}:{port}")

    @classmethod
    def get_instance(cls, host: str = "127.0.0.1", port: int = 9870, queue_size: int = 100):
        """Get the singleton instance.

        Args:
            host: Target host address (only used on first call)
            port: Target port number (only used on first call)
            queue_size: Maximum queue size (only used on first call)

        Returns:
            The singleton DataStreamSender instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(host, port, queue_size)
        return cls._instance

    def _init_socket(self):
        """Initialize UDP socket.

        UDP is connectionless, so no connection establishment is needed.
        """
        try:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass

            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[DataStreamSender] UDP socket initialized for {self.host}:{self.port}")

        except Exception as e:
            self._last_error = str(e)
            print(f"[DataStreamSender] Failed to initialize socket: {e}")

    def _sender_loop(self):
        """Background thread loop for sending queued messages via UDP."""
        while self._running:
            try:
                # Get message from queue (blocking with timeout)
                try:
                    data = self._queue.get(timeout=0.1)
                except Exception:
                    continue

                if not self._enabled or self._socket is None:
                    continue

                # Send the data via UDP
                try:
                    # Convert numpy arrays to lists for JSON serialization
                    json_data = self._serialize_data(data)
                    message = json.dumps(json_data) + "\n"
                    message_bytes = message.encode('utf-8')

                    # UDP sendto - no connection needed
                    self._socket.sendto(message_bytes, (self.host, self.port))
                    self._send_count += 1

                except Exception as e:
                    # Log errors but continue
                    self._error_count += 1
                    self._last_error = str(e)
                    if self._error_count % 100 == 0:
                        print(f"[DataStreamSender] Send error: {e}")

            except Exception as e:
                print(f"[DataStreamSender] Unexpected error in sender loop: {e}")

    def _serialize_data(self, data: Any) -> Any:
        """Convert data to JSON-serializable format.

        Recursively converts numpy arrays and tensors to lists.

        Args:
            data: Data to serialize

        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data

    def send(self, data: Dict[str, Any]) -> bool:
        """Send data asynchronously to the target server.

        Args:
            data: Dictionary of data to send (will be JSON-serialized)

        Returns:
            True if data was queued successfully, False if queue is full
        """
        if not self._enabled:
            return False

        try:
            self._queue.put_nowait(data)
            return True
        except Full:
            # Queue is full, drop the message
            if self._error_count % 100 == 0:
                print("[DataStreamSender] Queue full, dropping message")
            return False

    def enable(self):
        """Enable data sending."""
        self._enabled = True
        print("[DataStreamSender] Enabled")

    def disable(self):
        """Disable data sending."""
        self._enabled = False
        print("[DataStreamSender] Disabled")

    def get_stats(self) -> Dict[str, Any]:
        """Get sender statistics.

        Returns:
            Dictionary containing statistics
        """
        return {
            'enabled': self._enabled,
            'send_count': self._send_count,
            'error_count': self._error_count,
            'queue_size': self._queue.qsize(),
            'last_error': self._last_error,
            'socket_initialized': self._socket is not None,
        }

    def close(self):
        """Close the sender and cleanup resources."""
        self._running = False
        self._enabled = False

        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2.0)

        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

        print("[DataStreamSender] Closed")


# Convenience function for easy access
def send_data_stream(data: Dict[str, Any]) -> bool:
    """Convenience function to send data using the singleton DataStreamSender.

    Args:
        data: Dictionary of data to send

    Returns:
        True if data was queued successfully, False otherwise

    Example:
        send_data_stream({
            "timestamp": time.time(),
            "robot_position": [x, y, z],
            "joint_angles": joint_array
        })
    """
    return DataStreamSender.get_instance().send(data)
