# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import k1_walk.tasks  # noqa: F401


def main():
    """Agent that maintains initial joint positions."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # Get action information
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager._terms["joint_pos"]
    action_joint_names = action_term._joint_names
    action_joint_indices = action_term._joint_ids[0]  # Get indices for first environment

    # Get default positions for action-controlled joints
    all_default_joint_pos = robot.data.default_joint_pos
    initial_joint_pos = all_default_joint_pos[:, action_joint_indices]

    print(f"[INFO]: Action-controlled joints: {action_joint_names}")
    print(f"[INFO]: Initial joint positions: {initial_joint_pos[0]}")
    print(f"[INFO]: Action space shape: {env.action_space.shape}")
    print(f"[INFO]: use_default_offset: {action_term.cfg.use_default_offset}")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Since use_default_offset=True, zero actions maintain the default (initial) positions
            # Zero offset from default position = maintain initial position
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
