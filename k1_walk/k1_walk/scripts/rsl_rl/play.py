# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--viser", action="store_true", default=False, help="Run headless with viser web visualization.")
parser.add_argument("--viser_port", type=int, default=8080, help="Port for viser web server.")
parser.add_argument("--viser_env_id", type=int, default=0, help="Environment ID to visualize in viser.")
parser.add_argument("--log_data", action="store_true", default=False, help="Enable data logging.")
parser.add_argument("--log_steps", type=int, default=5000, help="Maximum number of steps to log.")
parser.add_argument("--log_env_ids", type=str, default="0", help="Comma-separated environment IDs to log (e.g., '0,1,2').")
parser.add_argument("--log_output_dir", type=str, default="review/play_data", help="Directory to save logged data.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# viser mode runs headless
if args_cli.viser:
    args_cli.headless = True


def _configure_single_gpu_rendering(args: argparse.Namespace) -> None:
    """Force Isaac Sim renderer to stay on a single selected GPU."""
    extra_kit_args = [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
    ]
    if isinstance(args.device, str):
        if args.device.startswith("cuda:"):
            gpu_index = args.device.split(":", 1)[1]
            extra_kit_args.append(f"--/renderer/activeGpu={gpu_index}")
        elif args.device == "cuda":
            extra_kit_args.append("--/renderer/activeGpu=0")

    existing_kit_args = args.kit_args.split()
    for extra_arg in extra_kit_args:
        if extra_arg not in existing_kit_args:
            existing_kit_args.append(extra_arg)
    args.kit_args = " ".join(existing_kit_args)


_configure_single_gpu_rendering(args_cli)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import k1_walk.tasks  # noqa: F401
from k1_walk.tasks.manager_based.k1_walk.mdp.data_logger import RobotDataLogger


class ViserVisualizer:
    """Visualizer for robot using viser web interface."""

    def __init__(self, urdf_path: str, port: int = 8080):
        """Initialize viser visualizer.

        Args:
            urdf_path: Path to the URDF file.
            port: Port for viser web server.
        """
        import viser
        import viser.extras

        self.urdf_path = Path(urdf_path)
        self.port = port

        # Start viser server
        self.server = viser.ViserServer(port=port)
        print(f"[INFO] Viser server started at http://localhost:{port}")

        # Load URDF
        self.urdf = viser.extras.ViserUrdf(
            self.server,
            urdf_or_path=self.urdf_path,
            root_node_name="/robot",
        )

        # Add ground plane
        self.server.scene.add_grid(
            "/ground",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
            plane="xy",
            cell_color=(128, 128, 128),
        )

        # Add axes helper
        self.server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)

        # Store joint names from URDF
        self.joint_names = list(self.urdf._joint_frames)
        print(f"[INFO] Loaded URDF joints: {self.joint_names}")

        # Store arrow handles for velocity visualization
        self._command_arrow = None
        self._velocity_arrow = None

    def update(
        self,
        joint_positions: dict[str, float],
        root_position: tuple[float, float, float] | None = None,
        root_orientation: tuple[float, float, float, float] | None = None,
        command_velocity: tuple[float, float, float] | None = None,
        actual_velocity: tuple[float, float, float] | None = None,
    ):
        """Update robot visualization.

        Args:
            joint_positions: Dictionary mapping joint names to positions (radians).
            root_position: Optional (x, y, z) position of the robot base.
            root_orientation: Optional (w, x, y, z) quaternion for robot base orientation.
            command_velocity: Optional (vx, vy, vz) commanded velocity in world frame.
            actual_velocity: Optional (vx, vy, vz) actual velocity in world frame.
        """
        # Update joint positions
        self.urdf.update_cfg(joint_positions)

        # Update root transform if provided
        if root_position is not None or root_orientation is not None:
            import viser.transforms as vtf

            pos = root_position if root_position is not None else (0.0, 0.0, 0.0)
            quat = root_orientation if root_orientation is not None else (1.0, 0.0, 0.0, 0.0)

            # Convert to viser transform (wxyz -> wxyz is the same)
            transform = vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3(np.array(quat)),
                translation=np.array(pos),
            )
            self.server.scene.add_frame(
                "/robot",
                wxyz=transform.rotation().wxyz,
                position=transform.translation(),
                axes_length=0.0,
                axes_radius=0.0,
            )

        # Draw velocity arrows
        if root_position is not None:
            arrow_z = root_position[2] + 0.1  # Draw arrows slightly above robot base
            arrow_scale = 0.5  # Scale arrow length (1 m/s = 0.5m arrow)

            # Draw command velocity arrow (green)
            if command_velocity is not None:
                cmd_vx, cmd_vy, _ = command_velocity
                cmd_magnitude = np.sqrt(cmd_vx**2 + cmd_vy**2)
                if cmd_magnitude > 0.01:
                    end_x = root_position[0] + cmd_vx * arrow_scale
                    end_y = root_position[1] + cmd_vy * arrow_scale
                    self._command_arrow = self.server.scene.add_spline_catmull_rom(
                        "/command_velocity",
                        positions=np.array([
                            [root_position[0], root_position[1], arrow_z],
                            [end_x, end_y, arrow_z],
                        ]),
                        color=(0, 255, 0),  # Green for command
                        line_width=4.0,
                    )
                    # Arrow head
                    self.server.scene.add_icosphere(
                        "/command_velocity_head",
                        radius=0.03,
                        position=(end_x, end_y, arrow_z),
                        color=(0, 255, 0),
                    )
                else:
                    # Hide arrow when velocity is too small by drawing at robot position
                    self.server.scene.add_spline_catmull_rom(
                        "/command_velocity",
                        positions=np.array([
                            [root_position[0], root_position[1], arrow_z],
                            [root_position[0], root_position[1], arrow_z],
                        ]),
                        color=(0, 255, 0),
                        line_width=0.0,
                    )
                    self.server.scene.add_icosphere(
                        "/command_velocity_head",
                        radius=0.0,
                        position=(root_position[0], root_position[1], arrow_z),
                        color=(0, 255, 0),
                    )

            # Draw actual velocity arrow (red)
            if actual_velocity is not None:
                act_vx, act_vy, _ = actual_velocity
                act_magnitude = np.sqrt(act_vx**2 + act_vy**2)
                if act_magnitude > 0.01:
                    end_x = root_position[0] + act_vx * arrow_scale
                    end_y = root_position[1] + act_vy * arrow_scale
                    self._velocity_arrow = self.server.scene.add_spline_catmull_rom(
                        "/actual_velocity",
                        positions=np.array([
                            [root_position[0], root_position[1], arrow_z],
                            [end_x, end_y, arrow_z],
                        ]),
                        color=(255, 0, 0),  # Red for actual
                        line_width=4.0,
                    )
                    # Arrow head
                    self.server.scene.add_icosphere(
                        "/actual_velocity_head",
                        radius=0.03,
                        position=(end_x, end_y, arrow_z),
                        color=(255, 0, 0),
                    )
                else:
                    # Hide arrow when velocity is too small by drawing at robot position
                    self.server.scene.add_spline_catmull_rom(
                        "/actual_velocity",
                        positions=np.array([
                            [root_position[0], root_position[1], arrow_z],
                            [root_position[0], root_position[1], arrow_z],
                        ]),
                        color=(255, 0, 0),
                        line_width=0.0,
                    )
                    self.server.scene.add_icosphere(
                        "/actual_velocity_head",
                        radius=0.0,
                        position=(root_position[0], root_position[1], arrow_z),
                        color=(255, 0, 0),
                    )

    def close(self):
        """Stop the viser server."""
        self.server.stop()


# PLACEHOLDER: Extension template (do not remove this comment)


def main_with_viser():
    """Play with RSL-RL agent using viser for visualization."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create isaac environment (no rendering needed)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # Initialize viser visualizer
    urdf_path = Path.home() / "k1_walking" / "booster_assets" / "robots" / "K1" / "K1_22dof.urdf"
    visualizer = ViserVisualizer(str(urdf_path), port=args_cli.viser_port)

    # Get the unwrapped environment for accessing robot state
    unwrapped_env = env.unwrapped

    # Get robot asset
    robot = unwrapped_env.scene["robot"]

    # Get joint names from the robot
    joint_names = robot.joint_names
    print(f"[INFO] Robot joint names: {joint_names}")

    # Environment ID to visualize
    env_id = args_cli.viser_env_id
    if env_id >= env_cfg.scene.num_envs:
        print(f"[WARNING] env_id {env_id} >= num_envs {env_cfg.scene.num_envs}, using env_id=0")
        env_id = 0

    # reset environment
    obs, _ = env.get_observations()

    print(f"[INFO] Visualizing environment {env_id} in viser")
    print(f"[INFO] Open http://localhost:{args_cli.viser_port} in your browser")
    print("[INFO] Press Ctrl+C to stop")

    # Helper function to convert velocity from robot frame to world frame
    def robot_to_world_velocity(vel_robot, yaw):
        """Convert velocity from robot frame to world frame using yaw angle."""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vx_world = vel_robot[0] * cos_yaw - vel_robot[1] * sin_yaw
        vy_world = vel_robot[0] * sin_yaw + vel_robot[1] * cos_yaw
        return (vx_world, vy_world, 0.0)

    def quat_to_yaw(quat):
        """Extract yaw angle from quaternion (w, x, y, z)."""
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    print("[INFO] Arrow legend: GREEN = commanded velocity, RED = actual velocity")

    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)

            # Get joint positions and root state for the specified environment
            joint_pos = robot.data.joint_pos[env_id].cpu().numpy()
            root_pos = robot.data.root_pos_w[env_id].cpu().numpy()
            root_quat = robot.data.root_quat_w[env_id].cpu().numpy()  # (w, x, y, z)

            # Get command velocity (in robot frame)
            command = unwrapped_env.command_manager.get_command("base_velocity")[env_id].cpu().numpy()
            # command is [vx, vy, ang_vel_z]
            cmd_vel_robot = (float(command[0]), float(command[1]), 0.0)

            # Get actual velocity (in world frame)
            root_lin_vel_w = robot.data.root_lin_vel_w[env_id].cpu().numpy()
            actual_vel_world = (float(root_lin_vel_w[0]), float(root_lin_vel_w[1]), float(root_lin_vel_w[2]))

            # Convert command velocity from robot frame to world frame
            yaw = quat_to_yaw(root_quat)
            cmd_vel_world = robot_to_world_velocity(cmd_vel_robot, yaw)

            # Build joint position dictionary
            joint_pos_dict = {}
            for i, name in enumerate(joint_names):
                if i < len(joint_pos):
                    joint_pos_dict[name] = float(joint_pos[i])

            # Update viser visualization
            # Note: Convert quaternion from (w, x, y, z) to viser format (w, x, y, z) - same format
            visualizer.update(
                joint_positions=joint_pos_dict,
                root_position=(float(root_pos[0]), float(root_pos[1]), float(root_pos[2])),
                root_orientation=(float(root_quat[0]), float(root_quat[1]), float(root_quat[2]), float(root_quat[3])),
                command_velocity=cmd_vel_world,
                actual_velocity=actual_vel_world,
            )

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        visualizer.close()
        env.close()


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # initialize data logger if enabled
    logger = None
    if args_cli.log_data:
        # Parse environment IDs
        env_ids = [int(x.strip()) for x in args_cli.log_env_ids.split(",")]
        # Get the unwrapped environment for logging
        unwrapped_env = env.unwrapped
        # Unwrap RecordVideo wrapper if present
        if args_cli.video:
            unwrapped_env = unwrapped_env.env
        logger = RobotDataLogger(unwrapped_env, max_steps=args_cli.log_steps, env_ids=env_ids)
        print(f"[INFO] Data logging enabled for environments: {env_ids}")
        print(f"[INFO] Max steps to log: {args_cli.log_steps}")
        print(f"[INFO] Output directory: {args_cli.log_output_dir}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # initialize progress bar if logging is enabled
    pbar = None
    if logger is not None:
        pbar = tqdm(total=args_cli.log_steps, desc="Logging progress", unit="step")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # log data if enabled
        if logger is not None:
            if logger.record(command_name="base_velocity"):
                if pbar is not None:
                    pbar.update(1)
                    pbar.close()
                print(f"\n[INFO] Data logging reached maximum steps: {logger.max_steps}")
                break
            # update progress bar
            if pbar is not None:
                pbar.update(1)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close progress bar if still open
    if pbar is not None:
        pbar.close()

    # save logged data
    if logger is not None and logger.step_count > 0:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(args_cli.log_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save as pickle
        pkl_path = os.path.join(output_dir, f"robot_data_{timestamp}.pkl")
        logger.save(pkl_path)

        # Save as CSV
        csv_dir = os.path.join(output_dir, f"csv_{timestamp}")
        logger.save_csv(csv_dir)

        print(f"[INFO] Data logging complete. Logged {logger.step_count} steps.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    if args_cli.viser:
        main_with_viser()
    else:
        main()
    # close sim app
    simulation_app.close()
