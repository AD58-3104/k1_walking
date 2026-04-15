# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint from FastSAC agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher
import isaaclab.app.app_launcher as app_launcher_module

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a FastSAC checkpoint.")
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
parser.add_argument(
    "--streaming-mode",
    "--streaming_mode",
    dest="streaming_mode",
    type=str,
    choices=("off", "public", "private"),
    default=None,
    help="Launch Isaac Lab with WebRTC streaming: off/public/private.",
)
# append FastSAC cli arguments
cli_args.add_fast_sac_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.streaming_mode is not None:
    args_cli.livestream = {"off": 0, "public": 1, "private": 2}[args_cli.streaming_mode]


def _configure_rendering_env(args: argparse.Namespace) -> None:
    """Prefer NVIDIA EGL/Vulkan for headless rendering and streaming."""
    livestream_enabled = getattr(args, "livestream", -1) in {1, 2}
    if not (args.headless or args.video or livestream_enabled):
        return

    # Avoid the X/GLX path in headless/streaming runs.
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    nvidia_icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.is_file():
        current_icd = os.environ.get("VK_ICD_FILENAMES", "")
        if "nvidia_icd.json" not in current_icd:
            os.environ["VK_ICD_FILENAMES"] = str(nvidia_icd)
        current_driver_files = os.environ.get("VK_DRIVER_FILES", "")
        if "nvidia_icd.json" not in current_driver_files:
            os.environ["VK_DRIVER_FILES"] = str(nvidia_icd)

    # Prevent Mesa's implicit device-selection layer from reordering Vulkan devices.
    os.environ.setdefault("NODEVICE_SELECT", "1")
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "*MESA*")


_configure_rendering_env(args_cli)

if getattr(args_cli, "livestream", -1) in {1, 2}:
    args_cli.headless = True
    args_cli.enable_cameras = True
    if args_cli.streaming_mode is not None:
        # Use the official Isaac Sim streaming experience instead of the generic Isaac Lab rendering app.
        args_cli.livestream = 0
        if not args_cli.experience:
            args_cli.experience = "isaacsim.exp.full.streaming.kit"
    elif not args_cli.experience:
        args_cli.experience = "isaaclab.python.headless.rendering.kit"

    extra_kit_args = [
        "--/app/vulkan=true",
        "--no-window",
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        "--/renderer/activeGpu=0",
    ]
    existing_kit_args = args_cli.kit_args.split()
    for extra_arg in extra_kit_args:
        if extra_arg not in existing_kit_args:
            existing_kit_args.append(extra_arg)
    args_cli.kit_args = " ".join(existing_kit_args)


def _patch_rendering_mode_fallback() -> None:
    """Allow AppLauncher to find Isaac Lab rendering presets from Isaac Sim experiences."""
    original_set_rendering_mode_settings = AppLauncher._set_rendering_mode_settings
    isaaclab_rendering_dir = Path(app_launcher_module.__file__).resolve().parents[4] / "apps" / "rendering_modes"

    def _set_rendering_mode_settings_with_fallback(self, launcher_args: dict) -> None:
        import carb
        import flatdict
        import toml
        from isaacsim.core.utils.carb import set_carb_setting

        rendering_mode = launcher_args.get("rendering_mode")
        if not self._enable_cameras and rendering_mode is None:
            return
        if rendering_mode is None:
            rendering_mode = "balanced"

        rendering_mode_explicitly_passed = launcher_args.pop("rendering_mode_explicit", False)
        if self._xr and not rendering_mode_explicitly_passed:
            rendering_mode = "xr"
            launcher_args["rendering_mode"] = "xr"

        repo_path = Path(carb.tokens.get_tokens_interface().resolve("${app}")).resolve().parent
        preset_filename = repo_path / "apps" / "rendering_modes" / f"{rendering_mode}.kit"
        if not preset_filename.is_file():
            fallback_filename = isaaclab_rendering_dir / f"{rendering_mode}.kit"
            if fallback_filename.is_file():
                preset_filename = fallback_filename
            else:
                return original_set_rendering_mode_settings(self, launcher_args)

        with open(preset_filename) as file:
            preset_dict = toml.load(file)
        preset_dict = dict(flatdict.FlatDict(preset_dict, delimiter="."))

        carb_setting = carb.settings.get_settings()
        for key, value in preset_dict.items():
            key = "/" + key.replace(".", "/")
            set_carb_setting(carb_setting, key, value)

    AppLauncher._set_rendering_mode_settings = _set_rendering_mode_settings_with_fallback


_patch_rendering_mode_fallback()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True

# viser mode runs headless
if args_cli.viser:
    args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch
from tqdm import tqdm

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Import FastSAC components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../FastSAC_standalone/isaaclab_fast_sac"))
from isaaclab_fast_sac import FastSacRunner, FastSacRunnerCfg, FastSacVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import k1_walk.tasks  # noqa: F401
from k1_walk.tasks.manager_based.k1_walk.mdp.data_logger import RobotDataLogger

from pathlib import Path
import numpy as np


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
            urdf_path=str(self.urdf_path),
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
        self.joint_names = list(self.urdf.joint_names)
        print(f"[INFO] Loaded URDF joints: {self.joint_names}")

    def update(
        self,
        joint_positions: dict[str, float],
        root_position: tuple[float, float, float] | None = None,
        root_orientation: tuple[float, float, float, float] | None = None,
    ):
        """Update robot visualization.

        Args:
            joint_positions: Dictionary mapping joint names to positions (radians).
            root_position: Optional (x, y, z) position of the robot base.
            root_orientation: Optional (w, x, y, z) quaternion for robot base orientation.
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

    def close(self):
        """Stop the viser server."""
        self.server.stop()


def main_with_viser():
    """Play with FastSAC agent using viser for visualization."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: FastSacRunnerCfg = cli_args.parse_fast_sac_cfg(args_cli.task, args_cli)

    # Set agent device if specified
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "fast_sac", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("fast_sac", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment (no rendering needed)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for FastSAC
    env = FastSacVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    fast_sac_runner = FastSacRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    fast_sac_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = fast_sac_runner.get_inference_policy(device=env.unwrapped.device)

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
    obs = env.get_observations()

    print(f"[INFO] Visualizing environment {env_id} in viser")
    print(f"[INFO] Open http://localhost:{args_cli.viser_port} in your browser")
    print("[INFO] Press Ctrl+C to stop")

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


# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with FastSAC agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: FastSacRunnerCfg = cli_args.parse_fast_sac_cfg(args_cli.task, args_cli)

    # Set agent device if specified
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "fast_sac", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("fast_sac", args_cli.task)
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

    # wrap around environment for FastSAC
    env = FastSacVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

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
    fast_sac_runner = FastSacRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    fast_sac_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = fast_sac_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    livestream_enabled = getattr(args_cli, "streaming_mode", None) in {"public", "private"} or getattr(
        args_cli, "livestream", -1
    ) in {1, 2}

    if livestream_enabled:
        env.unwrapped.render()

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
            if livestream_enabled:
                env.unwrapped.render()

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
