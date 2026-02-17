# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os

# Add isaaclab_fast_sac to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../../scripts/holosoma/holosoma/src/isaaclab_fast_sac"))

from isaaclab.utils import configclass
from isaaclab_fast_sac import FastSacRunnerCfg, FastSacAlgorithmCfg


@configclass
class K1FastSacRunnerCfg(FastSacRunnerCfg):
    """FastSAC configuration for K1 locomotion task."""

    seed: int = 42
    device: str = "cuda:0"
    max_iterations: int = 10000
    save_interval: int = 200
    experiment_name: str = "k1_fast_sac"
    run_name: str = ""
    logger: str = "tensorboard"
    wandb_project: str = "isaaclab"

    # Observation groups
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    clip_actions: float | None = None
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    # Algorithm configuration
    algorithm: FastSacAlgorithmCfg = FastSacAlgorithmCfg(
        # Learning rates
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,

        # Replay buffer
        buffer_size=1024,
        num_steps=1,

        # Discount factor
        gamma=0.97,
        tau=0.125,

        # Batch and update settings
        batch_size=8192,
        learning_starts=10,
        policy_frequency=4,
        num_updates=8,

        # Entropy
        target_entropy_ratio=0.0,
        alpha_init=0.001,
        use_autotune=True,

        # Distributional critic
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,

        # Network architecture
        critic_hidden_dim=768,
        actor_hidden_dim=512,
        use_layer_norm=True,
        num_q_networks=2,

        # Action settings
        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,
        action_scale=None,
        action_bias=None,

        # Optimization
        max_grad_norm=0.0,
        weight_decay=0.001,

        # Performance
        compile=True,
        amp=True,
        amp_dtype="bf16",
        obs_normalization=True,

        # Logging
        logging_interval=100,
    )


@configclass
class K1RoughFastSacRunnerCfg(K1FastSacRunnerCfg):
    """FastSAC configuration for K1 rough terrain locomotion task."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_rough_fast_sac"
        self.max_iterations = 15000
