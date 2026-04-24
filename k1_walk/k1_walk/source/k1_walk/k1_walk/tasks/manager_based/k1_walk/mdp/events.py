# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joint_targets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """指定関節の PD 目標値 (joint_pos_target) を default_joint_pos に書き戻す。

    アクション空間に含まれない関節は、JointPositionAction から target が一切書き込まれず、
    初期化時の 0（URDFデフォルト）のまま残るため、高 stiffness の PD がゼロへ引き戻してしまう。
    リセット時にこの関数を呼んで default に戻すことで固定姿勢を維持できる。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
