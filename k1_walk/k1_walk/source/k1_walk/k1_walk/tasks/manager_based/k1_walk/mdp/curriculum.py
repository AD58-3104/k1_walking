from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_reward_weight_interval(ManagerTermBase):
    """指定したスタートステップ以降、指定したインターバルごとに報酬項の重みを増加させるカリキュラム"""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        start_steps: int,
        interval_steps: int,
        weight_increment: float,
    ) -> float:
        # update term settings
        if env.common_step_counter > start_steps:
            if (env.common_step_counter - start_steps) % interval_steps == 0:
                num_intervals = (env.common_step_counter - start_steps) // interval_steps
                self._term_cfg.weight = weight + num_intervals * weight_increment
                env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight

class modify_reward_weight_incremental(ManagerTermBase):
    """指定したスタートステップ以降、指定した終了ステップまで、報酬の重みを指定した値まで増加させるカリキュラム"""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        self.init_weight: float = self._term_cfg.weight


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_steps: int,
        end_steps: int,
        target_weight: float,
    ) -> float:
        # update term settings
        if env.common_step_counter > start_steps:
            if env.common_step_counter > end_steps:
                self._term_cfg.weight = target_weight
            else:
                # Calculate the weight increment based on the progress between start and end steps
                progress = (env.common_step_counter - start_steps) / (end_steps - start_steps)
                self._term_cfg.weight = self.init_weight + progress * (target_weight - self.init_weight)
                env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight