from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import torch


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

class modify_reward_weight_by_episode_length(ManagerTermBase):
    """Holosoma寄りの更新則で、報酬重みを平均エピソード長に応じて段階的に変更する。"""

    _AVERAGE_EPISODE_LENGTH_ATTR: ClassVar[str] = "_reward_curriculum_average_episode_length"
    _AVERAGE_EPISODE_LENGTH_STEP_ATTR: ClassVar[str] = "_reward_curriculum_average_episode_length_last_step"

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        params = cfg.params or {}
        term_name = params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        self.init_weight: float = self._term_cfg.weight
        self.level_down_threshold = float(params.get("level_down_threshold", 150.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 750.0))
        self.degree = float(params.get("degree", 0.001))
        self.current_scale = float(params.get("initial_scale", 0.0))
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        base_num_compute_average_epl = float(params.get("num_compute_average_epl", 1000.0))
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        self.num_compute_average_epl = max(1, int(base_num_compute_average_epl * env.num_envs / base_denominator))

        if not hasattr(env, self._AVERAGE_EPISODE_LENGTH_ATTR):
            setattr(env, self._AVERAGE_EPISODE_LENGTH_ATTR, torch.tensor(0.0, device=env.device, dtype=torch.float))
        if not hasattr(env, self._AVERAGE_EPISODE_LENGTH_STEP_ATTR):
            setattr(env, self._AVERAGE_EPISODE_LENGTH_STEP_ATTR, -1)

    def _resolve_env_ids(self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        if torch.is_tensor(env_ids):
            return env_ids.to(device=env.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    def _update_average_episode_length(
        self, env: ManagerBasedRLEnv, env_ids: Sequence[int] | slice | torch.Tensor
    ) -> torch.Tensor:
        last_step = getattr(env, self._AVERAGE_EPISODE_LENGTH_STEP_ATTR)
        average_episode_length = getattr(env, self._AVERAGE_EPISODE_LENGTH_ATTR)
        if last_step == env.common_step_counter:
            return average_episode_length

        env_ids_tensor = self._resolve_env_ids(env, env_ids)
        if env_ids_tensor.numel() == 0:
            return average_episode_length

        episode_lengths = env.episode_length_buf[env_ids_tensor].to(dtype=torch.float)
        current_average = torch.mean(episode_lengths, dtype=torch.float)
        weight = min(env_ids_tensor.numel() / self.num_compute_average_epl, 1.0)
        average_episode_length = average_episode_length * (1.0 - weight) + current_average * weight
        setattr(env, self._AVERAGE_EPISODE_LENGTH_ATTR, average_episode_length)
        setattr(env, self._AVERAGE_EPISODE_LENGTH_STEP_ATTR, env.common_step_counter)
        return average_episode_length


    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        target_weight: float,
    ) -> float:
        average_episode_length = self._update_average_episode_length(env, env_ids)

        if float(average_episode_length) < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif float(average_episode_length) > self.level_up_threshold:
            self.current_scale = 1.0 - (1.0 - self.current_scale) * (1.0 - self.degree)

        self.current_scale = min(max(self.current_scale, self.min_scale), self.max_scale)
        self._term_cfg.weight = self.init_weight + self.current_scale * (target_weight - self.init_weight)
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
