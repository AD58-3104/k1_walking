"""探索空間の定義とHydra引数の構築"""

from typing import Dict
import optuna

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from optuna_config import RewardSearchSpace


def suggest_reward_weights(
    trial: optuna.Trial,
    search_space: RewardSearchSpace
) -> Dict[str, float]:
    """報酬重みをサンプリング

    Args:
        trial: Optunaトライアル
        search_space: 報酬重みの探索空間

    Returns:
        報酬項目名と重みの辞書
    """
    weights = {}
    search_ranges = search_space.get_search_ranges()

    for field_name, (low, high, log_scale) in search_ranges.items():
        if log_scale:
            weights[field_name] = trial.suggest_float(
                f"reward_{field_name}", low, high, log=True
            )
        else:
            weights[field_name] = trial.suggest_float(
                f"reward_{field_name}", low, high
            )
    return weights


def build_hydra_args(reward_weights: Dict[str, float]) -> str:
    """Hydra形式の引数文字列を構築

    Args:
        reward_weights: 報酬項目名と重みの辞書

    Returns:
        Hydra形式の引数文字列（例: "env.rewards.xxx.weight=1.0 env.rewards.yyy.weight=2.0"）
    """
    args = []

    for name, weight in reward_weights.items():
        args.append(f"env.rewards.{name}.weight={weight}")

    return " ".join(args)
