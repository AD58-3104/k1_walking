"""Optuna最適化の設定クラス"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional


@dataclass
class OptunaConfig:
    """Optuna最適化の基本設定"""

    # スタディ設定
    study_name: str = "k1_walk_reward_optimization"
    storage_path: str = "optuna_studies/study.db"
    n_trials: int = 50
    n_startup_trials: int = 10  # TPEサンプラーの初期ランダムトライアル数

    # 学習設定
    base_command: str = "bash train_rough.sh"
    task: str = "K1-Walk-Train-rough"
    num_envs: int = 4096
    max_iterations: int = 3000

    # 評価設定
    eval_metric: str = "Episode/length"  # 平均エピソード長を最大化
    eval_step: Optional[int] = None  # Noneなら最終ステップ
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "Episode/reward",
        "Rewards/track_lin_vel_xy_exp",
        "Rewards/alive_bonus",
    ])

    # タイムアウト設定（分）
    trial_timeout_minutes: int = 45


@dataclass
class RewardSearchSpace:
    """報酬重みの探索空間

    各フィールドは (min, max, log_scale) のタプル
    log_scale=True の場合は対数スケールでサンプリング
    """

    # タスク報酬
    track_lin_vel_xy_exp: Tuple[float, float, bool] = (0.5, 5.0, False)
    track_ang_vel_z_exp: Tuple[float, float, bool] = (0.3, 3.0, False)
    feet_height_bezier: Tuple[float, float, bool] = (1.0, 15.0, False)
    alive_bonus: Tuple[float, float, bool] = (3.0, 30.0, False)

    # シェイピング報酬（ポテンシャル系）
    orientation_potential: Tuple[float, float, bool] = (5.0, 60.0, False)
    height_potential: Tuple[float, float, bool] = (5.0, 50.0, False)
    joint_regularization_potential: Tuple[float, float, bool] = (0.001, 0.05, True)
    upper_body_joint_regularization: Tuple[float, float, bool] = (0.2, 2.0, False)

    # その他の報酬
    feet_slide: Tuple[float, float, bool] = (-0.3, -0.01, False)
    feet_air_time: Tuple[float, float, bool] = (1.0, 15.0, False)

    # ペナルティ系
    action_rate_l2_legs: Tuple[float, float, bool] = (-0.5, -0.01, False)
    lin_vel_z_pen: Tuple[float, float, bool] = (-20.0, -1.0, False)
    feet_close_penalty: Tuple[float, float, bool] = (-5.0, -0.1, False)

    def get_search_ranges(self) -> Dict[str, Tuple[float, float, bool]]:
        """探索範囲の辞書を返す"""
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
