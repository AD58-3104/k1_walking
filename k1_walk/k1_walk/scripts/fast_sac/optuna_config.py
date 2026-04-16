"""Optuna最適化の設定クラス"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class OptunaConfig:
    """Optuna最適化の基本設定"""

    # スタディ設定
    study_name: str = "k1_walk_reward_optimization"
    storage_path: str = "optuna_studies/study.db"
    n_trials: int = 40
    n_startup_trials: int = 10  # TPEサンプラーの初期ランダムトライアル数

    # 学習設定
    base_command: str = "bash train_rough.sh"
    task: str = "K1-Walk-Train-rough"
    num_envs: int = 4096
    max_iterations: int = 5000
    extra_args: str = ""  # train_rough.shに渡す追加引数（例: "--seed 42 env.scene.terrain.terrain_type=plane"）

    # 評価設定
    eval_metric: str = "Episode/length"  # 平均エピソード長を最大化
    eval_metric_direction: str = "maximize"  # "maximize" または "minimize"
    eval_step: Optional[int] = None  # Noneなら最終ステップ
    secondary_metrics: Dict[str, str] = field(default_factory=lambda: {
        # "Episode/reward": "maximize",
        "Rewards/track_lin_vel_xy_exp": "maximize",
        # "Rewards/alive_bonus": "maximize",
    })  # キー: メトリクス名, 値: "maximize" または "minimize"

    # タイムアウト設定（分）
    trial_timeout_minutes: int = 20


@dataclass
class RewardSearchSpace:
    """報酬重みの探索空間

    各フィールドは (min, max, log_scale) のタプル
    log_scale=True の場合は対数スケールでサンプリング
    """

    # タスク報酬
    track_lin_vel_xy_exp: Tuple[float, float, bool] = (2.0, 6.0, False)
    track_ang_vel_z_exp: Tuple[float, float, bool] = (1.5, 4.5, False)
    feet_height_bezier: Tuple[float, float, bool] = (5.0, 15.0, False)
    alive_bonus: Tuple[float, float, bool] = (3.0, 30.0, False)

    # シェイピング報酬（ポテンシャル系）
    # orientation_potential: Tuple[float, float, bool] = (5.0, 60.0, False)
    # height_potential: Tuple[float, float, bool] = (5.0, 50.0, False)

    # その他の報酬
    # feet_slide: Tuple[float, float, bool] = (-0.3, -0.01, False)
    # feet_air_time: Tuple[float, float, bool] = (1.0, 15.0, False)

    # ペナルティ系
    # joint_regularization_potential: Tuple[float, float, bool] = (0.001, 0.05, True)
    # upper_body_joint_regularization: Tuple[float, float, bool] = (0.2, 2.0, False)
    # feet_parallel_to_ground: Tuple[float, float, bool] = (0.5, 5.0, False)
    # action_rate_l2_legs: Tuple[float, float, bool] = (-2.0, -0.2, False)
    # action_rate_l2_arms: Tuple[float, float, bool] = (-1.2, -0.1, False)
    # ang_vel_xy_l2: Tuple[float, float, bool] = (-1.0, -0.1, False)

    def get_search_ranges(self) -> Dict[str, Tuple[float, float, bool]]:
        """探索範囲の辞書を返す"""
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
