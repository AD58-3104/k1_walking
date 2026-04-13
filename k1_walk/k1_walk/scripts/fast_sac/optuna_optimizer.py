#!/usr/bin/env python3
"""
Optuna による FastSAC 報酬重み最適化スクリプト

使用方法:
    python optuna_optimizer.py --study-name my_study --n-trials 50
    python optuna_optimizer.py --resume --study-name my_study  # 再開
    python optuna_optimizer.py --n-trials 3 --max-iterations 1000  # テスト
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from optuna_config import OptunaConfig, RewardSearchSpace
from optuna_utils import (
    suggest_reward_weights,
    build_hydra_args,
    find_latest_event_file,
    extract_metrics_from_tensorboard,
    compute_objective_value,
    TrialRunner,
)


class OptunaOptimizer:
    """FastSAC報酬重み最適化のメインクラス"""

    def __init__(self, config: OptunaConfig):
        self.config = config
        self.script_dir = Path(__file__).parent.resolve()
        self.log_root = self.script_dir / "logs" / "fast_sac"

        # 探索空間
        self.reward_space = RewardSearchSpace()

        # トライアルランナー
        self.runner = TrialRunner(
            base_command=config.base_command,
            task=config.task,
            num_envs=config.num_envs,
            max_iterations=config.max_iterations,
            log_root=self.log_root,
            working_dir=self.script_dir,
            timeout_minutes=config.trial_timeout_minutes
        )

    def create_or_load_study(self) -> optuna.Study:
        """Optunaスタディを作成または読み込み"""
        storage_path = self.script_dir / self.config.storage_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        storage = f"sqlite:///{storage_path}"

        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            seed=42
        )

        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            sampler=sampler,
            direction="maximize",
            load_if_exists=True
        )

        return study

    def objective(self, trial: optuna.Trial) -> float:
        """目的関数"""

        # パラメータのサンプリング
        reward_weights = suggest_reward_weights(trial, self.reward_space)

        # Hydra引数を構築
        hydra_args = build_hydra_args(reward_weights)

        # 学習実行
        success, log_dir = self.runner.run(trial.number, hydra_args)

        if not success or log_dir is None:
            return float("-inf")

        # メトリクス抽出
        event_file = find_latest_event_file(log_dir)
        if event_file is None:
            print(f"Trial {trial.number}: No event file found in {log_dir}")
            return float("-inf")

        all_metrics = [self.config.eval_metric] + self.config.secondary_metrics
        metrics = extract_metrics_from_tensorboard(
            event_file,
            all_metrics,
            step=self.config.eval_step
        )

        if not metrics:
            print(f"Trial {trial.number}: No metrics extracted")
            return float("-inf")

        # 目的関数値を計算
        objective_value = compute_objective_value(
            metrics,
            primary_metric=self.config.eval_metric,
        )

        # 副次メトリクスを記録（可視化用）
        for metric_name, value in metrics.items():
            trial.set_user_attr(metric_name.replace("/", "_"), value)

        print(f"Trial {trial.number}: {self.config.eval_metric}={objective_value:.2f}")

        return objective_value

    def optimize(self) -> None:
        """最適化を実行"""
        study = self.create_or_load_study()

        print(f"\n{'='*60}")
        print(f"Starting Optuna Optimization")
        print(f"Study: {self.config.study_name}")
        print(f"N trials: {self.config.n_trials}")
        print(f"Max iterations per trial: {self.config.max_iterations}")
        print(f"Eval metric: {self.config.eval_metric}")
        print(f"Storage: {self.config.storage_path}")
        print(f"{'='*60}\n")

        try:
            study.optimize(
                self.objective,
                n_trials=self.config.n_trials,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")

        # 結果を表示
        self._print_results(study)

        # 結果をJSONに保存
        self._save_results(study)

    def _print_results(self, study: optuna.Study) -> None:
        """最適化結果を表示"""
        print(f"\n{'='*60}")
        print("Optimization Results")
        print(f"{'='*60}")

        if study.best_trial is None:
            print("No completed trials.")
            return

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value ({self.config.eval_metric}): {study.best_value:.4f}")

        print("\nBest parameters:")
        for key, value in sorted(study.best_params.items()):
            print(f"  {key}: {value:.6f}")

        print(f"\nTotal trials: {len(study.trials)}")
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        print(f"Completed: {completed}, Failed: {failed}")

        # ベストパラメータでのHydra引数を表示
        print("\nTo use the best parameters, run:")
        best_hydra_args = self._params_to_hydra_args(study.best_params)
        print(f"  bash train_rough.sh {best_hydra_args}")

    def _params_to_hydra_args(self, params: dict) -> str:
        """OptunaパラメータをHydra引数に変換"""
        args = []
        for key, value in params.items():
            # "reward_xxx" -> "xxx"
            if key.startswith("reward_"):
                reward_name = key[len("reward_"):]
                args.append(f"env.rewards.{reward_name}.weight={value}")
        return " ".join(args)

    def _save_results(self, study: optuna.Study) -> None:
        """結果をJSONファイルに保存"""
        results_dir = self.script_dir / "optuna_studies" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"{self.config.study_name}_{timestamp}.json"

        results = {
            "study_name": self.config.study_name,
            "config": {
                "eval_metric": self.config.eval_metric,
                "max_iterations": self.config.max_iterations,
                "num_envs": self.config.num_envs,
            },
            "n_trials": len(study.trials),
            "timestamp": timestamp,
        }

        if study.best_trial is not None:
            results["best_trial"] = study.best_trial.number
            results["best_value"] = study.best_value
            results["best_params"] = study.best_params
            results["best_hydra_args"] = self._params_to_hydra_args(study.best_params)

        # 全トライアルの結果
        results["trials"] = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            results["trials"].append(trial_data)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_path}")


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Optuna による FastSAC 報酬重み最適化"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="k1_walk_reward_optimization",
        help="Optunaスタディ名"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="トライアル数"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3000,
        help="各トライアルの最大イテレーション数"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="環境数"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="トライアルのタイムアウト（分）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="前回の最適化を再開"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 設定を構築
    config = OptunaConfig(
        study_name=args.study_name,
        n_trials=args.n_trials,
        max_iterations=args.max_iterations,
        num_envs=args.num_envs,
        trial_timeout_minutes=args.timeout,
    )

    # 最適化を実行
    optimizer = OptunaOptimizer(config)
    optimizer.optimize()


if __name__ == "__main__":
    main()
