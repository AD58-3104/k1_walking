"""Optuna最適化用ユーティリティ関数（RSL-RL用）"""

import os
import subprocess
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna
from tensorboard.backend.event_processing import event_accumulator

from optuna_config import RewardSearchSpace


def suggest_reward_weights(trial: optuna.Trial, search_space: RewardSearchSpace) -> Dict[str, float]:
    """報酬重みをサンプリング"""
    weights = {}
    for name, (low, high, log_scale) in search_space.get_search_ranges().items():
        if log_scale:
            weights[name] = trial.suggest_float(f"reward_{name}", low, high, log=True)
        else:
            weights[name] = trial.suggest_float(f"reward_{name}", low, high)
    return weights


def build_hydra_args(reward_weights: Dict[str, float]) -> str:
    """報酬重みをHydra引数に変換"""
    args = []
    for name, weight in reward_weights.items():
        args.append(f"env.rewards.{name}.weight={weight}")
    return " ".join(args)


def find_latest_event_file(log_dir: Path) -> Optional[Path]:
    """ログディレクトリから最新のTensorBoardイベントファイルを検索"""
    event_files = list(log_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=lambda p: p.stat().st_mtime)


def extract_metrics_from_tensorboard(
    event_file: Path,
    metric_names: List[str],
    step: Optional[int] = None
) -> Dict[str, float]:
    """TensorBoardイベントファイルからメトリクスを抽出

    Args:
        event_file: イベントファイルのパス
        metric_names: 抽出するメトリクス名のリスト
        step: 特定のステップを指定（Noneなら最終ステップ）

    Returns:
        メトリクス名とその値の辞書
    """
    try:
        ea = event_accumulator.EventAccumulator(
            str(event_file.parent),
            size_guidance={
                event_accumulator.SCALARS: 0,  # 全て読む
            }
        )
        ea.Reload()

        available_tags = ea.Tags().get("scalars", [])
        metrics = {}

        for metric_name in metric_names:
            if metric_name not in available_tags:
                continue

            events = ea.Scalars(metric_name)
            if not events:
                continue

            if step is not None:
                # 指定ステップに最も近いイベントを取得
                closest_event = min(events, key=lambda e: abs(e.step - step))
                metrics[metric_name] = closest_event.value
            else:
                # 最終ステップの値を取得
                metrics[metric_name] = events[-1].value

        return metrics

    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return {}


def compute_objective_value(
    metrics: Dict[str, float],
    primary_metric: str,
) -> float:
    """目的関数値を計算

    Args:
        metrics: メトリクスの辞書
        primary_metric: 主要メトリクス名

    Returns:
        目的関数値
    """
    if primary_metric not in metrics:
        return float("-inf")
    return metrics[primary_metric]


class TrialRunner:
    """学習トライアルを実行するクラス"""

    def __init__(
        self,
        base_command: str,
        task: str,
        num_envs: int,
        max_iterations: int,
        log_root: Path,
        working_dir: Path,
        timeout_minutes: int = 30,
        extra_args: str = ""
    ):
        self.base_command = base_command
        self.task = task
        self.num_envs = num_envs
        self.max_iterations = max_iterations
        self.log_root = log_root
        self.working_dir = working_dir
        self.timeout_seconds = timeout_minutes * 60
        self.extra_args = extra_args

    def run(self, trial_number: int, hydra_args: str) -> Tuple[bool, Optional[Path]]:
        """学習を実行

        Returns:
            (成功フラグ, ログディレクトリ)のタプル
        """
        # コマンドを構築
        cmd_parts = [
            self.base_command,
            f"--num_envs {self.num_envs}",
            f"--max_iterations {self.max_iterations}",
            hydra_args,
        ]

        if self.extra_args:
            cmd_parts.append(self.extra_args)

        cmd = " ".join(cmd_parts)

        print(f"\n{'='*60}")
        print(f"Trial {trial_number}: Starting training")
        print(f"Command: {cmd}")
        print(f"{'='*60}\n")

        # 学習を実行
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                text=True
            )

            start_time = time.time()
            log_dir = None

            # プロセスの出力を監視
            while process.poll() is None:
                elapsed = time.time() - start_time

                # タイムアウトチェック
                if elapsed > self.timeout_seconds:
                    print(f"Trial {trial_number}: Timeout after {self.timeout_seconds}s")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(5)
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    return False, None

                # 出力を読む（ログディレクトリを検出）
                try:
                    line = process.stdout.readline()
                    if line:
                        print(line, end="")
                        # ログディレクトリの検出
                        if "Logging experiment in directory:" in line:
                            log_dir = Path(line.split(":")[-1].strip())
                        elif "Exact experiment name requested from command line:" in line:
                            # 別の方法でログディレクトリを推測
                            pass
                except:
                    pass

                time.sleep(0.1)

            # プロセス完了後の残り出力を読む
            remaining_output = process.stdout.read()
            if remaining_output:
                print(remaining_output)
                for line in remaining_output.split("\n"):
                    if "Logging experiment in directory:" in line:
                        log_dir = Path(line.split(":")[-1].strip())

            return_code = process.returncode
            success = return_code == 0

            if not success:
                print(f"Trial {trial_number}: Training failed with return code {return_code}")

            # ログディレクトリが見つからない場合、推測で検索
            if log_dir is None or not log_dir.exists():
                log_dir = self._find_latest_log_dir()

            return success, log_dir

        except Exception as e:
            print(f"Trial {trial_number}: Exception occurred: {e}")
            return False, None

    def _find_latest_log_dir(self) -> Optional[Path]:
        """最新のログディレクトリを検索"""
        log_dirs = list(self.log_root.glob("**/"))
        if not log_dirs:
            return None

        # イベントファイルを含むディレクトリを検索
        valid_dirs = []
        for d in log_dirs:
            if list(d.glob("events.out.tfevents.*")):
                valid_dirs.append(d)

        if not valid_dirs:
            return None

        return max(valid_dirs, key=lambda p: p.stat().st_mtime)
