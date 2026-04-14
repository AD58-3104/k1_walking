"""単一トライアルの学習実行"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple


class TrialRunner:
    """単一トライアルの学習を実行"""

    def __init__(
        self,
        base_command: str,
        task: str,
        num_envs: int,
        max_iterations: int,
        log_root: Path,
        working_dir: Path,
        timeout_minutes: int = 45,
        extra_args: str = ""
    ):
        """
        Args:
            base_command: 実行するベースコマンド（例: "bash train_rough.sh"）
            task: タスク名
            num_envs: 環境数
            max_iterations: 最大イテレーション数
            log_root: ログのルートディレクトリ
            working_dir: 作業ディレクトリ
            timeout_minutes: タイムアウト（分）
            extra_args: 追加の引数
        """
        self.base_command = base_command
        self.task = task
        self.num_envs = num_envs
        self.max_iterations = max_iterations
        self.log_root = log_root
        self.working_dir = working_dir
        self.timeout_seconds = timeout_minutes * 60
        self.extra_args = extra_args

    def run(
        self,
        trial_number: int,
        hydra_args: str
    ) -> Tuple[bool, Optional[Path]]:
        """トライアルを実行

        Args:
            trial_number: トライアル番号
            hydra_args: Hydra形式の追加引数

        Returns:
            (成功フラグ, ログディレクトリパス)
        """
        experiment_name = f"optuna_trial_{trial_number:04d}"

        # コマンド構築
        command = (
            f"{self.base_command} "
            f"--task {self.task} "
            f"--num_envs {self.num_envs} "
            f"--max_iterations {self.max_iterations} "
            f"--headless "
            f"--experiment_name {experiment_name} "
            f"{self.extra_args} "
            f"{hydra_args}"
        ).strip()

        print(f"\n{'='*60}")
        print(f"Trial {trial_number}: Starting")
        print(f"Command: {command}")
        print(f"{'='*60}\n")

        try:
            # 環境変数を継承して実行
            env = os.environ.copy()

            result = subprocess.run(
                command,
                shell=True,
                cwd=self.working_dir,
                timeout=self.timeout_seconds,
                env=env,
            )

            success = result.returncode == 0

            if not success:
                print(f"Trial {trial_number}: Failed with return code {result.returncode}")

            # ログディレクトリを特定
            log_dir = self._find_log_dir(experiment_name)

            return success, log_dir

        except subprocess.TimeoutExpired:
            print(f"Trial {trial_number}: Timeout after {self.timeout_seconds}s")
            return False, None
        except Exception as e:
            print(f"Trial {trial_number}: Error - {e}")
            return False, None

    def _find_log_dir(self, experiment_name: str) -> Optional[Path]:
        """実験のログディレクトリを特定"""
        experiment_dir = self.log_root / experiment_name
        if not experiment_dir.exists():
            return None

        # 最新のタイムスタンプディレクトリを取得
        subdirs = sorted(
            [d for d in experiment_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )

        return subdirs[0] if subdirs else None
