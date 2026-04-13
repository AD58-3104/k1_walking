"""TensorBoardログからメトリクスを抽出"""

from pathlib import Path
from typing import Dict, List, Optional


def find_latest_event_file(log_dir: Path) -> Optional[Path]:
    """最新のTensorBoardイベントファイルを取得

    Args:
        log_dir: ログディレクトリ

    Returns:
        最新のイベントファイルのパス、見つからない場合はNone
    """
    if not log_dir.exists():
        return None

    event_files = sorted(
        log_dir.rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime
    )
    return event_files[-1] if event_files else None


def extract_metrics_from_tensorboard(
    event_file: Path,
    metrics: List[str],
    step: Optional[int] = None
) -> Dict[str, float]:
    """TensorBoardログから指定されたメトリクスを抽出

    Args:
        event_file: イベントファイルのパス
        metrics: 抽出するメトリクス名のリスト
        step: 指定されたステップ（Noneなら最終ステップ）

    Returns:
        メトリクス名と値のDict
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is not installed. Install it with `pip install tensorboard`"
        ) from exc

    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0}
    )
    accumulator.Reload()

    results = {}
    tags = accumulator.Tags().get("scalars", [])

    for metric in metrics:
        if metric not in tags:
            continue

        events = accumulator.Scalars(metric)
        if not events:
            continue

        if step is None:
            selected_event = events[-1]
        else:
            selected_event = None
            for event in events:
                if event.step > step:
                    break
                selected_event = event
            if selected_event is None:
                continue

        results[metric] = float(selected_event.value)

    return results


def compute_objective_value(
    metrics: Dict[str, float],
    primary_metric: str = "Episode/length",
    min_episode_length: float = 100.0,
    penalty_weight: float = 0.1
) -> float:
    """目的関数値を計算

    Args:
        metrics: 抽出されたメトリクス
        primary_metric: 主評価メトリクス
        min_episode_length: 最小エピソード長（これ以下の場合ペナルティ）
        penalty_weight: ペナルティの重み

    Returns:
        目的関数値（最大化）
    """
    if primary_metric not in metrics:
        return float("-inf")

    value = metrics[primary_metric]

    # エピソード長が短すぎる場合のペナルティ
    if value < min_episode_length:
        value -= penalty_weight * (min_episode_length - value)

    return value
