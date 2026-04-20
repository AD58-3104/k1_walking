#!/usr/bin/env python3
"""TensorBoardのイベントファイルからRewardsを読み込んで1つのグラフに描画するスクリプト"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # インタラクティブなバックエンドを使用
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_rewards_from_event_file(event_file: str) -> dict:
    """イベントファイルからRewards関連のスカラーを読み込む"""
    ea = EventAccumulator(event_file)
    ea.Reload()

    rewards_data = {}
    for tag in ea.Tags()['scalars']:
        if tag.startswith('Rewards/Episode_Reward/'):
            reward_name = tag.replace('Rewards/Episode_Reward/', '')
            scalars = ea.Scalars(tag)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            rewards_data[reward_name] = {'steps': steps, 'values': values}

    return rewards_data


def find_event_files(logs_dir: str) -> list:
    """logsディレクトリからすべてのイベントファイルを検索"""
    pattern = os.path.join(logs_dir, '**', 'events.out.tfevents.*')
    return glob.glob(pattern, recursive=True)


def print_ratio_table(rewards_data: dict, baseline: str = 'alive_bonus'):
    """alive_bonusを基準として他の報酬の比率を一覧表示"""
    if not rewards_data:
        print("No rewards data found")
        return

    if baseline not in rewards_data:
        print(f"Baseline reward '{baseline}' not found in data")
        print(f"Available rewards: {list(rewards_data.keys())}")
        return

    # 各報酬の最終値を取得
    baseline_value = rewards_data[baseline]['values'][-1] if rewards_data[baseline]['values'] else 0

    if baseline_value == 0:
        print(f"Warning: baseline '{baseline}' has value 0, cannot compute ratio")
        return

    print(f"\n{'='*70}")
    print(f"Reward Ratios (baseline: {baseline} = {baseline_value:.4f})")
    print(f"{'='*70}")
    print(f"{'Reward Name':<40} {'Value':>12} {'Ratio':>12}")
    print(f"{'-'*70}")

    # 比率でソート（絶対値の降順）
    ratios = []
    for name, data in rewards_data.items():
        if data['values']:
            final_value = data['values'][-1]
            ratio = final_value / baseline_value
            ratios.append((name, final_value, ratio))

    ratios.sort(key=lambda x: abs(x[2]), reverse=True)

    for name, value, ratio in ratios:
        print(f"{name:<40} {value:>12.4f} {ratio:>12.4f}")

    print(f"{'='*70}\n")


def plot_rewards(rewards_data: dict, title: str = "Rewards", output_path: str = None):
    """すべてのRewardsを1つのグラフに描画"""
    if not rewards_data:
        print("No rewards data found")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for reward_name, data in rewards_data.items():
        ax.plot(data['steps'], data['values'], label=reward_name, alpha=0.8)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward Value')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show(block=True)  # ウィンドウが閉じられるまでブロック


def main():
    parser = argparse.ArgumentParser(description='Plot rewards from TensorBoard event files')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing TensorBoard logs')
    parser.add_argument('--run', type=str, default=None,
                        help='Specific run name to plot (e.g., optuna_trial_0001)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (e.g., rewards.png)')
    parser.add_argument('--list', action='store_true',
                        help='List available runs')
    parser.add_argument('--ratio', action='store_true',
                        help='Show reward ratios table (baseline: alive_bonus)')
    parser.add_argument('--baseline', type=str, default='alive_bonus',
                        help='Baseline reward for ratio calculation (default: alive_bonus)')
    args = parser.parse_args()

    event_files = find_event_files(args.logs_dir)

    if not event_files:
        print(f"No event files found in {args.logs_dir}")
        return

    # 利用可能なrunを表示
    if args.list:
        runs = set()
        for ef in event_files:
            parts = Path(ef).parts
            # logs/fast_sac/run_name/date/events... の形式を想定
            if len(parts) >= 3:
                run_name = parts[2] if parts[0] == 'logs' else parts[1]
                runs.add(run_name)
        print("Available runs:")
        for run in sorted(runs):
            print(f"  {run}")
        return

    # 特定のrunを指定した場合
    if args.run:
        event_files = [ef for ef in event_files if args.run in ef]
        if not event_files:
            print(f"No event files found for run: {args.run}")
            return
        # 最新のイベントファイルを使用
        event_files = sorted(event_files, key=os.path.getmtime, reverse=True)
        event_file = event_files[0]
        run_name = args.run
    else:
        # runが指定されていない場合は最新のイベントファイルを使用
        event_files = sorted(event_files, key=os.path.getmtime, reverse=True)
        event_file = event_files[0]
        run_name = Path(event_file).parts[2] if len(Path(event_file).parts) >= 3 else "Unknown"

    print(f"Using event file: {event_file}")
    rewards_data = load_rewards_from_event_file(event_file)

    # 比率表を表示
    if args.ratio:
        print_ratio_table(rewards_data, baseline=args.baseline)
    else:
        plot_rewards(rewards_data, title=f"Rewards - {run_name}", output_path=args.output)


if __name__ == '__main__':
    main()
