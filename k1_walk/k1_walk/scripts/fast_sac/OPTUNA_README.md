# Optuna による FastSAC 報酬重み最適化

K1-Walk-Train-rough タスクの報酬関数の重みを Optuna（ベイズ最適化）で自動調整するスクリプトです。

## 概要

- **評価指標**: 平均エピソード長（Episode/length）を最大化
- **最適化手法**: TPE（Tree-structured Parzen Estimator）サンプラー
- **最適化対象**: 13個の報酬重みパラメータ
- **途中再開**: SQLiteストレージにより中断後も再開可能

## クイックスタート

```bash
cd k1_walk/k1_walk/scripts/fast_sac

# テスト実行（2トライアル、500イテレーション）
bash run_optuna.sh --n-trials 2 --max-iterations 500

# 本番実行（50トライアル、約17時間）
bash run_optuna.sh --study-name my_study --n-trials 50
```

## コマンドオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--study-name` | `k1_walk_reward_optimization` | Optunaスタディ名 |
| `--n-trials` | `50` | トライアル数 |
| `--max-iterations` | `3000` | 各トライアルの最大イテレーション数 |
| `--num-envs` | `4096` | 環境数 |
| `--timeout` | `45` | トライアルのタイムアウト（分） |
| `--resume` | - | 前回の最適化を再開 |

## 使用例

```bash
# 基本的な使用
bash run_optuna.sh --study-name my_study --n-trials 50

# イテレーション数を変更（高精度だが時間がかかる）
bash run_optuna.sh --max-iterations 5000 --n-trials 30

# 前回の最適化を再開
bash run_optuna.sh --resume --study-name my_study

# 環境数を減らす（GPUメモリが少ない場合）
bash run_optuna.sh --num-envs 2048 --n-trials 50
```

## 最適化対象パラメータ

### 報酬関数の重み（13項目）

| パラメータ | 現在値 | 探索範囲 | 説明 |
|-----------|--------|----------|------|
| `track_lin_vel_xy_exp` | 2.0 | [0.5, 5.0] | 線速度追跡（xy） |
| `track_ang_vel_z_exp` | 1.5 | [0.3, 3.0] | 角速度追跡（z） |
| `feet_height_bezier` | 5.0 | [1.0, 15.0] | 足の高さ制御 |
| `alive_bonus` | 10.0 | [3.0, 30.0] | 生存ボーナス |
| `orientation_potential` | 20.0 | [5.0, 60.0] | 姿勢ポテンシャル |
| `height_potential` | 25.0 | [5.0, 50.0] | 高さポテンシャル |
| `joint_regularization_potential` | 0.004 | [0.001, 0.05] | 関節正則化（対数スケール） |
| `upper_body_joint_regularization` | 0.8 | [0.2, 2.0] | 上半身関節正則化 |
| `feet_slide` | -0.08 | [-0.3, -0.01] | 足の滑りペナルティ |
| `feet_air_time` | 5.0 | [1.0, 15.0] | 足の飛行時間 |
| `action_rate_l2_legs` | -0.1 | [-0.5, -0.01] | 脚アクション変化率ペナルティ |
| `lin_vel_z_pen` | -6.0 | [-20.0, -1.0] | 垂直速度ペナルティ |
| `feet_close_penalty` | -1.0 | [-5.0, -0.1] | 足が近すぎるペナルティ |

## 出力

### 実行中の出力

各トライアルの開始時にコマンドと進捗が表示されます：

```
============================================================
Trial 0: Starting
Command: bash train_rough.sh --task K1-Walk-Train-rough ...
============================================================

Trial 0: Episode/length=450.23
```

### 最適化完了後の出力

```
============================================================
Optimization Results
============================================================

Best trial: 23
Best value (Episode/length): 892.4500

Best parameters:
  reward_alive_bonus: 15.234567
  reward_feet_air_time: 8.123456
  ...

Total trials: 50
Completed: 48, Failed: 2

To use the best parameters, run:
  bash train_rough.sh env.rewards.alive_bonus.weight=15.234567 ...

Results saved to: optuna_studies/results/my_study_20260413_120000.json
```

### 結果ファイル

結果は以下の場所に保存されます：

- **SQLiteデータベース**: `optuna_studies/study.db`（途中再開用）
- **JSON結果**: `optuna_studies/results/{study_name}_{timestamp}.json`

JSON結果には以下が含まれます：
- ベストパラメータ
- 全トライアルの結果
- 各トライアルのメトリクス

## 探索対象パラメータのカスタマイズ

`optuna_config.py` の `RewardSearchSpace` クラスを編集して、探索対象のパラメータを増減できます。

### パラメータを減らす場合

不要なフィールドをコメントアウトまたは削除します：

```python
@dataclass
class RewardSearchSpace:
    # 重要なパラメータのみ残す（例：5個に絞る）
    track_lin_vel_xy_exp: Tuple[float, float, bool] = (0.5, 5.0, False)
    alive_bonus: Tuple[float, float, bool] = (3.0, 30.0, False)
    orientation_potential: Tuple[float, float, bool] = (5.0, 60.0, False)
    lin_vel_z_pen: Tuple[float, float, bool] = (-20.0, -1.0, False)
    feet_air_time: Tuple[float, float, bool] = (1.0, 15.0, False)

    # 以下はコメントアウトして除外
    # track_ang_vel_z_exp: Tuple[float, float, bool] = (0.3, 3.0, False)
    # feet_height_bezier: Tuple[float, float, bool] = (1.0, 15.0, False)
    # height_potential: Tuple[float, float, bool] = (5.0, 50.0, False)
    # joint_regularization_potential: Tuple[float, float, bool] = (0.001, 0.05, True)
    # upper_body_joint_regularization: Tuple[float, float, bool] = (0.2, 2.0, False)
    # feet_slide: Tuple[float, float, bool] = (-0.3, -0.01, False)
    # action_rate_l2_legs: Tuple[float, float, bool] = (-0.5, -0.01, False)
    # feet_close_penalty: Tuple[float, float, bool] = (-5.0, -0.1, False)
```

パラメータを減らすと収束が速くなります（5個なら20-30トライアルで十分な場合も）。

### パラメータを増やす場合

`k1_walk_env_cfg.py` の `K1Rewards` クラスにある他の報酬項を追加できます。

```python
@dataclass
class RewardSearchSpace:
    # 既存のパラメータ
    track_lin_vel_xy_exp: Tuple[float, float, bool] = (0.5, 5.0, False)
    # ... 他の既存パラメータ ...

    # 新しく追加する例
    ang_vel_xy_l2: Tuple[float, float, bool] = (-1.0, -0.01, False)
    action_rate_l2_arms: Tuple[float, float, bool] = (-0.5, -0.01, False)
```

### 追加可能なパラメータ一覧

`k1_walk_env_cfg.py` に定義されている報酬項のうち、現在最適化対象に含まれていないもの：

| パラメータ名 | 現在値 | 推奨探索範囲 | 説明 |
|-------------|--------|-------------|------|
| `terminate_penalty` | -10.0 | [-30.0, -1.0] | 終了ペナルティ |
| `ang_vel_xy_l2` | -0.1 | [-1.0, -0.01] | 角速度xyペナルティ |
| `action_rate_l2_arms` | -0.1 | [-0.5, -0.01] | 腕アクション変化率 |
| `feet_parallel_to_ground` | 0.0 | [0.0, 10.0] | 足の平行性（現在無効） |

### 探索範囲の形式

各フィールドは `(min, max, log_scale)` のタプルで定義します：

- `min`: 最小値
- `max`: 最大値
- `log_scale`: `True` で対数スケールサンプリング（桁が大きく異なる場合に有効）

```python
# 線形スケール（通常の範囲）
alive_bonus: Tuple[float, float, bool] = (3.0, 30.0, False)

# 対数スケール（0.001〜0.05のような範囲）
joint_regularization_potential: Tuple[float, float, bool] = (0.001, 0.05, True)
```

### 注意事項

- パラメータ名は `k1_walk_env_cfg.py` の報酬項の名前と**完全に一致**させる必要があります
- パラメータを増やすと収束に必要なトライアル数が増えます（目安：パラメータ数 × 5〜10トライアル）
- 探索範囲を広げすぎると収束が遅くなります

## 推定実行時間

| 設定 | 1トライアル | 50トライアル |
|------|------------|-------------|
| `--max-iterations 3000` | 約20分 | 約17時間 |
| `--max-iterations 5000` | 約35分 | 約29時間 |
| `--max-iterations 1000` | 約7分 | 約6時間 |

## トラブルシューティング

### Optunaがインストールされていない場合

```bash
source ~/.bash_functions
_labpython -m pip install optuna
```

### GPUメモリ不足の場合

環境数を減らして実行：

```bash
bash run_optuna.sh --num-envs 2048 --n-trials 50
```

### 途中で中断した場合

Ctrl+C で中断後、`--resume` オプションで再開：

```bash
bash run_optuna.sh --resume --study-name my_study
```

## ファイル構成

```
fast_sac/
├── optuna_optimizer.py       # メインの最適化スクリプト
├── optuna_config.py          # 設定クラス（探索空間含む）
├── run_optuna.sh             # 実行ラッパー
├── optuna_utils/
│   ├── __init__.py
│   ├── search_space.py       # パラメータサンプリング
│   ├── metrics_extractor.py  # TensorBoardログ抽出
│   └── trial_runner.py       # 学習実行
└── optuna_studies/           # SQLiteストレージ・結果
    ├── study.db
    └── results/
        └── {study_name}_{timestamp}.json
```
