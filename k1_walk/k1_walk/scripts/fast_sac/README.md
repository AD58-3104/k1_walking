# FastSAC Training for IsaacLab

このディレクトリには、IsaacLabのタスクをFastSAC (Soft Actor-Critic)でトレーニングするためのスクリプトが含まれています。

## 概要

FastSACは、大規模並列シミュレーションに最適化された効率的なSoft Actor-Critic (SAC)の実装です。詳細は[FastTD3論文](https://arxiv.org/abs/2505.22642)を参照してください。

## ファイル構成

- `train.py` - メインのトレーニングスクリプト
- `play.py` - 学習済みモデルの実行スクリプト
- `cli_args.py` - コマンドライン引数のハンドラ
- `train.sh` - トレーニング用のシェルスクリプト
- `play.sh` - プレイ用のシェルスクリプト

## 使い方

### トレーニング

基本的なトレーニング:

```bash
cd /home/satoshi/k1_walking/k1_walk/k1_walk/scripts/fast_sac
./train.sh
```

カスタム設定でトレーニング:

```bash
./train.sh --num_envs 8192 --max_iterations 5000 --seed 42
```

分散トレーニング（複数GPU）:

```bash
# 2つのGPUで実行
source ~/.bash_functions
_labpython -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py \
  --task K1-Walk-Train-fast-sac --num_envs 16394 --headless --distributed
```

### プレイ（推論）

学習済みモデルの実行:

```bash
./play.sh
```

ビデオを録画:

```bash
./play.sh --video --video_length 500
```

特定のチェックポイントを指定:

```bash
./play.sh --checkpoint /path/to/model_00001000.pt
```

データロギングを有効化:

```bash
./play.sh --log_data --log_steps 5000 --log_env_ids "0,1,2"
```

## 主なパラメータ

### トレーニングパラメータ

- `--task`: タスク名（デフォルト: K1-Walk-Train-fast-sac）
- `--num_envs`: 並列環境数（デフォルト: 16394）
- `--max_iterations`: 最大イテレーション数
- `--seed`: ランダムシード
- `--headless`: ヘッドレスモードで実行
- `--experiment_name`: 実験名（ログディレクトリ用）
- `--run_name`: 実行名のサフィックス
- `--logger`: ロガータイプ（tensorboard または wandb）

### プレイパラメータ

- `--checkpoint`: 読み込むチェックポイントファイル
- `--video`: ビデオ録画を有効化
- `--video_length`: ビデオの長さ（ステップ数）
- `--real-time`: リアルタイムで実行
- `--log_data`: データロギングを有効化
- `--log_steps`: ログするステップ数
- `--log_env_ids`: ログする環境ID（カンマ区切り）

## エージェント設定

FastSACエージェントの設定は以下のファイルで定義されています:

```
k1_walk/source/k1_walk/k1_walk/tasks/manager_based/k1_walk/agents/fast_sac_cfg.py
```

主な設定項目:

- **学習率**: actor_learning_rate, critic_learning_rate, alpha_learning_rate
- **リプレイバッファ**: buffer_size, batch_size
- **ネットワーク構造**: actor_hidden_dim, critic_hidden_dim
- **最適化**: gamma, tau, num_updates
- **エントロピー**: target_entropy_ratio, alpha_init
- **その他**: compile (torch.compile), amp (自動混合精度)

## ログとチェックポイント

トレーニングログとチェックポイントは以下のディレクトリに保存されます:

```
logs/fast_sac/{experiment_name}/{timestamp}/
```

TensorBoardでログを確認:

```bash
tensorboard --logdir logs/fast_sac/
```

## トラブルシューティング

### インポートエラー

isaaclab_fast_sacモジュールが見つからない場合、パスが正しく設定されているか確認してください:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../holosoma/holosoma/src/isaaclab_fast_sac"))
```

### CUDA / GPU関連

- GPUメモリ不足の場合: `--num_envs` を減らす、または `batch_size` を調整
- 複数GPUで実行する場合: `--distributed` フラグを使用

### パフォーマンス最適化

- `compile=True`: torch.compileを有効化（初回実行は遅いが、その後は高速）
- `amp=True`: 自動混合精度を有効化（メモリ使用量を削減）
- `num_updates`: 各ステップでの更新回数を調整

## 参考

- [FastTD3論文](https://arxiv.org/abs/2505.22642)
- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [SAC論文](https://arxiv.org/abs/1801.01290)
