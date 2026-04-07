# プロジェクト
強化学習を利用して、2足歩行ロボット(booster-K1ロボット)に歩行を学習させるためのプロジェクト

# 構造
- k1_walk/k1_walk/scripts
  - 学習を実行するスクリプトが入っている
  - 現在はscripts/rsl_rlで主に学習を行っている
  - scripts/holosomaは、実績がある歩行の強化フレームワーク。このt1_29dof関連のタスクは調整の参考になる。
- k1_walk/k1_walk/source
  - 学習の設定とタスクが定義されている
  - mdp以下に観測や報酬関数が定義されている
  - k1_walk_env_cfg.pyが今使っている歩行タスクのコンフィグ

# pythonの実行方法
- isaaclabやisaacsimの環境が入ったpythonは、.bash_functionsに書いてある_labpythonコマンドで実行可能である。
- 学習は、rsl_rl/train_rough.shで実行する
