# autoresearch

Booster-K1ロボットを対象として、歩行タスクを行うSACによる強化学習をllmに最適化させる。対象ロボットは、下半身の全ての関節と、上半身の肩の左右のピッチ関節とロール関節を動かして、安定して速度に追従する歩行を学習することを目的とする。

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — training scripts. Do not modify.
   - `train_for_llm.sh` - training running scripts. Do not modify.
   - `fast_sac_cfg.py` — the file you modify. Model architecture, optimizer, training loop.
   - `k1_walk_env_cfg.py` - the file you modify. Reinforcement learning parameters. Such as, rewards, curriculum, emvironment.
   - `mdp` - Concrete definitions of reinforcement setting, such as reward functions, curriculum classes, observation functions. Do not modify.
   - `show_final_reward_table.py` - Print training result statistics from tensorboard logs. Do not modify.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 3030 episodes** You launch it simply as: `bash train_for_llm.sh`.

**What you CAN do:**
- Modify `k1_walk_env_cfg.py` — あなたが変更可能なファイルの1つです。これには、rlの報酬の重みやDRの設定値など、学習時の環境設定に関するハイパーパラメータが記載してあります。まずはこのファイルだけを変更することを優先してください。
- Modify `fast_sac_cfg.py` - あなたが変更可能なファイルのもう1つです。これには、SACの学習時のハイパーパラメータを記載します。
- Execute `train_for_llm.sh` - あなたが実行可能なファイルです。これを実行すると学習が実行されます。
- Execute `show_final_reward_table.py` - あなたが実行可能なファイルです。これを実行すると最新の学習結果が表示されます。

**What you CANNOT do:**
- `k1_walk_env_cfg.py`と`fast_sac_cfg.py` 以外のファイルは全て変更してはいけません。参照専用です。
- Install new packages or add dependencies.
- `train_for_llm.sh`と`show_final_reward_table.py` 以外のファイルは実行してはいけません。
- `logs` ディレクトリ内をを変更または削除してはいけません。

**The goal is simple: get the highest episode length and the heighest Episode_Reward/track_lin_vel_xy_exp.** Since the training episode length is fixed, you don't need to worry about training time. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, and rl enviroment parameters. The only constraint is that the code runs without crashing.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 1 episode length improvement that adds 20 lines of hacky code? Probably not worth it. A 1 episode lengths improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**baseline**: baselineは、logs/fast_sac/good_strideとする。この数値に近づくことを目指す。

## Output format

Once the script finishes it prints a summary like this:

```
Event file: logs/fast_sac/runner_setting3_exp_10/2026-04-03_21-04-42/events.out.tfevents.1775217906.satoshi-server.610289.0
| Metric                                         | Step |      Value |
| ---------------------------------------------- | ----: | ----------: |
| Curriculum/action_rate_arms_cur                | 2900 |  -0.050000 |
| Curriculum/action_rate_legs_cur                | 2900 |  -0.100000 |
| Curriculum/ang_vel_xy_cur                      | 2900 |  -0.010000 |
| Curriculum/base_jerk_cur                       | 2900 |  -0.005000 |
| Curriculum/feet_close_cur                      | 2900 |  -1.000000 |
| Curriculum/feet_parallel_to_ground_cur         | 2900 |   0.300000 |
| Curriculum/joint_regularization_potential_cur  | 2900 |   0.000500 |
| Curriculum/lin_vel_z_cur                       | 2900 |  -0.500000 |
| Curriculum/orientation_potential_cur           | 2900 |  -0.010000 |
| Curriculum/upper_body_joint_regularization_cur | 2900 |  -0.000500 |
| Episode_Reward/action_rate_l2_arms             | 2900 |  -0.007996 |
| Episode_Reward/action_rate_l2_legs             | 2900 |  -0.061319 |
| Episode_Reward/alive_bonus                     | 2900 |   0.663758 |
| Episode_Reward/ang_vel_xy_l2                   | 2900 |  -0.012444 |
| Episode_Reward/base_jerk                       | 2900 |  -0.006650 |
| Episode_Reward/feet_air_time                   | 2900 |   0.009179 |
| Episode_Reward/feet_close_penalty              | 2900 |  -0.003601 |
| Episode_Reward/feet_height_bezier              | 2900 |   0.513111 |
| Episode_Reward/feet_parallel_to_ground         | 2900 |  -0.000175 |
| Episode_Reward/joint_regularization_potential  | 2900 |  -0.042903 |
| Episode_Reward/lin_vel_z_pen                   | 2900 |  -0.007784 |
| Episode_Reward/orientation_potential           | 2900 |  -0.000062 |
| Episode_Reward/track_ang_vel_z_exp             | 2900 |   0.006744 |
| Episode_Reward/track_lin_vel_xy_exp            | 2900 |   0.025049 |
| Episode_Reward/upper_body_joint_regularization | 2900 |  -0.000256 |
| Episode_Termination/bad_orientation            | 2900 |  35.630001 |
| Episode_Termination/base_contact               | 2900 |   3.500000 |
| Episode_Termination/root_height_low            | 2900 |  23.700001 |
| Episode_Termination/time_out                   | 2900 |   0.000000 |
| Metrics/base_velocity/error_vel_xy             | 2900 |   0.197195 |
| Metrics/base_velocity/error_vel_yaw            | 2900 |   1.028022 |
| length                                         | 2900 | 135.240005 |
| reward                                         | 2900 |  21.809999 |
```

Note that the script is configured to always stop after 2200 episodes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the show_final_reward_table.py outputs:

```
source ~/.bash_functions; _labpython show_final_reward_table.py | grep -e "length" -e "track_lin_vel_xy_exp"
```

## Logging results

When an experiment is done, log it to logs directory. This directory has a lot of subdirectory but, each subdirectory has `tensorboard events.` files and `params` directory. `tensorboard events.`ファイルは、tensorboardのログファイルで、`show_final_reward_table.py`を使って読み込む事ができる。このスクリプトは以下のように利用することができる

```
usage: show_final_reward_table.py [-h] [--log-root LOG_ROOT] [--run-dir RUN_DIR] [--event-file EVENT_FILE] [--tag-prefix TAG_PREFIX [TAG_PREFIX ...]] [--step STEP] [--sort-by {name,value,step}] [--descending]

Print the latest TensorBoard reward scalars as a Markdown table.

options:
  -h, --help            show this help message and exit
  --log-root LOG_ROOT   Root directory that contains experiment run directories.
  --run-dir RUN_DIR     Specific run directory to inspect. If omitted, the latest run under --log-root is used.
  --event-file EVENT_FILE
                        Specific TensorBoard event file to inspect.
  --tag-prefix TAG_PREFIX [TAG_PREFIX ...]
                        Only include scalar tags starting with these prefixes. Can specify multiple.
  --step STEP           If set, show the latest scalar value recorded at or before this global step.
  --sort-by {name,value,step}
                        Column used to sort the output rows.
  --descending          Sort in descending order.
```

logsディレクトリのサブディレクトリとして存在する`params`ディレクトリには、その時の試行におけるfast_sacのエージェントの設定を表す`agent.yaml`とその時のrl環境のパラメータを表す`env.yaml`が含まれます。過去のパラメータ変更履歴を見て、チューニングの指針を決めたい場合は、これらのファイルも参照してください。

## The experiment loop

学習ループを回す前に、まず`autoresearch/<tag>`というブランチを1つ作って下さい。`<tag>`はその日の日付にしてください。全ての作業はそのブランチの上で行って下さい。

LOOP FOREVER:

1. Tune `k1_walk_env_cfg.py` and `fast_sac_cfg.py` with an experimental idea by directly hacking the code.
2. git commit
3. Run the experiment: `bash train_for_llm.sh`
4. Read out the results: `source ~/.bash_functions; _labpython show_final_reward_table.py"`
5. If the grep output is empty, the run crashed. Run `tail -n 70 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
6. If episode length and track_lin_vel_xy_exp improved (lower), you "advance" the branch, keeping the git commit
7. If episode length and track_lin_vel_xy_exp is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.自律的に実行するために、ユーザに許可を求める必要があるコマンドは実行しないでください。