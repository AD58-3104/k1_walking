BASE_COMMAND = "/home/satoshi/learning_practice/IsaacLab/./isaaclab.sh -p train.py"
COMMON_ARGS = "--task K1-Walk-Train-fast-sac --num_envs 8196 --headless --max_iteration 20000"
ADDITIONAL_ARGS = ""   # コマンドラインからの追加引数を受け取る

rewards = "env.rewards"
currs = "env.curriculum"

arg_list =[
    [
    ],
    [
        rewards + ".feet_slide.weight=-0.1",
    ],
    [
        currs + ".body_lin_acc_cur.params.target_weight=-6e-4"
    ],
    [
        currs + ".base_jerk_cur.params.target_weight=-0.09"
    ],
    [
        rewards + ".feet_slide.weight=-0.1",
        currs + ".body_lin_acc_cur.params.target_weight=-6e-4"
    ],
    [
        rewards + ".feet_slide.weight=-0.1",
        currs + ".base_jerk_cur.params.target_weight=-0.09"
    ],
    [
        rewards + ".feet_slide.weight=-0.1",
        currs + ".base_jerk_cur.params.target_weight=-0.09",
        currs + ".body_lin_acc_cur.params.target_weight=-6e-4"
    ],
    [
        currs + ".base_jerk_cur.params.target_weight=-0.09",
        currs + ".body_lin_acc_cur.params.target_weight=-6e-4",
    ]
]