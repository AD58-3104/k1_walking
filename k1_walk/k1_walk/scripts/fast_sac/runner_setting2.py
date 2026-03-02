BASE_COMMAND = "/home/satoshi/learning_practice/IsaacLab/./isaaclab.sh -p train.py"
COMMON_ARGS = "--task K1-Walk-Train-fast-sac --num_envs 8196 --headless --device 'cuda:1'"
ADDITIONAL_ARGS = ""   # コマンドラインからの追加引数を受け取る

rewards = "env.rewards"
currs = "env.curriculum"

arg_list =[
    [
        rewards + ".feet_slide.weight=-0.5"
    ],
    [
        rewards + ".feet_slide.weight=-1.0"
    ],
    [
        rewards + ".lin_vel_z_pen.weight=-0.5"
    ],
    [
        rewards + ".ang_vel_xy_l2.weight=-0.1"
    ],
    [
        rewards + ".ang_vel_xy_l2.weight=-0.05"
    ],
]
