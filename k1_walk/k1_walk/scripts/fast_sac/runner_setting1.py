BASE_COMMAND = "/home/satoshi/learning_practice/IsaacLab/./isaaclab.sh -p train.py"
COMMON_ARGS = "--task K1-Walk-Train-fast-sac --num_envs 8196 --headless"
ADDITIONAL_ARGS = ""   # コマンドラインからの追加引数を受け取る

rewards = "env.rewards"
currs = "env.curriculum"

arg_list =[
    [
        rewards + ".feet_height_bezier.weight=10.0"
    ],
    [
        rewards + ".feet_air_time.weight=20.0"
    ],
    [
        rewards + ".joint_regularization_potential.weight=2e-3"
    ],
    [
        rewards + ".joint_regularization_potential.weight=5e-3"
    ],
    [
        rewards + ".ang_vel_xy_l2=null"
    ],
    [
        rewards + ".orientation_potential=null"
    ]

]