BASE_COMMAND = "/home/satoshi/learning_practice/IsaacLab/./isaaclab.sh -p train.py"
COMMON_ARGS = "--task K1-Walk-Train-fast-sac --num_envs 8196 --headless --device 'cuda:1'"
ADDITIONAL_ARGS = ""   # コマンドラインからの追加引数を受け取る

rewards = "env.rewards"
currs = "env.curriculum"

arg_list =[
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".lin_vel_z_pen.weight=-0.05",
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".lin_vel_z_pen.weight=-5.0"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".action_rate_cur.params.target_weight=-0.2"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".action_rate_cur.params.target_weight=-10.0"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".base_jerk_cur.params.target_weight=-0.3"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".base_jerk_cur.params.target_weight=-0.003"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".bad_gait_cur.params.target_weight=-0.1"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        currs + ".bad_gait_cur.params.target_weight=-12.0"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".orientation_potential.weight=100.0"
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".stride_length.weight=8.0 ",
        rewards + ".stride_length.params.sigma=0.15 "
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".stride_length.weight=8.0 ",
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
    [
        rewards + ".joint_regularization_potential.params.enable_potential=False ",
        rewards + ".stride_length.params.sigma=0.15 "
        "env.rewards.joint_regularization_potential.params.enable_potential=False env.rewards.joint_regularization_potential.weight=4e-5"
    ],
]
