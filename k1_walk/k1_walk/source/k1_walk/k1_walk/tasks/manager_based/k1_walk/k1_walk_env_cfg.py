# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import  IdealPDActuatorCfg

# ローカルのmdpモジュールをインポート（カスタム関数 + 標準mdp関数を含む）
from . import mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import math

from pathlib import Path

PROJECT_HOME_DIR = Path.home() / "k1_walking" / "K1_serial"

BOOSTER_K1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_HOME_DIR}/K1_serial.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # これオフでもいいらしいすね 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.0, 0.0, 0.72 - 0.0841),
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            "ALeft_Shoulder_Pitch" : 0.2, 
            "ARight_Shoulder_Pitch" : 0.2,   # ここはラジアンで指定するっぽい
            "Left_Shoulder_Roll" : -1.04,  # 地面に対して水平が0度なので、内側に動かすで正しい？ 
            "Left_Elbow_Pitch" : 0.5, 
            "Left_Elbow_Yaw" : 0.0, 
            "Right_Shoulder_Roll" : 1.04,
            "Right_Elbow_Pitch" : -0.5, 
            "Right_Elbow_Yaw" : 0.0, 

            "Left_Hip_Pitch" : -0.312, 
            "Left_Hip_Roll" : 0.0, 
            "Left_Hip_Yaw" : 0.0, 
            "Left_Knee_Pitch" : 0.669, 
            "Left_Ankle_Pitch" : -0.363, 
            "Left_Ankle_Roll" : 0.0, 
            
            "Right_Hip_Pitch" : -0.312, 
            "Right_Hip_Roll" : 0.0, 
            "Right_Hip_Yaw" : 0.0, 
            "Right_Knee_Pitch" : 0.669, 
            "Right_Ankle_Pitch" : -0.363, 
            "Right_Ankle_Roll" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*" : 30.0,
                ".*Hip_Roll": 20.0,
                ".*Hip_Yaw.*": 20.0,
                ".*_Knee_Pitch": 40.0,
                ".*_Ankle_Pitch": 20.0,
                ".*_Ankle_Roll": 15.0,
            },
            velocity_limit_sim={
                ".*Hip_Pitch.*" : 18.0, #rad/s
                ".*Hip_Roll": 18.0,
                ".*Hip_Yaw.*": 18.0,
                ".*_Knee_Pitch": 18.0,
                ".*_Ankle_Pitch": 18.0,
                ".*_Ankle_Roll": 18.0,
            },
            stiffness={
                ".*Hip_Yaw.*": 200.0,
                ".*Hip_Roll": 200.0,
                ".*Hip_Pitch.*": 200.0,
                ".*_Knee_Pitch": 200.0,
                ".*_Ankle_.*": 50.0,
            },
            damping={
                ".*Hip_Yaw.*": 2.0,
                ".*Hip_Roll": 2.0,
                ".*Hip_Pitch.*": 2.0,
                ".*_Knee_Pitch": 2.0,
                ".*_Ankle_.*": 1.0,
            },
        ),
        "arms": IdealPDActuatorCfg(    # K1_locomotionには無いのでコメントアウト
            joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit=100,
            velocity_limit=50.0,
            stiffness=40.0,
            damping=10.0,
        ),
        # "bodies": IdealPDActuatorCfg(
        #     joint_names_expr=["AAHead_Yaw", "Head_Pitch"],
        #     effort_limit=100.0,
        #     velocity_limit=100.0,
        #     stiffness=100.0,
        #     damping=10.0,
        # )
    },
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    contact_foot_right = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[])
    contact_foot_left = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_foot_link", # 足に当たった場合を検知する
                                        update_period=0.0,
                                        history_length=1,
                                        track_air_time=True,
                                        filter_prim_paths_expr=[])

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[
        ".*_Hip_.*",
        ".*_Knee_.*",
        ".*_Ankle_.*",
        ".*_Shoulder_Pitch",
        # ".*_Shoulder_.*",
        # ".*_Elbow_.*",
    ], scale=1.0, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # body_height = ObsTerm(func=mdp.body_height, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # foot_contact_states = ObsTerm(
        #         func=mdp.feet_contact,
        #         params={
        #             "sensor_cfg_right": SceneEntityCfg("contact_foot_right", body_names="right_foot_link"),
        #             "sensor_cfg_left": SceneEntityCfg("contact_foot_left", body_names="left_foot_link"),
        #         },
        #     )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        last_action = ObsTerm(func=mdp.last_action)
        # phase_time = ObsTerm(func=mdp.phase_time) rawなphaseはむしろノイズになる可能性があるらしいです。
        sincos_phase = ObsTerm(func=mdp.sincos_phase)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Observations for critic-only privileged state."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        foot_height = ObsTerm(
            func=mdp.foot_height,
            params={
                "foot_cfg_right": SceneEntityCfg("robot", body_names="right_foot_link"),
                "foot_cfg_left": SceneEntityCfg("robot", body_names="left_foot_link"),
            }
        )    # 要らなそうなやつを消してみる

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PrivilegedCfg):
        """Alias of privileged observations for frameworks expecting a critic group name."""

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    critic: CriticCfg = CriticCfg()



@configclass
class K1Rewards:
    """Reward terms for the MDP."""

    # ------------- タスク報酬
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.5, 
        params={"command_name": "base_velocity", "std": 0.25}
    )

    feet_height_bezier = RewTerm(
        func=mdp.feet_height_bezier, weight=5.0,
        params={
            "sigma": 0.08,
            "swing_height": 0.12,
            "stance_ratio": 0.40,
        },
    )  # メモ：報酬は遊脚のみ与えるようにするとか.

    alive_bonus = RewTerm(
        func=mdp.is_alive,
        weight= 10.0,
    )

    # ------------- ビヘイビア報酬

    # bad_gait_penalty = RewTerm(
    #     func=mdp.bad_gait_penalty,
    #     weight=-0.0,
    #     params={
    #         "min_air_time": 0.3,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
    #     }
    # )

    # ------------- シェイピング報酬（ポテンシャル系）

    orientation_potential = RewTerm(
        func=mdp.orientation_potential,
        weight=-1e-2,
        params={
            "sigma": 0.04,
            "discount_factor": 0.985,
            "enable_potential": False,
            }
    )

    joint_regularization_potential = RewTerm(
        func=mdp.joint_reqularization_potential,
        weight=4.0e-5,
        params={
            "sigma": 0.25,
            "pitch_slack": [0.01, 0.01, 5.0], # hip_pitch, knee_pitch, ankle_pitch
            "roll_slack": [1.0, 5.0],  # hip_roll, ankle_roll
            "yaw_slack": 0.3,
            "enable_exp_func": True,
            "enable_potential": False,
            }
    )

    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground, weight=3.0 * 0.1,
        params={
            "sigma": 0.1,
            "discount_factor": 0.985,
            "enable_potential": True,
                }
    )

    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,   # あまり高いとジャンプが最適解になる
    #     weight=-0.08,  # 元-0.1
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
    #     },
    # )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "command_name": "base_velocity",
            "threshold": 0.0,
        }
    )

    # ------------- シェイピング報酬（ペナルティ系）
    action_rate_l2_legs = RewTerm(
        func=mdp.action_rate_l2_subset,
        weight=-0.01,
        params={
            "joint_name_patterns": [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
            "action_term_name": "joint_pos",
        },
    )

    action_rate_l2_arms = RewTerm(
        func=mdp.action_rate_l2_subset,
        weight=-0.01,
        params={
            "joint_name_patterns": [".*_Shoulder_.*", ".*_Elbow_.*"],
            "action_term_name": "joint_pos",
        },
    )

    base_jerk = RewTerm(
        func=mdp.base_jerk,
        weight=-0.005,
    )

    ang_vel_xy_l2 = RewTerm(  # まだ追加した事は無いが、将来的に追加するかも
        func=mdp.ang_vel_xy_l2,
        weight=-0.01
    )

    # body_lin_acc = RewTerm(
    #     func=mdp.body_lin_acc_l2,
    #     weight=-0.075e-4,
    # )

    lin_vel_z_pen = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )

    feet_close_penalty = RewTerm(
        func=mdp.feet_close_penalty,
        weight=-1.0,
        params={
            "feet_distance_threshold": 0.09,
        }
    )

    # joint_jerk = RewTerm(
    #     func=mdp.joint_jerk,
    #     weight=-1e-7,
    # )

    # joint_torque = RewTerm(
    #     func=mdp.joint_torques_l2, 
    #     weight=-3e-7, 
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_.*", ".*_Hip_.*", ".*_Knee_.*"])}
    # )

    # joint_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-1e-9 * 0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_.*", ".*_Hip_.*", ".*_Knee_.*"])},
    # )

    upper_body_joint_regularization = RewTerm(
        func=mdp.upper_body_joint_regularization,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"]),
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    orientation_potential_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "orientation_potential",
            "target_weight" : -5.0,
            "init_levelup_threshold" : 100,
        }
    )

    joint_regularization_potential_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "joint_regularization_potential",
            "target_weight" : 3.0e-4,
            "init_levelup_threshold" : 100,
        }
    )

    feet_parallel_to_ground_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "feet_parallel_to_ground",
            "target_weight" : 10.0,
            "init_levelup_threshold" : 100,
        }
    )

    action_rate_legs_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "action_rate_l2_legs",
            "target_weight" : -1.7,
            "init_levelup_threshold" : 100,
        })

    action_rate_arms_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "action_rate_l2_arms",
            "target_weight" : -0.35,
            "init_levelup_threshold" : 100,
        })

    base_jerk_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "base_jerk",
            "target_weight" : -0.03,
            "init_levelup_threshold" : 100,
        }
    )

    ang_vel_xy_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "ang_vel_xy_l2",
            "target_weight" : -0.11,
            "init_levelup_threshold" : 100,
        }
    )

    # body_lin_acc_cur = CurrTerm(
    #     func=mdp.modify_reward_weight_by_episode_length_linearly,
    #     params = {
    #         "term_name" : "body_lin_acc",
    #         "target_weight" : -1.5e-4 * 0.5,
    #     }
    # )

    lin_vel_z_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "lin_vel_z_pen",
            "target_weight" : -12.5,
            "init_levelup_threshold" : 100,
        }
    )

    feet_close_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "feet_close_penalty",
            "target_weight" : -10.0,
            "init_levelup_threshold" : 100,
        }
    )

    # feet_slide_cur = CurrTerm(
    #     func=mdp.modify_reward_weight_by_episode_length_linearly,
    #      params = {
    #         "term_name" : "feet_slide",
    #         "target_weight" : -0.2 * 0.5,
    #      }
    # )

    # joint_acc_cur = CurrTerm(
    #     func=mdp.modify_reward_weight_by_episode_length_linearly,
    #     params = {
    #         "term_name" : "joint_acc",
    #         "target_weight" : -2.0e-8 * 0.5# -2e-8
    #     }
    # )

    upper_body_joint_regularization_cur = CurrTerm(
        func=mdp.modify_reward_weight_by_episode_length_linearly,
        params = {
            "term_name" : "upper_body_joint_regularization",
            "target_weight" : 2e-4,
            "init_levelup_threshold" : 100,
        }
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.95, 1.0),
            "dynamic_friction_range": (0.95, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-0.3, 0.3),
            "torque_range": (-0.3, 0.3),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk"]), "threshold": 1.0},
    )

    # Do not modify this parameter!!!! 
    root_height_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"), "minimum_height": 0.4},
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.4},
    )


@configclass
class K1RoughEnvCfg(ManagerBasedRLEnvCfg):
    rewards: K1Rewards = K1Rewards()
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.device = "cuda:0"  # Explicitly set to GPU
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Scene
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Randomization
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"


@configclass
class K1RoughEnvCfg_PLAY(K1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None



@configclass
class K1FlatEnvCfg(K1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
