# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.anubis import ANUBIS_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class EventCfg:
    robot_initial_joint = EventTerm(
      func=mdp.randomize_actuator_gains,
      mode="reset",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
          "stiffness_distribution_params": (1.0, 1e5),
          "damping_distribution_params": (1.0, 1e5),
          "operation": "abs",
          "distribution": "uniform",
      },
  )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint.*"),
            "stiffness_distribution_params": (1.0, 1e5),
            "damping_distribution_params": (1.0, 1e5),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    gripper_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
            "stiffness_distribution_params": (1.0, 1e4),
            "damping_distribution_params": (1.0, 1e4),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    

@configclass
class Real2simEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    
    action_space = 7 # 17 for anubis
    # +------------------------------------+
    # |   Active Action Terms (shape: 7)   |
    # +-------+----------------+-----------+
    # | Index | Name           | Dimension |
    # +-------+----------------+-----------+
    # |   0   | arm_action     |         6 |
    # |   1   | gripper_action |         1 |
    # +-------+----------------+-----------+  For Franka
    #+-------------------------------------+
    #|   Active Action Terms (shape: 17)   |
    #+-------+-----------------+-----------+
    #| Index | Name            | Dimension |
    #+-------+-----------------+-----------+
    #|   0   | armL_action     |         6 |
    #|   1   | armR_action     |         6 |
    #|   2   | gripperL_action |         1 |
    #|   3   | gripperR_action |         1 |
    #|   4   | base_action     |         3 |
    #+-------+-----------------+-----------+  For anubis
    
    observation_space = 30 # 55 for anubis
    
    # +-----------------------------------------------------------+
    # | Active Observation Terms in Group: 'policy' (shape: (30,)) |
    # +----------+-------------------------------------+----------+
    # |  Index   | Name                                |  Shape   |
    # +----------+-------------------------------------+----------+
    # |    0     | joint_pos                           |   (9,)   |
    # |    1     | joint_vel                           |   (9,)   |
    # |    2     | cabinet_joint_pos                   |   (1,)   |
    # |    3     | cabinet_joint_vel                   |   (1,)   |
    # |    4     | rel_ee_drawer_distance              |   (3,)   |
    # |    5     | actions                             |   (7,)   |
    # +----------+-------------------------------------+----------+  For Franka

    #+-----------------------------------------------------+
    #| Active Observation Terms in Group: 'policy' (shape: (55,)) |
    #+--------------+-----------------------+--------------+
    #|    Index     | Name                  |    Shape     |
    #+--------------+-----------------------+--------------+
    #|      0       | joint_pos             |    (19,)     |
    #|      1       | joint_vel             |    (19,)     |
    #|      2       | actions               |    (17,)     |
    #+--------------+-----------------------+--------------+  For anubis
            
    state_space = 30

    # simulation
    # [TODO] Match dt with the real robot
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    # robot_cfg: ArticulationCfg = ANUBIS_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)

    # events
    events: EventCfg = EventCfg()
    
    # - controllable joint
    panda_joint1 = "panda_joint1"
    panda_joint2 = "panda_joint2"
    panda_joint3 = "panda_joint3"
    panda_joint4 = "panda_joint4"
    panda_joint5 = "panda_joint5"
    panda_joint6 = "panda_joint6"
    panda_joint7 = "panda_joint7"
    panda_finger_joint1 = "panda_finger_joint1"
    panda_finger_joint2 = "panda_finger_joint2"
    
    # TODO Is this 1 correct?
    # - action scale
    action_scale = 1.0  # [N]
    
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    
    """               
    clip = {
                "link1_joint": (-0.523599, 1.91986),
                "link12_joint": (0.174533, 2.79253),
                "link13_joint": (-1.5708, 1.74533),
                "link14_joint": (-1.5708, 1.57085),
                "link15_joint": (-1.74533, 1.74533),
                "arm1_base_joint": (-0.523599, 0.523599),
            }
    """
            