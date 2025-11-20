# Cabinet Move Environment Configuration
  
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg

from . import mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_R_frame: FrameTransformerCfg = MISSING
    ee_L_frame: FrameTransformerCfg = MISSING
    
    # plane   
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    

    # Cabinet
    
    Brick = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/brick",
        spawn=sim_utils.UsdFileCfg(
            usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/Brick.usd",
            activate_contact_sensors=False,
            scale=(0.8, 0.8, 0.8), 
        ),
    )


    front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=1/30,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(-0.4, 0.0, 1.5),
            rot=(0.62721, 0.32651, -0.32651, -0.62721),
            convention="opengl",
        ),
    )

    wrist_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",
        update_period=1/30,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, -0.03),
            rot=(0.25882, 0.96593, 0.0, 0.0),
            convention="opengl",
        ),
    )
    
    wrist_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",
        update_period=1/30,
        height=128,
        width=128,
        data_types=["rgb"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, -0.03),
            rot=(0.25882, 0.96593, 0.0, 0.0),
            convention="opengl",
        ),
    )
    
    


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    armL_action: mdp.JointPositionActionCfg = MISSING
    armR_action: mdp.JointPositionActionCfg = MISSING
    gripperL_action: mdp.BinaryJointPositionActionCfg = MISSING
    gripperR_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""        
        ee_6d_pos = ObsTerm(func=mdp.ee_6d_pos)
        language_instruction = ObsTerm(func=mdp.language_instruction)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=['OmniR_roller_.*', 'OmniFR_roller_.*', 'OmniFL_roller_.*']),
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (1.7, 1.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # gripper_physics_material = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper1_joint", "gripper1R_joint", "gripper2_joint", "gripper2R_joint"]),
    #         "stiffness_distribution_params": (1, 1000),
    #         "damping_distribution_params": (1, 1000),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )


    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (0.5, 0.75),
    #         "dynamic_friction_range": (0.75, 1.0),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=['arm1_base_link_joint','arm2_base_link_joint','link11_joint', 'link21_joint', 'link12_joint', 'link22_joint', 'link13_joint', 'link23_joint', 'link14_joint', 'link24_joint', 'link15_joint', 'link25_joint']),
    #         "stiffness_distribution_params": (1.0, 1000),
    #         "damping_distribution_params": (1.0, 1000),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.open)

    
##  # Environment configuration
##  
    
@configclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""
    
    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=4096, env_spacing=2.8)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 100  # 60Hz
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625