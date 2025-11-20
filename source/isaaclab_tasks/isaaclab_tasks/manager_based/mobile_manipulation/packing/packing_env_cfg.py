# Cabinet Move Environment Configuration
  
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

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
class PackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the packing scene with a robot and a two desk.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_R_frame: FrameTransformerCfg = MISSING
    ee_L_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object : RigidObjectCfg | DeformableObjectCfg = MISSING
    object0 : RigidObjectCfg | DeformableObjectCfg = MISSING
    object1 : RigidObjectCfg | DeformableObjectCfg = MISSING
    object2 : RigidObjectCfg | DeformableObjectCfg = MISSING
    shelf : RigidObjectCfg | DeformableObjectCfg = MISSING
    # bag : RigidObjectCfg | DeformableObjectCfg = MISSING
    
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # Tables
    
    # table1 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table1",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1, 1, 1], rot=[0.707, 0, 0, -0.707]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(1.0, 1.0, 1.0), 
    #         ),
    # )

    
    # table2 = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table2",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1, -1, 1], rot=[0.707, 0, 0, -0.707]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(1.0, 1.0, 1.0), 
    #         ),
    # )
    
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    camera_Head = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
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
    
    camera_ee_r = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link1/ee_r_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, -0.05),
            rot=(0.25882, 0.96593, 0.0, 0.0),
            convention="opengl",
        ),
    )
    
    camera_ee_l = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link2/ee_l_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, -0.05),
            rot=(0.25882, 0.96593, 0.0, 0.0),
            convention="opengl",
        ),
    )

    

##
# MDP settings
##


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

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=['Omni.*','link1.*', 'link2.*', 'ee_link.*', 'arm.*', 'gripper.*']),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="can"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    
##  # Environment configuration
##  
    
@configclass
class PackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""
    
    # Scene settings
    scene: PackingSceneCfg = PackingSceneCfg(num_envs=4096, env_spacing=2.8)
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
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625