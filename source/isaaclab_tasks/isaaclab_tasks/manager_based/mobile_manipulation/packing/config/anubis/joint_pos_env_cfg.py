# Environment configuration for the Anubis robot in the Cabinet task for RL training.
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from isaaclab_tasks.manager_based.mobile_manipulation.packing import mdp

from isaaclab_tasks.manager_based.mobile_manipulation.packing.packing_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    PackingEnvCfg,
)
##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip

@configclass
class AnubisPackingEnvCfg(PackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.armR_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "arm1_base_joint": (-0.523599, 0.523599),
                "link11_joint": (-0.523599, 1.91986),
                "link12_joint": (0.174533, 2.79253),
                "link13_joint": (-1.5708, 1.74533),
                "link14_joint": (-1.5708, 1.57085),
                "link15_joint": (-1.74533, 1.74533),
            }
        )
        self.actions.armL_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link2.*","arm2.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "link21_joint": (-0.523599, 1.91986),
                "link22_joint": (0.174533, 2.79253),
                "link23_joint": (-1.5708, 1.74533),
                "link24_joint": (-1.5708, 1.57085),
                "link25_joint": (-1.74533, 1.74533),
                "arm2_base_joint": (-0.523599, 0.523599),
            }
        )
        self.actions.gripperR_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper1.*"],
            open_command_expr={"gripper1.*": 0.04},
            close_command_expr={"gripper1.*": 0.0},
        )

        self.actions.gripperL_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper2.*"],
            open_command_expr={"gripper2.*": 0.04},
            close_command_expr={"gripper2.*": 0.0},
        )

        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["Omni.*"],
        )
        

        shelf = RigidObjectCfg(
            init_state=RigidObjectCfg.InitialStateCfg(pos=[1, 1, 0], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Assets/shelf.usd",
                scale=(1.0, 1.0, 1.0),
            )
        )

        self.scene.shelf = shelf.replace(
            prim_path="{ENV_REGEX_NS}/shelf",
        )


        can_cfg = RigidObjectCfg(
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 1, 1.1], rot=[0.70711, -0.70711, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Assets/can.usd",
                scale=(0.5, 0.5, 0.5),
            )
        )
        
        self.scene.object = can_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Object",
        )

        for i in range(3):
            can_cfg = RigidObjectCfg(
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=[0.7 + i * 0.1, -1 , 1.1],  # adjust Y to avoid overlap
                    rot=[0.70711, -0.70711, 0, 0]
                ),
                spawn=UsdFileCfg(
                    usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Assets/can.usd",
                    scale=(0.5, 0.5, 0.5),
                )
            )
            if i == 0:
                self.scene.object0 = can_cfg.replace(
                    prim_path="{ENV_REGEX_NS}/Can_0",
                )
            elif i == 1:
                self.scene.object1 = can_cfg.replace(
                    prim_path="{ENV_REGEX_NS}/Can_1",
                )
            elif i == 2:
                self.scene.object2 = can_cfg.replace(
                    prim_path="{ENV_REGEX_NS}/Can_2",
                ) 



        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_R_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/vr_headset_frame",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link1",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )
        self.scene.ee_L_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/vr_headset_frame",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link2",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )

@configclass
class AnubisPackingEnvCfg_PLAY(AnubisPackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 6
        # disable randomization for play
        self.observations.policy.enable_corruption = False
