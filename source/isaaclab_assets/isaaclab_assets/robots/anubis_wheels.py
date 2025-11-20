# exaFLOPs

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

ANUBIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/IsaacLab/source/isaaclab_assets/data/Robots/MM/anubis/anubis_omni.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=20.0,
            max_contact_impulse=20.0,
            stabilization_threshold=0.001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),
    ), # --/renderer/shadercache/driverDiskCache/flush=true
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            # "OmniR": 0.0,
            # "OmniFR": 0.0,
            # "OmniFL": 0.0,

            # arm <-> base (rad)
            "arm1_base_link_joint": 0.0,
            "arm2_base_link_joint": 0.0,

            # Right arm
            "link11_joint": -0.69289571,
            "link12_joint": 2.34048653,
            "link13_joint": -0.07679449,
            "link14_joint": 0.52359878,
            "link15_joint": -0.17453293,
            
            # Left arm
            "link21_joint": -0.69289571,
            "link22_joint": 2.34048653,
            "link23_joint": -0.07679449,
            "link24_joint": -0.52359878,
            "link25_joint": 0.17453293,
            
            # finger
            "gripper.*": 0.04,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["OmniR", "OmniFR", "OmniFL"],
            effort_limit_sim=1e2,
            velocity_limit_sim=1e2,
            stiffness=0.01745,
            damping=1745,  # tip:: For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints. This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
            friction=0.75,
        ),
        "dummy_spheres": ImplicitActuatorCfg(
            joint_names_expr=["OmniR_roller_.*", "OmniFR_roller_.*", "OmniFL_roller_.*"],
            effort_limit_sim=1e1,
            velocity_limit_sim=1e2,
            stiffness=0,
            damping=0.00001,
            friction=1.5,
        ),
        "arm_base": ImplicitActuatorCfg(
            joint_names_expr=["arm.*"],
            effort_limit_sim=3000,
            velocity_limit_sim=100,
            stiffness= 6e2,
            damping= 100,
        ),
        "arm_link": ImplicitActuatorCfg(
            joint_names_expr=["link.*"],
            effort_limit_sim=3000,
            velocity_limit_sim=100,
            stiffness= 6e2,
            damping= 100,
        ),
        "anubis_right_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper1.*"],
            effort_limit_sim=300.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
            friction= 2.15,
        ),
        "anubis_left_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper2.*"],
            effort_limit_sim=30000.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
            friction= 2.15,
        ),
    },
)



ANUBIS_PD_CFG = ANUBIS_CFG.copy()
# ANUBIS_PD_CFG.spawn.rigid_props.disable_gravity = True
# ANUBIS_PD_CFG.actuators["arm_link"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_link"].damping ==400.0
# ANUBIS_PD_CFG.actuators["arm_base"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_base"].damping ==400.0
