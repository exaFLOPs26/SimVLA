# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##


@configclass
class BrickPoscoSceneCfg(InteractiveSceneCfg):
	"""Configuration for a cart-pole scene."""
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
	robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",init_state=ArticulationCfg.InitialStateCfg(pos=(0.0,0.0 ,4.0 ),rot=(1.0, 0.0, 0.0, 0.0))
			)

	Brick = AssetBaseCfg(
		prim_path="{ENV_REGEX_NS}/brick",
		spawn=sim_utils.UsdFileCfg(
			usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/Brick1.usd",
			activate_contact_sensors=False,
			scale=(1.0, 1.0, 1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
		),
	)
	Last_Brick = RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/last_brick",
		
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(-0.38952, 2.11861, 2.40992),
			rot=(0.1072, 0.0, 0.0, -0.9942),
		),
		spawn=sim_utils.UsdFileCfg(
			usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/last_brick.usd",
			activate_contact_sensors=False,
			scale=(1.09, 1.09, 1.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			rigid_props=sim_utils.RigidBodyPropertiesCfg(),
		)
	)
##
# MDP settings
##


@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""
	joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
	"""Observation specifications for the MDP."""

	@configclass
	class PolicyCfg(ObsGroup):
		"""Observations for policy group."""

		# observation terms (order preserved)
		joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
		joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

		def __post_init__(self) -> None:
			self.enable_corruption = False
			self.concatenate_terms = True

	# observation groups
	policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
	"""Configuration for events."""

	# reset
	reset_cart_position = EventTerm(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
			"position_range": (-1.0, 1.0),
			"velocity_range": (-0.5, 0.5),
		},
	)

	reset_pole_position = EventTerm(
		func=mdp.reset_joints_by_offset,
		mode="reset",
		params={
			"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
			"position_range": (-0.25 * math.pi, 0.25 * math.pi),
			"velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
		},
	)


@configclass
class RewardsCfg:
	"""Reward terms for the MDP."""

	# (1) Constant running reward
	alive = RewTerm(func=mdp.is_alive, weight=1.0)
	# (2) Failure penalty
	terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""

	# (1) Time out
	time_out = DoneTerm(func=mdp.time_out, time_out=True)
	# (2) Cart out of bounds
	cart_out_of_bounds = DoneTerm(
		func=mdp.joint_pos_out_of_manual_limit,
		params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
	)


##
# Environment configuration
##


@configclass
class BrickPoscoEnvCfg(ManagerBasedRLEnvCfg):
	# Scene settings
	scene: BrickPoscoSceneCfg = BrickPoscoSceneCfg(num_envs=32, env_spacing=8.0)
	# Basic settings
	observations: ObservationsCfg = ObservationsCfg()
	actions: ActionsCfg = ActionsCfg()
	events: EventCfg = EventCfg()
	# MDP settings
	rewards: RewardsCfg = RewardsCfg()
	terminations: TerminationsCfg = TerminationsCfg()

	# Post initialization
	def __post_init__(self) -> None:
		"""Post initialization."""
		# general settings
		self.decimation = 2
		self.episode_length_s = 5
		# viewer settings
		self.viewer.eye = (8.0, 0.0, 5.0)
		# simulation settings
		self.sim.dt = 1 / 120
		self.sim.render_interval = self.decimation
