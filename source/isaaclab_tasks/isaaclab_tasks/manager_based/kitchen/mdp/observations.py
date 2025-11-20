# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


# def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#	  """The distance between the end-effector and the object."""
#	  ee_tf_data: FrameTransformerData = env.scene["ee_R_frame"].data
#	  object_data: ArticulationData = env.scene["object"].data

#	  return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


# def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#	  """The distance between the end-effector and the object."""
#	  ee_tf_data: FrameTransformerData = env.scene["ee_R_frame"].data
#	  cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

#	  return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


# def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
#	  """The position of the fingertips relative to the environment origins."""
#	  ee_tf_data: FrameTransformerData = env.scene["ee_R_frame"].data
#	  fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

#	  return fingertips_pos.view(env.num_envs, -1)


# def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
#	  """The position of the end-effector relative to the environment origins."""
#	  ee_tf_data: FrameTransformerData = env.scene["ee_R_frame"].data
#	  ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

#	  return ee_pos


# def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
#	  """The orientation of the end-effector in the environment frame.

#	  If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
#	  """
#	  ee_tf_data: FrameTransformerData = env.scene["ee_R_frame"].data
#	  ee_quat = ee_tf_data.target_quat_w[..., 0, :]
#	  # make first element of quaternion positive
#	  return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat

def mobile_base(env: ManagerBasedRLEnv) -> torch.Tensor:
	pos = env.scene.articulations["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
	quat = env.scene.articulations["robot"].data.root_quat_w
	return torch.cat((pos, quat), dim=1)


def ee_6d_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
	observation = []
	
	# Robot base position	
	robot_data = env.scene.articulations["robot"].data
	root_pos_w = robot_data.root_pos_w.clone()
	root_quat_w = robot_data.root_quat_w.clone()

	# 1. ee_link2 (left arm)
	ee_L_pos_w = env.scene.articulations["robot"].data.body_pos_w[:,81]
	ee_L_quat_w = env.scene.articulations["robot"].data.body_quat_w[:,81]
	
	# 2. ee_link1 (right arm)
	ee_R_pos_w = env.scene.articulations["robot"].data.body_pos_w[:,78]
	ee_R_quat_w = env.scene.articulations["robot"].data.body_quat_w[:,78]
		
	ee_l_pose_b, ee_l_quat_b = math_utils.subtract_frame_transforms( root_pos_w, root_quat_w, ee_L_pos_w, ee_L_quat_w)
	ee_r_pose_b, ee_r_quat_b = math_utils.subtract_frame_transforms( root_pos_w, root_quat_w, ee_R_pos_w, ee_R_quat_w)
	observation.append(ee_l_pose_b)

	quat = ee_l_quat_b.clone()
		# Normalize quaternion
	quat = F.normalize(quat, dim=-1)
	w, x, y, z = quat.unbind(-1)

		# Compute rotation matrix elements
	B = quat.shape[0]
	R = torch.stack([
		1 - 2*y*y - 2*z*z,	 2*x*y - 2*z*w,		2*x*z + 2*y*w,
		2*x*y + 2*z*w,		 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
		2*x*z - 2*y*w,		 2*y*z + 2*x*w,		1 - 2*x*x - 2*y*y
	], dim=-1).reshape(B, 3, 3)

	# Take first 2 columns and flatten
	rot6d = R[:, :, :2].reshape(B, 6)
	observation.append(rot6d)

	observation.append(ee_r_pose_b)
	quat = ee_r_quat_b.clone()
		# Normalize quaternion
	quat = F.normalize(quat, dim=-1)
	w, x, y, z = quat.unbind(-1)

		# Compute rotation matrix elements
	B = quat.shape[0]
	R = torch.stack([
		1 - 2*y*y - 2*z*z,	 2*x*y - 2*z*w,		2*x*z + 2*y*w,
		2*x*y + 2*z*w,		 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
		2*x*z - 2*y*w,		 2*y*z + 2*x*w,		1 - 2*x*x - 2*y*y
	], dim=-1).reshape(B, 3, 3)

	# Take first 2 columns and flatten
	rot6d = R[:, :, :2].reshape(B, 6)
		

	observation.append(rot6d)

	# Gripper
	robot= env.scene["robot"]
	finger_joint_L1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
	finger_joint_R1 = robot.data.joint_pos[:, -3].clone().unsqueeze(1)

	observation.append(finger_joint_L1)
	observation.append(finger_joint_R1)
	return torch.cat(observation, dim=-1)

def base_qpos(env: ManagerBasedRLEnv) -> torch.Tensor:
	# compute angular velocity (yaw rate)
	base_qpos = env.scene.articulations["robot"].data.root_lin_vel_b
	base_qpos[:,1] *= -1
	base_qpos[:,2] = env.scene.articulations["robot"].data.root_ang_vel_b[:,2]
	# debug print
	return base_qpos

# Joint position
def joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
	return env.scene.articulations["robot"].data.joint_pos

def joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
	return env.scene.articulations["robot"].data.joint_vel

# Force
# EEF distance to all 
