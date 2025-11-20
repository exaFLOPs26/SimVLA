"""
Whole-body motion planning with cuRobo
"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Motion Planning Anubis in IsaacLab")
parser.add_argument(
	"--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."	
)
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v1", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=25, help="Number of environments to simulate.")
parser.add_argument("--robot", type=str, default="anubis", help="Which robot to use in the task.")
parser.add_argument("--record", type=bool, default=False, help="Whether to record the simulation.")
parser.add_argument(
	"--dataset_file", type=str, default="./datasets/anubis/Isaac_Kitchen_v1_1.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
	"--num_demos", type=int, default=4096, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
	"--num_success_steps",
	type=int,
	default=10,
	help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)
# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Import basic packages
import torch
import gymnasium as gym
import ipdb
# Import Omniverse logger
import omni.log
import omni.ui as ui
import numpy as np
import json
# Import for reset
from isaaclab.devices import Se3Keyboard_BMM
# Import for record Demo
import os
import time
import random
import contextlib
import isaaclab_mimic.envs	# noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
# Import for cuRobo
from isaacsim.core.utils.types import ArticulationAction
from curobo.util_file import (
	get_robot_configs_path,
	join_path,
	load_yaml,
)
from curobo.util.usd_helper import UsdHelper
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import (
	MotionGen,
	MotionGenConfig,
	MotionGenPlanConfig,
)
from curobo.geom.types import WorldConfig, Mesh, Sphere
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from isaacsim.core.api.objects import cuboid
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R
# TODO Do I need RateLimiter?
class RateLimiter:
	"""Convenience class for enforcing rates in loops."""

	def __init__(self, hz: int):
		"""Initialize a RateLimiter with specified frequency.

		Args:
			hz: Frequency to enforce in Hertz.
		"""
		self.hz = hz
		self.last_time = time.time()
		self.sleep_duration = 1.0 / hz
		self.render_period = min(0.033, self.sleep_duration)

	def sleep(self, env: gym.Env):
		"""Attempt to sleep at the specified rate in hz.

		Args:
			env: Environment to render during sleep periods.
		"""
		next_wakeup_time = self.last_time + self.sleep_duration
		while time.time() < next_wakeup_time:
			time.sleep(self.render_period)
			env.sim.render()

		self.last_time = self.last_time + self.sleep_duration

		# detect time jumping forwards (e.g. loop is too slow)
		if self.last_time < time.time():
			while self.last_time < time.time():
				self.last_time += self.sleep_duration
def compute_wheel_velocities_torch(vx, vy, wz, wheel_radius, l):
	theta = torch.tensor([0, 2 * torch.pi / 3, 4 * torch.pi / 3], device=vx.device)
	M = torch.stack([
		-torch.sin(theta),
		torch.cos(theta),
		torch.full_like(theta, l)
	], dim=1)  # Shape: (3, 3)

	base_vel = torch.stack([vx, vy, wz], dim=-1)  # Shape: (B, 3)
	wheel_velocities = (1 / wheel_radius) * base_vel @ M.T	# Shape: (B, 3)
	return wheel_velocities
def world2base(env, ee_pos_w, ee_quat_w, env_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Converts a single world-frame pose to a base-frame pose for a specific environment.
	"""
	robot_data = env.scene.articulations["robot"].data
	
	# Select the root pose for the *specific environment* using env_idx
	root_pos_w = robot_data.root_pos_w[env_idx]
	print(root_pos_w)
	root_quat_w = robot_data.root_quat_w[env_idx]

	# Add a batch dimension of 1 for the math_utils function
	root_pos_w = root_pos_w.unsqueeze(0)
	root_quat_w = root_quat_w.unsqueeze(0)
	ee_pos_w = ee_pos_w.unsqueeze(0)
	ee_quat_w = ee_quat_w.unsqueeze(0)

	# Compute the pose of the body in the root frame
	ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
		root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
	)
	# Return the single pose, removing the batch dimension
	return ee_pose_b.squeeze(0), ee_quat_b.squeeze(0)

def q_inverse(q):
	"""Calculates the inverse of a batch of quaternions."""
	# q_inv = q_conjugate / q_norm^2
	q_conj = q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)
	q_norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
	return q_conj / q_norm_sq

def q_mul(q1, q2):
	"""Multiplies two batches of quaternions."""
	w1, x1, y1, z1 = q1.unbind(-1)
	w2, x2, y2, z2 = q2.unbind(-1)
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
	z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
	return torch.stack((w, x, y, z), -1)
def quat_to_rotvec_torch(q, eps=1e-8):
	"""
	q: (B, 4) quaternions in (w, x, y, z) format
	returns: (B, 3) rotation vectors
	"""
	q = torch.nn.functional.normalize(q, dim=-1)  # ensure unit quaternion

	w, x, y, z = q.unbind(-1)
	angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))  # (B,)

	sin_half = torch.sqrt(1.0 - w * w + eps)  # (B,)
	axis = torch.stack((x, y, z), dim=-1) / sin_half.unsqueeze(-1)	# (B, 3)

	rotvec = axis * angle.unsqueeze(-1)
	# handle small angles: if angle ~ 0, set rotvec ~ 0
	rotvec = torch.where(angle.unsqueeze(-1) > eps, rotvec, torch.zeros_like(rotvec))
	return rotvec

def compute_eef_deltas_batched(motion_gen, plans, cmd_indices, joint_names, eef_link_name):
	"""
	Computes end-effector delta poses for a batch of arms executing plans.
	
	Args:
		motion_gen: The cuRobo MotionGen object.
		plans: A list of cuRobo plan objects for the executing arms.
		cmd_indices: A tensor of current command indices for each plan.
		joint_names: The list of joint names for ordering.
		eef_link_name: The name of the end-effector link in the URDF.

	Returns:
		A tensor of delta poses (dx, dy, dz, drx, dry, drz) of shape (batch_size, 6).
	"""
	batch_size = len(plans)
	device = cmd_indices.device

	q_t_list, q_tp1_list = [], []
	arm_joint_names = [name for name in joint_names if "gripper" not in name]
	# 1. Gather current (t) and next (t+1) joint positions for the entire batch
	for i in range(batch_size):
		plan = plans[i]
		idx = cmd_indices[i].item()

		# Get current joint state
		q_t_state = plan.get_ordered_joint_state(arm_joint_names)[idx]
		q_t_list.append(q_t_state.position)

		# Get next joint state, handling the end of the trajectory
		if (idx + 1) < len(plan):
			q_tp1_state = plan.get_ordered_joint_state(arm_joint_names)[idx + 1]
			q_tp1_list.append(q_tp1_state.position)
		else:
			# If at the end, the delta is zero, so next state is same as current
			q_tp1_list.append(q_t_state.position)
			
	# Stack lists into batched tensors
	q_t = torch.stack(q_t_list)
	q_tp1 = torch.stack(q_tp1_list)

	kinematics = motion_gen.kinematics
	pose_t_tuple = kinematics.forward(q_t, link_name=eef_link_name)
	
	position_t = pose_t_tuple[0].clone()
	quaternion_t = pose_t_tuple[1].clone()
	pose_tp1_tuple = kinematics.forward(q_tp1, link_name=eef_link_name)

	position_tp1 = pose_tp1_tuple[0]
	quaternion_tp1 = pose_tp1_tuple[1]
	delta_pos = position_tp1 - position_t

	r_t = R.from_quat(quaternion_t.detach().cpu().numpy()[:, [1, 2, 3, 0]])
	r_tp1 = R.from_quat(quaternion_tp1.detach().cpu().numpy()[:, [1, 2, 3, 0]])
	delta_r = r_tp1 * r_t.inv()
	delta_rotvec = torch.tensor(delta_r.as_rotvec(), dtype=torch.float32)
	device = delta_pos.device
	delta_rotvec = delta_rotvec.to(device)	

#	q_t_inv = q_inverse(quaternion_t)
#	delta_quat = q_mul(q_t_inv, quaternion_tp1)
#	delta_rotvec = quat_to_rotvec_torch(delta_quat)
	return torch.cat((delta_pos, delta_rotvec), dim=-1)

def pre_process_actions(
	delta_pose_L: torch.Tensor,
	gripper_command_L: bool,
	delta_pose_R: torch.Tensor,
	gripper_command_R: bool,
	delta_pose_base: torch.Tensor,
) -> torch.Tensor:
	"""Pre-process actions for the environment."""

	# Convert base motion to wheel velocities
	delta_pose_base_wheel = compute_wheel_velocities_torch(
		delta_pose_base[:, 0], delta_pose_base[:, 1], delta_pose_base[:, 2],
		wheel_radius=0.23, l=0.05
	)[:, [2, 1, 0]]

	batch_size = delta_pose_L.shape[0]

	# Convert gripper commands to tensors
	gripper_command_L = torch.as_tensor(gripper_command_L, device=delta_pose_L.device, dtype=torch.bool).reshape(-1, 1)
	gripper_command_R = torch.as_tensor(gripper_command_R, device=delta_pose_R.device, dtype=torch.bool).reshape(-1, 1)

	# Expand if needed
	if gripper_command_L.shape[0] == 1:
		gripper_command_L = gripper_command_L.expand(batch_size, 1)
	if gripper_command_R.shape[0] == 1:
		gripper_command_R = gripper_command_R.expand(batch_size, 1)

	# Convert gripper bools to velocities
	gripper_vel_L = torch.where(gripper_command_L, -1.0, 1.0)
	gripper_vel_R = torch.where(gripper_command_R, -1.0, 1.0)

	# Concatenate all action components
	action = torch.cat([
		delta_pose_L, delta_pose_R,
		gripper_vel_L, gripper_vel_R,
		delta_pose_base_wheel
	], dim=1)
	# Dummy spheres for omniwheel
	padding = torch.zeros(action.shape[0], 60, device=action.device)

	return torch.cat([action, padding], dim=1)

import numpy as np
from pxr import Gf

def pose_to_gf_matrix_tensor(position: torch.Tensor, quaternion: torch.Tensor) -> Gf.Matrix4d:
	"""
	position: torch.Tensor of shape (3,)
	quaternion: torch.Tensor of shape (4,) in w, x, y, z order
	"""
	# Convert to numpy
	pos = position.detach().cpu().numpy().astype(np.float64)
	quat_wxyz = quaternion.detach().cpu().numpy().astype(np.float64)

	# Reorder to x, y, z, w
	quat = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
	x, y, z, w = quat

	# Compute rotation matrix
	rot = np.array([
		[1 - 2*(y*y + z*z), 2*(x*y - z*w),	   2*(x*z + y*w)],
		[2*(x*y + z*w),		1 - 2*(x*x + z*z), 2*(y*z - x*w)],
		[2*(x*z - y*w),		2*(y*z + x*w),	   1 - 2*(x*x + y*y)]
	], dtype=np.float64)

	# Build 4x4 matrix
	mat = np.eye(4, dtype=np.float64)
	mat[:3, :3] = rot
	mat[3, :3] = pos

	return Gf.Matrix4d(mat.tolist())

def main():
	# Rate limiter
	rate_limiter = RateLimiter(args_cli.step_hz)
	# Save Dataset
	output_dir = os.path.dirname(args_cli.dataset_file)
	output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Parse configuration
	env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
	env_cfg.env_name = args_cli.task
	# Success termination check
	success_term = None
	if hasattr(env_cfg.terminations, "success"):
		success_term = env_cfg.terminations.success
		env_cfg.terminations.success = None
	else:
		omni.log.warn(
			"No success termination term was found in the environment."
			" Will not be able to mark recorded demos as successful."
		)

	# modify configuration such that the environment runs indefinitely until
	# the goal is reached or other termination conditions are met
	env_cfg.terminations.time_out = None

	# TODO: What is this concatenate_terms?
	env_cfg.observations.policy.concatenate_terms = False

# TODO: Check if this recorder matches to Lerobot format
	env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
	env_cfg.recorders.dataset_export_dir_path = output_dir
	env_cfg.recorders.dataset_filename = output_file_name
	env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

	# TODO: Add reset on vr button
	# add teleoperation key for env reset
	should_reset_recording_instance = False
	running_recording_instance = True


	# TODO: Do we need failure demos?
	def reset_recording_instance():
		"""Reset the current recording instance.

		This function is triggered when the user indicates the current demo attempt
		has failed and should be discarded. When called, it marks the environment
		for reset, which will start a fresh recording instance. This is useful when:
		- The robot gets into an unrecoverable state
		- The user makes a mistake during demonstration
		- The objects in the scene need to be reset to their initial positions
		"""
		nonlocal should_reset_recording_instance
		should_reset_recording_instance = True

	def start_recording_instance():
		"""Start or resume recording the current demonstration.

		This function enables active recording of robot actions. It's used when:
		- Beginning a new demonstration after positioning the robot
		- Resuming recording after temporarily stopping to reposition
		- Continuing demonstration after pausing to adjust approach or strategy

		The user can toggle between stop/start to reposition the robot without
		recording those transitional movements in the final demonstration.
		"""
		nonlocal running_recording_instance
		running_recording_instance = True

	def stop_recording_instance():
		"""Temporarily stop recording the current demonstration.

		This function pauses the active recording of robot actions, allowing the user to:
		- Reposition the robot or hand tracking device without recording those movements
		- Take a break without terminating the entire demonstration
		- Adjust their approach before continuing with the task

		The environment will continue rendering but won't record actions or advance
		the simulation until recording is resumed with start_recording_instance().
		"""
		nonlocal running_recording_instance
		running_recording_instance = False

	# For reset
	teleop_interface2 = Se3Keyboard_BMM(
			pos_sensitivity=0.005 , rot_sensitivity=0.01
		)
	teleop_interface2.add_callback("R", reset_recording_instance)
	

	# Sparse goal & initial pos and rot 
	device = args_cli.device
	num_envs = args_cli.num_envs
	all_goals = []
	file_path = f'/root/IsaacLab/scripts/simvla/goals/{args_cli.task}.json'
	sensors = ["front", "back", "wrist_right", "wrist_left"]
	output_root = "/root/IsaacLab/simvla/segmentation_outputs"
	for sensor in sensors:
	    os.makedirs(os.path.join(output_root, f"segmentation_{sensor}"), exist_ok=True)
		os.makedirs(os.path.join(output_root, f"segmentation_label_{sensor}"), exist_ok=True)

	with open(file_path, 'r') as file:
		# Load the JSON data directly from the file object
		data = json.load(file)
		# 1. Init pos
		init_pos = random.choice(data["initial_pos_ranges"])
		env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (init_pos[0][1], init_pos[0][2])
		env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (init_pos[1][1], init_pos[1][2])
		
		# 2. Init rot
		init_rot = data["initial_rot_yaw_range"]
		env_cfg.events.robot_init_pos.params["pose_range"]["yaw"] = (init_rot[0][1], init_rot[0][2])

		# 3. Goal
		goals = data["goals"]
		for i in range(num_envs):
			goal = random.choice(goals)
			all_goals.append(
				[
					(x[0], torch.tensor(x[1], dtype=torch.float32, device=device))
					for x in goal
				]
			)
	env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
	# Reset environment
	env.reset()
	# Success demo
	current_recorded_demo_count = 0
	success_step_count = torch.zeros(num_envs, device=env.device, dtype=torch.long)
	
	label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

	instruction_display = InstructionDisplay(args_cli.task)
	if args_cli.task.lower() != "handtracking":
		window = EmptyWindow(env, "Instruction")
		with window.ui_window_elements["main_vstack"]:
			demo_label = ui.Label(label_text)
			subtask_label = ui.Label("")
			instruction_display.set_labels(subtask_label, demo_label)

	subtasks = {}

	
	# define goal types with integer ids for tensor operations
	TASK_IDS = {"N": 0, "A_l": 1, "A_r": 2, "G_l": 3, "G_r": 4, "N_s": 5}
	N_ID, AL_ID, AR_ID, GL_ID, GR_ID, NS_ID = TASK_IDS["N"], TASK_IDS["A_l"], TASK_IDS["A_r"], TASK_IDS["G_l"], TASK_IDS["G_r"], TASK_IDS["N_s"]
	
	# --- Find the longest goal sequence to determine padding ---
	max_sequence_length = max(len(goals) for goals in all_goals)
	max_payload_size = 7

	all_task_ids_list = []
	all_payloads_list = []

	# Special ID for padded steps, so we can ignore them
	PADDING_TASK_ID = -1

	for env_idx in range(num_envs):
		goals = all_goals[env_idx]
		task_ids_list_env = []
		payloads_list_env = []

		for task, data in goals:
			task_ids_list_env.append(TASK_IDS[task])

			payload = torch.zeros(max_payload_size, device=device)
			if task in ["N", "A_l", "A_r", "N_s"]:
				payload[:data.numel()] = data
			elif task in ["G_l", "G_r"]:
				payload[0] = 1.0 if data.item() else -1.0
			payloads_list_env.append(payload)

		# --- Pad the current environment's script to the max length ---
		num_goals = len(goals)
		padding_needed = max_sequence_length - num_goals

		task_ids_list_env.extend([PADDING_TASK_ID] * padding_needed)
		payloads_list_env.extend([torch.zeros(max_payload_size, device=device)] * padding_needed)
		
		all_task_ids_list.append(torch.tensor(task_ids_list_env, device=device, dtype=torch.long))
		all_payloads_list.append(torch.stack(payloads_list_env))

	task_ids_tensor = torch.stack(all_task_ids_list)
	# Shape: (num_envs, max_sequence_length, max_payload_size)
	payloads_tensor = torch.stack(all_payloads_list)	# The final tensors that define the "script" for all environments
	
	# --- Per-environment state variables ---
	env_goal_indices = torch.zeros(num_envs, device=device, dtype=torch.long)
	env_cmd_plans = [None] * num_envs
	env_cmd_indices = torch.zeros(num_envs, device=device, dtype=torch.long)
	env_r_gripper_dist_prev = torch.full((num_envs,), 10.0, device=device)

	print("Setup for parallel execution complete.")
	# Hyperparameter (Navigation)
	goal_reached_distance = 0.02
	goal_reached_yaw = 0.1
	default_speed = 10.0
	min_speed = 3.0
	slowdown_radius = 0.3
	# Hyperparameter (cuRobo)
	num_targets = 0
	n_obstacle_cuboids = 30
	n_obstacle_mesh = 775
	target_pose = None
	tensor_args = TensorDeviceType(device="cuda:0")
	rotation_threshold = 10
	position_threshold = 10
	trajopt_dt = None
	optimize_dt = True
	trajopt_tsteps = 32
	trim_steps = None
	max_attempts = 2
	interpolation_dt = 0.03
	enable_finetune_trajopt = True
	cmd_plan = None
	robot_cfg_path = get_robot_configs_path()
	# Gripper
	gripper_command_L = False
	gripper_command_R = False
	r_dis_pre = 10

	if args_cli.robot == "anubis":
		right_arm = "anubis_right_arm.yml"
		left_arm = "anubis_left_arm.yml"
		r_robot_cfg = load_yaml(join_path(robot_cfg_path, right_arm))["robot_cfg"]
		l_robot_cfg = load_yaml(join_path(robot_cfg_path, left_arm))["robot_cfg"]
		r_j_names = r_robot_cfg["kinematics"]["cspace"]["joint_names"]
		l_j_names = l_robot_cfg['kinematics']['cspace']['joint_names']
		r_j_index = env.scene.articulations['robot'].find_joints(r_j_names)[0]
		l_j_index = env.scene.articulations['robot'].find_joints(l_j_names)[0]
		# First, get the world configuration from the USD stage
		usd_helper = UsdHelper()
		usd_helper.load_stage(env.sim.stage)
		root_pos = env.scene.articulations['robot'].data.root_pos_w
		root_quat = env.scene.articulations['robot'].data.root_quat_w
		r_T_w = pose_to_gf_matrix_tensor(root_pos[0,:], root_quat[0,:])
		
		world_cfg = usd_helper.get_obstacles_from_stage(
			only_paths= ["env_0"], r_T_w=r_T_w,
		)
		print("Initializing Motion Generator for Right Arm...")
		motion_gen_config_r = MotionGenConfig.load_from_robot_config(
			r_robot_cfg,
			world_cfg,
			tensor_args,
			rotation_threshold = rotation_threshold,
			position_threshold = position_threshold,
			collision_checker_type = CollisionCheckerType.MESH,
			num_trajopt_seeds = 12,
			num_graph_seeds = 12,
			interpolation_dt = interpolation_dt,
			collision_cache = {"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
			optimize_dt = optimize_dt,
			trajopt_dt = trajopt_dt,
			trajopt_tsteps = trajopt_tsteps,
			trim_steps = trim_steps,
		)
		motion_gen_r = MotionGen(motion_gen_config_r)
		motion_gen_r.warmup(enable_graph=True, warmup_js_trajopt=False)
		
		print("Initializing Motion Generator for Left Arm...")
		motion_gen_config_l = MotionGenConfig.load_from_robot_config(
			l_robot_cfg,
			world_cfg,
			tensor_args,
			rotation_threshold = rotation_threshold,
			position_threshold = position_threshold,
			collision_checker_type = CollisionCheckerType.MESH,
			num_trajopt_seeds = 12,
			num_graph_seeds = 12,
			interpolation_dt = interpolation_dt,
			collision_cache = {"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
			optimize_dt = optimize_dt,
			trajopt_tsteps = trajopt_tsteps,
			trim_steps = trim_steps,
		)
		motion_gen_l = MotionGen(motion_gen_config_l)
		motion_gen_l.warmup(enable_graph=True, warmup_js_trajopt=False)

		# --- Create a reusable plan configuration ---
		plan_config = MotionGenPlanConfig(
			enable_graph=False,
			enable_graph_attempt=4,
			max_attempts=max_attempts,
			enable_finetune_trajopt=enable_finetune_trajopt,
			time_dilation_factor=0.5,
		)

		print("cuRobo Motion Generators are ready.")
	new_gripper_commands_R = torch.zeros(num_envs, dtype=torch.bool, device=device)
	new_gripper_commands_L = torch.zeros(num_envs, dtype=torch.bool, device=device)

	reset_r = torch.zeros(num_envs, 7, dtype= torch.float32, device=device)

	with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
		while simulation_app.is_running():
			if not running_recording_instance:
				continue	
			
			for sensor in sensors:
				seg_tensor = env.scene.sensors.get(sensor).data.output["instance_id_segmentation_fast"]

				# keep seg_tensor on GPU during simulation
				# only clone â CPU when saving
				seg_to_save = seg_tensor.to(dtype=torch.uint8).contiguous().clone().cpu()

				imageio.imwrite(
					os.path.join(output_root, f"segmentation_{sensor}", f"{timestep:06d}.png"),
					seg_to_save
				)

				seg_labels = env.scene.sensors.get(sensor).data.info.get("instance_id_segmentation_fast")["idToLabels"]

				# JSON always has to be CPU/python dict
				with open(
					os.path.join(output_root, f"segmentation_label_{sensor}", f"{timestep:06d}.json"),
					"w"
				) as f:
					json.dump(seg_labels, f, indent=2)


			# --- Initialize action tensors for all environments ---
			pose_L = torch.zeros((num_envs, 6), device=device)
			pose_R = torch.zeros((num_envs, 6), device=device)
			delta_pose_base = torch.zeros((num_envs, 3), device=device)
			# --- Create a mask for environments that are still active (not finished all goals) ---
			active_envs_mask = torch.logical_not(env_goal_indices >= max_sequence_length)
			if not torch.any(active_envs_mask):
				print("All environments have completed all goals.")
				break
			# Filter global indices to get only active environments
			finished_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
			env_indices = torch.arange(num_envs, device=device)
			active_env_indices = env_indices[active_envs_mask]
			current_task_ids = task_ids_tensor[active_env_indices, env_goal_indices[active_envs_mask]]
			# --- Create boolean masks for each task type on the ACTIVE environments ---
			nav_mask = current_task_ids == N_ID
			arm_l_mask = current_task_ids == AL_ID
			arm_r_mask = current_task_ids == AR_ID
			grip_l_mask = current_task_ids == GL_ID
			grip_r_mask = current_task_ids == GR_ID
			nav_s_mask = current_task_ids == NS_ID
			# =================== Mobile Base Motion ("N") ===================
			if torch.any(nav_mask):
				# Get the global indices of environments that need a nav plan
				nav_indices_global = active_env_indices[nav_mask]

				# Get robot's absolute WORLD position and orientation
				root_pos_w = env.scene.articulations['robot'].data.root_link_pos_w[nav_indices_global, :2]
				w, x, y, z = env.scene.articulations['robot'].data.root_link_quat_w[nav_indices_global].unbind(-1)
				current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

				# Get the origins for ONLY the navigating environments
				nav_env_origins = env.scene.env_origins[nav_indices_global]

				# Convert the robot's world position into its position relative to its own environment
				base_pos_e = root_pos_w - nav_env_origins[:, :2]

				# Get the environment-relative goals
				nav_goal_step_indices = env_goal_indices[nav_indices_global]
				nav_goals = payloads_tensor[nav_indices_global, nav_goal_step_indices]
				goal_pos_e, goal_yaw = nav_goals[:, :2], nav_goals[:, 2]

				# Now the subtraction is correct because both positions are in the same environment frame
				delta_pos = goal_pos_e - base_pos_e
				distance = torch.linalg.norm(delta_pos, dim=1)
			
				moving_mask = distance > goal_reached_distance
				rotating_mask = ~moving_mask
				if torch.any(moving_mask):
					m_delta_pos = delta_pos[moving_mask]
					m_current_yaw = current_yaw[moving_mask]
					m_distance = distance[moving_mask]
					
					cos_yaw, sin_yaw = torch.cos(m_current_yaw), torch.sin(m_current_yaw)
					vx_local = m_delta_pos[:, 0] * cos_yaw + m_delta_pos[:, 1] * sin_yaw
					vy_local = -m_delta_pos[:, 0] * sin_yaw + m_delta_pos[:, 1] * cos_yaw
					
					speed_scale = torch.where(m_distance > slowdown_radius, default_speed, min_speed + (default_speed - min_speed) * (m_distance / slowdown_radius))
					
					nav_actions = torch.zeros(torch.sum(moving_mask), 3, device=device)
					nav_actions[:, 0] = (-1) * speed_scale * vx_local
					nav_actions[:, 1] = (-1) * speed_scale * vy_local
					
					delta_pose_base[nav_indices_global[moving_mask]] = nav_actions

				if torch.any(rotating_mask):
					r_current_yaw, r_goal_yaw = current_yaw[rotating_mask], goal_yaw[rotating_mask]
					delta_yaw_g = (r_goal_yaw - r_current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
					angular_speed = torch.sign(delta_yaw_g) * 10.0
					
					rot_actions = torch.zeros(torch.sum(rotating_mask), 3, device=device)
					rot_actions[:, 2] = angular_speed
					
					delta_pose_base[nav_indices_global[rotating_mask]] = rot_actions

					is_aligned = torch.abs(delta_yaw_g) < goal_reached_yaw
					if torch.any(is_aligned):
						delta_pose_base[nav_indices_global[rotating_mask][is_aligned]] = 0.0
						finished_mask[nav_indices_global[rotating_mask][is_aligned]] = True
			
			# =================== Mobile Base straight Motion ("N_s") ===================
			if torch.any(nav_s_mask):
				# Get the global indices of env that need a nav_s plan
				nav_s_indices_global = active_env_indices[nav_s_mask]

				# Get robot's absolute World position and orientation
				root_pos_w = env.scene.articulations['robot'].data.root_link_pos_w[nav_s_indices_global, :2]
				w, x, y, z = env.scene.articulations['robot'].data.root_link_quat_w[nav_s_indices_global].unbind(-1)
				current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

				# Get the origins for ONLY the navigating environments
				nav_env_origins = env.scene.env_origins[nav_s_indices_global]

				# Convert the robot's world position into its position relative to its own environment
				base_pos_e = root_pos_w - nav_env_origins[:, :2]

				# Get the environment-relative goals
				nav_goal_step_indices = env_goal_indices[nav_s_indices_global]
				nav_goals = payloads_tensor[nav_s_indices_global, nav_goal_step_indices]
				goal_pos_e, goal_yaw = nav_goals[:, :2], nav_goals[:, 2]
				
				# Get the distance
				delta_pos = goal_pos_e - base_pos_e
				distance = torch.linalg.norm(delta_pos, dim=1)
				
				moving_mask = distance > goal_reached_distance
				rotating_mask = ~moving_mask

				if torch.any(moving_mask):
					m_delta_pos = delta_pos[moving_mask]
					m_current_yaw = current_yaw[moving_mask]
					delta_yaw = torch.atan2(m_delta_pos[:,1], m_delta_pos[:,0])
					m_distance = distance[moving_mask]	
					pre_rotating_mask = (torch.abs(delta_yaw - m_current_yaw) > goal_reached_yaw) 
					post_rotating_mask = ~pre_rotating_mask

					if torch.any(pre_rotating_mask):
						rows_to_update = nav_s_indices_global[moving_mask][pre_rotating_mask]
						delta_pose_base[rows_to_update, 2] = torch.sign(delta_yaw - m_current_yaw) * 10
					if torch.any(post_rotating_mask):
						cos_yaw, sin_yaw = torch.cos(m_current_yaw), torch.sin(m_current_yaw)
						vx_local = m_delta_pos[:, 0] * cos_yaw + m_delta_pos[:, 1] * sin_yaw
						vy_local = -m_delta_pos[:, 0] * sin_yaw + m_delta_pos[:, 1] * cos_yaw
						
						speed_scale = torch.where(m_distance > slowdown_radius, default_speed, min_speed + (default_speed - min_speed) * (m_distance / slowdown_radius))
						
						nav_actions = torch.zeros(torch.sum(post_rotating_mask), 3, device=device)
						nav_actions[:, 0] = (-1) * speed_scale * vx_local
						nav_actions[:, 1] = (-1) * speed_scale * vy_local
						
						delta_pose_base[nav_s_indices_global[moving_mask][post_rotating_mask]] = nav_actions
				
				if torch.any(rotating_mask):
					r_current_yaw, r_goal_yaw = current_yaw[rotating_mask], goal_yaw[rotating_mask]
					delta_yaw_g = (r_goal_yaw - r_current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
					angular_speed = torch.sign(delta_yaw_g) * 10.0
					
					rot_actions = torch.zeros(torch.sum(rotating_mask), 3, device=device)
					rot_actions[:, 2] = angular_speed
					
					delta_pose_base[nav_s_indices_global[rotating_mask]] = rot_actions

					is_aligned = torch.abs(delta_yaw_g) < goal_reached_yaw
					if torch.any(is_aligned):
						delta_pose_base[nav_s_indices_global[rotating_mask][is_aligned]] = 0.0
						finished_mask[nav_s_indices_global[rotating_mask][is_aligned]] = True
			# =================== Right Arm Motion ("A_r") ===================
			if torch.any(arm_r_mask):
				arm_r_indices = active_env_indices[arm_r_mask]
				needs_plan_indices = [i.item() for i in arm_r_indices if env_cmd_plans[i.item()] is None]

				if needs_plan_indices: 
					goal_step_indices = env_goal_indices[needs_plan_indices]
					arm_r_goals = payloads_tensor[needs_plan_indices, goal_step_indices]
					reset_mask = (arm_r_goals[:, 0] == 999)
					place_mask = (arm_r_goals[:, 3] == 999)
					not_reset_and_not_place = ~(reset_mask | place_mask)

					# 1. Grasp
					reset_r[not_reset_and_not_place, :3] = env.scene["robot"].data.body_pos_w[not_reset_and_not_place, 78]
					reset_r[not_reset_and_not_place, 3:] = env.scene["robot"].data.body_quat_w[not_reset_and_not_place, 78]
					# 2. Reset
					arm_r_goals[reset_mask] = reset_r[reset_mask]				
					# 3. Place
					arm_r_goals[place_mask][3:] = reset_r[place_mask]

					for i, env_idx in enumerate(needs_plan_indices):
						# Get current joint state data for the specific environment
						sim_data = env.scene.articulations['robot'].data
						joint_pos = sim_data.joint_pos[env_idx, r_j_index].tolist()
						joint_vel = sim_data.joint_vel[env_idx, r_j_index].tolist()

						motion_gen_r.robot_cfg.kinematics.cspace.retract_config = joint_pos
						
						# Convert goal to world frame and then to base frame
						goal_env = arm_r_goals[i]
						env_origin_pos = env.scene.env_origins[env_idx]
						goal_pos_w = goal_env[:3] + env_origin_pos
						goal_quat_w = goal_env[3:7]
						goal_ee_pose_b, goal_ee_quat_b = world2base(env, goal_pos_w, goal_quat_w, env_idx)
#						target = cuboid.VisualCuboid(
#							f"/World/envs/env_{env_idx}/target_r",
#							position = goal_pos_w.tolist(),
#							orientation = goal_quat_w.tolist(),
#							color=np.array([1.0, 1.0, 1.0]),
#							size=0.05,
#						)
						cu_js = JointState(
							position=tensor_args.to_device(joint_pos),
							velocity=tensor_args.to_device(joint_vel),
							acceleration=tensor_args.to_device(joint_vel) * 0.0,
							jerk=tensor_args.to_device(joint_vel) * 0.0,
							joint_names=r_j_names,
						)
						cu_js = cu_js.get_ordered_joint_state(motion_gen_r.kinematics.joint_names)
						ik_goal = Pose(position=goal_ee_pose_b, quaternion=goal_ee_quat_b)
						usd_helper.load_stage(env.sim.stage)
						root_pos = env.scene.articulations['robot'].data.root_pos_w
						root_quat = env.scene.articulations['robot'].data.root_quat_w
						r_T_w = pose_to_gf_matrix_tensor(root_pos[env_idx,:], root_quat[env_idx,:])

						obstacles = usd_helper.get_obstacles_from_stage(
							only_paths=[f"env_{env_idx}"], 
							r_T_w=r_T_w,
							ignore_substring=[
								"/World/envs/env_{env_idx}/Floor",
								"/World/envs/env_{env_idx}/Robot",
#	"/World/envs/env_{env_idx}/obj0",
#								"/World/envs/env_{env_idx}/mug",
#								"/World/envs/env_{env_idx}/plate",
#	"/World/envs/env_{env_idx}/bowl",
							]
						).get_collision_check_world()
						motion_gen_r.update_world(world_cfg)	
						result = motion_gen_r.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
						if result.success.item():
							cmd_plan = result.get_interpolated_plan()
							cmd_plan = motion_gen_r.get_full_js(cmd_plan)
							# The joint names in the plan can differ from the robot's joint names
							plan_j_names = cmd_plan.joint_names
							# Filter r_j_names to only include joints present in the plan
							common_j_names = [name for name in r_j_names if name in plan_j_names]
							env_cmd_plans[env_idx] = cmd_plan.get_ordered_joint_state(common_j_names)
							env_cmd_indices[env_idx] = 0
						else:
							finished_mask[env_idx] = True
							print(f"Env {env_idx} failed to find a plan for A_r.")

				is_executing_mask = torch.tensor([env_cmd_plans[i.item()] is not None for i in arm_r_indices], device=device)
				if torch.any(is_executing_mask):
					executing_indices = arm_r_indices[is_executing_mask]
					
					# Get the plans and indices for the executing environments
					plans_to_exec = [env_cmd_plans[i.item()] for i in executing_indices]
					indices_to_exec = env_cmd_indices[executing_indices]

					# Replace the entire for loop with one function call!
					delta_poses = compute_eef_deltas_batched(
						motion_gen=motion_gen_r,
						plans=plans_to_exec,
						cmd_indices=indices_to_exec,
						joint_names=r_j_names,
						eef_link_name="ee_link1"  # <-- IMPORTANT: Change to your actual link name
					)
					
					# Update the main pose_R tensor
					pose_R[executing_indices] = delta_poses
					
					# Increment command indices and check for completion
					env_cmd_indices[executing_indices] += 1
					plan_lengths = torch.tensor([len(p) for p in plans_to_exec], device=device)
					finished_exec_mask = (env_cmd_indices[executing_indices] >= plan_lengths)
					finished_mask[executing_indices[finished_exec_mask]] = True

			# =================== Left Arm Motion ("A_l") ===================
			if torch.any(arm_l_mask):
				arm_l_indices = active_env_indices[arm_l_mask]
				needs_plan_indices = [i.item() for i in arm_l_indices if env_cmd_plans[i.item()] is None]
				
				if needs_plan_indices:
					goal_step_indices = env_goal_indices[needs_plan_indices]
					arm_l_goals = payloads_tensor[needs_plan_indices, goal_step_indices]
					for i, env_idx in enumerate(needs_plan_indices):
						goal_env = arm_l_goals[i]
						env_origin_pos = env.scene.env_origins[env_idx]
						goal_pos_w = goal_env[:3] + env_origin_pos
						goal_quat_w = goal_env[3:7]  # Orientation remains the same
						goal_ee_pose_b, goal_ee_quat_b = world2base(env, goal_pos_w, goal_quat_w, env_idx)
						joint_pos = env.scene.articulations['robot'].data.joint_pos[env_idx, l_j_index].tolist()
						cu_js = JointState(position=tensor_args.to_device(joint_pos), joint_names=l_j_names)
						cu_js = cu_js.get_ordered_joint_state(motion_gen_l.kinematics.joint_names)
						ik_goal = Pose(position=goal_ee_pose_b, quaternion=goal_ee_quat_b)
						result = motion_gen_l.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
						if result.success.item():
							cmd_plan = result.get_interpolated_plan()
							env_cmd_plans[env_idx] = cmd_plan.get_ordered_joint_state(l_j_names)
							env_cmd_indices[env_idx] = 0
						else: finished_mask[env_idx] = True
				is_executing_mask = torch.tensor([env_cmd_plans[i.item()] is not None for i in arm_l_indices], device=device)
				if torch.any(is_executing_mask):
					executing_indices = arm_l_indices[is_executing_mask]
					
					plans_to_exec = [env_cmd_plans[i.item()] for i in executing_indices]
					indices_to_exec = env_cmd_indices[executing_indices]
					
					delta_poses = compute_eef_deltas_batched(
						motion_gen=motion_gen_l, # You might need a separate motion_gen for the left arm
						plans=plans_to_exec,
						cmd_indices=indices_to_exec,
						joint_names=l_j_names,
						eef_link_name="ee_link2"  # <-- IMPORTANT: Change to your actual link name
					)
					
					pose_L[executing_indices] = delta_poses

					env_cmd_indices[executing_indices] += 1
					plan_lengths = torch.tensor([len(p) for p in plans_to_exec], device=device)
					finished_exec_mask = (env_cmd_indices[executing_indices] >= plan_lengths)
					finished_mask[executing_indices[finished_exec_mask]] = True

			# =================== Gripper Motion ("G_r" & "G_l") ===================
			if torch.any(grip_r_mask):
				grip_r_env_indices = active_env_indices[grip_r_mask]
				grip_r_goal_step_indices = env_goal_indices[grip_r_env_indices]
				grip_r_goals = payloads_tensor[grip_r_env_indices, grip_r_goal_step_indices]
				
				# Latch: only update command if not already finished.
				# The command should stay True for the current goal until the next goal starts.
				current_commands = (grip_r_goals[:, 0] > 0)
				new_gripper_commands_R[grip_r_env_indices] = current_commands

				r_gripper_pos = env.scene._articulations['robot'].data.body_pos_w[grip_r_env_indices][:, [79, 80]]
				r_dist_curr = torch.linalg.norm(r_gripper_pos[:, 0] - r_gripper_pos[:, 1], dim=1)

				is_stopped = (env_r_gripper_dist_prev[grip_r_env_indices] - r_dist_curr).abs() < 0.001
				finished_mask[grip_r_env_indices[is_stopped]] = True

				env_r_gripper_dist_prev[grip_r_env_indices] = r_dist_curr
			
			if torch.any(grip_l_mask):
				grip_l_env_indices = active_env_indices[grip_l_mask]
				grip_l_goal_step_indices = env_goal_indices[grip_l_env_indices]
				grip_l_goals = payloads_tensor[grip_l_env_indices, grip_l_goal_step_indices]

				# Latch: only update command if not already finished.
				# The command should stay True for the current goal until the next goal starts.
				current_commands = (grip_l_goals[:, 0] > 0)
				new_gripper_commands_L[grip_l_env_indices] = current_commands

				l_gripper_pos = env.scene._articulations['robot'].data.body_pos_w[grip_l_env_indices][:, [79, 80]]
				l_dist_curr = torch.linalg.norm(l_gripper_pos[:, 0] - l_gripper_pos[:, 1], dim=1)

				is_stopped = (env_l_gripper_dist_prev[grip_l_env_indices] - l_dist_curr).abs() < 0.001
				finished_mask[grip_l_env_indices[is_stopped]] = True

				env_l_gripper_dist_prev[grip_l_env_indices] = l_dist_curr

			# =================== Goal State Progression ===================
			if torch.any(finished_mask):
				finished_indices_in_active = torch.where(finished_mask[active_env_indices])[0]
				
				# Filter out finished environments from the lists of plans and indices
				for i in finished_indices_in_active.tolist():
					global_idx = active_env_indices[i].item()
					env_cmd_plans[global_idx] = None
				
				env_cmd_indices[finished_mask] = 0
				env_r_gripper_dist_prev[finished_mask] = 10.0
				
				# Increment goal index for finished environments
				env_goal_indices[finished_mask] += 1
				
				# Check for full demo completion
				is_demo_completed = env_goal_indices >= max_sequence_length
				if torch.any(is_demo_completed):
					completed_demo_indices = torch.where(is_demo_completed)[0].tolist()
					# Check for success and export
					is_successful = success_term.func(env, **success_term.params)[completed_demo_indices]
					successful_demos = [completed_demo_indices[i] for i, success in enumerate(is_successful) if success]
					
					if successful_demos:
						env.recorder_manager.record_pre_reset(successful_demos, force_export_or_skip=False)
						env.recorder_manager.set_success_to_episodes(
							successful_demos, torch.tensor([[True]] * len(successful_demos), dtype=torch.bool, device=device)
						)
						env.recorder_manager.export_episodes(successful_demos)
					
					# Reset the environments that have completed their demo, regardless of success
#env.sim.reset(indices=completed_demo_indices)
#					env.recorder_manager.reset(indices=completed_demo_indices)
#					env.reset(indices=completed_demo_indices)

			# --- Final action processing and simulation step ---
			actions = pre_process_actions(
				pose_L, new_gripper_commands_L,
				pose_R, new_gripper_commands_R,
				delta_pose_base
			)
			obv = env.step(actions)

			# UI and success tracking logic
			if subtasks is not None:
				if subtasks == {}:
					subtasks = obv[0].get("subtask_terms")
				elif subtasks:
					show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
		
			if success_term is not None:
				# Get a boolean tensor for success across all environments
				success_per_env = success_term.func(env, **success_term.params).squeeze()
				# Increment success counter for successful environments
				success_step_count[success_per_env] += 1
				# Reset counter for unsuccessful environments
				success_step_count[~success_per_env] = 0

				# Check for environments that have been successful for enough steps
				finished_success_mask = success_step_count >= args_cli.num_success_steps
				finished_success_indices = torch.where(finished_success_mask)[0].tolist()

				if finished_success_indices:
					# Check if the demo has already been completed in this step
					newly_completed_demos = [i for i in finished_success_indices if env_goal_indices[i].item() < max_sequence_length]

					if newly_completed_demos:
						# Mark the goals for these environments as finished
						env_goal_indices[newly_completed_demos] = max_sequence_length

						# Export the successful demos
						env.recorder_manager.record_pre_reset(newly_completed_demos, force_export_or_skip=False)
						env.recorder_manager.set_success_to_episodes(
							newly_completed_demos, torch.tensor([[True]] * len(newly_completed_demos), dtype=torch.bool, device=device)
						)
						env.recorder_manager.export_episodes(newly_completed_demos)
						# Reset the environments
#env.sim.reset(indices=newly_completed_demos)
#						env.recorder_manager.reset(indices=newly_completed_demos)
#						env.reset(indices=newly_completed_demos)
						success_step_count[newly_completed_demos] = 0

			# Update and display demo count
			if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
				current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
				label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
				print(label_text)
				instruction_display.show_demo(label_text)

			# Final check for exiting the loop
			if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
				print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
				break
			
			if rate_limiter:
				rate_limiter.sleep(env)

		# close the simulator
		env.close()

if __name__ == "__main__":
	# run the main function
	main()
	# close sim app
	simulation_app.close()
