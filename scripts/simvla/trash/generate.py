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
parser.add_argument("--task", type=str, default="Isaac-Kitchen-v01-01", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
import torch, queue, threading
from pathlib import Path
import gymnasium as gym
import ipdb
import gc
# Import Omniverse logger
import omni.log
import numpy as np
import json
# Import for reset
from isaaclab.devices import Se3Keyboard_BMM
# Import for record Demo
import imageio

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
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from isaacsim.core.api.objects import cuboid
import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R
from isaaclab.sensors import TiledCameraCfg, TiledCamera
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
def print_mem(tag=""):
	torch.cuda.synchronize()
	print(f"{tag} | alloc={torch.cuda.memory_allocated()/1024**2:.1f} MB, "
		  f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB")

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

def classify_yaw_rotation(quat_wxyz_tensor, tolerance_degrees=10):
	"""
	Classifies the yaw rotation of a quaternion tensor.

	Args:
		quat_wxyz_tensor (torch.Tensor): A tensor of shape [1, 4] or [4]
										 representing a quaternion in (w, x, y, z) format.
		tolerance_degrees (float): How many degrees to allow as a buffer.
	"""
	
	# 1. Convert tensor to NumPy array on CPU
	# .squeeze() removes the extra dimension (e.g., from [1, 4] to [4])
	q_wxyz = quat_wxyz_tensor.squeeze().cpu().numpy()
	
	# 2. Re-order from (w, x, y, z) to (x, y, z, w) for SciPy
	q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
	
	try:
		# 3. Create a Rotation object
		r = Rotation.from_quat(q_xyzw)
	except ValueError as e:
		print(f"Error: Invalid quaternion. {e}")
		return None
		
	# 4. Convert to Euler angles.
	# 'zyx' means the rotations are applied in order: Z (yaw), then Y (pitch), then X (roll).
	# We read the first angle (index 0), which is the yaw rotation.
	euler_angles = r.as_euler('zyx', degrees=True)
	yaw = euler_angles[0]
	
	# 5. Normalize the angle to be between 0 and 360
	yaw_normalized = yaw % 360
	
	# 6. Classify the rotation based on the yaw angle
	if abs(yaw_normalized - 0) < tolerance_degrees or abs(yaw_normalized - 360) < tolerance_degrees:
		return 0
	elif abs(yaw_normalized - 90) < tolerance_degrees:
		return 90
	elif abs(yaw_normalized - 180) < tolerance_degrees:
		return 180
	elif abs(yaw_normalized - 270) < tolerance_degrees:
		return 270
	else:
		# It's not close to any of the 90-degree intervals
		return f"Unclassified ({yaw_normalized:.2f} degrees)"


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
	arm_joint_names = [name for name in joint_names] # if "gripper" not in name]
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
	mat[:3, 3] = pos
	
	return mat
#	return Gf.Matrix4d(mat.tolist())

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
#env_cfg.terminations.time_out = None

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
	device = args_cli.device
	num_envs = args_cli.num_envs

# -------------------- Start of replacement: multiprocessing shared-memory writers --------------------
	import multiprocessing as mp
	from multiprocessing.shared_memory import SharedMemory
	import atexit
	import math

# Writer pool configuration
	NUM_SHM_BUFFERS = 32  # how many shared-memory buffers to rotate through (tune)
	NUM_WRITERS = max(2, min(8, (os.cpu_count() or 4) // 2))  # number of writer processes (tune)

	fps = args_cli.step_hz
	video_root = Path(f"simvla/videos/{args_cli.task}")
	video_root.mkdir(parents=True, exist_ok=True)
	camera_names = ["front", "back", "wrist_left", "wrist_right"]

# We'll lazily determine H,W from the sensor after env exists (right after env.reset())
# but to create shared memory we need H,W. We'll create the pools after first read below.

# Create queues for interprocess comm
	manager = mp.Manager()
	work_q = mp.JoinableQueue(maxsize=1024)     # tasks for writers: (shm_idx:int, env_id:int, cam_idx:int, frame_ts:int)
	free_q = mp.Queue()                         # available shared-memory buffer indices

# Container for shm names; writer procs will open by name
	shm_names = manager.list()  # will be filled after we know H,W
	frame_shape = manager.list([0, 0])  # placeholder for H,W (H,W stored as two ints)
# Keep created SharedMemory objects so main can write into them, then close/unlink on exit
	shm_objects = []
# Start writer processes only after first frame shape known. We'll lazily initialize below.
	writers_started = False
	writer_processes = []

# writer process function

	def writer_process_main(work_q, free_q, shm_names, frame_shape, video_root_str, camera_names_list, fps):
		import imageio, os, time
		from multiprocessing.shared_memory import SharedMemory
		video_root = Path(video_root_str)
		H, W, N = int(frame_shape[0]), int(frame_shape[1]), int(frame_shape[2])
		writers = {}  # (env_id, cam_idx) -> writer

		def get_writer(env_id, cam_idx):
			key = (env_id, cam_idx)
			if key not in writers:
				env_dir = video_root / f"env_{env_id}"
				env_dir.mkdir(exist_ok=True, parents=True)
				path = env_dir / f"{camera_names_list[cam_idx]}.mp4"
				writers[key] = imageio.get_writer(str(path), fps=fps)
			return writers[key]

		while True:
			task = work_q.get()
			if task is None:
				work_q.task_done()
				break
			shm_idx, env_id, cam_idx, ts = task
			shm = SharedMemory(name=shm_names[shm_idx])
			if env_id == -1:
				batch = np.ndarray((N, H, W, 3), dtype=np.uint8, buffer=shm.buf)
				# write each env frame into its per-env writer (or change as needed)
				for eid in range(N):
					frame = batch[eid].copy()
					w = get_writer(eid, cam_idx)
					w.append_data(frame)
			else:
				# legacy single-env write
				frame = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf).copy()
				w = get_writer(env_id, cam_idx)
				w.append_data(frame)
			free_q.put(shm_idx)
			shm.close()
			work_q.task_done()

	def create_shm_pool(num_buffers, H, W, num_env):
		total_bytes = num_env * H * W * 3
		names = []
		shms = []
		for i in range(num_buffers):
			shm = SharedMemory(create=True, size=total_bytes)
			names.append(shm.name)
			shms.append(shm)
			free_q.put(i)
		for n in names:
			shm_names.append(n)
		frame_shape[0] = H
		frame_shape[1] = W
		# store num_env as manager-list entry too
		if len(frame_shape) < 3:
			frame_shape.append(num_env)
		else:
			frame_shape[2] = num_env
		return shms

	def start_writers_if_needed(H, W, num_env):
		nonlocal writers_started, shm_objects, writer_processes
		if writers_started:
			return
		shm_objects = create_shm_pool(NUM_SHM_BUFFERS, H, W, num_env)
		writer_processes = []
		for i in range(NUM_WRITERS):
			p = mp.Process(target=writer_process_main,
						   args=(work_q, free_q, shm_names, frame_shape, str(video_root), camera_names, fps),
						   daemon=True)
			p.start()
			writer_processes.append(p)
		writers_started = True


# cleanup on exit
	def _cleanup_on_exit():
		# send stop sentinel to writer processes
		for _ in range(len(writer_processes)):
			try:
				work_q.put_nowait(None)
			except Exception:
				pass
		# wait for them
		for p in writer_processes:
			p.join(timeout=5)
		# close & unlink shared memory
		for shm in shm_objects:
			try:
				shm.close()
				shm.unlink()
			except Exception:
				pass

	atexit.register(_cleanup_on_exit)

# -------------------- End of replacement setup --------------------

	# Sparse goal & initial pos and rot 
	all_goals = []
	file_path = f'/root/IsaacLab/scripts/simvla/goals/{args_cli.task}.json'
	sensors = ["front"] #, "back", "wrist_right", "wrist_left"]
	output_root = "/root/IsaacLab/simvla/segmentation_outputs"
	for sensor in sensors:
		os.makedirs(os.path.join(output_root, f"segmentation_{sensor}"), exist_ok=True)
		os.makedirs(os.path.join(output_root, f"segmentation_label_{sensor}"), exist_ok=True)
		os.makedirs(os.path.join(output_root, f"rgb_{sensor}"), exist_ok=True)

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
		kitchen_type = data["kitchen_type"]
		if kitchen_type == "island":
			island_min = [data["island_bound"][0],data["island_bound"][2]]
			island_max = [data["island_bound"][1],data["island_bound"][3]]


	# To opimize env
	env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
	# Reset environment
	env.reset()
	# Success demo
	current_recorded_demo_count = 0
	success_step_count = torch.zeros(num_envs, device=env.device, dtype=torch.long)


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
				if len(data.size()) == 1:
					payload[:data.numel()] = data
				else:
					idx = torch.randint(data.size(0), (1,)) 
					rand_data = data[idx]  
					payload[:data.numel()] = rand_data
					print(rand_data)
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
	ik_goals = [None] * num_envs 
	print("Setup for parallel execution complete.")
	# Hyperparameter (Navigation)
	goal_reached_distance = 0.02
	goal_reached_yaw = 0.1
	default_speed = 20.0
	min_speed = 3.0
	slowdown_radius = 0.3
	# Hyperparameter (cuRobo)
	num_targets = 0
	n_obstacle_cuboids = 100
	n_obstacle_mesh = 8000
	target_pose = None
	tensor_args = TensorDeviceType(device="cuda:0")
	rotation_threshold = 10
	position_threshold = 10
	trajopt_dt = None
	optimize_dt = True
	trajopt_tsteps = 32
	trim_steps = None
	max_attempts = 20
	interpolation_dt = 0.03
	enable_finetune_trajopt = True
	cmd_plan = None
	robot_cfg_path = get_robot_configs_path()
	dummy = torch.tensor([1e10, 1e10, 0], device='cuda:0')
	# Gripper
	gripper_command_L = False
	gripper_command_R = False

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
		
		world_cfg = usd_helper.get_obstacles_from_stage_exaFLOPs(
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
			collision_activation_distance=1e-3,
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
	timestep = 0
	with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
		while simulation_app.is_running():
			if not running_recording_instance:
				continue
			# --- Initialize action tensors for all environments ---
#timestep+=1
#			print(timestep)
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
				if kitchen_type == "island":
					island_min_xy = torch.tensor(island_min, device=device) # Bottom-left corner
					island_max_xy = torch.tensor(island_max, device=device)   # Top-right corner
					island_center_pos_e = (island_min_xy + island_max_xy) / 2.0
					# 1. Check whether island is between the goal and current position.
					bb_min = torch.min(base_pos_e, goal_pos_e)
					bb_max = torch.max(base_pos_e, goal_pos_e)
					island_is_between = (
						(island_center_pos_e[0] > bb_min[:, 0]) & (island_center_pos_e[0] < bb_max[:, 0]) &
						(island_center_pos_e[1] > bb_min[:, 1]) & (island_center_pos_e[1] < bb_max[:, 1])
					)

					# Only apply island navigation logic to the relevant environments
					if torch.any(island_is_between):
						sub_goal_pos_e = goal_pos_e.clone()

						sub_mask = island_is_between
						start_pos = base_pos_e[sub_mask]
						end_pos = goal_pos_e[sub_mask]
						
						start_x, start_y = start_pos.unbind(dim=-1)
						goal_x, goal_y = end_pos.unbind(dim=-1)

						waypoint_hv = torch.stack([goal_x, start_y], dim=1)
						waypoint_vh = torch.stack([start_x, goal_y], dim=1)

						h_leg_collides = (
							(start_y > island_min_xy[1]) & (start_y < island_max_xy[1]) &
							(torch.min(start_x, goal_x) < island_max_xy[0]) &
							(torch.max(start_x, goal_x) > island_min_xy[0])
						)
						v_leg_collides = (
							(start_x > island_min_xy[0]) & (start_x < island_max_xy[0]) &
							(torch.min(start_y, goal_y) < island_max_xy[1]) &
							(torch.max(start_y, goal_y) > island_min_xy[1])
						)

						can_go_h = ~h_leg_collides
						can_go_v = ~v_leg_collides

						chosen_waypoint = end_pos.clone()
						mask_v_only = can_go_v & ~can_go_h
						if torch.any(mask_v_only):
							chosen_waypoint[mask_v_only] = waypoint_vh[mask_v_only]

						mask_h_only = can_go_h & ~can_go_v
						if torch.any(mask_h_only):
							chosen_waypoint[mask_h_only] = waypoint_hv[mask_h_only]

						mask_both = can_go_h & can_go_v
						if torch.any(mask_both):
							dist_h = torch.abs(goal_x[mask_both] - start_x[mask_both])
							dist_v = torch.abs(goal_y[mask_both] - start_y[mask_both])
							
							h_is_shorter = dist_h < dist_v
							# Create a temporary choice tensor for the 'mask_both' subset
							temp_choices = torch.where(h_is_shorter.unsqueeze(1), waypoint_hv[mask_both], waypoint_vh[mask_both])
							chosen_waypoint[mask_both] = temp_choices

						sub_goal_pos_e[sub_mask] = chosen_waypoint
						goal_pos_e = sub_goal_pos_e


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
					reset_env = torch.nonzero(reset_mask, as_tuple=True)[0].tolist()
					place_mask = (arm_r_goals[:, 3] == 999)
					place_env = torch.nonzero(place_mask, as_tuple=True)[0].tolist() 
					not_reset_and_not_place = ~(reset_mask | place_mask)
					grasp_env = torch.nonzero(not_reset_and_not_place, as_tuple=True)[0].tolist()
					# 1. Grasp, for reset
					if torch.any(not_reset_and_not_place):
						reset_r[grasp_env, :3] = env.scene["robot"].data.body_pos_w[grasp_env, 78]
						reset_r[grasp_env, 3:] = env.scene["robot"].data.body_quat_w[grasp_env, 78]
					# 2. Reset
					if torch.any(reset_mask):
						arm_r_goals[reset_env] = reset_r[reset_env]				
					# 3. Place
					if torch.any(place_mask):
						arm_r_goals[place_env, 3:] = env.scene["robot"].data.body_quat_w[place_env, 78]

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
						print("goal_pos_w: ", goal_pos_w)
						print("goal_quat_w: ", goal_quat_w)
						goal_ee_pose_b, goal_ee_quat_b = world2base(env, goal_pos_w, goal_quat_w, env_idx)
						cu_js = JointState(
							position=tensor_args.to_device(joint_pos),
							velocity=tensor_args.to_device(joint_vel),
							acceleration=tensor_args.to_device(joint_vel) * 0.0,
							jerk=tensor_args.to_device(joint_vel) * 0.0,
							joint_names=r_j_names,
						)
						cu_js = cu_js.get_ordered_joint_state(motion_gen_r.kinematics.joint_names)
						ik_goal = Pose(position=goal_ee_pose_b, quaternion=goal_ee_quat_b)
						ik_goals[env_idx] = ik_goal
						usd_helper.load_stage(env.sim.stage)
						root_pos = env.scene.articulations['robot'].data.root_pos_w
						root_quat = env.scene.articulations['robot'].data.root_quat_w
						r_T_w = pose_to_gf_matrix_tensor(root_pos[env_idx,:], root_quat[env_idx,:])
#						r_T_w= pose_to_gf_matrix_tensor(dummy, root_quat[env_idx,:])
						obstacles = usd_helper.get_obstacles_from_stage_exaFLOPs(
							only_paths=[f"env_0"], 
							r_T_w=r_T_w,
							ignore_substring=[
								"/World/envs/env_{env_idx}/Floor",
								"/World/envs/env_{env_idx}/Robot",
								"/curobo"
							]
						).get_collision_check_world()
# [obj.pose for obj in motion_gen_r.world_model.objects if "bottle" in obj.name]
# env.scene.rigid_objects["bottle0"].data.body_pos_w 
# env.scene.articulations['robot'].data.root_pos_w
# env.scene.articulations['robot'].data.body_pos_w[0,-6]
# env.scene.articulations['robot'].data.body_pos_w[0,-6]-env.scene.articulations['robot'].data.root_pos_w[0]
# [meshy.name for meshy in motion_gen_r.world_model.mesh if 'mug' in meshy.name ]
						motion_gen_r.clear_world_cache()
						motion_gen_r.update_world(obstacles)
						result = motion_gen_r.plan_single(cu_js.unsqueeze(0), ik_goals[env_idx], plan_config)
						if not result.success.item():
							r_T_w= pose_to_gf_matrix_tensor(dummy, root_quat[env_idx,:])
							obstacles = usd_helper.get_obstacles_from_stage_exaFLOPs(
								only_paths=[f"env_0"], 
								r_T_w=r_T_w,
								ignore_substring=[
									"/World/envs/env_{env_idx}/Floor",
									"/World/envs/env_{env_idx}/Robot",
									"/curobo"
								]
							).get_collision_check_world()
							motion_gen_r.clear_world_cache()
							motion_gen_r.update_world(obstacles)
							result = motion_gen_r.plan_single(cu_js.unsqueeze(0), ik_goals[env_idx], plan_config)

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

					# Check if the current EEF is far from the goal
					pos_threshold = 0.02
					rot_threshold_deg = 10
					max_replans =2

					for j, env_idx in enumerate(executing_indices[finished_exec_mask].tolist()):
						curr_eef_pos, curr_eef_quat = world2base(env, env.scene.articulations['robot'].data.body_pos_w[env_idx,-6], env.scene.articulations['robot'].data.body_quat_w[env_idx,-6],env_idx)
						pos_err = torch.norm(curr_eef_pos - ik_goals[env_idx].position)
						r_eef = R.from_quat(curr_eef_quat.cpu().numpy()[[1, 2, 3, 0]])
						r_goal = R.from_quat(ik_goals[env_idx].quaternion.squeeze(0).cpu().numpy()[[1, 2, 3, 0]])
						rot_err_deg = np.linalg.norm((r_goal.inv() * r_eef).as_rotvec()) * 180 / np.pi
						if pos_err > pos_threshold:
							print(f"[Replan] Env {env_idx}: EEF not at goal "
									f"(pos_err={pos_err:.3f}, rot_err={rot_err_deg:.1f}Â°). Replanning...")

							# get current joint state again
							sim_data = env.scene.articulations['robot'].data
							joint_pos = sim_data.joint_pos[env_idx, r_j_index].tolist()
							joint_vel = sim_data.joint_vel[env_idx, r_j_index].tolist()

							motion_gen_r.robot_cfg.kinematics.cspace.retract_config = joint_pos
							cu_js = JointState(
								position=tensor_args.to_device(joint_pos),
								velocity=tensor_args.to_device(joint_vel),
								acceleration=tensor_args.to_device(joint_vel) * 0.0,
								jerk=tensor_args.to_device(joint_vel) * 0.0,
								joint_names=r_j_names,
							)
							cu_js = cu_js.get_ordered_joint_state(motion_gen_r.kinematics.joint_names)
							# replan attempt loop
							success = False
							for _ in range(max_replans):
								r_T_w= pose_to_gf_matrix_tensor(dummy, root_quat[env_idx,:])
								obstacles = usd_helper.get_obstacles_from_stage_exaFLOPs(
										only_paths=[f"env_0"], 
										r_T_w=r_T_w
										).get_collision_check_world()
								motion_gen_r.clear_world_cache()
								motion_gen_r.update_world(obstacles)
								result = motion_gen_r.plan_single(cu_js.unsqueeze(0), ik_goals[env_idx], plan_config)
								if result.success.item():
									cmd_plan = result.get_interpolated_plan()
									cmd_plan = motion_gen_r.get_full_js(cmd_plan)
									env_cmd_plans[env_idx] = cmd_plan.get_ordered_joint_state(r_j_names)
									env_cmd_indices[env_idx] = 0
									success = True
									print(f"[Replan Success] Env {env_idx}: new plan found.")
									break

							if not success:
								print(f"[Replan Failed] Env {env_idx}: Could not reach goal after replans.")
								finished_mask[env_idx] = True
						else:
							print(f"[OK] Env {env_idx}: EEF reached goal within threshold.")
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
						ik_goals[env_idx] = ik_goal
						result = motion_gen_l.plan_single(cu_js.unsqueeze(0), ik_goals[env_idx], plan_config)
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
			# env step
			actions = pre_process_actions(
				pose_L, new_gripper_commands_L,
				pose_R, new_gripper_commands_R,
				delta_pose_base
			)
		
			# --- Replace per-env loop with single per-camera SHM copy (one task per camera) ---
			before_cam = time.time()


			for cam_idx, cam in enumerate(camera_names):
				frames_torch = env.scene.sensors[cam].data.output["rgb"]  # (N, H, W, 3)
				N, H, W, C = frames_torch.shape

				if not writers_started:
					start_writers_if_needed(H, W, num_env=N)

				frames_cpu = frames_torch.detach().to("cpu", non_blocking=True).byte().numpy()

				try:
					shm_idx = free_q.get(timeout=0.05)  # small block; adjust to behavior you want
				except Exception:
					print(f"â ï¸ No free shared-buffer â dropped camera frame: cam={cam}")
					continue

				shm_name = shm_names[shm_idx]
				shm = SharedMemory(name=shm_name)
				arr = np.ndarray((N, H, W, 3), dtype=np.uint8, buffer=shm.buf)
				arr[:] = frames_cpu  # bulk copy
				shm.close()
				work_q.put((shm_idx, -1, int(cam_idx), int(time.time_ns())))
				del frames_cpu, frames_torch


			torch.cuda.empty_cache()
			after_cam = time.time()
			print("after cam: ", after_cam - before_cam)


			obv = env.step(actions)
			after_env = time.time()
			print("env step: ", after_env-after_cam)
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
						print("ð§¹ Finalizing videos...")
# signal writers to stop and wait for completion
# block until work queue drains
						work_q.join()

# Tell writer processes to stop
						for _ in range(len(writer_processes)):
							work_q.put(None)

# Give them time to finish, then join
						for p in writer_processes:
							p.join(timeout=5)

# cleanup shared memory will be handled by atexit registered _cleanup_on_exit()
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
