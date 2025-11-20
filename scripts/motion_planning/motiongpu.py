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
parser.add_argument("--task", type=str, default="Cabinet-anubis-ik-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--robot", type=str, default="anubis", help="Which robot to use in the task.")
parser.add_argument("--record", type=bool, default=False, help="Whether to record the simulation.")
parser.add_argument(
	"--dataset_file", type=str, default="./datasets/anubis/kitchen_ik_cpu.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
	"--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
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
# Omniverse logger
import omni.log
import omni.ui as ui
from isaacsim.core.utils.types import ArticulationAction
# For reset
from isaaclab.devices import Se3Keyboard_BMM
# Record Demo
import os
import time
import contextlib
import isaaclab_mimic.envs	# noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
# TODO: This is only for cabinet
from isaaclab_tasks.manager_based.mobile_manipulation.cabinet import mdp
# Camera 
import numpy as np
import ipdb
# If I need to do pointclouds
# import omni.replicator.core as rep
# from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
# cuRobo
from curobo.util_file import (
	get_assets_path,
	get_filename,
	get_path_of_dir,
	get_robot_configs_path,
	get_world_configs_path,
	join_path,
	load_yaml,
)
from curobo.util.usd_helper import UsdHelper
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import (
	MotionGen,
	MotionGenConfig,
	MotionGenPlanConfig,
	PoseCostMetric,
)
from curobo.geom.types import WorldConfig, Mesh, Sphere
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from isaacsim.core.api.objects import cuboid
import isaaclab.utils.math as math_utils
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

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

def world2base(env, ee_pos_w, ee_quat_w) -> tuple[torch.Tensor, torch.Tensor]:
	# convert world frame to base frame
	robot_data = env.scene.articulations["robot"].data
	root_pos_w = robot_data.root_pos_w
	root_quat_w = robot_data.root_quat_w
	# compute the pose of the body in the root frame
	ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0))
	return ee_pose_b, ee_quat_b

def get_eef_pose(cmd_state, l_or_r, j_names, idx_list, robot, data):
	art_action = ArticulationAction(
		cmd_state.position.cpu().numpy(),
		cmd_state.velocity.cpu().numpy(),
		joint_indices=idx_list
	)

	# Forward kinematics
	q = np.zeros(robot.nq)

	if l_or_r == "A_r":
		ee_link = "ee_link1"	
	elif l_or_r == "A_l":
		ee_link = "ee_link2"
	
	j_names_filtered = [item for item in j_names if item not in ['gripper2R_joint', 'gripper2_joint',"gripper1R_joint", "gripper1_joint"]]
	for i, name in enumerate(j_names_filtered):
		joint_id = robot.getJointId(name)
		idx = robot.joints[joint_id].idx_q
		q[idx] = art_action.joint_positions[i]

	pin.framesForwardKinematics(robot, data, q)
	pin.updateFramePlacements(robot, data)

	eef_frame_id = robot.getFrameId(ee_link)
	eef_pose = data.oMf[eef_frame_id]

	pos = eef_pose.translation
	rot_matrix = eef_pose.rotation
	quat = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

	return torch.tensor(pos), torch.tensor(quat)

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
    axis = torch.stack((x, y, z), dim=-1) / sin_half.unsqueeze(-1)  # (B, 3)

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

#   # 1. Convert xyzw to wxyz (PyTorch format for multiplication)
#   quat_t_wxyz = torch.cat([quaternion_t[..., 3:4], quaternion_t[..., 0:3]], dim=-1)
#   quat_tp1_wxyz = torch.cat([quaternion_tp1[..., 3:4], quaternion_tp1[..., 0:3]], dim=-1)
#   quat_t_inv = quat_t_wxyz * torch.tensor([1, -1, -1, -1], device=quat_t_wxyz.device, dtype=quat_t_wxyz.dtype)
#   w1, x1, y1, z1 = quat_t_inv[..., 0:1], quat_t_inv[..., 1:2], quat_t_inv[..., 2:3], quat_t_inv[..., 3:4]
#   w2, x2, y2, z2 = quat_tp1_wxyz[..., 0:1], quat_tp1_wxyz[..., 1:2], quat_tp1_wxyz[..., 2:3], quat_tp1_wxyz[..., 3:4]
#
#   delta_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#   delta_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#   delta_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#   delta_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#
#   delta_quat_wxyz = torch.cat([delta_w, delta_x, delta_y, delta_z], dim=-1)
#
#   angle = 2 * torch.acos(torch.clamp(delta_quat_wxyz[..., 0], -1.0, 1.0))
#   # The axis is the vector part of the quaternion normalized
#   axis = delta_quat_wxyz[..., 1:] / torch.sin(angle / 2).unsqueeze(-1)
#   delta_rotvec = angle.unsqueeze(-1) * axis


    r_t = R.from_quat(quaternion_t.detach().cpu().numpy()[:, [1, 2, 3, 0]])
    r_tp1 = R.from_quat(quaternion_tp1.detach().cpu().numpy()[:, [1, 2, 3, 0]])
    delta_r = r_tp1 * r_t.inv()
    delta_rotvec = torch.tensor(delta_r.as_rotvec(), dtype=torch.float32)
    device = delta_pos.device
    delta_rotvec = delta_rotvec.to(device)

#   q_t_inv = q_inverse(quaternion_t)
#   delta_quat = q_mul(q_t_inv, quaternion_tp1)
#   delta_rotvec = quat_to_rotvec_torch(delta_quat)
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
		wheel_radius=0.103, l=0.05
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


def main():
	rate_limiter = RateLimiter(args_cli.step_hz)

	# get directory path and file name (without extension) from cli arguments
	output_dir = os.path.dirname(args_cli.dataset_file)
	output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
	# create directory if it does not exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# parse configuration
	env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
	env_cfg.env_name = args_cli.task
	env_cfg.sim.use_gpu_pipeline = True
	env_cfg.sim.enable_gpu_dynamics = True	# PhysX GPU
	# extract success checking function to invoke in the main loop
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

	teleop_interface2 = Se3Keyboard_BMM(
			pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
		)
	teleop_interface2.add_callback("R", reset_recording_instance)
	
	# create environment
	env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
	# reset environment
	env.reset()
	current_recorded_demo_count = 0
	success_step_count = 0

	label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

	instruction_display = InstructionDisplay(args_cli.task)
	if args_cli.task.lower() != "handtracking":
		window = EmptyWindow(env, "Instruction")
		with window.ui_window_elements["main_vstack"]:
			demo_label = ui.Label(label_text)
			subtask_label = ui.Label("")
			instruction_display.set_labels(subtask_label, demo_label)

	subtasks = {}
	
	# Whole body motion planning
	# Goal states
	goals = []
	current_goal = None
	goal_index = -1	
	
	goal_reached_distance = 0.01
	goal_reached_yaw = 0.05
	default_speed = 20.0
	min_speed = 5.0
	slowdown_radius = 0.2
	# cuRobo
	num_targets = 0
	n_obstacle_cuboids = 30
	n_obstacle_mesh = 100
	target_pose = None
	tensor_args = TensorDeviceType(device=env.device)
	rotation_threshold = 1
	position_threshold = 1
	trajopt_dt = None
	optimize_dt = True
	trajopt_tsteps = 32
	trim_steps = None
	max_attempts = 2
	interpolation_dt = 0.03
	enable_finetune_trajopt = True
	cmd_plan = None
	robot_cfg_path = get_robot_configs_path()
	
	if args_cli.robot == "anubis":
		right_arm = "anubis_right_arm.yml"
		left_arm = "anubis_left_arm.yml"
		r_robot_cfg = load_yaml(join_path(robot_cfg_path, right_arm))["robot_cfg"]
		l_robot_cfg = load_yaml(join_path(robot_cfg_path, left_arm))["robot_cfg"]
		r_j_names = r_robot_cfg["kinematics"]["cspace"]["joint_names"]
		l_j_names = l_robot_cfg['kinematics']['cspace']['joint_names']
		r_j_index = env.scene.articulations['robot'].find_joints(r_j_names)[0]
		l_j_index = env.scene.articulations['robot'].find_joints(l_j_names)[0]
		# Load the URDF file for forward kinematics
		urdf_file = l_robot_cfg["kinematics"]["urdf_path"]
		robot = pin.buildModelFromUrdf(urdf_file)
		data = robot.createData()	
	
	# Define goals
	# The goal is given by world frame
#goals.append(("A_r", torch.tensor([0.1, -0.2, 1.0, 0.6396, -0.5940, 0.3203, -0.3682], device=env.device)))
	goals.append(("N", torch.tensor([0.67542, -0.68647, 0], device=env.device)))
	goals.append(("A_r", torch.tensor([1.05875, -0.60828, 0.9534,-0.0039, -0.7134,	0.0066, -0.7008], device=env.device)))
	goals.append(("G_r", torch.tensor( True, device=env.device)))  # Gripper close
	goals.append(("N", torch.tensor([0.32542, -0.68647, 0], device=env.device)))
#	goals.append(("A_r", torch.tensor([0.5, 0.0, 1.0, 0.6396, -0.5940, 0.3203, -0.3682], device=env.device)))
	goals.append(("G_r", torch.tensor( False, device=env.device)))	# Gripper close
	goals.append(("G_l", torch.tensor( True, device=env.device)))  # Gripper close

	gripper_command_L = False
	gripper_command_R = False
	r_dis_pre = 10
	with contextlib.suppress(KeyboardInterrupt):
		while simulation_app.is_running():
			if not running_recording_instance:
				continue
			# Define actions
			start = time.time()
			pose_L = torch.zeros((env.num_envs, 6), device=env.device)
			pose_R = torch.zeros((env.num_envs, 6), device=env.device)
			delta_pose_base = torch.zeros((env.num_envs, 3), device=env.device)
			if current_goal is None:
				goal_index += 1
				if goal_index >= len(goals):
					print("All goals reached.")
					break
#break	# or loop: base_goal_index = 0
				else:
					current_goal = goals[goal_index]
					print(f"Switching to new goal:", current_goal)
			# Mobile base motion planning
			if current_goal[0] == "N":		
				# Get current position of the robot
				base_pos = env.scene.articulations['robot'].data.root_link_pos_w
#print(f"Current position: {base_pos[0,:2].tolist()}, Goal position: {current_goal[1]}")
				
				# Get current orientation of the robot
				w, x, y, z = env.scene.articulations['robot'].data.root_link_quat_w.unbind(-1)
				siny_cosp = 2 * (w * z + x * y)
				cosy_cosp = 1 - 2 * (y * y + z * z)
				current_yaw = torch.atan2(siny_cosp, cosy_cosp)
				delta_pos = current_goal[1][:2] - base_pos[0,:2]
				desired_yaw = torch.atan2(delta_pos[1], delta_pos[0])
				goal_yaw = current_goal[1][2]

#print(f"Current yaw: {current_yaw.item()}, Desired yaw: {desired_yaw.item()}")

				delta_yaw = (desired_yaw - current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
				delta_yaw_g = (goal_yaw - current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
				
				# Compute delta position as before
				distance = torch.norm(delta_pos)

				if distance > goal_reached_distance:
					# --- Move toward the goal position ---
					cos_yaw = torch.cos(current_yaw)
					sin_yaw = torch.sin(current_yaw)

					vx_local = delta_pos[0] * cos_yaw + delta_pos[1] * sin_yaw
					vy_local = -delta_pos[0] * sin_yaw + delta_pos[1] * cos_yaw

					if distance > slowdown_radius:
						speed_scale = default_speed
					else:
						speed_scale = min_speed + (default_speed - min_speed) * (distance / slowdown_radius)

					angular_speed = 0.0  # No rotation during travel

					delta_pose_base[:, 0] = (-1) * speed_scale * vx_local
					delta_pose_base[:, 1] = (-1) * speed_scale * vy_local
					delta_pose_base[:, 2] = angular_speed
				else:
					# --- Position reached: align to final orientation ---
					# Rotate in place
					rotation_speed = 5.0  # constant angular velocity (positive = CCW, negative = CW)

# Determine direction to turn
					if delta_yaw_g > 0:
						angular_speed = rotation_speed
					else:
						angular_speed = -rotation_speed

# Stop linear motion, rotate only
					delta_pose_base[:, 0:2] = 0.0
					delta_pose_base[:, 2] = angular_speed
					if torch.abs(delta_yaw_g) < goal_reached_yaw:
						print("Goal reached!")
						current_goal = None
						delta_pose_base[:, :] = 0.0
						
			
			# Arm motion planning
			elif current_goal[0] == "A_l":
				# 1. Convert goal world frame to robot base frame
				goal_ee_pose_b, goal_ee_quat_b = world2base(env, current_goal[1][:3], current_goal[1][3:])		
				
				if cmd_plan is None:
					# 2.Planning
					sim_data = env.scene.articulations['robot']
					joint_pos = sim_data.data.joint_pos[0, l_j_index].tolist()
					joint_vel = sim_data.data.joint_vel[0, l_j_index].tolist()
					l_robot_cfg["kinematics"]["cspace"]["retract_config"] = joint_pos
					target = cuboid.VisualCuboid(
						"/World/target_l",
						position = current_goal[1][:3].squeeze(0).tolist(),
						orientation = current_goal[1][3:].squeeze(0).tolist(),
						color=np.array([1.0, 0, 0]),
						size=0.05,
					)
					
					usd_helper = UsdHelper()
					usd_helper.load_stage(env.sim.stage)
					world_cfg = usd_helper.get_obstacles_from_stage(
						only_paths=["collisions"], reference_prim_path="/World/envs/env_0/Robot/base_link"
					)
					motion_gen_config = MotionGenConfig.load_from_robot_config(
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
						trajopt_dt = trajopt_dt,
						trajopt_tsteps = trajopt_tsteps,
						trim_steps = trim_steps,
					)
					motion_gen = MotionGen(motion_gen_config)
					
					#TODO Is warmup needed?
					motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
					plan_config = MotionGenPlanConfig(
						enable_graph=False,
						enable_graph_attempt=4,
						max_attempts=max_attempts,
						enable_finetune_trajopt=enable_finetune_trajopt,
						time_dilation_factor=0.5,
					)

					cmd_idx = 0
					i = 0
					spheres = None
					target_orientation = None
					pose_metric = None
					initialized = False
					
					cu_js = JointState(
						position = tensor_args.to_device(joint_pos),
						velocity = tensor_args.to_device(joint_vel),
						acceleration = tensor_args.to_device(joint_vel) * 0.0,
						jerk = tensor_args.to_device(joint_vel) * 0.0,
						joint_names = l_j_names
					)
					cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
					ee_translation_goal = goal_ee_pose_b.squeeze(0).tolist()
					ee_orientation_goal = goal_ee_quat_b.squeeze(0).tolist()
					ik_goal = Pose(
						position=tensor_args.to_device(ee_translation_goal),
						quaternion=tensor_args.to_device(ee_orientation_goal),
					)
					plan_config.pose_cost_metric = pose_metric
					result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
					succ = result.success.item()
					print(f"succ:{succ}")
					if succ:
						num_targets += 1
						cmd_plan = result.get_interpolated_plan()
						cmd_plan = motion_gen.get_full_js(cmd_plan)
						# get only joint names that are in both:
						idx_list = []
						common_js_names = []
						for x in l_j_names:
							if x in cmd_plan.joint_names:
								idx_list.append(sim_data.find_joints(x)[0][0])
								common_js_names.append(sim_data.find_joints(x)[1][0])

						cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
						cmd_idx = 0

					else:
						carb.log_warn("Plan did not converge to a solution: " + str(result.status))
				else:	
					# Compute poses at t and t+1
					eef_pos_t, eef_quat_t = get_eef_pose(cmd_plan[cmd_idx], current_goal[0], l_j_names, idx_list, robot, data)
					if (cmd_idx+1) == len(cmd_plan):					
						eef_pos_tp1, eef_quat_tp1 = get_eef_pose(cmd_plan[cmd_idx], current_goal[0], l_j_names, idx_list, robot, data)
						current_goal = None
						cmd_plan = None
					else:
						eef_pos_tp1, eef_quat_tp1 = get_eef_pose(cmd_plan[cmd_idx + 1], current_goal[0], l_j_names, idx_list, robot, data)
					# Delta Position
					delta_pos = (eef_pos_tp1 - eef_pos_t) 
					# Delat Orientation
					r_t		= R.from_quat(eef_quat_t.numpy()[[1, 2, 3, 0]])		# WXYZ â XYZW
					r_tp1	= R.from_quat(eef_quat_tp1.numpy()[[1, 2, 3, 0]])	# WXYZ â XYZW
					delta_r = r_tp1 * r_t.inv()
					delta_rotvec = torch.tensor(delta_r.as_rotvec(), dtype=torch.float32)
					pose_L = torch.cat((delta_pos, delta_rotvec))	
					cmd_idx += 1

			elif current_goal[0] == "A_r":	
				# 1. Convert goal world frame to robot base frame
				goal_ee_pose_b, goal_ee_quat_b = world2base(env, current_goal[1][:3], current_goal[1][3:])		
				
				if cmd_plan is None:
					# 2.Planning
					sim_data = env.scene.articulations['robot']
					joint_pos = sim_data.data.joint_pos[0, r_j_index].tolist()
					joint_vel = sim_data.data.joint_vel[0, r_j_index].tolist()
					r_robot_cfg["kinematics"]["cspace"]["retract_config"] = joint_pos
#					target = cuboid.VisualCuboid(
#						"/World/target_r",
#						position = current_goal[1][:3].squeeze(0).tolist(), #goal_ee_pose_b.squeeze(0).tolist(),
#						orientation = current_goal[1][3:].squeeze(0).tolist(), #goal_ee_quat_b.squeeze(0).tolist(),
#						color=np.array([1.0, 0, 0]),
#						size=0.08,	
#						)
					
					usd_helper = UsdHelper()
					usd_helper.load_stage(env.sim.stage)
					env.scene.articulations["robot"].data.body_pos_w
					world_cfg = usd_helper.get_obstacles_from_stage(
						only_paths=["collisions", "env_0"], reference_prim_path="/World/envs/env_0/Robot/base_link"
					)
					ipdb.set_trace()
					motion_gen_config = MotionGenConfig.load_from_robot_config(
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
					motion_gen = MotionGen(motion_gen_config)
					#TODO Is warmup needed?
					motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
					plan_config = MotionGenPlanConfig(
						enable_graph=False,
						enable_graph_attempt=4,
						max_attempts=max_attempts,
						enable_finetune_trajopt=enable_finetune_trajopt,
						time_dilation_factor=0.5,
					)

					cmd_idx = 0
					i = 0
					spheres = None
					target_orientation = None
					pose_metric = None
					initialized = False
					
					
					cu_js = JointState(
						position = tensor_args.to_device(joint_pos),
						velocity = tensor_args.to_device(joint_vel),
						acceleration = tensor_args.to_device(joint_vel) * 0.0,
						jerk = tensor_args.to_device(joint_vel) * 0.0,
						joint_names = r_j_names
					)
					cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
					ee_translation_goal = goal_ee_pose_b.squeeze(0).tolist()
					ee_orientation_goal = goal_ee_quat_b.squeeze(0).tolist()
					ik_goal = Pose(
						position=tensor_args.to_device(ee_translation_goal),
						quaternion=tensor_args.to_device(ee_orientation_goal),
					)
					plan_config.pose_cost_metric = pose_metric
					result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
					succ = result.success.item()
					print(f"succ:{succ}")
					if succ:
						num_targets += 1
						cmd_plan = result.get_interpolated_plan()
						cmd_plan = motion_gen.get_full_js(cmd_plan)
						# get only joint names that are in both:
						idx_list = []
						common_js_names = []
						for x in r_j_names:
							if x in cmd_plan.joint_names:
								idx_list.append(sim_data.find_joints(x)[0][0])
								common_js_names.append(sim_data.find_joints(x)[1][0])

						cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
						cmd_idx = 0

					else:
						carb.log_warn("Plan did not converge to a solution: " + str(result.status))
				else:	
					# Compute poses at t and t+1
					eef_pos_t, eef_quat_t = get_eef_pose(cmd_plan[cmd_idx], current_goal[0], r_j_names, idx_list, robot, data)
					if (cmd_idx+1) == len(cmd_plan):					
						eef_pos_tp1, eef_quat_tp1 = get_eef_pose(cmd_plan[cmd_idx], current_goal[0], r_j_names, idx_list, robot, data)
						current_goal = None
						cmd_plan = None
					else:
						eef_pos_tp1, eef_quat_tp1 = get_eef_pose(cmd_plan[cmd_idx + 1], current_goal[0], r_j_names, idx_list, robot, data)
					# Delta Position
					delta_pos = (eef_pos_tp1 - eef_pos_t) 
					# Delat Orientation
					r_t		= R.from_quat(eef_quat_t.numpy()[[1, 2, 3, 0]])		# WXYZ â XYZW
					r_tp1	= R.from_quat(eef_quat_tp1.numpy()[[1, 2, 3, 0]])	# WXYZ â XYZW
					delta_r = r_tp1 * r_t.inv()
					delta_pos_tensor = (delta_pos.detach() if isinstance(delta_pos, torch.Tensor) else torch.as_tensor(delta_pos)).to(device=env.device, dtype=torch.float32)
					delta_rotvec = torch.tensor(delta_r.as_rotvec(), dtype=torch.float32, device=env.device)
					pose_R = torch.cat((delta_pos_tensor, delta_rotvec)).unsqueeze(0) 
					cmd_idx += 1
			elif current_goal[0] == "G_l":
				gripper_command_L = current_goal[1].item()
#print(f"Gripper L command: {gripper_command_L}")		
				l_gripper = env.scene._articulations['robot'].data.body_pos_w[0, [82,83]]
				if torch.norm(l_gripper[0] - l_gripper[1]) < 0.005 :
					current_goal = None
				
			elif current_goal[0] == "G_r":
				gripper_command_R = current_goal[1].item()
#					print(f"Gripper R command: {gripper_command_R}")   
				r_gripper = env.scene._articulations['robot'].data.body_pos_w[0, [79, 80]]
				r_dis_curr = torch.norm(r_gripper[0] - r_gripper[1])
#					print(r_dis_pre - r_dis_curr)
				if r_dis_pre - r_dis_curr < 0.001:
					current_goal = None			
					r_dis_pre = 10
				else:
					r_dis_pre = r_dis_curr

					
			# convert to torch
			actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
			obv = env.step(actions)
			# TODO: Check about subtask
			if subtasks is not None:
				if subtasks == {}:
					subtasks = obv[0].get("subtask_terms")
				elif subtasks:
					show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)

			if success_term is not None:
				if bool(success_term.func(env, **success_term.params)[0]):
					success_step_count += 1
					if success_step_count >= args_cli.num_success_steps:
						# recorder_manager : isaaclab.managers.recorder_manager
						env.recorder_manager.record_pre_reset([0], force_export_or_skip=False) # 366
						env.recorder_manager.set_success_to_episodes(
							[0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
						)
						env.recorder_manager.export_episodes([0])
						should_reset_recording_instance = True
				else:
					success_step_count = 0

			# print out the current demo count if it has changed
			if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
				current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
				label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
				print(label_text)

			if should_reset_recording_instance:
				env.sim.reset()
				env.recorder_manager.reset()
				env.reset()
				should_reset_recording_instance = False
				success_step_count = 0
				instruction_display.show_demo(label_text)
				env.action_manager.get_term('armL_action')._ik_controller.reset()
				env.action_manager.get_term('armR_action')._ik_controller.reset()

			if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
				print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
				break

			# check that simulation is stopped or not
			if env.sim.is_stopped():
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
