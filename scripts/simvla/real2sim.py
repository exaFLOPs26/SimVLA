"""
Real2Sim
"""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Real2Sim")
parser.add_argument(
	"--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Real2Sim-v1", help="Name of the task.")
parser.add_argument("--robot", type=str, default="anubis", help="Which robot to use in the task.")
parser.add_argument("--step_hz", type=int, default=50, help="Environment stepping rate in Hz.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=900, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1, help="Interval between video recordings (in steps).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.video:
	args_cli.enable_cameras = True

# Omniverse logger
import omni.log

# Teleoperation
import torch
import gymnasium as gym
from isaaclab.devices import Se3Keyboard_BMM, Oculus_mobile
from isaaclab.simvla import timestep
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
from isaaclab.utils.dict import print_dict


# Camera 
import numpy as np
from scipy.spatial.transform import Rotation as R
import ipdb
from datasets import load_dataset

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
	scale = (1.0 / wheel_radius)    # make sure float32
	base_vel = base_vel.float()               # ensure float32
	M = M.float()                             # ensure float32

	wheel_velocities = scale * (base_vel @ M.T)

	return wheel_velocities

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

	padding = torch.zeros(action.shape[0], 60, device=action.device)

	return torch.cat([action, padding], dim=1)



def main():
	rate_limiter = RateLimiter(args_cli.step_hz)

	# parse configuration
	env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=128)
	env_cfg.env_name = args_cli.task
	env_cfg.terminations.time_out = None
	env_cfg.observations.policy.concatenate_terms = False
	
	# Offline data 
	offline_data = load_dataset("chomeed/anubis_restock_tuna")
	action_data = offline_data["train"]["action_quat"]
	qpos_data = offline_data["train"]["observation.state"]
	# Initial joint state 
	init_joint = qpos_data[0]
	env_cfg.scene.robot.init_state.joint_pos["arm1_base_link_joint"] = init_joint[7]
	env_cfg.scene.robot.init_state.joint_pos["link11_joint"] = init_joint[8]
	env_cfg.scene.robot.init_state.joint_pos["link12_joint"] = init_joint[9]
	env_cfg.scene.robot.init_state.joint_pos["link13_joint"] = init_joint[10]
	env_cfg.scene.robot.init_state.joint_pos["link14_joint"] = init_joint[11]
	env_cfg.scene.robot.init_state.joint_pos["link15_joint"] = init_joint[12]
	env_cfg.scene.robot.init_state.joint_pos["arm2_base_link_joint"] = init_joint[0]
	env_cfg.scene.robot.init_state.joint_pos["link21_joint"] = init_joint[1]	
	env_cfg.scene.robot.init_state.joint_pos["link22_joint"] = init_joint[2]
	env_cfg.scene.robot.init_state.joint_pos["link23_joint"] = init_joint[3]
	env_cfg.scene.robot.init_state.joint_pos["link24_joint"] = init_joint[4]
	env_cfg.scene.robot.init_state.joint_pos["link25_joint"] = init_joint[5]
   
	# Full joint state for MSE
	env_cfg.terminations.sim2ruin.params["qpos"] = qpos_data	

	# create environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None).unwrapped
	env.reset()
	 # wrap for video recording
	if args_cli.video:
		video_kwargs = {
			"video_folder": "/root/IsaacLab/videos/real2sim",
			"step_trigger": lambda step: step % args_cli.video_interval == 0,
			"video_length": args_cli.video_length,
			"disable_logger": True,
		}
		print("[INFO] Recording videos during training.")
		print_dict(video_kwargs, nesting=4)
		env = gym.wrappers.RecordVideo(env, **video_kwargs)

	# Frame transformation
	arr = np.array(action_data)  # or np.stack(action_data) if it's nested

	arr[:, 0] += -0.095
	arr[:, 2] += 0.823356
	arr[:, 8] += -0.095
	arr[:, 10] += 0.823356
	with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
		while simulation_app.is_running():
			global timestep
			indices = timestep.to(torch.int64).cpu().numpy()
			curr_action = torch.tensor(arr[indices,:])
			pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base = curr_action[:,:7], curr_action[:, 7], curr_action[:, 8:15], curr_action[:, 15], curr_action[:, 16:19]
#				 # pre-process actions
#				 pose_L = pose_L.astype("float32")
#				 pose_R = pose_R.astype("float32")
#				 delta_pose_base = delta_pose_base.astype("float32")
#				 # convert to torch
#				 pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
#				 pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
#				 delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)
			actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
			# Send timestep to sim2ruin terminationcfg
			obv = env.step(actions)
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
