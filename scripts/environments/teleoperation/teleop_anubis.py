"""Script to run a keyboard teleoperation with anubis in Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Bimanual Mobile Manipulator(BMM) Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="oculus_mobile", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Cabinet-anubis-teleop-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--bimanual", type=bool, default=True, help="Whether to use bimanual teleoperation.")
parser.add_argument("--EEF_control", type=str, default="delta", help="Control mode: 'delta' or 'abs'.")

"""
Teleoperation devices:
- Oculus_abs: Bimanual absolute control with Oculus Quest 2
- Oculus_droid: Bimanual delta control with Oculus Quest 2 from DROID setup
- Oculus_mobile: Bimanual delta control with Oculus Quest 2 from scratch
- keyboard_bmm: Bimanual delta control with keyboard and mouse
"""

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# parse the arguments

app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import ipdb
import omni.log
import numpy as np
from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse, Se3Keyboard_BMM, Oculus_mobile, Oculus_abs, Oculus_droid
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
from scipy.spatial.transform import Rotation

def get_ee_state(env, ee_name):
    # arm
    # ipdb.set_trace()
    ee = env.scene[ee_name].data
    pos = ee.target_pos_source[0, 0]
    rot = ee.target_quat_source[0, 0]
    
    # Gripper
    if ee_name == "ee_L_frame":
        body_pos = env.scene._articulations['robot'].data.body_pos_w[0, -2:]
    else:
        body_pos = env.scene._articulations['robot'].data.body_pos_w[0, -4:-2]
    gripper_dist = torch.norm(body_pos[0] - body_pos[1])*-1*20.8+0.05 # To match [0.05, -1.65] the real robot
    return torch.cat((pos, rot, gripper_dist.unsqueeze(0))).unsqueeze(0)


def pre_process_actions_abs(env, abs_pose_L: torch.Tensor, gripper_command_L: bool, abs_pose_R, gripper_command_R: bool, delta_pose_base, num_envs, device) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose_base
    else:
        # ipdb.set_trace()
        abs_pose_L = torch.tensor(abs_pose_L, dtype=torch.float, device=device)
        abs_pose_R = torch.tensor(abs_pose_R, dtype=torch.float, device=device)
        delta_pose_base = torch.tensor(delta_pose_base, dtype=torch.float, device=device)
        # resolve gripper command
        gripper_vel_L = torch.zeros(abs_pose_L.shape[0], 1, device=device)
        gripper_vel_L[:] = -1.0 if gripper_command_L else 1.0

        gripper_vel_R = torch.zeros(abs_pose_R.shape[0], 1, device=device)
        gripper_vel_R[:] = -1.0 if gripper_command_R else 1.0
        # compute actions

        # Ensure gripper velocities and base poses have the correct shapes  
        
        gripper_vel_L = gripper_vel_L.reshape(-1, 1)  # Shape: (batch_size, 1)
        gripper_vel_R = gripper_vel_R.reshape(-1, 1)  # Shape: (batch_size, 1)
        
        dummy_zeros = torch.zeros((num_envs, 60), device=device)
        return torch.concat([abs_pose_L, abs_pose_R, gripper_vel_L, gripper_vel_R, delta_pose_base, dummy_zeros], dim=1)

def compute_wheel_velocities_torch(vx, vy, wz, wheel_radius, l):
    theta = torch.tensor([0, 2 * torch.pi / 3, 4 * torch.pi / 3], device=vx.device)
    M = torch.stack([
        -torch.sin(theta),
        torch.cos(theta),
        torch.full_like(theta, l)
    ], dim=1)  # Shape: (3, 3)

    base_vel = torch.stack([vx, vy, wz], dim=-1)  # Shape: (B, 3)
    wheel_velocities = (1 / wheel_radius) * base_vel @ M.T  # Shape: (B, 3)
    return wheel_velocitiesa


def pre_process_actions(delta_pose_L: torch.Tensor, gripper_command_L: bool, delta_pose_R, gripper_command_R: bool, delta_pose_base) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose_base
    else:
        
        # Create output tensors
        # TODO Check if wheel_radius is for real wheels or the cylinders inside the wheels
        delta_pose_base_wheel = compute_wheel_velocities_torch(
            delta_pose_base[:, 0], delta_pose_base[:, 1], delta_pose_base[:, 2],
            wheel_radius=0.103, l=0.05
        )# Shape: (batch_size, 3)
        delta_pose_base_wheel = delta_pose_base_wheel[:, [2, 1, 0]]

        batch_size = delta_pose_L.shape[0]

        gripper_command_L = torch.as_tensor(gripper_command_L, device=delta_pose_L.device).unsqueeze(-1)
        gripper_command_R = torch.as_tensor(gripper_command_R, device=delta_pose_R.device).unsqueeze(-1)

        # Expand or repeat to match the batch size
        if gripper_command_L.shape[0] == 1:
            gripper_command_L = gripper_command_L.repeat(batch_size, 1)
        if gripper_command_R.shape[0] == 1:
            gripper_command_R = gripper_command_R.repeat(batch_size, 1)

        gripper_vel_L = torch.where(gripper_command_L > 0.5,
                                    torch.tensor(-1.0, device=gripper_command_L.device),
                                    torch.tensor(1.0, device=gripper_command_L.device))
        gripper_vel_R = torch.where(gripper_command_R > 0.5,
                                    torch.tensor(-1.0, device=gripper_command_R.device),
                                    torch.tensor(1.0, device=gripper_command_R.device))

        # Ensure final shape is (batch_size, 1)
        gripper_vel_L = gripper_vel_L.reshape(-1, 1)
        gripper_vel_R = gripper_vel_R.reshape(-1, 1)

        # ipdb.set_trace()
        
        action = torch.concat([
            delta_pose_L, delta_pose_R,
            gripper_vel_L, gripper_vel_R,
            delta_pose_base_wheel
        ], dim=1)  # Shape: (batch_size, 17)
        
        dummy_zeros = torch.zeros(action.shape[0], 60, device=action.device)
        
        return torch.concat([action, dummy_zeros], dim=1) 

def main():
    """Running teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)


    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "keyboard_bmm":
        teleop_interface = Se3Keyboard_BMM(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.08 * args_cli.sensitivity, base_sensitivity = 0.7 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "oculus_mobile":
        teleop_interface = Oculus_mobile(
            pos_sensitivity=1.0 * args_cli.sensitivity, rot_sensitivity=5.0 * args_cli.sensitivity, base_sensitivity = 0.3 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "oculus":
        if args_cli.EEF_control.lower() == "delta":
            teleop_interface = Oculus_droid()
        elif args_cli.EEF_control.lower() == "abs":
            teleop_interface = Oculus_abs()
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.000001 * args_cli.sensitivity, rot_sensitivity=0.000001 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
    

    teleop_interface2 = Se3Keyboard_BMM(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )

    teleop_interface2.add_callback("R", reset_recording_instance)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    print("Starting teleoperation...Press LG")
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # # Bimanual teleoperation
            # if args_cli.bimanual == True:
            #     ee_l_state = get_ee_state(env, "ee_L_frame")
            #     ee_r_state = get_ee_state(env, "ee_R_frame")
            #     obs_dict = {"left_arm": ee_l_state, "right_arm": ee_r_state}
            # else:
            #     ee_r_state = get_ee_state(env, "ee_R_frame")
            #     obs_dict = {"right_arm": ee_r_state}
            
            pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base = teleop_interface.advance()
            
            # pre-process actions
            pose_L = pose_L.astype("float32")
            pose_R = pose_R.astype("float32")
            delta_pose_base = delta_pose_base.astype("float32")
            # convert to torch
            pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
            pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
            delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)

            if args_cli.EEF_control.lower() == "delta":
                actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
            else: # abs
                actions = pre_process_actions_abs(env, pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base, env.num_envs, env.device)
            
            # apply actions
            env.step(actions)
            # ipdb.set_trace()

            if should_reset_recording_instance:
                env.reset()
                teleop_interface.reset()
                should_reset_recording_instance = False
                
                env.action_manager.get_term('armL_action')._ik_controller.reset()
                env.action_manager.get_term('armR_action')._ik_controller.reset()

                # for i in range(env.num_envs):
                #     print("env", i, "stiffness",env.scene.articulations["robot"].data.joint_stiffness[i])
                #     print("env", i, "damping",env.scene.articulations["robot"].data.joint_damping[i])


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
