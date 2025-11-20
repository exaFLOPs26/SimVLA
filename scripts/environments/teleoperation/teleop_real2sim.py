"""
Script for real-to-sim teleoperation using a joystick.
by exaFLOPS

Contribution:
    1. Teleoperate simulation & real world using a joystick in the same time
    2. Real in Sim(superset)
    3. Cross-embodiement teleoperation
"""
import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for real2sim")

parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="oculus_droid", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Isaac-Real2Sim-Franka-IK-Rel-v0", help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# parse the arguments
app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import ipdb
import omni.log
import numpy as np
from isaaclab.devices import Oculus_droid, Se3Keyboard_BMM
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Pause and reset simulation for multi stages

from isaaclab.sim import SimulationContext                                 
import omni.ui as ui                                                                        
import carb.input                                                             
from carb.input import KeyboardEventType, KeyboardInput     
from real_state_subscriber import RealStateSubscriber                  
import rclpy
from rclpy.node import Node
import threading
from isaaclab.envs.mdp.terminations import real2ruin_subscriber

class PauseResetController():
    def __init__(self):
        # SimulationContext singleton
        self.sim = SimulationContext.instance()                                
        self.paused = False
        self.win = None
        self.stage = 0

        # Start sim synchronously
        self.sim.reset()                                                      
        self.sim.play()                                                        
    
    def _enter_pause(self, stage : int):
        # Pause the sim
        self.sim.pause()                                                         
        self.paused = True
        self.stage = stage
        # Create a simple, floating window (no flags needed)
        self.win = ui.Window("Paused", width=300, height=100)                    
        # Bring it to the front as a modal
        self.win.set_top_modal()                                                 

        # Populate the window
        with self.win.frame:
            with ui.VStack(spacing=10, alignment=ui.Alignment.CENTER):
                if self.stage % 2 == 0:
                    ui.Label("Stage 1: Arm movement", alignment=ui.Alignment.CENTER)
                    ui.Label("(Press A to reset)", alignment=ui.Alignment.CENTER)
                    
                elif self.stage % 2 == 1:
                    ui.Label("Stage 2: Gripper", alignment=ui.Alignment.CENTER)
                    ui.Label("(Press A to reset)", alignment=ui.Alignment.CENTER)
                    

    def _do_reset(self):
        # Hide overlay
        if self.win:
            self.win.visible = False
            self.win = None

        # Reset & resume
        self.sim.reset()                                                         
        self.sim.play()                                                          
        self.paused = False
        stage += 1

# Get the simulation state of the end effector
def get_ee_state(env, ee_name):
    ee = env.scene[ee_name].data

    # Get pos and rot for all environments (assuming [num_envs, ...] shape)
    pos = ee.target_pos_source[:, 0]  # Shape: [num_envs, 3]
    rot = ee.target_quat_source[:, 0]  # Shape: [num_envs, 4]

    # Gripper distance for each environment
    if ee_name == "ee_L_frame":
        # shape: [num_envs, 2, 3]
        body_pos = env.scene._articulations['robot'].data.body_pos_w[:, -2:]
    else:
        body_pos = env.scene._articulations['robot'].data.body_pos_w[:, -4:-2]

    # Compute gripper distance for each env
    gripper_dist = torch.norm(body_pos[:, 0] - body_pos[:, 1], dim=1) * -1 * 20.8 + 0.05  # shape: [num_envs]

    # Concatenate everything: [num_envs, 3 + 4 + 1] = [num_envs, 8]
    return torch.cat((pos, rot, gripper_dist.unsqueeze(1)), dim=1)


def pre_process_actions_mobile(delta_pose_L: torch.Tensor, gripper_command_L: bool, delta_pose_R, gripper_command_R: bool, delta_pose_base) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    val_L = -1.0 if gripper_command_L else 1.0
    val_R = -1.0 if gripper_command_R else 1.0

    gripper_vel_L = torch.full((delta_pose_L.shape[0], 1), val_L, device=delta_pose_L.device)
    gripper_vel_R = torch.full((delta_pose_R.shape[0], 1), val_R, device=delta_pose_R.device)

    return torch.cat([delta_pose_L, delta_pose_R, gripper_vel_L, gripper_vel_R, delta_pose_base], dim=1)

def pre_process_actions(
    teleop_data: tuple[np.ndarray, np.ndarray],  # adjust type hint if needed
    num_envs: int, 
    device: str
) -> torch.Tensor:
    delta_pose, gripper_command = teleop_data

    # Ensure delta_pose is shape (num_envs, 6)
    delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device)

    # Ensure gripper_command is shape (num_envs, 1)
    gripper_command = torch.tensor(gripper_command, dtype=torch.float, device=device).view(-1, 1)

    # Map command to velocity (custom logic – change as needed)
    gripper_vel = torch.where(
        gripper_command > 0.5,
        torch.tensor(1.0, device=device),   # True → 1.0
        torch.tensor(-1.0, device=device)   # False → -1.0
    )
    return torch.cat([delta_pose, gripper_vel], dim=1)


def main():
    """Main function for real2sim teleoperation."""
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # modify configuration
    env_cfg.terminations.time_out = None
    
    rclpy.init()
    real_sub = RealStateSubscriber(topic='vrpolicy_obs_publisher')
    threading.Thread(target=lambda: rclpy.spin(real_sub), daemon=True).start()
    real2ruin_subscriber = real_sub
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if args_cli.teleop_device.lower() == "oculus_droid":
        teleop_interface = Oculus_droid()
        pause_reset = PauseResetController()
        teleop_interface.add_callback("B", pause_reset._enter_pause)
        teleop_interface.add_callback("A", pause_reset._do_reset)
        
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
        )
    
    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # [TODO] Just in case
    teleop_interface2 = Se3Keyboard_BMM(
            pos_sensitivity=0.005 , rot_sensitivity=0.01 
        )

    teleop_interface2.add_callback("R", reset_recording_instance)

    # reset environment
    env.reset()
    teleop_interface.reset()
    
    # Simulation environment
    while simulation_app.is_running():
        # run everthing in inference mode
        with torch.inference_mode():
            # single arm teleop
            if args_cli.teleop_device.lower() == "oculus_droid": 
                # Right arm
                ee_r_state = get_ee_state(env, "ee_frame")
                obs_dict = {"left_arm": 0 , "right_arm": ee_r_state}
                teleop_data = teleop_interface.advance_onearm(obs_dict)
              
            # Bimanual teleop  
            else:
                ee_l_state = get_ee_state(env, "ee_L_frame")
                ee_r_state = get_ee_state(env, "ee_R_frame")
                
                obs_dict = {"left_arm": ee_l_state, "right_arm": ee_r_state}
                
                pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base = teleop_interface.advance(obs_dict)

                pose_L = pose_L.astype("float32")
                pose_R = pose_R.astype("float32")
                delta_pose_base = delta_pose_base.astype("float32")
                # convert to torch
                pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
                pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
                delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)
                teleop_data = (pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
                
            if args_cli.teleop_device.lower() == "mobile":
                actions = pre_process_actions_mobile(teleop_data, env.num_envs, env.device)

            else:
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
            env.step(actions)

            if should_reset_recording_instance:
                env.reset()
                teleop_interface.reset()
                should_reset_recording_instance = False
                
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
