"""
Teleoperate in IsaacLab to convert into LeRobot format."

1. actions
    1.1 ee_6d_pose + mobile base
2. Observations
    2.1 images
        2.1.1 front
        2.1.2 wrist_left
        2.1.3 wrist_right
    2.2 states
        2.2.1 ee_6d_pose + mobile base
        2.2.3 language_instruction
"""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperate Anubis in IsaacLab")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--teleop_device", type=str, default="oculus_mobile", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Cabinet-anubis-teleop-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--robot", type=str, default="anubis", help="Which robot to use in the task.")
parser.add_argument("--record", type=bool, default=True, help="Whether to record the simulation.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/anubis/brick.hdf5", help="File path to export recorded demos."
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

# Omniverse logger
import omni.log
import omni.ui as ui

# Teleoperation
import torch
import gymnasium as gym
from isaaclab.devices import Se3Keyboard_BMM, Oculus_mobile

# Record Demo
import os
import time
import contextlib
import isaaclab_mimic.envs  # noqa: F401
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
from isaaclab.sensors.camera import Camera
from isaaclab.utils import convert_dict_to_backend

# If I need to do pointclouds
# import omni.replicator.core as rep
# from isaaclab.sensors.camera.utils import create_pointcloud_from_depth


# TODO: Do I need RateLimiter?
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
    print("vx, vy, wz", vx, vy, wz)
    theta = torch.tensor([0, 2 * torch.pi / 3, 4 * torch.pi / 3], device=vx.device)
    M = torch.stack([
        -torch.sin(theta),
        torch.cos(theta),
        torch.full_like(theta, l)
    ], dim=1)  # Shape: (3, 3)

    base_vel = torch.stack([vx, vy, wz], dim=-1)  # Shape: (B, 3)
    wheel_velocities = (1 / wheel_radius) * base_vel @ M.T  # Shape: (B, 3)
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
    
    # 1. Record Demo
    
    rate_limiter = RateLimiter(args_cli.step_hz)
    
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

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
    
    # create controller 
    if args_cli.robot == "anubis":
        teleop_interface = Oculus_mobile(
            pos_sensitivity=0.5 * args_cli.sensitivity, rot_sensitivity=0.5 * args_cli.sensitivity, base_sensitivity = 0.8 * args_cli.sensitivity
        )
    elif args_cli.robot == "franka":
        teleop_interface = Se3Keyboard_BMM(
            pos_sensitivity=1.0 * args_cli.sensitivity, rot_sensitivity=5.0 * args_cli.sensitivity, base_sensitivity = 0.3 * args_cli.sensitivity
        )
    else:
        raise ValueError(
            f"Invalid robot to teleoperate '{args_cli.robot}'."
        )
        
    teleop_interface2 = Se3Keyboard_BMM(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )
    teleop_interface2.add_callback("R", reset_recording_instance)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # reset environment
    env.reset()
    teleop_interface.reset()
    
    current_recorded_demo_count = 0
    success_step_count = 0
    
    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    instruction_display = InstructionDisplay(args_cli.teleop_device)
    if args_cli.teleop_device.lower() != "handtracking":
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    subtasks = {}
    
    
    # simulate environment
    print("Starting teleoperation...Press LG")
    
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            
            if running_recording_instance:
                # get signal from teleoperation interface
                pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base = teleop_interface.advance()
                
                # pre-process actions
                pose_L = pose_L.astype("float32")
                pose_R = pose_R.astype("float32")
                delta_pose_base = delta_pose_base.astype("float32")
                # convert to torch
                pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
                pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
                delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)
                
                actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)

                obv = env.step(actions)
                
                # TODO: Check about subtask
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()
            
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
                teleop_interface.reset()
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
