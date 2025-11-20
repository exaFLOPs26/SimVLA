"""
Motion Planning with pre-defined goal pose

ScLERP is not that stable
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

import torch
import gymnasium as gym

# Omniverse logger
import omni.log
import omni.ui as ui

# For reset
from isaaclab.devices import Se3Keyboard_BMM

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
import ipdb
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

def get_ee_state(env, ee_name):
    # arm
    ee = env.scene[ee_name].data
    pos = ee.target_pos_source[0, 0]
    rot = ee.target_quat_source[0, 0]
    
    return torch.cat((pos, rot)).unsqueeze(0)

def rotation_matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3x3 rotation matrix to a 3D rotation vector (axis-angle).
    Works in PyTorch, stays on the same device.
    """
    cos_theta = ((R.trace() - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta.abs() < 1e-5:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    else:
        lnR = (theta / (2 * torch.sin(theta))) * (R - R.T)
        return torch.stack([lnR[2, 1], lnR[0, 2], lnR[1, 0]])

def skew(w):
    wx, wy, wz = w[0], w[1], w[2]
    return torch.tensor([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ], device=w.device, dtype=w.dtype)

def matrix_log_SE3(T):
    R = T[:3, :3]
    p = T[:3, 3]

    # Clamp trace for numerical stability
    cos_theta = ((R.trace() - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta.abs() < 1e-5:
        omega = torch.zeros(3, device=T.device, dtype=T.dtype)
        V_inv = torch.eye(3, device=T.device, dtype=T.dtype)
    else:
        lnR = (theta / (2 * torch.sin(theta))) * (R - R.T)
        omega = torch.tensor([lnR[2, 1], lnR[0, 2], lnR[1, 0]], device=T.device, dtype=T.dtype)
        skew_omega = skew(omega)
        theta2 = theta * theta
        A = torch.sin(theta) / theta
        B = (1 - torch.cos(theta)) / theta2
        V_inv = torch.eye(3, device=T.device, dtype=T.dtype) - 0.5 * skew_omega + \
                (1 / theta2) * (1 - A / (2 * B)) * (skew_omega @ skew_omega)

    v = V_inv @ p
    twist = torch.zeros((4, 4), device=T.device, dtype=T.dtype)
    twist[:3, :3] = skew(omega)
    twist[:3, 3] = v
    return twist

def matrix_exp_SE3(twist):
    omega_hat = twist[:3, :3]
    v = twist[:3, 3]

    omega = torch.tensor([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]], device=twist.device, dtype=twist.dtype)
    theta = torch.norm(omega)

    if theta.abs() < 1e-5:
        R = torch.eye(3, device=twist.device, dtype=twist.dtype)
        V = torch.eye(3, device=twist.device, dtype=twist.dtype)
    else:
        omega_unit = omega / theta
        skew_omega = skew(omega_unit)
        R = torch.eye(3, device=twist.device, dtype=twist.dtype) + \
            torch.sin(theta) * skew_omega + (1 - torch.cos(theta)) * (skew_omega @ skew_omega)
        V = torch.eye(3, device=twist.device, dtype=twist.dtype) + \
            ((1 - torch.cos(theta)) / theta) * skew_omega + \
            ((theta - torch.sin(theta)) / (theta ** 2)) * (skew_omega @ skew_omega)

    T = torch.eye(4, device=twist.device, dtype=twist.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T

def pose_to_SE3(position, quat):
    # quat assumed [w, x, y, z] or [x, y, z, w]?  
    # Your example shows [w, x, y, z] at current_goal[1]: [1.0, 0.0, 0.0, 0.0]
    # But in your tensor you have [pos(3), quat(4)] in order: (x,y,z,w,x,y,z)?? 
    # Your example tensor current_goal[1]: tensor([0.5000, 0.0000, 0.2000, 1.0000, 0.0000, 0.0000, 0.0000])
    # So quat = [1.0, 0.0, 0.0, 0.0] = (w, x, y, z)
    # Let's unpack accordingly:

    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    R = torch.tensor([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], device=position.device, dtype=position.dtype)

    T = torch.eye(4, device=position.device, dtype=position.dtype)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

def quat_angle_diff(q1, q2):
    """Compute the absolute rotation angle difference between two quaternions."""
    dot = torch.sum(q1 * q2)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = 2 * torch.acos(dot.abs())
    return angle  # in radians

def slerp_se3(ee_start, ee_goal):
    # ee_start, ee_goal are tensors of shape (7,) with [pos(3), quat(4)]
    
    pos_s = ee_start[:3]
    quat_s = ee_start[3:]
    pos_g = ee_goal[:3]
    quat_g = ee_goal[3:]

    pos_dist = torch.norm(pos_g - pos_s)
    rot_angle = quat_angle_diff(quat_s, quat_g)
    motion_mag = pos_dist + rot_angle
    num_steps = max(int(motion_mag * 1000), 2)
    print(num_steps)
    T_s = pose_to_SE3(pos_s, quat_s)
    T_g = pose_to_SE3(pos_g, quat_g)

    T_rel = torch.linalg.inv(T_s) @ T_g
    log_T = matrix_log_SE3(T_rel)

    traj = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        t_scaled = t * t * (3 - 2 * t)  # smoothstep interpolation
        T_t = T_s @ matrix_exp_SE3(t_scaled * log_T)
        traj.append(T_t)
    
    return traj


def compute_deltas(traj, pos_sensitivity=1.0, rot_sensitivity=1.0):
    """Returns list of twist deltas: 6D vectors [dx, dy, dz, rx, ry, rz]"""
    deltas = []
    for i in range(len(traj) - 1):
        T_curr = traj[i]
        T_next = traj[i + 1]

        # Translation delta
        dp = (T_next[:3, 3] - T_curr[:3, 3]) * pos_sensitivity

        # Rotation delta matrix
        R_curr = T_curr[:3, :3]
        R_next = T_next[:3, :3]
        R_delta = R_next @ R_curr.T

        # Matrix to rotvec
        theta = torch.acos(torch.clamp((torch.trace(R_delta) - 1) / 2, -1.0, 1.0))
        if theta.abs() < 1e-5:
            rotvec = torch.zeros(3, device=traj[0].device, dtype=traj[0].dtype)
        else:
            skew_sym = (R_delta - R_delta.T) / (2 * torch.sin(theta))
            rotvec = theta * torch.tensor([
                skew_sym[2,1],
                skew_sym[0,2],
                skew_sym[1,0]
            ], device=traj[0].device, dtype=traj[0].dtype)

        rotvec = rotvec * rot_sensitivity

        # Combine into 6D twist delta
        twist = torch.cat([dp, rotvec])
        deltas.append(twist)

    return deltas



def main():
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
    # env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    # env_cfg.recorders.dataset_export_dir_path = output_dir
    # env_cfg.recorders.dataset_filename = output_file_name
    # env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

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

    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        demo_label = ui.Label(label_text)
        subtask_label = ui.Label("")
    subtasks = {}
    
    # Whole body motion planning
    goals = []
    current_goal = None
    goal_index = -1
    
    arm_index = -1
    deltas = []

    # Define goals
    #goals.append(("N", torch.tensor([0.3, 0.3, 0.0], device=env.device)))
    goals.append(("A_l", torch.tensor([-0.1206,  0.4385, -0.2500, -0.7140,  0.6530, -0.1729,  0.1843], device=env.device)))
    goals.append(("A_r", torch.tensor([ 0.1206,  0.3385, -0.1600,  0.7140, -0.6530, -0.1729,  0.1843], device=env.device)))
    goals.append(("G_l", torch.tensor( True, device=env.device)))  # Gripper close
    
    gripper_command_L = False
    gripper_command_R = False
    
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            if running_recording_instance:
                # Define actions
                pose_L = torch.zeros(6)
                pose_R = torch.zeros(6)

                delta_pose_base = torch.zeros((env.num_envs, 3), device=env.device)
                
                if current_goal is None:
                    goal_index += 1
                    if goal_index >= len(goals):
                        print("All goals reached.")
                        break  # or loop: base_goal_index = 0
                    current_goal = goals[goal_index]
                    print(f"Switching to new goal:", current_goal)
                
                if current_goal[0] == "N":      
                    # Get current position of the robot
                    base_pos = env.scene.articulations['robot'].data.root_link_pos_w
                    print(f"Current position: {base_pos[0,:2].tolist()}, Goal position: {current_goal[1]}")
                    
                    # Get current orientation of the robot
                    w, x, y, z = env.scene.articulations['robot'].data.root_link_quat_w.unbind(-1)
                    siny_cosp = 2 * (w * z + x * y)
                    cosy_cosp = 1 - 2 * (y * y + z * z)
                    current_yaw = torch.atan2(siny_cosp, cosy_cosp)
                    delta_pos = current_goal[1][:2] - base_pos[0,:2]
                    desired_yaw = torch.atan2(delta_pos[1], delta_pos[0])
                    goal_yaw = current_goal[1][2]
                    print(f"Current yaw: {current_yaw.item()}, Desired yaw: {desired_yaw.item()}")

                    delta_yaw = (desired_yaw - current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                    delta_yaw_g = (goal_yaw - current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                    yaw_threshold = 0.05   # ~6 degrees
                    
                    distance = torch.norm(delta_pos)
                    print(torch.abs(delta_yaw_g) > yaw_threshold)
                    print(distance) 
                    if distance > 0.019:
                        if torch.abs(delta_yaw) > yaw_threshold:
                            delta_pose_base[:, 2] = 5.0 * torch.sign(delta_yaw)  # Rotate to face goal
                        else:
                            delta_pose_base[:, 0] = -1.0  # Move forward
                    elif torch.abs(delta_yaw_g) > yaw_threshold:
                            delta_pose_base[:, 2] = 1.0 * torch.sign(delta_yaw_g)  # Rotate to match goal yaw
                    else:   
                        print("✅ Goal reached!")
                        current_goal = None  # Done! 
                elif current_goal[0] == "A_l":                   
                    if arm_index == -1:
                        # Read start pos and quat
                        ee_l_state = get_ee_state(env, "ee_L_frame").squeeze(0)
                        # Use ScLERP for interpolation
                        traj = slerp_se3(ee_l_state, current_goal[1])
                        deltas = compute_deltas(traj)  # Each delta is a 4x4 SE(3) delta transform on GPU
                        arm_index = 0
                        ipdb.set_trace()
                    else:
                        ee_l_state = get_ee_state(env, "ee_L_frame").squeeze(0)
                        
                        if arm_index < len(deltas):
                            delta_T = deltas[arm_index]
                            pose_L[:3] = delta_T[:3] 
                            pose_L[3:] = delta_T[3:]      # rotation delta (rotvec)
                            arm_index += 1
                            
                            print("ee_l_state:", ee_l_state)
                            print("Current goal:", current_goal[1])
                        else:
                            pos_dist = torch.norm(ee_l_state[:3] - current_goal[1][:3])
                            rot_angle = quat_angle_diff(ee_l_state[3:], current_goal[1][3:])
                            
                            if pos_dist < 0.01 and rot_angle < 0.1:
                                print("Left arm goal reached!")
                                current_goal = None
                                
                            else:
                                print("Left arm goal not reached, but trajectory completed.")
                                print("pos distance:", pos_dist.item(), "rotation angle:", rot_angle.item())
                            arm_index = -1
                        
                elif current_goal[0] == "A_r":
                    if arm_index == -1:
                        # Read start pos and quat
                        ee_r_state = get_ee_state(env, "ee_R_frame").squeeze(0)
                        # Use ScLERP for interpolation
                        traj = slerp_se3(ee_r_state, current_goal[1])
                        deltas = compute_deltas(traj)  # Each delta is a 4x4 SE(3) delta transform on GPU
                        arm_index = 0
                    else:
                        ee_r_state = get_ee_state(env, "ee_R_frame").squeeze(0)
                        
                        if arm_index < len(deltas):
                            delta_T = deltas[arm_index]
                            pose_R[:3] = delta_T[:3]      # translation delta
                            pose_R[3:] = delta_T[3:]      # rotation delta (rotvec)
                            arm_index += 1
                            ee_l_state = get_ee_state(env, "ee_L_frame").squeeze(0)
                            print("ee_l_state:", ee_l_state)
                            ee_r_state = get_ee_state(env, "ee_R_frame")
                            print("ee_r_state:", ee_r_state)
                            print("Current goal:", current_goal[1])
                        else:
                            pos_dist = torch.norm(ee_r_state[:3] - current_goal[1][:3])
                            rot_angle = quat_angle_diff(ee_r_state[3:], current_goal[1][3:])
                            
                            if pos_dist < 0.01 and rot_angle < 0.1:
                                print("Right arm goal reached!")
                                current_goal = None
                            else:
                                print("Right arm goal not reached, but trajectory completed.")
                            arm_index = -1
                        
                elif current_goal[0] == "G_l":
                    gripper_command_L = current_goal[1].item()
                    print(f"Gripper L command: {gripper_command_L}")
                    current_goal = None
                    
                elif current_goal[0] == "G_r":
                    gripper_command_R = current_goal[1].item()
                    print(f"Gripper R command: {gripper_command_R}")   
                    current_goal = None                     
                        
                # convert to torch
                pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
                pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
                delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)
                
                actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
                print("Actions:", actions)
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
                        # env.recorder_manager.record_pre_reset([0], force_export_or_skip=False) # 366
                        # env.recorder_manager.set_success_to_episodes(
                        #     [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        # )
                        # env.recorder_manager.export_episodes([0])
                        should_reset_recording_instance = True
                else:
                    success_step_count = 0
                    
            # print out the current demo count if it has changed
            # if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
            #     current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
            #     label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
            #     print(label_text)
            
            # if should_reset_recording_instance:
            #     env.sim.reset()
            #     env.recorder_manager.reset()
            #     env.reset()
            #     should_reset_recording_instance = False
            #     success_step_count = 0
            #     instruction_display.show_demo(label_text)               
            #     teleop_interface.reset()
            
            # if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
            #     print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
            #     break

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
