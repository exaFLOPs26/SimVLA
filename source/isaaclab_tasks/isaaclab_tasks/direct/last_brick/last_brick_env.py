# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

"""
@configclass
class EventCfg:
    brick_position = EventTerm(
        func=mdp.randomize_rigid_body_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                name="last_brick",
                body_names=".*"
            ),
            "pose_range": {
                "pos": ((0,0, -0.05, 0.0),
                        (0.0, 0.05, 0.0))
            },
            "distribution": "uniform",
            "operation": "add"
        },
    )
"""


import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def direction_to_quaternion_torch(forward: torch.Tensor,
                                  up: torch.Tensor | None = None) -> torch.Tensor:
    """
    Convert forward direction vector(s) to quaternion(s) (wxyz).

    Args:
        forward: torch.Tensor of shape (3,) or (N, 3)
        up:      optional torch.Tensor(3,) world up. Default = [0,0,1].

    Returns:
        torch.Tensor of shape (4,) or (N, 4) in wxyz order.
    """
    device = forward.device
    dtype = forward.dtype

    # Prepare up vector in numpy
    if up is None:
        up_np = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up_np = up.detach().cpu().numpy().astype(float)

    # Work on CPU numpy for SciPy
    f_t = forward.detach().cpu()
    single_input = False
    if f_t.ndim == 1:
        # (3,) -> (1,3)
        f_t = f_t.unsqueeze(0)
        single_input = True

    N = f_t.shape[0]
    q_list = []

    for i in range(N):
        f = f_t[i].numpy().astype(float)

        # normalize forward
        norm_f = np.linalg.norm(f)
        if norm_f < 1e-8:
            # if forward is zero, just pick identity orientation
            q_list.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
            continue
        f = f / norm_f

        u = up_np
        # if forward almost parallel to up, choose different up
        if float(np.abs(np.dot(f, u))) > 0.9999:
            u = np.array([1.0, 0.0, 0.0], dtype=float)

        r = np.cross(u, f)
        r = r / np.linalg.norm(r)

        u2 = np.cross(f, r)

        R_mat = np.vstack([r, u2, f]).T  # 3x3

        q_xyzw = R.from_matrix(R_mat).as_quat()  # xyzw
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        q_list.append(q_wxyz)

    q_np = np.stack(q_list, axis=0)  # (N,4)
    q = torch.from_numpy(q_np).to(device=device, dtype=dtype)

    if single_input:
        q = q[0]  # (4,)

    return q


def normalize_to_length(F: torch.Tensor, target_len=0.5):
    """
    F : (num_env, 3) force or direction vectors
    target_len : desired normalized magnitude (default = 0.5)

    Returns:
        F_scaled : normalized vectors with magnitude target_len
        scale    : original magnitudes (num_env,)
    """
    # Compute magnitude per env
    scale = torch.norm(F, dim=1)                  # (num_env,)

    # Avoid division by zero
    eps = 1e-8
    scale_safe = scale.clamp(min=eps).unsqueeze(1)  # (num_env,1)

    # Normalize to unit, then scale
    F_scaled = (F / scale_safe) * target_len      # (num_env,3)

    return F_scaled, scale



def normalize_force_vectors(F):
    """
    F: tensor of shape (num_env, 3)
       each row is a force vector [fx, fy, fz]

    returns:
        F_normed : normalized vectors (num_env, 3)
        scale    : magnitudes (num_env,)
    """

    # magnitude per env: shape (num_env,)
    scale = torch.norm(F, dim=1)

    # avoid division by zero
    eps = 1e-8
    scale_safe = scale.clamp(min=eps).unsqueeze(-1)

    # normalize: shape (num_env, 3)
    F_normed = F / scale_safe

    return F_normed, scale




@configclass
class LastBrickEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition
    action_space = 3
    observation_space = 30 # TODO how much observation to use?
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=7.0, replicate_physics=True)
    
    # domain randomization
    # events: EventCfg = EventCfg()

    # custom parameters/scales
    # - action scale
    action_scale = 1  # [N]
   # debug_vis = True


    # - reward scales
    rew_scale_x_pos = -0.2
    rew_scale_y_pos = -0.2
    rew_scale_z_pos = -1.5
    rew_scale_w_ori = -0.5
    rew_scale_x_ori = -0.5
    rew_scale_y_ori = -0.5
    rew_scale_z_ori = -0.5
    rew_scale_ang_vel = -2.0 
    rew_scale_downward = 0.3 # Scale for force applied when brick is stuck
    rew_scale_stuck_position = 0.3 # Scale for rewarding good behavior when stuck

    # initial brick position in world
    last_brick_initial_pos = (-0.42466, 2.11538, 2.40992) #(-0.42466, 2.17838, 2.40992)  2.19538
    last_brick_initial_ori = (0.1072, 0.0, 0.0, -0.9942)

    # out_of_bounds threshold
    last_brick_final_pos_threshold = (0.01, 0.01, 0.01)


    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)



class LastBrickEnv(DirectRLEnv):
    cfg: LastBrickEnvCfg

    def __init__(self, cfg: LastBrickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        device = self.device

        self._init_pos = torch.as_tensor(cfg.last_brick_initial_pos, dtype=torch.float32, device=device).unsqueeze(0)
        self._init_ori = torch.as_tensor(cfg.last_brick_initial_ori, dtype=torch.float32, device=device).unsqueeze(0)
        
        self._pos_thresh = torch.as_tensor(cfg.last_brick_final_pos_threshold, dtype=torch.float32, device=device).unsqueeze(0)
#        self.set_debug_vis(self.cfg.debug_vis)
    def _debug_vis_callback(self, event):
            # check if robot is initialized
            # note: this is needed in-case the robot is de-initialized. we can't access the data
        # translation
        base_pos_w = self.scene.rigid_objects["last_brick"].data.root_pos_w.clone()
            # -- resolve the scales and quaternions
        # orientation
        quat = direction_to_quaternion_torch(self.actions)
        F_normed , scale = normalize_to_length(self.actions, target_len=0.7)

         # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, quat, F_normed)
      
    def _set_debug_vis_impl(self, debug_vis: bool):
            # set visibility of markers
            # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                    # set their visibility to true
                self.goal_vel_visualizer.set_visibility(True)

            else:
                if hasattr(self, "goal_vel_visualizer"):
                    self.goal_vel_visualizer.set_visibility(False)
            


    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # brick env
        brick_env_cfg = sim_utils.UsdFileCfg(
            usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/posco_final2.usd",
            activate_contact_sensors=False,
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        brick_env_cfg.func("/World/envs/env_.*/brick_env", brick_env_cfg)
        
        # Left brick
        left_brick_cfg = RigidObjectCfg(
                prim_path= "/World/envs/env_.*/left_brick",
                init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-0.49950, 2.12402, 2.1879),
                        rot=(0.99022, 0.0, 0.0, 0.14781),
                ),
                spawn=sim_utils.UsdFileCfg(
                        usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/last_brick.usd",
                        activate_contact_sensors=False,
                        scale=(1.09, 1.09, 1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                )
        )
        self.scene.rigid_objects["left_brick"] = RigidObject(left_brick_cfg)

        # Last brick
        last_brick_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/last_brick",
                init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.42466, 2.20538, 2.40992), #(-0.40952, 2.11861, 2.40992)
                rot=(0.99357, 0.0, 0.0, 0.1132),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/last_brick.usd",
                activate_contact_sensors=False,
                scale=(0.915, 1.09, 1.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            )
        )
        self.scene.rigid_objects["last_brick"] = RigidObject(last_brick_cfg)
        # Right brick
        right_brick_cfg = RigidObjectCfg(
                prim_path =  "/World/envs/env_.*/right_brick",
                init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(-0.33268, 2.16236, 2.1879), #(-0.32319, 2.18651, 2.15922)
                        rot=(0.99655, 0.0, 0.0, 0.06976), #(0.99705, 0.0, 0.0, 0.06976)
                ),
                spawn=sim_utils.UsdFileCfg(
                        usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Env/last_brick.usd",
                        activate_contact_sensors=False,
                        scale=(1.09, 1.09, 1.00),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                )
        )
        self.scene.rigid_objects["right_brick"] = RigidObject(right_brick_cfg)
        self._set_debug_vis_impl(True)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        force = (self.actions * self.cfg.action_scale).unsqueeze(1)
#        force[:, :, 0] = 0
        force[:, :, 2] = -torch.abs(force[:, :, 2])
        # print(force)
        torque = torch.zeros_like(force)
        self.scene.rigid_objects["last_brick"].set_external_force_and_torque(forces=force,torques=torque) 
        self._debug_vis_callback(True)

    def _get_observations(self) -> dict:
        # TODO Observations 
        obs = torch.cat(
                (
                 self.scene.rigid_objects["left_brick"].data.root_pos_w,
                 self.scene.rigid_objects["left_brick"].data.root_quat_w,
                 
                 self.scene.rigid_objects["last_brick"].data.root_pos_w,
                 self.scene.rigid_objects["last_brick"].data.root_quat_w, 
                 self.scene.rigid_objects["last_brick"].data.root_link_vel_w,
                 self.scene.rigid_objects["last_brick"].data.root_link_ang_vel_w,

                 self.scene.rigid_objects["right_brick"].data.root_pos_w,
                 self.scene.rigid_objects["right_brick"].data.root_quat_w,
                
                 
                 ),
                dim = -1,
            )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # TODO Total rewards with weights
        lin_vel = self.scene.rigid_objects["last_brick"].data.root_link_vel_w
        lin_vel = lin_vel[:, :3]

        total_reward =  compute_rewards(
                            self.cfg.rew_scale_x_pos,
                            self.cfg.rew_scale_y_pos,
                            self.cfg.rew_scale_z_pos,
                            self.cfg.rew_scale_w_ori,
                            self.cfg.rew_scale_x_ori,
                            self.cfg.rew_scale_y_ori,
                            self.cfg.rew_scale_z_ori,
                            self.cfg.rew_scale_ang_vel,
                            self.cfg.rew_scale_downward,
                            self.cfg.rew_scale_stuck_position,
                            self.scene.rigid_objects["last_brick"].data.root_pos_w,
                            self.scene.rigid_objects["last_brick"].data.root_quat_w, 
                            self.scene.rigid_objects["left_brick"].data.root_pos_w,
                            self.scene.rigid_objects["left_brick"].data.root_quat_w,
                            self.scene.rigid_objects["right_brick"].data.root_pos_w,
                            self.scene.rigid_objects["right_brick"].data.root_quat_w,
                            self.scene.rigid_objects["last_brick"].data.root_link_ang_vel_w,
                            lin_vel
                        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # curr_pos = self.scene.rigid_objects["last_brick"].data.root_pos_w
        # left_pos = self.scene.rigid_objects["left_brick"].data.root_pos_w
        # right_pos = self.scene.rigid_objects["right_brick"].data.root_pos_w
        # target_pos = 0.5 * (left_pos + right_pos)

        # Success
        # success = ((torch.abs(curr_pos - target_pos)) <= self._pos_thresh).all(dim=-1)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out 

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.scene.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        
        pos = self._init_pos.expand(len(env_ids), -1) + self.scene.env_origins[env_ids]
        ori = self._init_ori.expand(len(env_ids), -1)

        reset_pos = torch.cat([pos, ori], dim=-1)
        reset_vel = torch.zeros(len(env_ids), 6, device=self.device)

        self.scene.rigid_objects["last_brick"].write_root_pose_to_sim(reset_pos, env_ids)
        self.scene.rigid_objects["last_brick"].write_root_velocity_to_sim(reset_vel, env_ids)


#@torch.jit.script
#def compute_rewards(
#   
#   rew_scale_x_pos: float,
#   rew_scale_y_pos: float,
#   rew_scale_z_pos: float,
#   rew_scale_w_ori: float,
#   rew_scale_x_ori: float,
#   rew_scale_y_ori: float,
#   rew_scale_z_ori: float,
#   rew_scale_ang_vel: float,
#   rew_scale_downward: float,
#   rew_scale_stuck_position: float,
#   
#
#   last_brick_pos: torch.Tensor, #(shape: [N, 3])
#   last_brick_ori:torch.Tensor, #(shape: [N, 4])
#
#   left_brick_pos:torch.Tensor, #(shape: [N, 3])
#   left_brick_ori:torch.Tensor, #(shape: [N, 4])
#
#   right_brick_pos:torch.Tensor, #(shape: [N, 3])
#   right_brick_ori:torch.Tensor, #(shape: [N, 4])
#
#   last_brick_ang_vel: torch.Tensor,
#   last_brick_lin_vel: torch.Tensor
#
#):
#
#   # Calculate target position & orientation
#   target_pos =  (right_brick_pos + left_brick_pos) / 2
#   target_ori = (right_brick_ori + left_brick_ori) / 2
#
#   # Penalty for bad position
#   rew_x_pos = rew_scale_x_pos * torch.abs(target_pos[:,0] - last_brick_pos[:, 0])
#   rew_y_pos = rew_scale_y_pos * torch.abs(target_pos[:,1] - last_brick_pos[:, 1])
#   rew_z_pos = rew_scale_z_pos * torch.abs(target_pos[:, 2] - last_brick_pos[:, 2])
#
#   # Penalty for bad orientation
#   # rew_w_ori = rew_scale_w_ori * torch.abs(target_ori[:, 0] - last_brick_ori[:, 0])
#   # rew_x_ori = rew_scale_x_ori * torch.abs(target_ori[:, 1] - last_brick_ori[:, 1])
#   # rew_y_ori = rew_scale_y_ori * torch.abs(target_ori[:, 2] - last_brick_ori[:, 2])
#   # rew_z_ori = rew_scale_z_ori * torch.abs(target_ori[:, 3] - last_brick_ori[:, 3])
#
#   # Penalty for high angular velocity
#   rew_high_lin_vel_penalty = -rew_scale_lin_vel * torch.abs(last_brick_lin_vel)
#   rew_high_ang_vel_penalty = -rew_scale_ang_vel * torch.abs(last_brick_ang_vel)
#
## --- 2) Penalty for being stuck (too slow -> not inserting) ---
## Add small epsilon to avoid divide-by-zero
#   eps = 1e-4
#   rew_stuck_lin_penalty = -rew_scale_stuck * (1.0 / (torch.abs(last_brick_lin_vel) + eps))
#   rew_stuck_ang_penalty = -rew_scale_stuck * (1.0 / (torch.abs(last_brick_ang_vel) + eps))
#
#   # Reward for force applied when brick's edge is stuck
#   to_gap = target_pos - last_brick_pos                              # [N,3]
#   down  = torch.tensor([0.0, 0.0, -1.0], device=to_gap.device, dtype=to_gap.dtype).expand_as(to_gap) # unit vector pointing down
#
#   dvec = to_gap + rew_scale_downward * down # Direction vector biased toward goal + downard direction
#   dhat = dvec / torch.clamp(torch.linalg.norm(dvec, dim=1, keepdim=True), min=1e-6)  # Normalize to unit direction
#
#   # Reward positive velocity along dhat (desired direction)
#   v_proj = torch.sum(last_brick_lin_vel * dhat, dim=1) # Project linear velocity to desired direction (dhat)
#   rew_progress = rew_scale_stuck_position * torch.relu(v_proj) # Reward only positive progress
#
#   total_reward = rew_x_pos + rew_y_pos + rew_z_pos + rew_ang_vel + rew_progress #+ rew_w_ori + rew_x_ori + rew_y_ori + rew_z_ori
#   return total_reward


@torch.jit.script
def compute_rewards(

    rew_scale_x_pos: float,
    rew_scale_y_pos: float,
    rew_scale_z_pos: float,
    rew_scale_w_ori: float,
    rew_scale_x_ori: float,
    rew_scale_y_ori: float,
    rew_scale_z_ori: float,
    rew_scale_ang_vel: float,
    rew_scale_downward: float,
    rew_scale_stuck_position: float,


    last_brick_pos: torch.Tensor, #(shape: [N, 3])
    last_brick_ori:torch.Tensor, #(shape: [N, 4])

    left_brick_pos:torch.Tensor, #(shape: [N, 3])
    left_brick_ori:torch.Tensor, #(shape: [N, 4])

    right_brick_pos:torch.Tensor, #(shape: [N, 3])
    right_brick_ori:torch.Tensor, #(shape: [N, 4])

    last_brick_ang_vel: torch.Tensor,
    last_brick_lin_vel: torch.Tensor

):

    # Calculate target position & orientation
    target_pos =  (right_brick_pos + left_brick_pos) / 2
    target_ori = (right_brick_ori + left_brick_ori) / 2

    # Penalty for bad position
    rew_x_pos = rew_scale_x_pos * torch.abs(target_pos[:,0] - last_brick_pos[:, 0])
    rew_y_pos = rew_scale_y_pos * torch.abs(target_pos[:,1] - last_brick_pos[:, 1])
    rew_z_pos = rew_scale_z_pos * torch.abs(target_pos[:, 2] - last_brick_pos[:, 2])

    # Penalty for bad orientation
    # rew_w_ori = rew_scale_w_ori * torch.abs(target_ori[:, 0] - last_brick_ori[:, 0])
    # rew_x_ori = rew_scale_x_ori * torch.abs(target_ori[:, 1] - last_brick_ori[:, 1])
    # rew_y_ori = rew_scale_y_ori * torch.abs(target_ori[:, 2] - last_brick_ori[:, 2])
    # rew_z_ori = rew_scale_z_ori * torch.abs(target_ori[:, 3] - last_brick_ori[:, 3])

    # Penalty for high angular velocity
    rew_ang_vel = rew_scale_ang_vel * torch.abs(last_brick_ang_vel[:, 1])



    # Reward for force applied when brick's edge is stuck
    to_gap = target_pos - last_brick_pos                              # [N,3]
    down  = torch.tensor([0.0, 0.0, -1.0], device=to_gap.device, dtype=to_gap.dtype).expand_as(to_gap) # unit vector pointing down

    dvec = to_gap + rew_scale_downward * down # Direction vector biased toward goal + downard direction
    dhat = dvec / torch.clamp(torch.linalg.norm(dvec, dim=1, keepdim=True), min=1e-6)  # Normalize to unit direction

    # Reward positive velocity along dhat (desired direction)
    v_proj = torch.sum(last_brick_lin_vel * dhat, dim=1) # Project linear velocity to desired direction (dhat)
    rew_progress = rew_scale_stuck_position * torch.relu(v_proj) # Reward only positive progress

    total_reward = rew_x_pos + rew_y_pos + rew_z_pos + rew_ang_vel + rew_progress #+ rew_w_ori + rew_x_ori + rew_y_ori + rew_z_ori
    return total_reward
