from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    

def open(
    env: ManagerBasedRLEnv,
    threshold: float = 0.3, # Pi/4
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet_02"),
) -> bool:
    # TODO: Check if this is reading the right drawer angle
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    drawer_angle_1 = env.scene[cabinet_cfg.name].data.joint_pos[:,-1]
    drawer_angle_2 = env.scene[cabinet_cfg.name].data.joint_pos[:,-2]
    drawer_angle_3 = env.scene[cabinet_cfg.name].data.joint_pos[:,-3]



    return ((torch.abs(drawer_angle_1) > threshold) or (torch.abs(drawer_angle_2) > threshold) or (torch.abs(drawer_angle_3) > threshold))