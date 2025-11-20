import torch
import math
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.simvla import timestep
import ipdb
   
if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv

def calculate_batched_mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
	squared_diff = (tensor1 - tensor2)**2
	mse_per_env = torch.mean(squared_diff, dim=1)
	return mse_per_env

def sim2ruin(
		env,
		qpos,
		thres: float,
) -> torch.Tensor:
	global timestep

	# TODO Also consider base obs in error

	# 1) sim state (num_envs, dim(robot joint pos))
	sim = env.scene.articulations["robot"].data.joint_pos[:, -16:][:, [1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10]]
	# 2) real state (num_envs, dim(qpos))
	real = torch.tensor(qpos[timestep+1], device=sim.device)[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
	# 3) MSE errors
	mse = calculate_batched_mse(sim, real)	  
	mask_reset = mse > thres
	reset_ids = torch.where(mask_reset)[0]
	timestep+=1
	# For env reset
	timestep[mask_reset] = 0
	print(mse)
	print(reset_ids)
	
	mask_done = (timestep == 1000)
	done_ids = torch.where(mask_done)[0]
	
	if len(done_ids) != 0:
		ipdb.set_trace()

	env.action_manager.get_term('armL_action')._ik_controller.reset(env_ids = reset_ids)
	env.action_manager.get_term('armR_action')._ik_controller.reset(env_ids = reset_ids)

	# 4) termination mask
	return mask_reset
