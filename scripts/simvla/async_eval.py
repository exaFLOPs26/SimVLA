"""
Evaluate a trained LeRobot ACT policy in an Isaac Lab environment across
multiple evaluation settings (lighting, textures, etc.) and seeds.

The policy is loaded from a Hugging Face repo or a local training output
directory using ACTPolicy.from_pretrained, so LeRobot handles:
  - the correct ACT architecture
  - input / output normalization
  - action chunking (predicting future action chunks, but returning one
	action per call via select_action).

Usage example:

	./isaaclab.sh -p eval_act_in_isaaclab.py \
		--task Isaac-Kitchen-v1103-00 \
		--policy_path exaFLOPs09/act_policy \
		--horizon 800 \
		--num_rollouts 10 \
		--log_dir ./act_eval_logs \
		--device cuda:0 \
		--headless

Assumptions:
  - Isaac Lab task exposes observations under obs_dict["policy"].
  - The keys in obs_dict["policy"] match the policy input feature names,
	e.g. "observation.state", "observation.images.front",
	"observation.images.wrist_left", "observation.images.wrist_right".
  - Action dimension is 23 (matching your dataset's "action" feature).
"""

import argparse
from isaaclab.app import AppLauncher

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser(
	description="Evaluate LeRobot ACT policy for Isaac Lab environment."
)

parser.add_argument(
	"--disable_fabric",
	action="store_true",
	default=False,
	help="Disable Fabric and use USD I/O operations.",
)
parser.add_argument(
	"--task",
	type=str,
	required=True,
	help="Name of the Isaac Lab task (e.g., Isaac-Kitchen-v1103-00).",
)
parser.add_argument(
	"--policy_path",
	type=str,
	required=True,
	help=(
		"Path or Hugging Face repo id for the pretrained ACT policy "
		"(e.g., exaFLOPs09/act_policy or ./outputs/train/act_your_dataset)"
	),
)
parser.add_argument(
	"--horizon",
	type=int,
	default=400,
	help="Step horizon per rollout.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
	"--num_rollouts",
	type=int,
	default=10,
	help="Number of rollouts per setting.",
)
parser.add_argument(
	"--num_seeds",
	type=int,
	default=3,
	help="Number of random seeds to evaluate.",
)
parser.add_argument(
	"--seeds",
	nargs="+",
	type=int,
	default=None,
	help="Specific seeds to use (overrides --num_seeds).",
)
parser.add_argument(
	"--log_dir",
	type=str,
	default="/tmp/act_policy_evaluation_results",
	help="Directory to write results to.",
)
parser.add_argument(
	"--log_file",
	type=str,
	default="results",
	help="Base name of output file.",
)
parser.add_argument(
	"--version_name",
	type=str,
	default="v0",
	help="version name",
)
parser.add_argument(
	"--enable_pinocchio",
	default=False,
	action="store_true",
	help="Enable Pinocchio (needed by some controllers/retargeters).",
)

# Pass AppLauncher args (e.g., --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
	import pinocchio  # noqa: F401

# Launch Omniverse app early
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------- Imports after app -----------------------
import copy
import gymnasium as gym
import os
import pathlib
import random
import torch
import numpy as np

from isaaclab_tasks.utils import parse_env_cfg
import robomimic.utils.torch_utils as TorchUtils  # for device util

#from lerobot.policies.act.modeling_act import ACTPolicy



import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import time
import json
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

#from lerobot.policies.act.configuration_act import ACTConfig
#from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from dataclasses import dataclass, field

from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import builtins
import logging
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, TypeVar

import packaging
import safetensors
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")
def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
	"""Log missing and unexpected keys when loading a model.

	Args:
		missing_keys (list[str]): Keys that were expected but not found.
		unexpected_keys (list[str]): Keys that were found but not expected.
	"""
	if missing_keys:
		logging.warning(f"Missing key(s) when loading model: {missing_keys}")
	if unexpected_keys:
		logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")

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



class ActionSelectKwargs(TypedDict, total=False):
	noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
	"""
	Base class for policy models.
	"""

	config_class: None
	name: None

	def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
		super().__init__()
		if not isinstance(config, PreTrainedConfig):
			raise ValueError(
				f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
				"`PreTrainedConfig`. To create a model from a pretrained model use "
				f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
			)
		self.config = config

	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		if not getattr(cls, "config_class", None):
			raise TypeError(f"Class {cls.__name__} must define 'config_class'")
		if not getattr(cls, "name", None):
			raise TypeError(f"Class {cls.__name__} must define 'name'")

	def _save_pretrained(self, save_directory: Path) -> None:
		self.config._save_pretrained(save_directory)
		model_to_save = self.module if hasattr(self, "module") else self
		save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

	@classmethod
	def from_pretrained(
		cls: builtins.type[T],
		pretrained_name_or_path: str | Path,
		*,
		config: PreTrainedConfig | None = None,
		force_download: bool = False,
		resume_download: bool | None = None,
		proxies: dict | None = None,
		token: str | bool | None = None,
		cache_dir: str | Path | None = None,
		local_files_only: bool = False,
		revision: str | None = None,
		strict: bool = False,
		**kwargs,
	) -> T:
		"""
		The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
		deactivated). To train it, you should first set it back in training mode with `policy.train()`.
		"""
		if config is None:
			config = PreTrainedConfig.from_pretrained(
				pretrained_name_or_path=pretrained_name_or_path,
				force_download=force_download,
				resume_download=resume_download,
				proxies=proxies,
				token=token,
				cache_dir=cache_dir,
				local_files_only=local_files_only,
				revision=revision,
				**kwargs,
			)
		model_id = str(pretrained_name_or_path)
		instance = cls(config, **kwargs)
		if os.path.isdir(model_id):
			print("Loading weights from local directory")
			model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
			policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
		else:
			try:
				model_file = hf_hub_download(
					repo_id=model_id,
					filename=SAFETENSORS_SINGLE_FILE,
					revision=revision,
					cache_dir=cache_dir,
					force_download=force_download,
					proxies=proxies,
					resume_download=resume_download,
					token=token,
					local_files_only=local_files_only,
				)
				policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
			except HfHubHTTPError as e:
				raise FileNotFoundError(
					f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
				) from e

		policy.to(config.device)
		policy.eval()
		return policy

	@classmethod
	def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
		# Create base kwargs
		kwargs = {"strict": strict}

		# Add device parameter for newer versions that support it
		if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
			kwargs["device"] = map_location

		# Load the model with appropriate kwargs
		missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
		log_model_loading_keys(missing_keys, unexpected_keys)

		# For older versions, manually move to device if needed
		if "device" not in kwargs and map_location != "cpu":
			logging.warning(
				"Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
				" This means that the model is loaded on 'cpu' first and then copied to the device."
				" This leads to a slower loading time."
				" Please update safetensors to version 0.4.3 or above for improved performance."
			)
			model.to(map_location)
		return model

	@abc.abstractmethod
	def get_optim_params(self) -> dict:
		"""
		Returns the policy-specific parameters dict to be passed on to the optimizer.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def reset(self):
		"""To be called whenever the environment is reset.

		Does things like clearing caches.
		"""
		raise NotImplementedError

	# TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
	@abc.abstractmethod
	def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
		"""_summary_

		Args:
			batch (dict[str, Tensor]): _description_

		Returns:
			tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
				is a Tensor, all other items should be logging-friendly, native Python types.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
		"""Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

		Child classes using action chunking should use this method within `select_action` to form the action chunk
		cached for selection.
		"""
		raise NotImplementedError

	@abc.abstractmethod
	def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
		"""Return one action to run in the environment (potentially in batch mode).

		When the model uses a history of observations, or outputs a sequence of actions, this method deals
		with caching.
		"""
		raise NotImplementedError

	def push_model_to_hub(
		self,
		cfg: TrainPipelineConfig,
	):
		api = HfApi()
		repo_id = api.create_repo(
			repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
		).repo_id

		# Push the files to the repo in a single commit
		with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
			saved_path = Path(tmp) / repo_id

			self.save_pretrained(saved_path)  # Calls _save_pretrained and stores model tensors

			card = self.generate_model_card(
				cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
			)
			card.save(str(saved_path / "README.md"))

			cfg.save_pretrained(saved_path)  # Calls _save_pretrained and stores train config

			commit_info = api.upload_folder(
				repo_id=repo_id,
				repo_type="model",
				folder_path=saved_path,
				commit_message="Upload policy weights, train config and readme",
				allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
				ignore_patterns=["*.tmp", "*.log"],
			)

			logging.info(f"Model pushed to {commit_info.repo_url.url}")

	def generate_model_card(
		self, dataset_repo_id: str, model_type: str, license: str | None, tags: list[str] | None
	) -> ModelCard:
		base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None  # Set a base model

		card_data = ModelCardData(
			license=license or "apache-2.0",
			library_name="lerobot",
			pipeline_tag="robotics",
			tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
			model_name=model_type,
			datasets=dataset_repo_id,
			base_model=base_model,
		)

		template_card = (
			files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
		)
		card = ModelCard.from_template(card_data, template_str=template_card)
		card.validate()
		return card 

@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
	"""Configuration class for the Action Chunking Transformers policy.

	Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

	The parameters you will most likely need to change are the ones which depend on the environment / sensors.
	Those are: `input_shapes` and 'output_shapes`.

	Notes on the inputs and outputs:
		- Either:
			- At least one key starting with "observation.image is required as an input.
			  AND/OR
			- The key "observation.environment_state" is required as input.
		- If there are multiple keys beginning with "observation.images." they are treated as multiple camera
		  views. Right now we only support all images having the same shape.
		- May optionally work without an "observation.state" key for the proprioceptive robot state.
		- "action" is required as an output key.

	Args:
		n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
			current step and additional steps going back).
		chunk_size: The size of the action prediction "chunks" in units of environment steps.
		n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
			This should be no greater than the chunk size. For example, if the chunk size size 100, you may
			set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
			environment, and throws the other 50 out.
		input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
			the input data name, and the value is a list indicating the dimensions of the corresponding data.
			For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
			indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
			include batch dimension or temporal dimension.
		output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
			the output data name, and the value is a list indicating the dimensions of the corresponding data.
			For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
			Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
		input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
			and the value specifies the normalization mode to apply. The two available modes are "mean_std"
			which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
			[-1, 1] range.
		output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
			original scale. Note that this is also used for normalizing the training targets.
		vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
		pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
			`None` means no pretrained weights.
		replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
			convolution.
		pre_norm: Whether to use "pre-norm" in the transformer blocks.
		dim_model: The transformer blocks' main hidden dimension.
		n_heads: The number of heads to use in the transformer blocks' multi-head attention.
		dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
			layers.
		feedforward_activation: The activation to use in the transformer block's feed-forward layers.
		n_encoder_layers: The number of transformer layers to use for the transformer encoder.
		n_decoder_layers: The number of transformer layers to use for the transformer decoder.
		use_vae: Whether to use a variational objective during training. This introduces another transformer
			which is used as the VAE's encoder (not to be confused with the transformer encoder - see
			documentation in the policy class).
		latent_dim: The VAE's latent dimension.
		n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
		temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
			ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
			1 when using this feature, as inference needs to happen at every step to form an ensemble. For
			more information on how ensembling works, please see `ACTTemporalEnsembler`.
		dropout: Dropout to use in the transformer layers (see code for details).
		kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
			is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
	"""

	# Input / output structure.
	n_obs_steps: int = 1
	chunk_size: int = 100
	n_action_steps: int = 100

	normalization_mapping: dict[str, NormalizationMode] = field(
		default_factory=lambda: {
			"VISUAL": NormalizationMode.MEAN_STD,
			"STATE": NormalizationMode.MEAN_STD,
			"ACTION": NormalizationMode.MEAN_STD,
		}
	)

	# Architecture.
	# Vision backbone.
	vision_backbone: str = "resnet18"
	pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
	replace_final_stride_with_dilation: int = False
	# Transformer layers.
	pre_norm: bool = False
	dim_model: int = 512
	n_heads: int = 8
	dim_feedforward: int = 3200
	feedforward_activation: str = "relu"
	n_encoder_layers: int = 4
	# Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
	# that means only the first layer is used. Here we match the original implementation by setting this to 1.
	# See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
	n_decoder_layers: int = 1
	# VAE.
	use_vae: bool = True
	latent_dim: int = 32
	n_vae_encoder_layers: int = 4

	# Inference.
	# Note: the value used in ACT when temporal ensembling is enabled is 0.01.
	temporal_ensemble_coeff: float | None = None

	# Training and loss computation.
	dropout: float = 0.1
	kl_weight: float = 10.0

	# Training preset
	optimizer_lr: float = 1e-5
	optimizer_weight_decay: float = 1e-4
	optimizer_lr_backbone: float = 1e-5

	def __post_init__(self):
		super().__post_init__()

		"""Input validation (not exhaustive)."""
		if not self.vision_backbone.startswith("resnet"):
			raise ValueError(
				f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
			)
		if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
			raise NotImplementedError(
				"`n_action_steps` must be 1 when using temporal ensembling. This is "
				"because the policy needs to be queried every step to compute the ensembled action."
			)
		if self.n_action_steps > self.chunk_size:
			raise ValueError(
				f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
				f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
			)
		if self.n_obs_steps != 1:
			raise ValueError(
				f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
			)

	def get_optimizer_preset(self) -> AdamWConfig:
		return AdamWConfig(
			lr=self.optimizer_lr,
			weight_decay=self.optimizer_weight_decay,
		)

	def get_scheduler_preset(self) -> None:
		return None

	def validate_features(self) -> None:
		if not self.image_features and not self.env_state_feature:
			raise ValueError("You must provide at least one image or the environment state among the inputs.")

	@property
	def observation_delta_indices(self) -> None:
		return None

	@property
	def action_delta_indices(self) -> list:
		return list(range(self.chunk_size))

	@property
	def reward_delta_indices(self) -> None:
		return None



#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTPolicy(PreTrainedPolicy):
	"""
	Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
	Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
	"""

	config_class = ACTConfig
	name = "act"

	def __init__(
		self,
		config: ACTConfig,
	):
		"""
		Args:
			config: Policy configuration class instance or None, in which case the default instantiation of
					the configuration class is used.
		"""
		super().__init__(config)
		config.validate_features()
		self.config = config

		self.model = ACT(config)

		if config.temporal_ensemble_coeff is not None:
			self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

		self.reset()

	def get_optim_params(self) -> dict:
		# TODO(aliberts, rcadene): As of now, lr_backbone == lr
		# Should we remove this and just `return self.parameters()`?
		return [
			{
				"params": [
					p
					for n, p in self.named_parameters()
					if not n.startswith("model.backbone") and p.requires_grad
				]
			},
			{
				"params": [
					p
					for n, p in self.named_parameters()
					if n.startswith("model.backbone") and p.requires_grad
				],
				"lr": self.config.optimizer_lr_backbone,
			},
		]

	def reset(self):
		"""This should be called whenever the environment is reset."""
		if self.config.temporal_ensemble_coeff is not None:
			self.temporal_ensembler.reset()
		else:
			self._action_queue = deque([], maxlen=self.config.n_action_steps)

	@torch.no_grad()
	def select_action(self, batch: dict[str, Tensor]) -> Tensor:
		"""Select a single action given environment observations.

		This method wraps `select_actions` in order to return one action at a time for execution in the
		environment. It works by managing the actions in a queue and only calling `select_actions` when the
		queue is empty.
		"""
		self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

		if self.config.temporal_ensemble_coeff is not None:
			actions = self.predict_action_chunk(batch)
			action = self.temporal_ensembler.update(actions)
			return action

		# Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
		# querying the policy.
		if len(self._action_queue) == 0:
			actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

			# `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
			# effectively has shape (n_action_steps, batch_size, *), hence the transpose.
			self._action_queue.extend(actions.transpose(0, 1))
		return self._action_queue.popleft()

	@torch.no_grad()
	def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
		"""Predict a chunk of actions given environment observations."""
		self.eval()

		if self.config.image_features:
			batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
			batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

		actions = self.model(batch)[0]
		return actions

	def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
		"""Run the batch through the model and compute the loss for training or validation."""
		if self.config.image_features:
			batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
			batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

		actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

		l1_loss = (
			F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
		).mean()

		loss_dict = {"l1_loss": l1_loss.item()}
		if self.config.use_vae:
			# Calculate Dââ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
			# each dimension independently, we sum over the latent dimension to get the total
			# KL-divergence per batch element, then take the mean over the batch.
			# (See App. B of https://huggingface.co/papers/1312.6114 for more details).
			mean_kld = (
				(-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
			)
			loss_dict["kld_loss"] = mean_kld.item()
			loss = l1_loss + mean_kld * self.config.kl_weight
		else:
			loss = l1_loss

		return loss, loss_dict


class ACTTemporalEnsembler:
	def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
		"""Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

		The weights are calculated as wáµ¢ = exp(-temporal_ensemble_coeff * i) where wâ is the oldest action.
		They are then normalized to sum to 1 by dividing by Î£wáµ¢. Here's some intuition around how the
		coefficient works:
			- Setting it to 0 uniformly weighs all actions.
			- Setting it positive gives more weight to older actions.
			- Setting it negative gives more weight to newer actions.
		NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
		results in older actions being weighed more highly than newer actions (the experiments documented in
		https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
		detrimental: doing so aggressively may diminish the benefits of action chunking).

		Here we use an online method for computing the average rather than caching a history of actions in
		order to compute the average offline. For a simple 1D sequence it looks something like:

		```
		import torch

		seq = torch.linspace(8, 8.5, 100)
		print(seq)

		m = 0.01
		exp_weights = torch.exp(-m * torch.arange(len(seq)))
		print(exp_weights)

		# Calculate offline
		avg = (exp_weights * seq).sum() / exp_weights.sum()
		print("offline", avg)

		# Calculate online
		for i, item in enumerate(seq):
			if i == 0:
				avg = item
				continue
			avg *= exp_weights[:i].sum()
			avg += item * exp_weights[i]
			avg /= exp_weights[: i + 1].sum()
		print("online", avg)
		```
		"""
		self.chunk_size = chunk_size
		self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
		self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
		self.reset()

	def reset(self):
		"""Resets the online computation variables."""
		self.ensembled_actions = None
		# (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
		self.ensembled_actions_count = None

	def update(self, actions: Tensor) -> Tensor:
		"""
		Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
		time steps, and pop/return the next batch of actions in the sequence.
		"""
		self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
		self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
		if self.ensembled_actions is None:
			# Initializes `self._ensembled_action` to the sequence of actions predicted during the first
			# time step of the episode.
			self.ensembled_actions = actions.clone()
			# Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
			# operations later.
			self.ensembled_actions_count = torch.ones(
				(self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
			)
		else:
			# self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
			# the online update for those entries.
			self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
			self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
			self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
			self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
			# The last action, which has no prior online average, needs to get concatenated onto the end.
			self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
			self.ensembled_actions_count = torch.cat(
				[self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
			)
		# "Consume" the first action.
		action, self.ensembled_actions, self.ensembled_actions_count = (
			self.ensembled_actions[:, 0],
			self.ensembled_actions[:, 1:],
			self.ensembled_actions_count[1:],
		)
		return action


class ACT(nn.Module):
	"""Action Chunking Transformer: The underlying neural network for ACTPolicy.

	Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
		- The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
		  model that encodes the target data (a sequence of actions), and the condition (the robot
		  joint-space).
		- A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
		  cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
		  have an option to train this model without the variational objective (in which case we drop the
		  `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

								 Transformer
								 Used alone for inference
								 (acts as VAE decoder
								  during training)
								ââââââââÃ¢ÂÂââââââââââââââââ
								â			  Outputs	â
								â				 â²	   â
								â	  âââââââºâââââââââ  â
				   ââââââââ		â	  â		 âTransf.â	â
				   â	  â		â	  âââââââºâdecoderâ  â
			  ââââââ´âââââ â	   â	 â		â		â  â
			  â			â â		â âââââ´ââââ¬ââºâ		  â  â
			  â VAE		â â		â â		  â  âââââââââ	â
			  â encoder â â		â âTransf.â				â
			  â			â â		â âencoderâ				â
			  âââââ²ââââââ â	   â â		 â			   â
				  â		  â		â ââ²âââ²ââ²ââ			 â
				  â		  â		â  â  â â				â
				inputs	  âââââââ¼âââ  â image emb.	   â
								â	 state emb.			â
								âââââââââââââââââââââââââ
	"""

	def __init__(self, config: ACTConfig):
		# BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
		# The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
		super().__init__()
		self.config = config

		if self.config.use_vae:
			self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
			self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
			# Projection layer for joint-space configuration to hidden dimension.
			if self.config.robot_state_feature:
				self.vae_encoder_robot_state_input_proj = nn.Linear(
					self.config.robot_state_feature.shape[0], config.dim_model
				)
			# Projection layer for action (joint-space target) to hidden dimension.
			self.vae_encoder_action_input_proj = nn.Linear(
				self.config.action_feature.shape[0],
				config.dim_model,
			)
			# Projection layer from the VAE encoder's output to the latent distribution's parameter space.
			self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
			# Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
			# dimension.
			num_input_token_encoder = 1 + config.chunk_size
			if self.config.robot_state_feature:
				num_input_token_encoder += 1
			self.register_buffer(
				"vae_encoder_pos_enc",
				create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
			)

		# Backbone for image feature extraction.
		if self.config.image_features:
			backbone_model = getattr(torchvision.models, config.vision_backbone)(
				replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
				weights=config.pretrained_backbone_weights,
				norm_layer=FrozenBatchNorm2d,
			)
			# Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
			# feature map).
			# Note: The forward method of this returns a dict: {"feature_map": output}.
			self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

		# Transformer (acts as VAE decoder when training with the variational objective).
		self.encoder = ACTEncoder(config)
		self.decoder = ACTDecoder(config)

		# Transformer encoder input projections. The tokens will be structured like
		# [latent, (robot_state), (env_state), (image_feature_map_pixels)].
		if self.config.robot_state_feature:
			self.encoder_robot_state_input_proj = nn.Linear(
				self.config.robot_state_feature.shape[0], config.dim_model
			)
		if self.config.env_state_feature:
			self.encoder_env_state_input_proj = nn.Linear(
				self.config.env_state_feature.shape[0], config.dim_model
			)
		self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
		if self.config.image_features:
			self.encoder_img_feat_input_proj = nn.Conv2d(
				backbone_model.fc.in_features, config.dim_model, kernel_size=1
			)
		# Transformer encoder positional embeddings.
		n_1d_tokens = 1  # for the latent
		if self.config.robot_state_feature:
			n_1d_tokens += 1
		if self.config.env_state_feature:
			n_1d_tokens += 1
		self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
		if self.config.image_features:
			self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

		# Transformer decoder.
		# Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
		self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

		# Final action regression head on the output of the transformer's decoder.
		self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

		self._reset_parameters()

	def _reset_parameters(self):
		"""Xavier-uniform initialization of the transformer parameters as in the original code."""
		for p in chain(self.encoder.parameters(), self.decoder.parameters()):
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
		"""A forward pass through the Action Chunking Transformer (with optional VAE encoder).

		`batch` should have the following structure:
		{
			[robot_state_feature] (optional): (B, state_dim) batch of robot states.

			[image_features]: (B, n_cameras, C, H, W) batch of images.
				AND/OR
			[env_state_feature]: (B, env_dim) batch of environment states.

			[action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
		}

		Returns:
			(B, chunk_size, action_dim) batch of action sequences
			Tuple containing the latent PDF's parameters (mean, log(ÏÂ²)) both as (B, L) tensors where L is the
			latent dimension.
		"""
		if self.config.use_vae and self.training:
			assert ACTION in batch, (
				"actions must be provided when using the variational objective in training mode."
			)

		batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

		# Prepare the latent for input to the transformer encoder.
		if self.config.use_vae and ACTION in batch and self.training:
			# Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
			cls_embed = einops.repeat(
				self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
			)  # (B, 1, D)
			if self.config.robot_state_feature:
				robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
				robot_state_embed = robot_state_embed.unsqueeze(1)	# (B, 1, D)
			action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

			if self.config.robot_state_feature:
				vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
			else:
				vae_encoder_input = [cls_embed, action_embed]
			vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

			# Prepare fixed positional embedding.
			# Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
			pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

			# Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
			# sequence depending whether we use the input states or not (cls and robot state)
			# False means not a padding token.
			cls_joint_is_pad = torch.full(
				(batch_size, 2 if self.config.robot_state_feature else 1),
				False,
				device=batch[OBS_STATE].device,
			)
			key_padding_mask = torch.cat(
				[cls_joint_is_pad, batch["action_is_pad"]], axis=1
			)  # (bs, seq+1 or 2)

			# Forward pass through VAE encoder to get the latent PDF parameters.
			cls_token_out = self.vae_encoder(
				vae_encoder_input.permute(1, 0, 2),
				pos_embed=pos_embed.permute(1, 0, 2),
				key_padding_mask=key_padding_mask,
			)[0]  # select the class token, with shape (B, D)
			latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
			mu = latent_pdf_params[:, : self.config.latent_dim]
			# This is 2log(sigma). Done this way to match the original implementation.
			log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

			# Sample the latent with the reparameterization trick.
			latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
		else:
			# When not using the VAE encoder, we set the latent to be all zeros.
			mu = log_sigma_x2 = None
			# TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
			latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
				batch[OBS_STATE].device
			)

		# Prepare transformer encoder inputs.
		encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
		encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
		# Robot state token.
		if self.config.robot_state_feature:
			encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
		# Environment state token.
		if self.config.env_state_feature:
			encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

		if self.config.image_features:
			# For a list of images, the H and W may vary but H*W is constant.
			# NOTE: If modifying this section, verify on MPS devices that
			# gradients remain stable (no explosions or NaNs).
			for img in batch[OBS_IMAGES]:
				cam_features = self.backbone(img)["feature_map"]
				cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
				cam_features = self.encoder_img_feat_input_proj(cam_features)

				# Rearrange features to (sequence, batch, dim).
				cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
				cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

				# Extend immediately instead of accumulating and concatenating
				# Convert to list to extend properly
				encoder_in_tokens.extend(list(cam_features))
				encoder_in_pos_embed.extend(list(cam_pos_embed))

		# Stack all tokens along the sequence dimension.
		encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
		encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

		# Forward pass through the transformer modules.
		encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
		# TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
		decoder_in = torch.zeros(
			(self.config.chunk_size, batch_size, self.config.dim_model),
			dtype=encoder_in_pos_embed.dtype,
			device=encoder_in_pos_embed.device,
		)
		decoder_out = self.decoder(
			decoder_in,
			encoder_out,
			encoder_pos_embed=encoder_in_pos_embed,
			decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
		)

		# Move back to (B, S, C).
		decoder_out = decoder_out.transpose(0, 1)

		actions = self.action_head(decoder_out)

		return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
	"""Convenience module for running multiple encoder layers, maybe followed by normalization."""

	def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
		super().__init__()
		self.is_vae_encoder = is_vae_encoder
		num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
		self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
		self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

	def forward(
		self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
	) -> Tensor:
		for layer in self.layers:
			x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
		x = self.norm(x)
		return x


class ACTEncoderLayer(nn.Module):
	def __init__(self, config: ACTConfig):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

		# Feed forward layers.
		self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
		self.dropout = nn.Dropout(config.dropout)
		self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

		self.norm1 = nn.LayerNorm(config.dim_model)
		self.norm2 = nn.LayerNorm(config.dim_model)
		self.dropout1 = nn.Dropout(config.dropout)
		self.dropout2 = nn.Dropout(config.dropout)

		self.activation = get_activation_fn(config.feedforward_activation)
		self.pre_norm = config.pre_norm

	def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
		skip = x
		if self.pre_norm:
			x = self.norm1(x)
		q = k = x if pos_embed is None else x + pos_embed
		x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
		x = x[0]  # note: [0] to select just the output, not the attention weights
		x = skip + self.dropout1(x)
		if self.pre_norm:
			skip = x
			x = self.norm2(x)
		else:
			x = self.norm1(x)
			skip = x
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		x = skip + self.dropout2(x)
		if not self.pre_norm:
			x = self.norm2(x)
		return x


class ACTDecoder(nn.Module):
	def __init__(self, config: ACTConfig):
		"""Convenience module for running multiple decoder layers followed by normalization."""
		super().__init__()
		self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
		self.norm = nn.LayerNorm(config.dim_model)

	def forward(
		self,
		x: Tensor,
		encoder_out: Tensor,
		decoder_pos_embed: Tensor | None = None,
		encoder_pos_embed: Tensor | None = None,
	) -> Tensor:
		for layer in self.layers:
			x = layer(
				x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
			)
		if self.norm is not None:
			x = self.norm(x)
		return x


class ACTDecoderLayer(nn.Module):
	def __init__(self, config: ACTConfig):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
		self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

		# Feed forward layers.
		self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
		self.dropout = nn.Dropout(config.dropout)
		self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

		self.norm1 = nn.LayerNorm(config.dim_model)
		self.norm2 = nn.LayerNorm(config.dim_model)
		self.norm3 = nn.LayerNorm(config.dim_model)
		self.dropout1 = nn.Dropout(config.dropout)
		self.dropout2 = nn.Dropout(config.dropout)
		self.dropout3 = nn.Dropout(config.dropout)

		self.activation = get_activation_fn(config.feedforward_activation)
		self.pre_norm = config.pre_norm

	def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
		return tensor if pos_embed is None else tensor + pos_embed

	def forward(
		self,
		x: Tensor,
		encoder_out: Tensor,
		decoder_pos_embed: Tensor | None = None,
		encoder_pos_embed: Tensor | None = None,
	) -> Tensor:
		"""
		Args:
			x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
			encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
				cross-attending with.
			encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
			decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
		Returns:
			(DS, B, C) tensor of decoder output features.
		"""
		skip = x
		if self.pre_norm:
			x = self.norm1(x)
		q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
		x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
		x = skip + self.dropout1(x)
		if self.pre_norm:
			skip = x
			x = self.norm2(x)
		else:
			x = self.norm1(x)
			skip = x
		x = self.multihead_attn(
			query=self.maybe_add_pos_embed(x, decoder_pos_embed),
			key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
			value=encoder_out,
		)[0]  # select just the output, not the attention weights
		x = skip + self.dropout2(x)
		if self.pre_norm:
			skip = x
			x = self.norm3(x)
		else:
			x = self.norm2(x)
			skip = x
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		x = skip + self.dropout3(x)
		if not self.pre_norm:
			x = self.norm3(x)
		return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
	"""1D sinusoidal positional embeddings as in Attention is All You Need.

	Args:
		num_positions: Number of token positions required.
	Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

	"""

	def get_position_angle_vec(position):
		return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

	sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
	return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
	"""2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

	The variation is that the position indices are normalized in [0, 2Ï] (not quite: the lower bound is 1/H
	for the vertical direction, and 1/W for the horizontal direction.
	"""

	def __init__(self, dimension: int):
		"""
		Args:
			dimension: The desired dimension of the embeddings.
		"""
		super().__init__()
		self.dimension = dimension
		self._two_pi = 2 * math.pi
		self._eps = 1e-6
		# Inverse "common ratio" for the geometric progression in sinusoid frequencies.
		self._temperature = 10000

	def forward(self, x: Tensor) -> Tensor:
		"""
		Args:
			x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
		Returns:
			A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
		"""
		not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
		# Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
		# they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
		y_range = not_mask.cumsum(1, dtype=torch.float32)
		x_range = not_mask.cumsum(2, dtype=torch.float32)

		# "Normalize" the position index such that it ranges in [0, 2Ï].
		# Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
		# are non-zero by construction. This is an artifact of the original code.
		y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
		x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

		inverse_frequency = self._temperature ** (
			2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
		)

		x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
		y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

		# Note: this stack then flatten operation results in interleaved sine and cosine terms.
		# pos_embed_x and pos_embed_y are (1, H, W, C // 2).
		pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
		pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
		pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

		return pos_embed


def get_activation_fn(activation: str) -> Callable:
	"""Return an activation function given a string."""
	if activation == "relu":
		return F.relu
	if activation == "gelu":
		return F.gelu
	if activation == "glu":
		return F.glu
	raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

# ----------------------- Helpers -----------------------
def compute_wheel_velocities_torch(vx, vy, wz, wheel_radius, l):
	theta = torch.tensor([2 * torch.pi / 3, 4 * torch.pi / 3, 0], device=vx.device)
	M = torch.stack([
		-torch.sin(theta),
		torch.cos(theta),
		torch.full_like(theta, l)
	], dim=1)  # Shape: (3, 3)

	base_vel = torch.stack([vx, vy, wz], dim=-1)  # Shape: (B, 3)
	wheel_velocities = (1 / wheel_radius) * base_vel @ M.T	# Shape: (B, 3)
	return wheel_velocities

def build_policy_observation(
	obs_dict: dict,
	policy: ACTPolicy,
	device: torch.device,
) -> dict:
	"""
	Convert Isaac Lab observation (obs_dict["policy"]) into the dict of
	tensors expected by LeRobot's ACTPolicy.select_action.

	We:
	  - Look at policy.config.input_features.keys() to know which feature
		names the policy expects.
	  - For features whose name contains "image" or "images", we interpret
		them as RGB images and:
			HWC uint8/float -> CHW float in [0,1], add batch dim.
	  - For others, we treat them as vector features (e.g. "observation.state")
		and:
			(D,) -> (1, D) float32 on the correct device.

	IMPORTANT:
	  This assumes that obs_dict["policy"] already has keys like
	  "observation.state", "observation.images.front", ...
	  If your Isaac Lab observations use different names, replace the
	  src[...] access below with your own mapping.
	"""
	src = obs_dict["policy"]
	ee_6d = src["ee_6D_pos"].to(device).float()
	# (num_env, D2)
	base_qpos = src["base_qpos"].to(device).float()

	# (num_env, D1 + D2)
	observation_state = torch.cat([ee_6d, base_qpos], dim=-1)

	# This key name should match what your LeRobot dataset / policy expects
	src["observation.state"] = observation_state

	policy_obs = {}


	for name in policy.config.input_features.keys():
		if name not in src:
			# If some optional feature isn't provided by the env,
			# you can either skip or raise an error.
			# For now we just skip silently.
			continue

		value = src[name]
		if not isinstance(value, torch.Tensor):
			value = torch.as_tensor(value)

		value = value.to(device=device)

		if ("image" in name) or ("images" in name):
			# Image feature
			# Expect HWC in 0..255 or 0..1 -> CHW in 0..1
			if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
				# HWC -> CHW
				value = value.permute(2, 0, 1)
			value = value.to(torch.float32)
			# If coming in as 0..255, normalize to 0..1
			if value.max() > 1.1:
				value = value / 255.0
			# Add batch dim: (C,H,W) -> (1,C,H,W)
			if value.ndim == 3:
				value = value.unsqueeze(0)
		else:
			# Non-image feature, e.g. "observation.state"
			value = value.to(torch.float32)
			# Add batch dim if it's a single vector
			if value.ndim == 1:
				value = value.unsqueeze(0)

		policy_obs[name] = value
	return policy_obs

import numpy as np
import imageio.v2 as imageio
from pathlib import Path

# ---- before the rollout loop ----
video_dir = Path(f"videos_eval/{args_cli.version_name}")
video_dir.mkdir(parents=True, exist_ok=True)

front_frames = []
wrist_left_frames = []
wrist_right_frames = []

def _to_hwc_uint8(t):
	"""IsaacLab tensor -> HWC uint8 RGB."""
	x = t.detach().cpu().numpy()
	# If CHW, move channels to last
	if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
		x = np.moveaxis(x, 0, -1)
	# Drop alpha if present
	if x.shape[-1] == 4:
		x = x[..., :3]
	# [0,1] -> [0,255]
	if x.dtype != np.uint8:
		if x.max() <= 1.0:
			x = (x * 255.0).clip(0, 255).astype(np.uint8)
		else:
			x = x.clip(0, 255).astype(np.uint8)
	return x


def rollout(
	policy: ACTPolicy,
	env: gym.Env,
	success_term,
	horizon: int,
	device: torch.device,
) -> bool:
	"""
	Single rollout of one ACT policy in one IsaacLab env.

	- Calls policy.reset() once at episode start
	- At every step:
		* build policy input dict from obs_dict["policy"]
		* call policy.select_action(...)
		* clip action to env.action_space and step.
	- Early stops when 'success' termination is triggered or env is done.

	Note: ACT outputs action chunks internally, but select_action() returns
	a single low-level action per call (it handles chunk caching), so we can
	just call it every timestep.
	"""
	policy.reset()
	obs_dict, _ = env.reset()
	rate_limiter = RateLimiter(args_cli.step_hz)	
	start = 0
	prev_l_xyz = 0
	prev_r_xyz = 0
	for _ in range(horizon):
		retry_mask = env.termination_manager.get_term("retry").clone()
		retry_idx = torch.where(retry_mask)[0]
		if torch.any(retry_mask):
			env.action_manager.get_term('armL_action')._ik_controller.reset(retry_mask)
			env.action_manager.get_term('armR_action')._ik_controller.reset(retry_mask)
			print(f"Reset Env {retry_idx} for OBB.")

		front_rgb		= env.scene.sensors["front"].data.output["rgb"][0]
		wrist_left_rgb	= env.scene.sensors["wrist_left"].data.output["rgb"][0]
		wrist_right_rgb = env.scene.sensors["wrist_right"].data.output["rgb"][0]

		# put into obs_dict for the policy (what you already had)
		obs_dict["policy"]["observation.images.front"]		 = front_rgb
		obs_dict["policy"]["observation.images.wrist_left"]  = wrist_left_rgb
		obs_dict["policy"]["observation.images.wrist_right"] = wrist_right_rgb
		front_frames.append(_to_hwc_uint8(front_rgb))
		wrist_left_frames.append(_to_hwc_uint8(wrist_left_rgb))
		wrist_right_frames.append(_to_hwc_uint8(wrist_right_rgb))
		obs_for_policy = build_policy_observation(obs_dict, policy, device)
		print(obs_for_policy["observation.state"][0][-3:])
		with torch.inference_mode():
			action = policy.select_action(obs_for_policy)
		import torch.nn.functional as F
		from isaaclab.utils.math import quat_from_matrix, quat_mul

		def rot6d_to_quat_wxyz(rot6d: torch.Tensor) -> torch.Tensor:
			"""
			Inverse of your quat -> 6D pipeline.

			rot6d: (..., 6)  from:
				R = quat_to_R(quat)
				rot6d = R[:, :, :2].reshape(B, 6)

			returns: (..., 4) quaternion in (w, x, y, z)
			"""
			orig_shape = rot6d.shape[:-1]  # everything except last dim
			rot6d_flat = rot6d.reshape(-1, 6)  # (B, 6)
			B = rot6d_flat.shape[0]

			# Recover first 2 columns of rotation matrix
			R_2 = rot6d_flat.view(B, 3, 2)		# (B, 3, 2)
			a1 = R_2[:, :, 0]					# (B, 3)
			a2 = R_2[:, :, 1]					# (B, 3)

			# GramâSchmidt to make them orthonormal
			b1 = F.normalize(a1, dim=-1)
			proj = (b1 * a2).sum(-1, keepdim=True)
			a2_orth = a2 - proj * b1
			b2 = F.normalize(a2_orth, dim=-1)
			b3 = torch.cross(b1, b2, dim=-1)

			# Rebuild rotation matrix with these as **columns**
			R = torch.stack([b1, b2, b3], dim=-1)	# (B, 3, 3)

			# Convert rotation matrix -> quaternion (w, x, y, z)
			quat = quat_from_matrix(R)	# (B, 4)

			return quat.view(*orig_shape, 4)

		def quat_inv_unit(q):
			"""Inverse of a unit quaternion (w, x, y, z)."""
			w, x, y, z = q.unbind(-1)
			return torch.stack((w, -x, -y, -z), dim=-1)

		def quat_delta_axis_angle(q_t, q_tp1, eps=1e-6):
			"""
			q_t, q_tp1: (..., 4) unit quaternions (w, x, y, z)
			returns: (..., 3) axis-angle vector (length = angle)
			"""
			# ensure unit
			q_t = F.normalize(q_t, dim=-1)
			q_tp1 = F.normalize(q_tp1, dim=-1)

			# relative quaternion: q_delta â q_t = q_tp1
			q_delta = quat_mul(q_tp1, quat_inv_unit(q_t))

			# flip sign to take shortest rotation
			mask = q_delta[..., 0] < 0
			q_delta[mask] = -q_delta[mask]

			w = q_delta[..., 0].clamp(-1.0, 1.0)
			xyz = q_delta[..., 1:]

			angle = 2.0 * torch.acos(w)						   # (...,)
			sin_half = torch.sqrt(1.0 - w*w).clamp_min(eps)    # (...,)
			axis = xyz / sin_half.unsqueeze(-1)				   # (..., 3)

			rot_vec = axis * angle.unsqueeze(-1)			   # (..., 3)
			return rot_vec

		l_rot6d = action[..., 3:9]
		r_rot6d = action[..., 12:18]
		l_quat = rot6d_to_quat_wxyz(l_rot6d)
		r_quat = rot6d_to_quat_wxyz(r_rot6d)
		if start == 0:	
			d_l_xyz = torch.zeros(3, device=device).unsqueeze(0)
			d_r_xyz = torch.zeros(3, device=device).unsqueeze(0)
			d_l_rot = torch.zeros(3, device=device).unsqueeze(0)
			d_r_rot = torch.zeros(3, device=device).unsqueeze(0)

			start = 1
		else:
			d_l_xyz = action[..., 0:3] - prev_l_xyz
			d_r_xyz = action[..., 9:12] - prev_r_xyz
			d_l_rot = quat_delta_axis_angle(prev_l_quat, l_quat)
			d_r_rot = quat_delta_axis_angle(prev_r_quat, r_quat)
		prev_l_xyz = action[...,0:3]
		prev_r_xyz = action[...,9:12]
		prev_l_quat = l_quat
		prev_r_quat = r_quat

		l_gripper = action[..., 18:19]
		r_gripper = action[..., 19:20]
		print("action from policy: ",action[...,20:23])
		base = compute_wheel_velocities_torch(
				action[..., 20], action[..., 21], action[...,22],
				wheel_radius=0.1, l=0.23
		)
		import ipdb
		ipdb.set_trace()
		action = torch.cat(
			(d_l_xyz, d_l_rot, d_r_xyz, d_r_rot, l_gripper, r_gripper, base),
			dim=1,
		)	
		# select_action usually returns a Tensor of shape (B, action_dim)
		# but may return a dict; handle both.
		if isinstance(action, dict):
			# by convention, ACT policies use "action" key. If your
			# policy uses a different key, change this.
			if "action" in action:
				action = action["action"]
			else:
				# fallback: take first value
				action = next(iter(action.values()))

		if not isinstance(action, torch.Tensor):
			action = torch.as_tensor(action, device=device, dtype=torch.float32)

		# Handle potential ACT chunk dimension: (B, H, A) -> take first time step.
		# (If ACT already returns (B,A), this does nothing.)
		if action.ndim == 3:
			action = action[:, 0, :]

		# Expect (1, action_dim)
		if action.ndim == 1:
			action = action.unsqueeze(0)

		# Clip to env bounds
		low = torch.as_tensor(env.action_space.low, device=device).view(1, -1)
		high = torch.as_tensor(env.action_space.high, device=device).view(1, -1)
		action = torch.maximum(torch.minimum(action, high), low)

		# Step env
#		obs_dict, _, terminated, truncated, _ = env.step(action, torch.tensor([0], device=device))
#		# success_term is an Isaac Lab termination func + params
#		if bool(success_term.func(env, **success_term.params)[0]):
#			return True
#		if terminated or truncated:
#			return False
		obs_tuple = env.step(action, torch.tensor([0], device=device))
		obs_dict = obs_tuple[0]
		if rate_limiter:
			rate_limiter.sleep(env)
	fps = 30  # or env.step_dt ** -1 if you want exact sim FPS

	imageio.mimwrite(video_dir / "front.mp4",		front_frames,		fps=fps, codec="libx264")
	imageio.mimwrite(video_dir / "wrist_left.mp4",	wrist_left_frames,	fps=fps, codec="libx264")
	imageio.mimwrite(video_dir / "wrist_right.mp4", wrist_right_frames, fps=fps, codec="libx264")
	
	return False


# ----------------------- Evaluation -----------------------
def evaluate_policy_on_setting(
	policy: ACTPolicy,
	env: gym.Env,
	device: torch.device,
	success_term,
	num_rollouts: int,
	horizon: int,
	seed: int,
	output_file: str,
	setting_name: str,
) -> float:
	"""
	Evaluate a single ACT policy under one eval setting (lighting/texture).
	"""
	# Seeds
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	# Newer gym API prefers env.reset(seed=...), but IsaacLab envs
	# often also support env.seed().
	if hasattr(env, "seed"):
		env.seed(seed)

	results = []
	for trial in range(num_rollouts):
		print(f"[Setting: {setting_name}] Trial {trial}")
		success = rollout(policy, env, success_term, horizon, device)
		results.append(success)
		with open(output_file, "a") as f:
			f.write(
				f"[Setting: {setting_name}] Trial {trial}: {success}\n"
			)

	success_rate = results.count(True) / len(results)

	with open(output_file, "a") as f:
		f.write(
			f"[Setting: {setting_name}] Successful trials: "
			f"{results.count(True)} / {len(results)}\n"
		)
		f.write(f"[Setting: {setting_name}] Success rate: {success_rate}\n")
		f.write(f"[Setting: {setting_name}] Results: {results}\n")
		f.write("-" * 80 + "\n\n")

	print(
		f"\n[Setting: {setting_name}] Successful trials: "
		f"{results.count(True)} / {len(results)}"
	)
	print(f"[Setting: {setting_name}] Success rate: {success_rate}\n")
	print(f"[Setting: {setting_name}] Results: {results}\n")

	return success_rate


def main() -> None:
	# -------------- Isaac Lab env config --------------
	env_cfg = parse_env_cfg(
		args_cli.task,
		device=args_cli.device,
		num_envs=1,
		use_fabric=not args_cli.disable_fabric,
	)

	# Obs in dict mode (like your previous robomimic-style script)
	env_cfg.observations.policy.concatenate_terms = False

	# Terminations & eval mode
	env_cfg.terminations.time_out = None
	env_cfg.recorders = None
	success_term = env_cfg.terminations.success
	env_cfg.terminations.success = None
	env_cfg.eval_mode = True

	file_path = f'/root/IsaacLab/scripts/simvla/goals/{args_cli.task}.json'

	with open(file_path, 'r') as file:
		# Load the JSON data directly from the file object
		data = json.load(file)
		if args_cli.version_name == "in_distribution":
			from datasets import load_dataset
			from isaaclab.utils.math import euler_xyz_from_quat
			repo_id = f"exaFLOPs09/{args_cli.task}"
			print(f"Loading dataset: {repo_id}")
			ds = load_dataset(repo_id, split="train")
			initial_pose = np.array(ds["initial_pose"])
			x, y, quat = random.choice(ds["initial_pose"])[0], random.choice(ds["initial_pose"])[1], random.choice(ds["initial_pose"])[2:]
			r,p, yaw = euler_xyz_from_quat(torch.tensor(quat, device='cuda:0').unsqueeze(0))

			env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (x,x+0.0001)
			env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (y,y+0.0001)
			env_cfg.events.robot_init_pos.params["pose_range"]["yaw"] = (yaw,yaw+0.0001)
		elif args_cli.version_type == "out_of_distribution":
			# 1. Init pos
			init_pos = random.choice(data["initial_pos_ranges"])
			env_cfg.events.robot_init_pos.params["pose_range"]["x"] = (init_pos[0][1], init_pos[0][2])
			env_cfg.events.robot_init_pos.params["pose_range"]["y"] = (init_pos[1][1], init_pos[1][2])
			# 2. Init rot
			init_rot = data["initial_rot_yaw_range"]
			env_cfg.events.robot_init_pos.params["pose_range"]["yaw"] = (init_rot[0][1], init_rot[0][2])

	
	# Create env
	env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
	env.reset()
	# Device
	if args_cli.device is not None:
		device = torch.device(args_cli.device)
	else:
		device = TorchUtils.get_torch_device(try_to_use_cuda=True)

	# Load ACT policy from HF or local path
	print(f"Loading ACT policy from: {args_cli.policy_path}")
	policy = ACTPolicy.from_pretrained(
		args_cli.policy_path,
	)
	policy.to(device)
	policy.eval()
	print("Policy loaded. Input features:", policy.config.input_features.keys())
	print("Policy output features:", policy.config.output_features.keys())

	# Seeds
	if args_cli.seeds is None:
		seeds = random.sample(range(0, 10000), args_cli.num_seeds)
	else:
		seeds = args_cli.seeds

	# Eval settings (match your earlier script)
	settings = [
		"vanilla",
		"light_intensity",
		"light_color",
		"light_texture",
		"table_texture",
		"robot_texture",
		"all",
	]

	# Logs
	os.makedirs(args_cli.log_dir, exist_ok=True)

	# Per-seed evaluation
	for seed in seeds:
		output_path = os.path.join(
			args_cli.log_dir, f"{args_cli.log_file}_seed_{seed}"
		)
		path = pathlib.Path(output_path)
		path.parent.mkdir(parents=True, exist_ok=True)

		results_summary = {}

		with open(output_path, "w") as f:
			f.write(f"Policy path: {args_cli.policy_path}\n")
			f.write(f"Task: {args_cli.task}\n")
			f.write(f"Seed: {seed}\n")
			f.write("=" * 80 + "\n\n")

		for setting in settings:
			env.cfg.eval_type = setting
			print(f"Evaluation setting: {setting}")
			print("=" * 80)

			with open(output_path, "a") as f:
				f.write(f"Evaluation setting: {setting}\n")
				f.write("=" * 80 + "\n\n")

			sr = evaluate_policy_on_setting(
				policy=policy,
				env=env,
				device=device,
				success_term=success_term,
				num_rollouts=args_cli.num_rollouts,
				horizon=args_cli.horizon,
				seed=seed,
				output_file=output_path,
				setting_name=setting,
			)
			results_summary[setting] = sr

			env.reset()

		# Write summary per seed
		with open(output_path, "a") as f:
			f.write("\nSummary (success rate per setting):\n")
			for setting, sr in results_summary.items():
				f.write(f"{setting}: {sr}\n")
			if len(results_summary) > 0:
				best_setting = max(
					results_summary, key=results_summary.get
				)
				f.write(
					f"\nBest setting: {best_setting} "
					f"with success rate {results_summary[best_setting]}\n"
				)

	env.close()


if __name__ == "__main__":
	main()
	simulation_app.close()

