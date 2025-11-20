"""Kitchen"""
  
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Kitchen-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_01:AnubisKitchenEnvCfg",
    }
)
gym.register(
    id="Isaac-Kitchen-v01-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_01_01:AnubisKitchenEnvCfg",
    }
)
gym.register(
    id="Isaac-Kitchen-v926-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_926_10:AnubisKitchenEnvCfg",
    }
)

gym.register(
    id="Isaac-Kitchen-v100-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_100_10:AnubisKitchenEnvCfg",
    }
)
gym.register(
    id="Isaac-Kitchen-v13",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_13:AnubisKitchenEnvCfg",
    }
)

gym.register(
    id="Isaac-Kitchen-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg2",
    }
)


gym.register(
    id="Isaac-Kitchen-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg3",
    }
)

gym.register(
    id="Isaac-Kitchen-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg4",
    }
)

gym.register(
    id="Isaac-Kitchen-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg5",
    }
)

gym.register(
    id="Isaac-Kitchen-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg2",
    }
)

gym.register(
    id="Isaac-Kitchen-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg7",
    }
)

gym.register(
    id="Isaac-Kitchen-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg8",
    }
)

gym.register(
    id="Isaac-Kitchen-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg9",
    }
)

gym.register(
    id="Isaac-Kitchen-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
	kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_env_cfg:AnubisKitchenEnvCfg10",
    }
)

gym.register(
    id="Isaac-Kitchen-v101-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v101-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_101_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v45-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_45_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v31-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_31_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v109-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_109_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v102-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_102_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1014-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1014_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1020-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1020_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1021-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1021_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v10212-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_10212_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1024-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1024_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v77-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_77_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1025-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1025_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1207-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1207_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1030-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1030_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1102-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1102_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v112-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_112_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1103-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1103_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1117-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1117_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v11171-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_11171_11:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-00",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_00:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-01",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_01:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-02",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_02:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_03:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-04",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_04:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_05:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-06",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_06:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-07",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_07:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-08",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_08:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-09",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_09:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_10:AnubisKitchenEnvCfg",
    },
)

gym.register(
    id="Isaac-Kitchen-v1119-11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_1119_11:AnubisKitchenEnvCfg",
    },
)
