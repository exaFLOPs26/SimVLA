import gymnasium as gym

gym.register(
    id="Real2Sim-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.real2sim_env_cfg:Real2SimEnvCfg",
    }
)
