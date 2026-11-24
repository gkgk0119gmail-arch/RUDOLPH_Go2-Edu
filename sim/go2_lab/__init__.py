"""3DOG Isaac Lab tasks: Go2 gait, posture and crouch walking."""

import gymnasium as gym

_TASKS = {
    "Go2-Factory-Walk": ("Go2FactoryWalkEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    "Go2-Factory-Walk-Play": ("Go2FactoryWalkEnvCfg_PLAY", "Go2FactoryWalkPPORunnerCfg"),
    "Go2-Posture-Walk": ("posture:Go2PostureWalkEnvCfg", "posture:Go2PostureWalkPPORunnerCfg"),
    "Go2-Posture-Walk-Play": ("posture:Go2PostureWalkEnvCfg_PLAY", "posture:Go2PostureWalkPPORunnerCfg"),
    "Go2-Crouch-Walk": ("crouch:Go2CrouchWalkEnvCfg", "crouch:Go2CrouchWalkPPORunnerCfg"),
    "Go2-Crouch-Walk-Play": ("crouch:Go2CrouchWalkEnvCfg_PLAY", "crouch:Go2CrouchWalkPPORunnerCfg"),
}

for _task_id, (_env_cfg, _agent_cfg) in _TASKS.items():
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env_cfg}" if ":" in _env_cfg else f"{__name__}.locomotion:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.{_agent_cfg}" if ":" in _agent_cfg else f"{__name__}.locomotion:{_agent_cfg}",
            "default_agent": "rsl_rl",
        },
    )
