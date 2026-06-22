"""A2C reference tracking for the Unbalanced Disk — Assignment 4.2.2

Extends the swing-up policy to track a reference angle within ±15° of the
upright position using a single policy (no switching controllers).

Key differences from a2c_unbalanced_disk_train.py:
  - Observation extended to [sin(θ), cos(θ), ω, sin(θ_ref), cos(θ_ref)]
  - theta_ref is randomised at each episode reset within ±REF_RANGE_DEG of π
  - Reward centred on theta_ref instead of π so the policy tracks the reference
"""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import sys
import time
from datetime import datetime   
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


SEED = 7
EPISODE_STEPS = 300
DEFAULT_SIMULATION_STEPS = 1200
ALGORITHM = "A2C"  # switch to "A2C" to use the original A2C setup
OBS_NOISE_OMEGA = 0.3  # rad/s std dev added to obs[2] during sim training

# Reference range: ±15° around the upright (π rad)
REF_RANGE_DEG = 15.0
REF_RANGE_RAD = np.deg2rad(REF_RANGE_DEG)

REWARD_WEIGHTS = {
    "swing_up":          0.5,
    "capture":           12.0,
    "top_speed_penalty": 1.0,
    "omega_penalty":     0.02,
    "action_penalty":    0.002,
    "stall_penalty":     0.5,
}

REWARD_PARAMS = {
    "balance_sigma_angle": 0.15,
    "balance_sigma_omega": 1.5,
    "top_gate_sigma":      0.25,
    "top_speed_sigma":     8.0,
    "omega_penalty_denom": 10.0,
    "stall_omega_sigma":   0.75,
}

A2C_HYPERPARAMS = {
    "seed":          SEED,
    "episode_steps": EPISODE_STEPS,
    "learning_rate": 7e-4,
    "n_steps":       32,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "net_arch_pi":   [64, 64],
    "net_arch_vf":   [64, 64],
}

PPO_HYPERPARAMS = {
    "seed":          SEED,
    "episode_steps": EPISODE_STEPS,
    "learning_rate": 2e-4,
    "n_steps":       1200,
    "batch_size":    120,
    "n_epochs":      8,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.08,
    "target_kl":     0.02,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "net_arch_pi":   [128, 128],
    "net_arch_vf":   [128, 128],
}

MODEL_PATHS = {
    "A2C": Path("ref_models/a2c_ref_track_v1"),
    "PPO": Path("ref_models/ppo_ref_track_noise_01"),
}


def default_hyperparams(algorithm: str | None = None) -> dict:
    algorithm = (algorithm or ALGORITHM).upper()
    if algorithm == "A2C":
        return A2C_HYPERPARAMS.copy()
    if algorithm == "PPO":
        return PPO_HYPERPARAMS.copy()
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def default_model_path(algorithm: str | None = None) -> Path:
    algorithm = (algorithm or ALGORITHM).upper()
    if algorithm not in MODEL_PATHS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return MODEL_PATHS[algorithm]


def set_algorithm(algorithm: str) -> tuple[Path, dict]:
    """Switch algorithm and restore its default model path/hyperparameters."""
    global ALGORITHM, MODEL_PATH, TRAIN_HYPERPARAMS
    ALGORITHM = algorithm.upper()
    MODEL_PATH = default_model_path(ALGORITHM)
    TRAIN_HYPERPARAMS = default_hyperparams(ALGORITHM)
    return MODEL_PATH, TRAIN_HYPERPARAMS


MODEL_PATH = default_model_path()
TRAIN_HYPERPARAMS = default_hyperparams()

ROBUSTNESS = {
    "action_delay":    0,
    "obs_noise_omega": OBS_NOISE_OMEGA,
}


# ── path setup ────────────────────────────────────────────────────────────────
def add_repo_to_path() -> None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "gym_unbalanced_disk").exists():
            sys.path.insert(0, str(parent))
            return


add_repo_to_path()
import gym_unbalanced_disk  # noqa: E402,F401


# ── helpers ───────────────────────────────────────────────────────────────────
def wrap_angle(theta: float) -> float:
    """Map any angle to [-π, π]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


# ── reward ────────────────────────────────────────────────────────────────────
def external_ref_reward(
    theta: float, omega: float, action: float, umax: float, theta_ref: float
) -> float:
    """Reward for reference tracking.

    theta_ref is the target angle for this episode (π ± up to 15°).
    The swing-up term still uses raw height so the policy climbs from the bottom.
    The capture / speed terms are centred on theta_ref so the policy settles there.
    """
    theta_wrapped = wrap_angle(theta)

    # General height reward — same as swing-up policy, reference-agnostic
    # Keeps pushing the pendulum toward the top region during swing-up
    swing_up_reward = ((1.0 - np.cos(theta_wrapped)) / 2.0) ** 2

    # Deviation from the reference (0 when exactly at theta_ref)
    ref_error = abs(wrap_angle(theta - theta_ref))

    # Capture bonus: rewards being near theta_ref AND slow
    top_gate      = np.exp(-(ref_error**2) / (2 * REWARD_PARAMS["top_gate_sigma"]**2))
    capture_bonus = (np.exp(-(ref_error**2) / (2 * REWARD_PARAMS["balance_sigma_angle"]**2))
                     * np.exp(-(omega**2) / (2 * REWARD_PARAMS["balance_sigma_omega"]**2)))

    top_speed_penalty = top_gate * (omega / REWARD_PARAMS["top_speed_sigma"]) ** 2
    omega_penalty     = (omega / REWARD_PARAMS["omega_penalty_denom"]) ** 2
    action_penalty    = (action / umax) ** 2

    # Stall penalty uses distance from π (not theta_ref) to force swing-up
    upright_error = np.pi - abs(theta_wrapped)
    stall_penalty = (np.exp(-(omega**2) / (2 * REWARD_PARAMS["stall_omega_sigma"]**2))
                     * (upright_error / np.pi) ** 2)

    return (
        REWARD_WEIGHTS["swing_up"]             * swing_up_reward
        + REWARD_WEIGHTS["capture"]            * capture_bonus
        - REWARD_WEIGHTS["top_speed_penalty"]  * top_speed_penalty
        - REWARD_WEIGHTS["omega_penalty"]      * omega_penalty
        - REWARD_WEIGHTS["action_penalty"]     * action_penalty
        - REWARD_WEIGHTS["stall_penalty"]      * stall_penalty
    )


# ── environment ───────────────────────────────────────────────────────────────
class DiskRefTrackEnv(gym.Env):
    """Unbalanced disk with a randomised reference angle.

    Observation: [sin(θ), cos(θ), ω, sin(θ_ref), cos(θ_ref)]  — shape (5,)
    At each episode reset theta_ref is sampled uniformly from
    [π - REF_RANGE_RAD, π + REF_RANGE_RAD].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.025,
        umax: float = 3.0,
        max_episode_steps: int = EPISODE_STEPS,
        ref_range_rad: float = REF_RANGE_RAD,
        action_delay: int = ROBUSTNESS["action_delay"],
        obs_noise_omega: float = ROBUSTNESS["obs_noise_omega"],
    ):
        super().__init__()
        self.umax = umax
        self.ref_range_rad = ref_range_rad
        self.action_delay = action_delay
        self.obs_noise_omega = obs_noise_omega
        self.theta_ref = np.pi  # initialised to pure upright; reset overrides this

        self.env = gym.make(
            "unbalanced-disk-sincos-v0",
            dt=dt,
            umax=umax,
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,
        )

        original_reset = self.env.unwrapped.reset

        def reset_with_options(seed=None, options=None):
            return original_reset(seed=seed)

        self.env.unwrapped.reset = reset_with_options

        self.action_space = gym.spaces.Box(
            low=np.array([-umax], dtype=np.float32),
            high=np.array([umax], dtype=np.float32),
            dtype=np.float32,
        )
        # Extended observation: base (3,) + reference sin/cos (2,) = (5,)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1., -1., -40., -1., -1.], dtype=np.float32),
            high=np.array([1.,  1.,  40.,  1.,  1.], dtype=np.float32),
        )

        self._action_buf: collections.deque = collections.deque(
            [0.0] * (self.action_delay + 1), maxlen=self.action_delay + 1
        )

    def _extend_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append sin/cos of theta_ref to the base observation."""
        ref_obs = np.array(
            [np.sin(self.theta_ref), np.cos(self.theta_ref)], dtype=np.float32
        )
        obs = np.concatenate([base_obs, ref_obs])
        if self.obs_noise_omega > 0.0:
            obs[2] += np.random.normal(0.0, self.obs_noise_omega)
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Sample a new reference for this episode
        self.theta_ref = np.pi + np.random.uniform(-self.ref_range_rad, self.ref_range_rad)

        obs, info = self.env.reset(seed=seed, options=options)
        self._action_buf = collections.deque(
            [0.0] * (self.action_delay + 1), maxlen=self.action_delay + 1
        )
        info["theta_ref"] = float(self.theta_ref)
        return self._extend_obs(obs.astype(np.float32)), info

    def step(self, action):
        u_commanded = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        self._action_buf.append(u_commanded)
        u_applied = self._action_buf[0]

        obs, env_reward, terminated, truncated, info = self.env.step(u_applied)

        theta = self.env.unwrapped.th
        omega = self.env.unwrapped.omega
        theta_wrapped = wrap_angle(theta)
        ref_error     = abs(wrap_angle(theta - self.theta_ref))
        upright_error = np.pi - abs(theta_wrapped)
        train_reward  = external_ref_reward(theta, omega, u_applied, self.umax, self.theta_ref)

        info = {
            **info,
            "theta":         float(theta),
            "theta_wrapped": float(theta_wrapped),
            "omega":         float(omega),
            "theta_ref":     float(self.theta_ref),
            "ref_error":     float(ref_error),        # deviation from reference
            "upright_error": float(upright_error),    # deviation from π
            "env_reward":    float(env_reward),
            "train_reward":  float(train_reward),
            "u_applied":     float(u_applied),
            "u_commanded":   float(u_commanded),
        }
        return self._extend_obs(obs.astype(np.float32)), float(train_reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


# ── training helpers ──────────────────────────────────────────────────────────
def make_env(
    seed: int = SEED,
    max_episode_steps: int = EPISODE_STEPS,
    obs_noise_omega: float = OBS_NOISE_OMEGA,
):
    def _init():
        env = DiskRefTrackEnv(
            max_episode_steps=max_episode_steps,
            obs_noise_omega=obs_noise_omega,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def build_model(env, device: str, entropy_coef: float):
    hp = TRAIN_HYPERPARAMS
    policy_kwargs = {
        "net_arch": {
            "pi": hp["net_arch_pi"],
            "vf": hp["net_arch_vf"],
        }
    }
    if ALGORITHM.upper() == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            seed=hp["seed"],
            verbose=0,
            learning_rate=hp["learning_rate"],
            n_steps=hp["n_steps"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            ent_coef=entropy_coef,
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            device=device,
        )
    if ALGORITHM.upper() != "PPO":
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")
    return PPO(
        "MlpPolicy",
        env,
        seed=hp["seed"],
        verbose=0,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        ent_coef=entropy_coef,
        clip_range=hp["clip_range"],
        target_kl=hp["target_kl"],
        vf_coef=hp["vf_coef"],
        max_grad_norm=hp["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        device=device,
    )


def load_model(model_path: Path, device: str):
    algorithm = ALGORITHM.upper()
    model_class = {"PPO": PPO, "A2C": A2C}.get(algorithm)
    if model_class is None:
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")

    try:
        return model_class.load(model_path, device=device)
    except (ModuleNotFoundError, ValueError, AttributeError) as exc:
        print(f"Standard model loading failed: {exc}")
        print("Retrying with compatibility overrides for saved NumPy/SB3 metadata.")
        return model_class.load(
            model_path,
            device=device,
            custom_objects={
                "action_noise": None,
                "_last_obs": None,
                "_last_episode_starts": None,
                "ep_info_buffer": None,
                "ep_success_buffer": None,
            },
        )


def rollout(
    model,
    max_steps: int,
    theta_ref_deg: float = 0.0,   # offset from upright in degrees (0 = pure upright)
    deterministic: bool = True,
    render: bool = False,
    obs_noise_omega: float = 0.0,
):
    """Run one episode with a fixed reference angle for evaluation."""
    env = DiskRefTrackEnv(max_episode_steps=max_steps, obs_noise_omega=obs_noise_omega)
    env.theta_ref = np.pi + np.deg2rad(theta_ref_deg)   # fix reference for eval
    obs, _ = env.reset(seed=SEED + 1)
    env.theta_ref = np.pi + np.deg2rad(theta_ref_deg)   # override after reset
    # rebuild obs with correct ref
    base = obs[:3]
    obs = env._extend_obs(base)

    rewards, actions, infos = [], [], []
    try:
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            actions.append(float(np.asarray(action).reshape(-1)[0]))
            infos.append(info)
            if render:
                env.render()
                time.sleep(1 / 50)
            if terminated or truncated:
                break
    finally:
        env.close()
    return np.asarray(rewards), np.asarray(actions), infos


# ── config save/load ──────────────────────────────────────────────────────────
def save_config(
    model_path: Path,
    entropy_coef: float,
    total_timesteps: int,
    training_time_s: float,
    rollout_summary: dict,
) -> Path:
    config = {
        "saved_at":        datetime.now().isoformat(timespec="seconds"),
        "model_path":      str(model_path) + ".zip",
        "algorithm":       ALGORITHM,
        "ref_range_deg":   REF_RANGE_DEG,
        "reward_weights":  REWARD_WEIGHTS,
        "reward_params":   REWARD_PARAMS,
        "reward_function": inspect.getsource(external_ref_reward),
        "robustness":      ROBUSTNESS,
        "hyperparameters": {**TRAIN_HYPERPARAMS, "entropy_coef": entropy_coef},
        "training": {
            "total_timesteps":  total_timesteps,
            "training_time_s":  round(training_time_s, 1),
            "steps_per_second": round(total_timesteps / training_time_s, 1),
        },
        "rollout_summary": rollout_summary,
    }
    config_path = Path(str(model_path) + "_config.json")
    config_path.write_text(json.dumps(config, indent=2))

    reward_path = Path(str(model_path) + "_reward.py")
    reward_path.write_text(
        f"# Reward function used to train {model_path}.zip\n"
        f"# Saved at {config['saved_at']}\n"
        f"# Weights: {REWARD_WEIGHTS}\n\n"
        + inspect.getsource(external_ref_reward)
    )
    return config_path


def load_config(model_path: Path) -> dict:
    config_path = Path(str(model_path) + "_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"No config found at {config_path}")
    return json.loads(config_path.read_text())


def print_config(config: dict) -> None:
    print(f"  saved at:       {config['saved_at']}")
    print(f"  model:          {config['model_path']}")
    print(f"  ref range:      ±{config.get('ref_range_deg', '?')}°")
    print(f"  reward weights: {config['reward_weights']}")
    print(f"  reward params:  {config['reward_params']}")
    print(f"  robustness:     {config['robustness']}")
    t = config["training"]
    print(f"  training:       {t['total_timesteps']:,} steps in {t['training_time_s']}s "
          f"({t['steps_per_second']} steps/s)")
    print(f"  rollout:        {config['rollout_summary']}")
    if "reward_function" in config:
        print("  reward function:")
        for line in config["reward_function"].splitlines():
            print(f"    {line}")
