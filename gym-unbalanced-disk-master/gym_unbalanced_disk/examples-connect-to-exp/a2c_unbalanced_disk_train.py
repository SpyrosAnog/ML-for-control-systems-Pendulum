from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


SEED = 7
EPISODE_STEPS = 300
DEFAULT_SIMULATION_STEPS = 1200
MODEL_PATH = Path("a2c_unbalanced_disk_long")
REWARD_WEIGHTS = {
    "swing_up": 0.5,
    "capture": 8.0,
    "top_speed_penalty": 1.0,
    "omega_penalty": 0.02,
    "action_penalty": 0.005,
    "stall_penalty": 0.5,
}


def add_repo_to_path() -> None:
    repo_root = Path.cwd()
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "gym_unbalanced_disk").exists():
            repo_root = parent
            break
    sys.path.insert(0, str(repo_root))


add_repo_to_path()
import gym_unbalanced_disk  # noqa: E402,F401 - registers the unbalanced disk Gym envs


def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def external_balance_reward(theta: float, omega: float, action: float, umax: float) -> float:
    theta_wrapped = wrap_angle(theta)
    upright_error = np.pi - abs(theta_wrapped)

    swing_up_reward = ((1.0 - np.cos(theta_wrapped)) / 2.0) ** 2
    top_gate = np.exp(-(upright_error**2) / (2 * 0.35**2))
    capture_bonus = np.exp(-(upright_error**2) / (2 * 0.25**2)) * np.exp(
        -(omega**2) / (2 * 1.5**2)
    )
    top_speed_penalty = top_gate * (omega / 8.0) ** 2
    omega_penalty = (omega / 10.0) ** 2
    action_penalty = (action / umax) ** 2
    stall_penalty = np.exp(-(omega**2) / (2 * 0.75**2)) * (upright_error / np.pi) ** 2

    return (
        REWARD_WEIGHTS["swing_up"] * swing_up_reward
        + REWARD_WEIGHTS["capture"] * capture_bonus
        - REWARD_WEIGHTS["top_speed_penalty"] * top_speed_penalty
        - REWARD_WEIGHTS["omega_penalty"] * omega_penalty
        - REWARD_WEIGHTS["action_penalty"] * action_penalty
        - REWARD_WEIGHTS["stall_penalty"] * stall_penalty
    )


class DiskSB3Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, dt: float = 0.025, umax: float = 3.0, max_episode_steps: int = EPISODE_STEPS):
        super().__init__()
        self.umax = umax
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
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32), info

    def step(self, action):
        u = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        obs, env_reward, terminated, truncated, info = self.env.step(u)

        theta = self.env.unwrapped.th
        omega = self.env.unwrapped.omega
        theta_wrapped = wrap_angle(theta)
        upright_error = np.pi - abs(theta_wrapped)
        train_reward = external_balance_reward(theta, omega, u, self.umax)

        info = {
            **info,
            "theta": float(theta),
            "theta_wrapped": float(theta_wrapped),
            "omega": float(omega),
            "upright_error": float(upright_error),
            "env_reward": float(env_reward),
            "train_reward": float(train_reward),
        }
        return obs.astype(np.float32), float(train_reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(seed: int = SEED, max_episode_steps: int = EPISODE_STEPS):
    def _init():
        env = DiskSB3Env(max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def build_model(env, device: str, entropy_coef: float) -> A2C:
    return A2C(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        learning_rate=7e-4,
        n_steps=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            }
        },
        device=device,
    )


def rollout(model: A2C, max_steps: int, deterministic: bool = True, render: bool = False):
    env = DiskSB3Env(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=SEED + 1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--simulation-steps", type=int, default=DEFAULT_SIMULATION_STEPS)
    parser.add_argument("--progress", action="store_true", help="Show SB3 progress bar during training.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    total_timesteps = args.episodes * args.episode_steps
    print("=== A2C unbalanced disk training ===")
    print(f"env id: unbalanced-disk-sincos-v0")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device: {args.device}")
    print(f"episodes: {args.episodes}")
    print(f"episode steps: {args.episode_steps}")
    print(f"total timesteps: {total_timesteps:,}")
    print(f"entropy coefficient: {args.entropy_coef}")
    print(f"reward weights: {REWARD_WEIGHTS}")
    print("training...")

    train_env = DummyVecEnv([make_env(SEED, args.episode_steps)])
    model = build_model(train_env, args.device, args.entropy_coef)
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=args.progress)
    elapsed = time.perf_counter() - start
    model.save(MODEL_PATH)
    train_env.close()
    print(f"training time: {elapsed:.1f}s ({total_timesteps / elapsed:.1f} steps/s)")
    print(f"saved model: {MODEL_PATH}.zip")

    rewards, actions, infos = rollout(
        model,
        max_steps=args.simulation_steps if args.render else args.episode_steps,
        deterministic=True,
        render=args.render,
    )
    upright_errors = np.array([info["upright_error"] for info in infos])
    omegas = np.array([info["omega"] for info in infos])
    env_rewards = np.array([info["env_reward"] for info in infos])
    print("=== deterministic rollout ===")
    print(f"steps: {len(rewards)}")
    print(f"external return: {rewards.sum():.2f}")
    print(f"environment return: {env_rewards.sum():.2f}")
    print(f"mean upright error: {upright_errors.mean():.3f} rad")
    print(f"final upright error: {upright_errors[-1]:.3f} rad")
    print(f"max |omega|: {np.abs(omegas).max():.2f} rad/s")
    print(f"max |u|: {np.abs(actions).max():.2f} V")


if __name__ == "__main__":
    main()
