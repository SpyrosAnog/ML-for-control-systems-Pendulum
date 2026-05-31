import time
import numpy as np
import gymnasium as gym
import gym_unbalanced_disk


class OldGymToGymnasiumWrapper(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        result = self.env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            return result

        obs = result
        info = {}
        return obs, info

    def step(self, action):
        result = self.env.step(action)

        if len(result) == 5:
            return result

        obs, reward, done, info = result

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info


raw_env = gym.make("unbalanced-disk-v0", disable_env_checker=True).unwrapped
env = OldGymToGymnasiumWrapper(raw_env)

obs, info = env.reset(seed=0)
print("Initial observation:", obs)

for k in range(200):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(k, obs, reward)

    env.render()
    time.sleep(1 / 24)

    if terminated or truncated:
        obs, info = env.reset()

env.close()