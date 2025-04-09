import gymnasium as gym
import aisd_examples

env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()

