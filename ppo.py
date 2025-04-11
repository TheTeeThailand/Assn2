import gymnasium as gym
import aisd_examples.envs.register_envs  # Ensure env registered
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

# Create and vectorize the environment
env_id = "CreateRedBall-v0"
vec_env = make_vec_env(env_id, n_envs=1)

# Create PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)
model.save("ppo_create_redball")
print("✅ PPO model saved: ppo_create_redball.zip")

# Evaluate the agent
eval_env = gym.make(env_id)
returns = []
num_episodes = 20

for ep in range(num_episodes):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        step += 1
        # Handle case where no red ball is detected (obs == -1)
        if isinstance(obs, int) or (hasattr(obs, "__len__") and len(obs) == 1 and obs[0] == -1):
            print(f"[STEP {step}] ❌ No red ball detected for RL")
            action = [2]  # stop
        else:
            action, _states = model.predict(obs)

        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    returns.append(total_reward)
    print(f"🎯 PPO Episode {ep+1}: return = {total_reward:.2f}")

# Plot the results
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("PPO Agent Performance")
plt.grid()
plt.savefig("ppo_returns.png")
print("📈 Saved ppo_returns.png")

