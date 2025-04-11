import gymnasium as gym
import aisd_examples.envs.register_envs  # Ensure env registered
import matplotlib.pyplot as plt

env = gym.make("CreateRedBall-v0")

num_episodes = 20
max_steps = 100
returns = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):
        # Default action
        action = 2  # stop

        # Handle observation
        if isinstance(obs, int) or isinstance(obs, float):
            x = int(obs)
            if x == -1:
                print(f"[STEP {step+1}] ❌ No red ball detected for RL")
                action = 2  # stop
            else:
                if x < 300:
                    action = 1  # turn right
                elif x > 340:
                    action = 0  # turn left
                else:
                    action = 2  # stop
                print(f"[STEP {step+1}] ✅ Red ball detected for RL: x={x} → action={action}")
        else:
            print(f"[STEP {step+1}] ❓ Unexpected observation: {obs}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    returns.append(total_reward)
    print(f"📦 Baseline Episode {episode+1}: return = {total_reward:.2f}")

env.close()

# Plot results
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Non-RL Baseline Agent")
plt.grid()
plt.savefig("non_rl_returns.png")
print("📊 Saved non_rl_returns.png")

