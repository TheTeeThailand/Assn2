import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import aisd_examples.envs.register_envs

# Q-learning settings
alpha = 0.1         # Learning rate
gamma = 0.99        # Discount factor
epsilon = 0.1       # Exploration rate
num_episodes = 20   # Total episodes
max_steps = 100     # Max steps per episode

# Create the environment
env = gym.make("CreateRedBall-v0")

# Observation and action space size
obs_size = env.observation_space.n
act_size = env.action_space.n

# Q-table initialization
Q = np.zeros((obs_size, act_size))

# Track returns for plotting
returns = []

for episode in range(num_episodes):
    print(f"📦 Episode {episode + 1}/{num_episodes}")
    obs, _ = env.reset()

    # If no ball → stop
    if isinstance(obs, int) and obs == -1:
        state = 0  # fallback safe state
        print("⚠️ Red ball not detected at reset.")
    else:
        state = int(obs)

    total_reward = 0

    for step in range(max_steps):
        print(f"[STEP {step + 1}]")

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(act_size)
        else:
            action = np.argmax(Q[state])

        obs, reward, terminated, truncated, info = env.step(action)

        if isinstance(obs, int) and obs == -1:
            next_state = state  # hold position if no ball
            print(f"❌ Ball lost → holding state {state}")
        else:
            next_state = int(obs)

        # Q-learning update
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])
        Q[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        total_reward += reward

        print(f"🤖 EP {episode + 1} state={state}, action={action}, reward={reward:.2f} → next_state={next_state}")
        state = next_state

        if terminated or truncated:
            break

    returns.append(total_reward)
    print(f"✅ Episode {episode + 1}/{num_episodes} Return = {total_reward}\n")

# Save learning curve
plt.plot(returns)
plt.title("Q-learning Episode Returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid()
plt.savefig("qlearning_returns.png")
print("📈 Saved qlearning_returns.png")

env.close()

