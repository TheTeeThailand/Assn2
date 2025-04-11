import matplotlib.pyplot as plt
import os

# Load return files (make sure these files exist)
def load_returns(filename):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return []
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Fallback for matplotlib .png
import matplotlib.image as mpimg

# Or load .png plots as reference
returns_files = {
    "Q-Learning": "qlearning_returns.png",
    "Non-RL": "non_rl_returns.png",
    "PPO": "ppo_returns.png",
    "DQN": "dqn_returns.png"
}

plt.figure(figsize=(10, 6))

for label, path in returns_files.items():
    try:
        img = mpimg.imread(path)
        plt.plot([], [], label=label)  # For legend placeholder
        print(f"✅ Loaded plot image: {path}")
    except Exception as e:
        print(f"⚠️ Could not load {path}: {e}")

# Optionally: add legend placeholder
plt.title("Return Comparison Across Agents")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.grid()
plt.savefig("summary_returns_comparison.png")
print("📈 Saved summary_returns_comparison.png")

