#!/bin/bash

echo "===================="
echo "🏁 Running Q-Learning"
echo "===================="
python3 qlearning.py

echo "===================="
echo "🔍 Running Non-RL"
echo "===================="
python3 non-rl.py

echo "===================="
echo "🚀 Running PPO"
echo "===================="
python3 ppo.py

echo "===================="
echo "🧠 Running DQN"
echo "===================="
python3 dqn.py

echo "===================="
echo "✅ All runs completed."
echo "===================="

