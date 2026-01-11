# 🐍 Snake Reinforcement Learning (DQN)

A reinforcement learning project where an AI agent learns to play a small-grid Snake game using **Deep Q-Learning (DQN)** with experience replay and target networks.

The environment is implemented in **pygame**, and the agent is trained using **PyTorch**.

---

## 🚀 Features

- DQN with:
  - Experience Replay Buffer
  - Target Network Updates
  - SmoothL1 Loss (Huber)
  - Adam Optimizer
- Checkpoint loading & saving:
  - Resume training across sessions
  - Store weights & episode index
- Two Modes:
  - **Training Mode** → continues learning
  - **Evaluation Mode** → loads trained model & plays with `epsilon = 0`
- Visual Rendering Toggle:
  - `y` → watch the game play out
  - `n` → much faster, headless training
- Reward shaping:
  - `+10` for eating food
  - `-10` for death
  - `-0.01` step penalty
  - `+0.1` when moving closer to food
  - `-0.1` when moving farther from food
- Checkpoint auto-saving every 10 episodes

---

## 📦 Dependencies

Create a Python venv first (recommended).

Install dependencies:

```bash
pip install torch pygame numpy
