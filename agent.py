# agent.py
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store one transition in the buffer.
        state / next_state: 1D or multi-dim (will be flattened later)
        action: int
        reward: float
        done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_steps: int = 1000,
        device: str = None,
    ):
        # Device (CPU/GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net in eval mode

        # Optimizer & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Step counter for target net updates
        self.train_steps = 0

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in replay memory."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def select_action(self, state, epsilon: float) -> int:
        """
        Epsilon-greedy action selection.
        state: np.array or list (any shape; will be flattened)
        """
        if random.random() < epsilon:
            # Explore
            return random.randrange(self.action_dim)

        # Exploit
        state_array = np.array(state, dtype=np.float32).flatten()
        state_tensor = torch.tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action

    def train_step(self):
        """
        Perform one gradient descent step on a batch from replay memory.
        Call this after every environment step once you have enough samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data yet

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        # Flatten if needed (handles grid inputs too)
        states = states.view(states.size(0), -1)
        next_states = next_states.view(next_states.size(0), -1)

        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

        # Loss
        loss = self.criterion(q_values, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.update_target_net()

    def update_target_net(self):
        """Copy weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save the policy network weights."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """Load weights into policy network and sync target network."""
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.update_target_net()