# main.py
import os
import random
from collections import deque
from enum import Enum

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


# ===========================
#   Game / Environment
# ===========================

GRID_SIZE = 5
CELL_SIZE = 32
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

SNAKE_COLOR = (80, 200, 80)
FOOD_COLOR = (200, 50, 50)
BG_COLOR = (20, 20, 20)
GRID_COLOR = (40, 40, 40)
TEXT_COLOR = (250, 250, 250)


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeEnv:
    """
    Snake environment with:
      - grid size 16x16
      - one apple at a time
      - state: flattened grid (0=empty, 1=snake, 2=food)
      - actions: 0=left, 1=straight, 2=right (relative turn)
    """

    def __init__(self, render: bool = True, fps: int = 60):
        self.render_enabled = render
        self.fps = fps

        if self.render_enabled:
            pygame.init()
            pygame.display.set_caption("Snake RL - 16x16")
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 24)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.score = 0
        self.game_over = False
        self.direction = Direction.RIGHT
        self.pending_direction = self.direction
        self.snake = []
        self.food = None
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = Direction.RIGHT
        self.pending_direction = self.direction
        self.score = 0
        self.game_over = False
        self._place_food()
        return self._get_state()

    def _place_food(self):
        all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        available = list(set(all_cells) - set(self.snake))
        self.food = random.choice(available) if available else None

    def _update_direction(self, action: int):
        """
        action: 0=left, 1=straight, 2=right relative to current direction
        """
        dx, dy = self.direction.value
        straight = (dx, dy)
        left = (-dy, dx)
        right = (dy, -dx)

        if action == 0:
            self.pending_direction = Direction(left)
        elif action == 1:
            self.pending_direction = Direction(straight)
        elif action == 2:
            self.pending_direction = Direction(right)

    def step(self, action: int):
        """
        Take one step in the environment using the given action.
        Returns: next_state, reward, done
        """
        if self.game_over:
            return self._get_state(), 0.0, True

        # Handle events so pygame window stays responsive
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # Update direction from action
        self._update_direction(action)
        self.direction = self.pending_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        done = False

        # Collision with wall
        if not (0 <= new_head[0] < GRID_SIZE and 0 <= new_head[1] < GRID_SIZE):
            self.game_over = True
            reward = -10.0
            done = True
            return self._get_state(), reward, done

        # Collision with self
        if new_head in self.snake:
            self.game_over = True
            reward = -10.0
            done = True
            return self._get_state(), reward, done

        # Move snake
        self.snake.insert(0, new_head)

        # Check food
        if self.food is not None and new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            # Move forward
            self.snake.pop()

        # Small living cost
        reward += -0.01

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Returns flattened grid: 0=empty, 1=snake, 2=food
        Shape: (GRID_SIZE * GRID_SIZE,)
        """
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for (x, y) in self.snake:
            grid[y, x] = 1.0
        if self.food is not None:
            fx, fy = self.food
            grid[fy, fx] = 2.0
        return grid.flatten()

    def draw_grid(self):
        if not self.render_enabled:
            return
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def render(self, extra_text: str = ""):
        """
        Render the game and optional HUD text.
        """
        if not self.render_enabled:
            return

        self.screen.fill(BG_COLOR)
        self.draw_grid()

        # Draw food
        if self.food is not None:
            fx, fy = self.food
            pygame.draw.rect(
                self.screen,
                FOOD_COLOR,
                (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw snake
        for (sx, sy) in self.snake:
            rect = pygame.Rect(sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, SNAKE_COLOR, rect)

        # Score
        score_surf = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        self.screen.blit(score_surf, (10, 10))

        # Extra HUD info (batch size, epsilon, etc.)
        if extra_text:
            info_surf = self.font.render(extra_text, True, TEXT_COLOR)
            self.screen.blit(info_surf, (10, 40))

        pygame.display.flip()
        self.clock.tick(self.fps)


# ===========================
#   DQN Model
# ===========================

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# ===========================
#   Replay Buffer
# ===========================

class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
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


# ===========================
#   DQN Agent
# ===========================

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        target_update_steps: int = 1000,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_array = np.array(state, dtype=np.float32).flatten()
        state_tensor = (
            torch.tensor(state_array, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        action = int(q_values.argmax(dim=1).item())
        return action

    def train_step(self):
        """
        One gradient step using a minibatch from replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        states = states.view(states.size(0), -1)
        next_states = next_states.view(next_states.size(0), -1)

        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ===========================
#   Training Loop + Checkpointing
# ===========================

CHECKPOINT_PATH = "snake_checkpoint.pt"


def save_checkpoint(agent: DQNAgent, episodes_trained: int):
    torch.save(
        {
            "policy_state": agent.policy_net.state_dict(),
            "target_state": agent.target_net.state_dict(),
            "episodes_trained": episodes_trained,
            "train_steps": agent.train_steps,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH} (episodes_trained={episodes_trained})")


def load_checkpoint(agent: DQNAgent):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found, starting from scratch.")
        return 0  # episodes_trained

    data = torch.load(CHECKPOINT_PATH, map_location=agent.device)
    agent.policy_net.load_state_dict(data["policy_state"])
    agent.target_net.load_state_dict(data["target_state"])
    agent.update_target_net()
    agent.train_steps = data.get("train_steps", 0)
    episodes_trained = data.get("episodes_trained", 0)
    print(
        f"Loaded checkpoint from {CHECKPOINT_PATH} "
        f"(episodes_trained={episodes_trained}, train_steps={agent.train_steps})"
    )
    return episodes_trained


def train():
    # Ask user if they want visuals
    choice = input("Enable visuals? (y/n): ").strip().lower()
    render = choice == "y"

    # episodes_to_run = how many *new* episodes this run will train
    episodes_to_run = 2000
    max_steps = 1000
    train_batches_per_step = 2

    env = SnakeEnv(render=render, fps=60)
    state_dim = env._get_state().shape[0]
    action_dim = 3  # left, straight, right

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=128,
        buffer_capacity=100_000,
        target_update_steps=1000,
    )

    # Try to resume from checkpoint
    episodes_trained_before = load_checkpoint(agent)
    # Global episode index will continue from here
    start_ep_idx = episodes_trained_before
    end_ep_idx = start_ep_idx + episodes_to_run

    # Epsilon schedule now uses the GLOBAL episode index, not just 0..episodes_to_run
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 300  # bigger = slower decay

    for ep_idx in range(start_ep_idx, end_ep_idx):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -1.0 * ep_idx / epsilon_decay
            )

            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            # multiple minibatch updates per environment step
            for _ in range(train_batches_per_step):
                agent.train_step()

            # HUD text showing batch size, epsilon, episode, score
            hud = (
                f"Ep:{ep_idx+1} | "
                f"Batch:{agent.batch_size}x{train_batches_per_step} | "
                f"Eps:{epsilon:.2f} | "
                f"Score:{env.score}"
            )

            if render:
                env.render(extra_text=hud)

            state = next_state
            total_reward += reward
            steps += 1

        print(
            f"Episode {ep_idx+1} | "
            f"Score={env.score} | "
            f"Reward={total_reward:.2f} | "
            f"Epsilon={epsilon:.3f}"
        )

        # Save checkpoint every 10 episodes
        if (ep_idx + 1) % 10 == 0:
            save_checkpoint(agent, episodes_trained=ep_idx + 1)

    # Final save at end of this run
    save_checkpoint(agent, episodes_trained=end_ep_idx)

    if env.render_enabled:
        pygame.quit()


if __name__ == "__main__":
    train()