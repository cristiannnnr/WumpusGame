import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class CyberneticAgent:
    def __init__(self, env, use_dqn=True):
        print("Cargando versión correcta de CyberneticAgent")
        self.env = env
        self.use_dqn = use_dqn

        input_dim = 8  # 2 pos + 3 percepts + 1 orientation + 2 chaos

        if self.use_dqn:
            self.q_net = self._build_dqn(input_dim)
            self.target_net = self._build_dqn(input_dim)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
            self.memory = deque(maxlen=10000)
            self.update_target_every = 20  # episodios
            self.episode_count = 0
        else:
            self.q_table = np.zeros((self.env.size ** 2 * 8 * 4 * 2, 6))

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

    def _build_dqn(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )

    def _state_to_tensor(self, state):
        pos = np.array(state['position'], dtype=np.float32) / self.env.size
        chaos = np.array(state['chaos'], dtype=np.float32)
        percepts = np.array(state['percepts'], dtype=np.float32)
        orientation = np.array([state['orientation'] / 3.0], dtype=np.float32)
        return torch.FloatTensor(np.concatenate([pos, percepts, orientation, chaos]))

    def _state_to_index(self, state):
        return hash((
            tuple(state['position']),
            tuple(state['percepts']),
            state['orientation'],
            tuple(state['chaos'])
        )) % self.q_table.shape[0]

    def act(self, state):
        chaos_level = state['chaos'][1]
        exploration_chance = self.epsilon * chaos_level

        if np.random.rand() < exploration_chance:
            return np.random.randint(6)

        if self.use_dqn:
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                return torch.argmax(self.q_net(state_tensor)).item()
        else:
            state_idx = self._state_to_index(state)
            return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, done):
        if self.use_dqn:
            self.memory.append((state, action, reward, next_state, done))
            loss = self._replay()

            if done:
                self.episode_count += 1
                if self.episode_count % self.update_target_every == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            s_idx = self._state_to_index(state)
            next_idx = self._state_to_index(next_state) if not done else None
            target = reward + self.gamma * np.max(self.q_table[next_idx]) if next_idx else reward
            self.q_table[s_idx, action] += 0.1 * (target - self.q_table[s_idx, action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss if self.use_dqn else 0

    def _replay(self):
        if len(self.memory) < 512:
            return 0

        batch = random.sample(self.memory, 512)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([self._state_to_tensor(s) for s in states])
        next_states = torch.stack([self._state_to_tensor(s) for s in next_states])

        current_q = self.q_net(states)[range(512), actions]
        next_q = self.target_net(next_states).max(1)[0].detach()
        targets = torch.FloatTensor(rewards) + self.gamma * next_q * (1 - torch.FloatTensor(dones))

        loss = nn.MSELoss()(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path="model.pt"):
        if self.use_dqn:
            torch.save(self.q_net.state_dict(), path)
            print(f"✅ Modelo guardado en {path}")

    def load(self, path="model.pt"):
        if self.use_dqn:
            self.q_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f"✅ Modelo cargado desde {path}")

    def reset_epsilon(self, value=0.0):
        self.epsilon = value
