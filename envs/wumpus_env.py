import gym
import numpy as np
import pygame
import os
from gym import spaces
from scipy.stats import entropy as scipy_entropy
import random

class WumpusCyberEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, mode='static', seed=None, pits=None, wumpus_pos=None, gold_pos=None):
        super().__init__()
        self.size = size
        self.mode = mode
        self.seed(seed)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32),
            'percepts': spaces.MultiBinary(3),
            'orientation': spaces.Discrete(4),
            'chaos': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        })

        self.custom_pits = pits
        self.custom_wumpus = wumpus_pos
        self.custom_gold = gold_pos

        pygame.init()
        self.cell_size = 100
        self.screen = pygame.display.set_mode((size * self.cell_size, size * self.cell_size))
        self.font = pygame.font.SysFont('Arial', 24)
        self._load_images()

        self.reset()

    def _load_images(self):
        asset_path = "assets"
        self.img_agent = pygame.image.load(os.path.join(asset_path, "agent.png"))
        self.img_wumpus = pygame.image.load(os.path.join(asset_path, "wumpus.png"))
        self.img_gold = pygame.image.load(os.path.join(asset_path, "gold.png"))
        self.img_pit = pygame.image.load(os.path.join(asset_path, "pit.png"))
        self.img_bg = pygame.image.load(os.path.join(asset_path, "background.png"))

        self.img_agent = pygame.transform.scale(self.img_agent, (80, 80))
        self.img_wumpus = pygame.transform.scale(self.img_wumpus, (80, 80))
        self.img_gold = pygame.transform.scale(self.img_gold, (80, 80))
        self.img_pit = pygame.transform.scale(self.img_pit, (80, 80))
        self.img_bg = pygame.transform.scale(self.img_bg, (self.size * self.cell_size, self.size * self.cell_size))

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self):
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.orientation = 0
        self.has_arrow = True
        self.has_gold = False
        self.chaos_params = np.array([0.0, 0.01], dtype=np.float32)
        self.gold_visible = True
        self.wumpus_alive = True

        if self.mode == 'static' or self.mode == 'dynamic':
            self.wumpus_pos = (self.size - 1, self.size - 1)
            self.gold_pos = (self.size - 2, self.size - 2)
            self.pits = [(1, 1), (self.size - 1, 1)]
        elif self.mode == 'random':
            cells = [(i, j) for i in range(self.size) for j in range(self.size) if not (i == 0 and j == 0)]
            self.np_random.shuffle(cells)
            self.wumpus_pos = cells[0]
            self.gold_pos = cells[1]
            self.pits = cells[2:4]
        else:
            self.wumpus_pos = self.custom_wumpus if self.custom_wumpus else (self.size - 1, self.size - 1)
            self.gold_pos = self.custom_gold if self.custom_gold else (self.size - 2, self.size - 2)
            self.pits = self.custom_pits if self.custom_pits else [(1, 1), (self.size - 1, 1)]

        return self._get_obs()

    def _get_obs(self):
        return {
            'position': self.agent_pos.copy(),
            'percepts': self._get_percepts(),
            'orientation': self.orientation,
            'chaos': self.chaos_params.copy()
        }

    def _get_percepts(self):
        x, y = self.agent_pos
        percepts = np.zeros(3, dtype=np.int32)
        adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        percepts[0] = any(pos in self.pits for pos in adjacent)
        if self.wumpus_alive and self.wumpus_pos:
            wx, wy = self.wumpus_pos
            w_adj = [(wx+1, wy), (wx-1, wy), (wx, wy+1), (wx, wy-1)]
            percepts[1] = int((x, y) in w_adj)
        percepts[2] = int(self.gold_visible and (x, y) == tuple(self.gold_pos))
        return percepts

    def step(self, action):
        reward = -1
        done = False
        info = {}

        if action < 4:
            self._move(action)
        elif action == 4:
            reward += self._shoot()
        elif action == 5:
            reward += self._grab_gold()

        if (tuple(self.agent_pos) == tuple(self.gold_pos)) and not self.has_gold:
            reward += self._grab_gold()
            print("Oro recogido automáticamente.")

        self._update_chaos(action)

        pos = tuple(self.agent_pos)
        if pos in self.pits:
            reward -= 1000
            done = True
            info['event'] = 'pit'
        elif self.wumpus_alive and self.wumpus_pos and pos == self.wumpus_pos:
            reward -= 1000
            done = True
            info['event'] = 'wumpus'
        elif self.has_gold and pos == (0, 0):
            reward += 2000
            done = True
            info['event'] = 'success'
            print("¡Victoria! Regresaste con el oro.")

        if self.mode == 'dynamic' and self.wumpus_alive:
            empty_cells = [
                (i, j) for i in range(self.size)
                for j in range(self.size)
                if (i, j) != tuple(self.agent_pos) and (i, j) not in self.pits and (i, j) != tuple(self.gold_pos)
            ]
            self.wumpus_pos = random.choice(empty_cells)

        obs = self._get_obs()
        return obs, reward, done, info

    def _update_chaos(self, action):
        probs = np.zeros(self.action_space.n, dtype=np.float32)
        probs[action] = 1.0
        self.chaos_params[0] = scipy_entropy(probs, base=2)
        x = float(self.chaos_params[1])
        r = 3.89
        self.chaos_params[1] = np.clip(r * x * (1 - x), 0.0, 1.0)

    def _move(self, direction):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[direction]
        new = self.agent_pos + np.array([dx, dy], dtype=np.int32)
        self.agent_pos = np.clip(new, 0, self.size - 1)
        self.orientation = direction

    def _shoot(self):
        if not self.has_arrow:
            return 0
        self.has_arrow = False
        x0, y0 = self.agent_pos
        if self.orientation == 0:
            traj = [(x0, y0 + i) for i in range(1, self.size)]
        elif self.orientation == 1:
            traj = [(x0 + i, y0) for i in range(1, self.size)]
        elif self.orientation == 2:
            traj = [(x0, y0 - i) for i in range(1, self.size)]
        else:
            traj = [(x0 - i, y0) for i in range(1, self.size)]
        if self.wumpus_pos in traj:
            self.wumpus_alive = False
            print("¡Has derrotado al Wumpus!")
            return 500
        return -50

    def _grab_gold(self):
        if (tuple(self.agent_pos) == tuple(self.gold_pos)) and not self.has_gold:
            self.has_gold = True
            self.gold_visible = False
            return 1000
        return 0

    def render(self, mode='human'):
        self.screen.blit(self.img_bg, (0, 0))
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (120, 100, 80), rect, 1)

        if self.wumpus_alive and self.wumpus_pos:
            self._blit_image(self.img_wumpus, self.wumpus_pos)
        if self.gold_visible:
            self._blit_image(self.img_gold, self.gold_pos)
        for pit in self.pits:
            self._blit_image(self.img_pit, pit)
        self._blit_image(self.img_agent, self.agent_pos)
        pygame.display.flip()

    def _blit_image(self, img, pos):
        x, y = pos
        self.screen.blit(img, (x * self.cell_size + 10, y * self.cell_size + 10))

    def close(self):
        pygame.quit()
