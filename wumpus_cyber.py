import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.stats import entropy

SEED = 1

seeds = [
    {
        'name': 'default',
        'size': 4,
        'pits': [(1, 1), (3, 1)],
        'wumpus_pos': (3, 3),
        'gold_pos': (2, 2),
        'agent_pos': (0, 0)
    },
    {
        'name': 'corners',
        'size': 4,
        'pits': [(0, 3), (3, 0)],
        'wumpus_pos': (3, 3),
        'gold_pos': (1, 1),
        'agent_pos': (0, 0)
    },
    {
        'name': 'clustered',
        'size': 4,
        'pits': [(1, 1), (1, 2)],
        'wumpus_pos': (1, 3),
        'gold_pos': (1, 0),
        'agent_pos': (0, 0)
    },
    {
        'name': 'random6x6',
        'size': 6,
        'pits': [(2, 5), (5, 1), (1, 4)],
        'wumpus_pos': (4, 3),
        'gold_pos': (0, 5),
        'agent_pos': (0, 0)
    }
]

class WumpusCyberEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed_config=None, mode='static'):
        super().__init__()
        self.seed_config = seed_config or {}
        self.size = seed_config.get('size', 4)
        self.mode = mode

        # Espacio de acciones: [↑, →, ↓, ←, Disparar, Agarrar]
        self.action_space = spaces.Discrete(6)

        # Espacio de observación
        self.observation_space = spaces.Dict({
            'position': spaces.Box(0, self.size - 1, (2,), int),
            'percepts': spaces.MultiBinary(3),  # [Brisa, Hedor, Brillo]
            'orientation': spaces.Discrete(4),
            'chaos': spaces.Box(0, 1, (2,))  # [Entropía, Lyapunov]
        })

        # Estado inicial
        self.reset()

        # Configuración PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.size * 100, self.size * 100))
        self.font = pygame.font.SysFont('Arial', 24)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Configurar posiciones estáticas
        self.agent_pos = np.array(self.seed_config.get('agent_pos', (0, 0)))
        self.wumpus_pos = self.seed_config.get('wumpus_pos',(2,3))
        self.gold_pos = np.array(self.seed_config.get('gold_pos', (2, 2))) #! Definir como array
        self.pits = self.seed_config.get('pits', [])

        # Estado interno
        self.orientation = 0
        self.has_arrow = True
        self.has_gold = False
        self.chaos_params = np.zeros(2)

        return self._get_obs(), {} 

    def _get_obs(self):
        return {
            'position': self.agent_pos,
            'percepts': self._get_percepts(),
            'orientation': self.orientation,
            'chaos': self.chaos_params
        }

    def _move_wumpus(self):
        if self.wumpus_pos is None:
            return  # Ya fue eliminado

        x, y = self.wumpus_pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)  # Para que el movimiento sea aleatorio

        for dx, dy in directions:
            new_pos = (x + dx, y + dy)

            # Verificar que no se salga del mapa
            if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
                continue

            # Verificar que no toque el oro, los pozos ni al agente
            if (tuple(new_pos) in self.pits or
                np.array_equal(new_pos, self.gold_pos) or
                np.array_equal(new_pos, self.agent_pos)):
                continue

            # Mover al Wumpus
            self.wumpus_pos = new_pos
            break  # Solo un movimiento por turno

    def _get_percepts(self):
        x, y = self.agent_pos
        percepts = np.zeros(3, dtype=int)

        # Brisa (pozos adyacentes)
        adjacent = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        percepts[0] = any(pos in self.pits for pos in adjacent)

        # Hedor (Wumpus adyacente)
        if self.wumpus_pos:
            wx, wy = self.wumpus_pos
            w_adjacent = [(wx + 1, wy), (wx - 1, wy), (wx, wy + 1), (wx, wy - 1)]
            percepts[1] = self.agent_pos.tolist() in w_adjacent

        # Brillo (oro en la posición actual)
        percepts[2] = np.array_equal(self.agent_pos, self.gold_pos)

        return percepts

    def step(self, action):
        reward = -1  # Costo por paso
        done = False
        info = {}

        # Ejecutar acción
        if action < 4:
            self._move(action)
        elif action == 4:
            reward += self._shoot()
        elif action == 5:
            reward += self._grab_gold()
        
        if self.mode != "static":
            self._move_wumpus()
        

        # Recoger oro automáticamente si está en la casilla
        if np.array_equal(self.agent_pos, self.gold_pos) and not self.has_gold:
            reward += self._grab_gold()  # ! Recoger automáticamente
            print("¡Oro recogido automáticamente! 💰")  # ! Debug

        # Actualizar parámetros caóticos
        self._update_chaos(action)

        # Verificar estados terminales
        if tuple(self.agent_pos) in self.pits:
            reward -= 1000
            done = True
            print("¡Caíste en un pozo! 💀")
        elif np.array_equal(self.agent_pos, self.wumpus_pos):
            reward -= 1000
            done = True
            print("¡El Wumpus te atrapó! 🐉")
        elif self.has_gold and np.array_equal(self.agent_pos, [0, 0]) and self.wumpus_pos is None:
            reward += 2000
            done = True
            print("¡Victoria con Wumpus eliminado! 🏆")

        return self._get_obs(), reward, done, info

    def _update_chaos(self, action):
        # Corregir cálculo de action_probs
        action_probs = np.zeros(6)
        action_probs[action] = 1.0  # Distribución one-hot válida
        self.chaos_params[0] = entropy(action_probs, base=2)  # Entropía en bits

        # Mapa logístico ajustado
        self.chaos_params[1] = 3.89 * self.chaos_params[1] * (1 - self.chaos_params[1])

    def _move(self, direction):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[direction]
        new_pos = self.agent_pos + [dx, dy]

        # Mantener dentro de los límites
        self.agent_pos = np.clip(new_pos, 0, self.size - 1)
        self.orientation = direction

    def _shoot(self):
        if not self.has_arrow:
            return 0

        self.has_arrow = False
        trajectory = {
            0: [tuple(self.agent_pos + [0, i]) for i in range(1, self.size)],
            1: [tuple(self.agent_pos + [i, 0]) for i in range(1, self.size)],
            2: [tuple(self.agent_pos - [0, i]) for i in range(1, self.size)],
            3: [tuple(self.agent_pos - [i, 0]) for i in range(1, self.size)]
        }[self.orientation]

        if self.wumpus_pos in trajectory:
            self.wumpus_pos = None
            return 500
        return -50

    def _grab_gold(self):
        if np.array_equal(self.agent_pos, self.gold_pos) and not self.has_gold:
            self.has_gold = True
            print("¡Oro recogido! 💰")  # ! Debug
            return 1000  # Recompensa por agarrar el oro
        return 0  # No hay oro para agarrar

    def render(self, mode='human'):
        self.screen.fill((30, 30, 30))  # Fondo oscuro

        # Dibujar cuadrícula
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * 100, y * 100, 100, 100)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

        # Dibujar elementos
        self._draw_element(self.wumpus_pos, (255, 0, 0), 'W')
        self._draw_element(self.gold_pos, (255, 215, 0), 'G')
        for pit in self.pits:
            self._draw_element(pit, (0, 0, 255), 'P')

        # Dibujar perceptos (brisa y hedor)
        # Brisa: casillas adyacentes a pozos
        breeze_color = (42, 46, 59)  # celeste claro semi-transparente
        for pit in self.pits:
            adj = [(pit[0] + 1, pit[1]), (pit[0] - 1, pit[1]), (pit[0], pit[1] + 1), (pit[0], pit[1] - 1)]
            for pos in adj:
                if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                    rect = pygame.Rect(pos[0] * 100, pos[1] * 100, 100, 100)
                    # Para transparencia hay que usar Surface aparte
                    s = pygame.Surface((100, 100), pygame.SRCALPHA)  
                    s.fill(breeze_color)  
                    self.screen.blit(s, rect.topleft)

        # Hedor: casillas adyacentes al Wumpus
        if self.wumpus_pos:
            stench_color = (59, 42, 42)  # naranja claro semi-transparente
            wx, wy = self.wumpus_pos
            adj = [(wx + 1, wy), (wx - 1, wy), (wx, wy + 1), (wx, wy - 1)]
            for pos in adj:
                if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size:
                    rect = pygame.Rect(pos[0] * 100, pos[1] * 100, 100, 100)
                    s = pygame.Surface((100, 100), pygame.SRCALPHA)
                    s.fill(stench_color)
                    self.screen.blit(s, rect.topleft)

        # Dibujar agente
        x, y = self.agent_pos
        center = (x * 100 + 50, y * 100 + 50)
        pygame.draw.circle(self.screen, (0, 255, 0), center, 30)

        pygame.display.flip()

    def _draw_element(self, pos, color, text):
        if pos is None:
            return
        x, y = pos
        rect = pygame.Rect(x * 100, y * 100, 100, 100)
        pygame.draw.rect(self.screen, color, rect)
        text_surf = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surf, (x * 100 + 40, y * 100 + 40))

    def close(self):
        pygame.quit()


class CyberneticAgent:
    def __init__(self, env, use_dqn=True):
        self.env = env
        self.use_dqn = use_dqn

        # Dimensión de entrada corregida (8 elementos)
        input_dim = 2 + 3 + 1 + 2  # position(2) + percepts(3) + orientation(1) + chaos(2)

        if self.use_dqn:
            self.q_net = self._build_dqn(input_dim)
            self.target_net = self._build_dqn(input_dim)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
            self.memory = deque(maxlen=10000)
        else:
            self.q_table = np.zeros((self.env.size ** 2 * 8 * 4 * 2, 6))

        # Parámetros cibernéticos
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.chaos_factor = 1.0

    def _build_dqn(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )

    def act(self, state):
        if np.random.random() < self.epsilon * self.chaos_factor:
            return np.random.randint(6)

        if self.use_dqn:
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                return torch.argmax(self.q_net(state_tensor)).item()
        else:
            state_idx = self._state_to_index(state)
            return np.argmax(self.q_table[state_idx])

    def _state_to_tensor(self, state):
        # Si es una tupla como (dict, dict), toma solo el primero
        if isinstance(state, tuple):
            state = state[0]

        position = state['position']           # np.array de 2 elementos
        percepts = state['percepts']           # np.array de 3 elementos
        orientation = np.array([state['orientation']])  # convertir a array para poder concatenar
        chaos = state['chaos']                 # np.array de 2 elementos

        # Concatenar todo en un solo array
        all_data = np.concatenate([position, percepts, orientation, chaos])
        
        # Convertir a tensor de tipo float
        return torch.FloatTensor(all_data)

    def _state_to_index(self, state):
        return hash((
            tuple(state['position']),
            tuple(state['percepts']),
            state['orientation'],
            tuple(state['chaos'])
        )) % self.q_table.shape[0]

    def update(self, state, action, reward, next_state, done):
        # Simplificar chaos_factor (eliminar dependencia de entropía)
        self.chaos_factor = 1.0  # !! Valor fijo para pruebas

        if self.use_dqn:
            self.memory.append((state, action, reward, next_state, done))
            self._replay()
        else:
            state_idx = self._state_to_index(state)
            next_idx = self._state_to_index(next_state) if not done else None
            self.q_table[state_idx, action] += 0.1 * (
                    reward + 0.95 * (np.max(self.q_table[next_idx]) if next_idx else 0) -
                    self.q_table[state_idx, action]
            )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _replay(self):
        if len(self.memory) < 512:
            return

        batch = random.sample(self.memory, 512)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Actualizar red neuronal
        states = torch.stack([self._state_to_tensor(s) for s in states])
        next_states = torch.stack([self._state_to_tensor(s) for s in next_states])

        current_q = self.q_net(states)[range(512), actions]
        next_q = self.target_net(next_states).max(1)[0].detach()
        targets = torch.FloatTensor(rewards) + 0.95 * next_q * (1 - torch.FloatTensor(dones))

        loss = nn.MSELoss()(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    env = WumpusCyberEnv(seed_config=seeds[SEED], mode='static')
    agent = CyberneticAgent(env, use_dqn=True)

    # Configurar pygame para actualizaciones más lentas
    clock = pygame.time.Clock()  # ! Controlar FPS

    # Estadísticas avanzadas
    recompensas = []
    victorias = []
    ventana = 50
    total_episodios = 1000

    plt.ion()  # Gráficos interactivos

    for episodio in range(total_episodios):
        estado = env.reset()
        done = False
        recompensa_episodio = 0
        victoria = 0

        print(f"\n--- EPISODIO {episodio} ---")  # ! Debug inicial

        while not done:
            # Manejar eventos y renderizar
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            env.render()
            clock.tick(4000)  # ! 20 FPS máximo

            # Obtener acción y ejecutar
            accion = agent.act(estado)
            
            prox_estado, recompensa, done, info = env.step(accion)

            # ! Debug por cada acción
            # print(f"Pos: {estado['position']} | Acción: {accion} | Recompensa: {recompensa}")

            agent.update(estado, accion, recompensa, prox_estado, done)
            recompensa_episodio += recompensa
            estado = prox_estado

            if done and recompensa > 1900:
                victoria = 1
                print("¡Victoria! 🏆")  # ! Debug de victoria

        # Actualizar estadísticas
        recompensas.append(recompensa_episodio)
        victorias.append(victoria)

        # Mostrar resumen cada 5 episodios
        if episodio % 5 == 0:
            avg_recompensa = np.mean(recompensas[-ventana:])
            tasa_victorias = np.mean(victorias[-ventana:]) * 100

            print(f"\n[Resumen Episodio {episodio}]")
            print(f"Recompensa Promedio: {avg_recompensa:.1f}")
            print(f"Tasa de Victorias: {tasa_victorias:.1f}%")
            print(f"Exploración: {agent.epsilon:.2f}")

    env.close()


if __name__ == "__main__":
    train()