# Wumpus World Adaptive Agent

This repo implements a reinforcement learning agent capable of navigating and solving a custom Wumpus World environment but combining environment simulation, cybernetic-inspired reward mechanisms, and Deep Q-Network (DQN) learning strategies to train an adaptive agent.

🔗 **Full repository available at:** [https://github.com/cristiannnnr/WumpusGame](https://github.com/cristiannnnr/WumpusGame)

---

## Code overview

### Environment 
- Gymnasium environment simulating the Wumpus World grid
  - Fully implements the Wumpus World environment based on a grid.
  - Configurable scenarios with different positions of pits, Wumpus, and gold.
  - Allows static and dynamic Wumpus modes.
  - Includes percepts: breeze (pits), stench (Wumpus proximity), and glitter (gold location).
  - Handles rendering using `pygame` for visual representation.

### Machine Learning Components
  - Implements the Deep Q-Network (DQN) algorithm.
  - Handles:
    - State encoding
    - Action selection
    - Experience replay
    - Q-Network architecture
  - Supports both tabular and DQN versions for experimentation.

### 📊 Experimentation & Training
  - Manages training loop, statistics collection, and agent learning.
  - Tracks average rewards and victory rates over time.
  - Renders live environment visuals.
  - Automatically plots training results after execution.

### Environment Initialization

The environment simulates a configurable Wumpus World grid, handling percepts and agent interaction:

```python
env = WumpusCyberEnv(seed_config=seeds[0], mode='static')
observation, _ = env.reset()
env.render()
```

---

## Running Experiments

### 1️⃣ Install Dependencies

Make sure you have Python 3.8+ installed. Then, install required packages:

```bash
pip install -r requirements.txt
```

### 2️⃣ Execute Training
You can directly run the training by executing:

```bash
python wumpus_cyber.py
```

- The agent will train for 100 episodes as default.

- The environment is visually rendered using pygame if you add --render.

- After training, two performance graphs are displayed:


### 👨‍💻 Author
Developed by Cristian Romero and Cesar Pulido.