import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TrafficLightEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super(TrafficLightEnv, self).__init__()
        self.num_lanes = 4
        self.max_cars = 20
        self.cars_per_pass = 5
        self.render_mode = render_mode

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_lanes)
        self.observation_space = spaces.Box(low=0, high=self.max_cars, shape=(self.num_lanes,), dtype=np.int32)

        # Initialize the state (number of cars in each lane)
        self.state = np.random.randint(0, self.max_cars + 1, size=(self.num_lanes,))

        # Track the last lane chosen
        self.last_lane_chosen = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, self.max_cars + 1, size=(self.num_lanes,))
        self.last_lane_chosen = None
        info = {}
        return self.state, info

    def step(self, action):
        reward = 0.0

        # Penalize for choosing a lane with 0 cars
        if self.state[action] == 0:
            reward -= 5.0  # Penalty for choosing an empty lane
        else:
            # Apply the action and manage the cars in the selected lane
            cars_before_action = self.state[action]
            if cars_before_action >= self.cars_per_pass:
                self.state[action] -= self.cars_per_pass
            else:
                self.state[action] = 0

            if cars_before_action > 0 and self.state[action] == 0:
                reward += 10.0

            reward -= 1.0  # Step penalty

            if np.all(self.state == 0):
                reward += 100  # Reward for clearing all lanes

        # Update the last lane chosen
        self.last_lane_chosen = action

        terminated = np.all(self.state == 0)
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"State: {self.state}")