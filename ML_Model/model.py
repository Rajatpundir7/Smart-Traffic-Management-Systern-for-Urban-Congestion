# train_dqn_traffic_light.py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from traffic_light_env import TrafficLightEnv  # Import your environment

# Create and wrap the environment with render_mode='human'
env = TrafficLightEnv(render_mode='human')
env = make_vec_env(lambda: env, n_envs=1)

# Define the DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("dqn_traffic_light")
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from traffic_light_env import TrafficLightEnv  # Import your environment

# Create and wrap the environment with render_mode='human'
env = TrafficLightEnv(render_mode='human')
env = make_vec_env(lambda: env, n_envs=1)

# Define the DQN model with increased verbosity
model = DQN('MlpPolicy', env, verbose=2)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("dqn_traffic_light")
