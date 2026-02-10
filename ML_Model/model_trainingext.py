# continue_train_dqn_traffic_light.py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from traffic_light_env import TrafficLightEnv  # Import your environment

# Create and wrap the environment with render_mode='human'
env = TrafficLightEnv(render_mode='human')
env = make_vec_env(lambda: env, n_envs=1)

# Load the existing model
model_path = "dqn_traffic_light"
try:
    model = DQN.load(model_path, env=env)
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    # If the model does not exist, initialize a new model
    print(f"No existing model found at {model_path}, creating a new model")
    model = DQN('MlpPolicy', env, verbose=1)

# Continue training the model
model.learn(total_timesteps=100000)

# Save the updated model
model.save(model_path)
print(f"Model saved to {model_path}")

# Optional: Load the model again to verify saving was successful
loaded_model = DQN.load(model_path)