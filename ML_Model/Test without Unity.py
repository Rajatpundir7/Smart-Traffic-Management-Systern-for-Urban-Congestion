import numpy as np
from stable_baselines3 import DQN
from traffic_light_env import TrafficLightEnv  # Assuming the environment is saved in a file named traffic_light_env.py


def main():
    # Load the trained model
    model = DQN.load("dqn_traffic_light")

    # Initialize the environment
    env = TrafficLightEnv()

    # Take input for the number of cars in each lane
    cars_in_lanes = []
    for i in range(env.num_lanes):
        cars = int(input(f"Enter the number of cars in lane {i + 1} (0 to {env.max_cars}): "))
        cars_in_lanes.append(cars)

    # Set the initial state of the environment
    env.state = np.array(cars_in_lanes)

    done = False
    steps = 0

    while not done:
        # Predict the action using the trained model
        action, _states = model.predict(env.state)

        # Take a step in the environment
        state, reward, done, info = env.step(action)

        # Print the chosen action and the resulting state
        print(f"Action taken: Lane {action + 1} chosen")
        print(f"State after action: {state}")

        steps += 1

    print(f"Number of steps taken to reach the done condition: {steps}")


if __name__ == "__main__":
    main()