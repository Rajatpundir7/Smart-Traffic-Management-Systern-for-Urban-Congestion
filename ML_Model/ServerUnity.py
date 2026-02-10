import socket
import numpy as np
import time
from stable_baselines3 import DQN
from traffic_light_env import TrafficLightEnv  # Assuming the environment is saved in a file named traffic_light_env.py

def main():
    # Load the trained model
    model = DQN.load("dqn_traffic_light")

    # Initialize the environment
    env = TrafficLightEnv()

    # Setup socket communication
    HOST = '127.0.0.1'  # Localhost
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Waiting for connection from Unity...")
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)

            # Take input for the number of cars in each lane
            cars_in_lanes = []
            for i in range(4):  # Assuming there are 4 lanes
                cars = int(input(f"Enter the number of cars in lane {i + 1} (0 to {env.max_cars}): "))
                cars_in_lanes.append(cars)

            # Set the initial state of the environment
            env.state = np.array(cars_in_lanes)

            done = False

            while not done:
                # Predict the action using the trained model
                action, _states = model.predict(env.state)

                # Print the chosen action
                print(f"Action taken: Lane {action + 1} chosen")

                # Send the chosen lane to Unity
                conn.sendall(str(action + 1).encode())

                # Wait for 5 seconds before sending the next action
                time.sleep(3)

                # Take a step in the environment
                state, reward, done, info = env.step(action)

                # Update the state for the next prediction
                env.state = state

                # Check if the state has reached [0, 0, 0, 0]
                if np.array_equal(env.state, [0, 0, 0, 0]):
                    conn.sendall(b'TERMINATE')
                    print("State reached [0, 0, 0, 0]. Terminating...")
                    break

    print("Simulation ended.")

if __name__ == "__main__":
    main()
