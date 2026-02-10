import threading
import time
import numpy as np
from stable_baselines3 import DQN
from traffic_light_env import TrafficLightEnv
from object_detection import run_object_detection, car_count_left, car_count_right

# Global variable to check if object detection is still running
object_detection_active = True

# Function to run traffic light model testing
def run_traffic_light_test():
    env = TrafficLightEnv(render_mode='human')
    model_path = "dqn_traffic_light"
    model = DQN.load(model_path, env=env)

    def simulate_initial_lane_values():
        return np.random.randint(0, env.max_cars + 1, size=2)

    lane_3_value, lane_4_value = simulate_initial_lane_values()
    state = np.array([car_count_left, car_count_right, lane_3_value, lane_4_value])

    last_add_time = time.time()

    while object_detection_active or not np.all(state[2:] == 0):  # Continue running as long as object detection is active or simulated lanes are not cleared
        action = model.predict(state, deterministic=True)[0]

        if action in [2, 3]:
            state[action] = max(state[action] - env.cars_per_pass, 0)
        cars_leaving = min(env.cars_per_pass, state[action])
        time.sleep(cars_leaving)

        if time.time() - last_add_time >= 10:
            state[2] += np.random.randint(0, 4)
            state[3] += np.random.randint(0, 4)
            last_add_time = time.time()

        state[0] = car_count_left
        state[1] = car_count_right

        # Debugging output to check if car_count_left and car_count_right are updated
        print(f"Lane 1 (Left): {state[0]}, Lane 2 (Right): {state[1]}")

        if np.all(state[2:] == 0) and not object_detection_active:
            print("All simulated lanes cleared and object detection ended!")
            break

        print(f"Chosen lane: {action + 1}, State: {state}")
        time.sleep(1)

# Function to monitor object detection status
def monitor_object_detection():
    global object_detection_active
    run_object_detection()
    object_detection_active = False

# Start the object detection in a separate thread
threading.Thread(target=monitor_object_detection).start()

# Run the traffic light testing in the main thread
run_traffic_light_test()
