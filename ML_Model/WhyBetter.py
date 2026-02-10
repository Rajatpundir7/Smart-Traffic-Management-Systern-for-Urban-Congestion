def traffic_management(lanes):
    steps = 0

    while any(lanes):  # Continue until all lanes are empty
        for i in range(len(lanes)):
            if not any(lanes):  # Check if all lanes are empty, then stop
                break
            print(f"Lane {i + 1} selected")
            if lanes[i] > 0:
                if lanes[i] >= 5:
                    lanes[i] -= 5
                else:
                    lanes[i] = 0
            steps += 1
            print(f"Cars left in lanes: {lanes}")

    print(f"Total steps taken: {steps}")


if __name__ == "__main__":
    lanes = []
    for i in range(4):
        while True:
            try:
                cars = int(input(f"Enter the number of cars in lane {i + 1} (max 20): "))
                if 0 <= cars <= 20:
                    lanes.append(cars)
                    break
                else:
                    print("Please enter a number between 0 and 20.")
            except ValueError:
                print("Please enter a valid number.")

    traffic_management(lanes)
