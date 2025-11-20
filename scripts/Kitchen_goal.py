import json
import ast
import math
def collect_kitchen_data():
	"""
	Collects goal state data and initial position ranges for a kitchen setup.
	"""
	kitchen_name = input("1. What is the kitchen number? ")
	kitchen_data = {
		"initial_pos_ranges": [],
		"initial_rot_yaw_range": "",
		"goals": []
	}

	print("-" * 20)

	# Loop to collect multiple initial position ranges
	pos_range_num = 1
	while True:
		print(f"2. What is the robot's possible init pos region {pos_range_num}? (or 'done' to finish)")
		x_min = float(input(" Enter the minimum x-coordinate: "))
		x_max = float(input(" Enter the maximum x-coordinate: "))
		y_min = float(input(" Enter the minimum y-coordinate: "))
		y_max = float(input(" Enter the maximum y-coordinate: "))

		 # Structured format: [["x", min, max], ["y", min, max]]
		pos_range_struct = [["x", x_min, x_max], ["y", y_min, y_max]]

		kitchen_data["initial_pos_ranges"].append(pos_range_struct)	
		
		add_more_pos = input("	 Do you want to add another position region? (yes/no) ")
		if add_more_pos.lower() == 'no':
			break
		
		pos_range_num += 1
	
	print("-" * 20)
	yaw_min_deg = float(input("3. Enter minimum yaw angle (in degrees): "))
	yaw_max_deg = float(input("4. Enter maximum yaw angle (in degrees): "))

	yaw_min_rad = math.radians(yaw_min_deg)
	yaw_max_rad = math.radians(yaw_max_deg)

	kitchen_data["initial_rot_yaw_range"] = [["yaw", yaw_min_rad, yaw_max_rad]]
	print("-" * 20)
	# Loop to collect goals, component by component
	goal_list = []
	goal_num = 1
	while True:
		goal_type = input(f"4. What is the goal type {goal_num}? (e.g., 'N', 'A_r', 'G_l', or 'done' to finish) ")
		if goal_type.lower() == 'done':
			break

		goal_value = input(f"	What is the value for goal {goal_num} ({goal_type})? (e.g., '(2,4,0)') ")
		if goal_type == "N":
			# Convert to list
		    value_list = list(ast.literal_eval(goal_value))
		    # Convert last element (yaw) to radians
		    value_list[-1] = math.radians(value_list[-1])
		    goal_list.append((goal_type, value_list))	
		else:
			goal_list.append((goal_type , ast.literal_eval(goal_value)))

		# Check if the user wants to add another component to the current goal
		add_more = input("	 Do you want to add another component to this goal? (yes/no) ")
		if add_more.lower() == 'no':
			kitchen_data["goals"].append(goal_list)
			goal_list = []

		goal_num += 1
		
	# If there are any remaining components in the list, add them as a final goal
	if goal_list:
		kitchen_data["goals"].append(goal_list)

	# Save the collected data to a JSON file
	file_name = f"Isaac-Kitchen-v{kitchen_name}.json"
	with open(file_name, "w") as f:
		json.dump(kitchen_data, f, indent=4)
	
	print("-" * 20)
	print(f"All data for kitchen number '{kitchen_name}' collected and saved to '{file_name}'.")

# Run the function
collect_kitchen_data()
