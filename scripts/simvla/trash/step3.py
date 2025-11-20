# Step 2: Read USD and make config file and goal generator

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

from pxr import UsdGeom, Gf, Usd
import omni.usd
import omni.physx
import torch
import math
import random
import ipdb
from shapely.geometry import box, Point
import json
import numpy as np
import re
import os
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from PIL import Image, ImageTk


def select_thumbnails_cached(thumbnail_path, cache_file=None):
	"""
	Same as select_thumbnails() but caches the selected indices in JSON.
	"""
	if cache_file is None:
		# default cache name: thumbnails folder name + ".json"
		cache_file = os.path.join(thumbnail_path, "selected_indices.json")

	# if cache exists, load and return
	if os.path.exists(cache_file):
		with open(cache_file, "r") as f:
			saved = json.load(f)
		return saved.get("selected_indices", [])

	# otherwise, run the GUI once
	selected_indices = select_thumbnails(thumbnail_path)

	# save the result for next time
	with open(cache_file, "w") as f:
		json.dump({"selected_indices": selected_indices}, f, indent=2)

	return selected_indices

def select_thumbnails(thumbnail_path, thumbs_per_row=5):
	"""
	Opens a Tkinter GUI with thumbnails arranged in a grid and checkboxes.
	Returns a list of indices the user selected.
	"""
	png_files = sorted([f for f in os.listdir(thumbnail_path) if f.lower().endswith(".png")])
	if not png_files:
		print("No PNG files found in", thumbnail_path)
		return []

	selected_indices = []

	root = tk.Tk()
	root.title("Select Grasp Preferences")

	canvas = tk.Canvas(root)
	scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
	scrollable_frame = tk.Frame(canvas)

	scrollable_frame.bind(
		"<Configure>",
		lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
	)

	canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
	canvas.configure(yscrollcommand=scrollbar.set)

	canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
	scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

	photo_images = []  # keep references alive
	var_list = []

	thumb_size = 128

	for idx, file in enumerate(png_files):
		row = idx // thumbs_per_row
		col = idx % thumbs_per_row

		frame = tk.Frame(scrollable_frame, relief=tk.RAISED, borderwidth=1)
		frame.grid(row=row, column=col, padx=5, pady=5)

		img_path = os.path.join(thumbnail_path, file)
		img = Image.open(img_path)
		img.thumbnail((thumb_size, thumb_size))
		photo = ImageTk.PhotoImage(img)
		photo_images.append(photo)

		label = tk.Label(frame, image=photo)
		label.pack()

		var = tk.BooleanVar()
		chk = tk.Checkbutton(frame, text=f"{idx}", variable=var)
		chk.pack()
		var_list.append(var)

	def confirm_selection():
		for i, var in enumerate(var_list):
			if var.get():
				selected_indices.append(i)
		root.destroy()

	confirm_btn = tk.Button(root, text="Confirm Selection", command=confirm_selection)
	confirm_btn.pack(pady=10, side=tk.BOTTOM)

	root.mainloop()
	return selected_indices
def merge_free_spaces(spaces, tolerance=1e-8):
	"""
	Merges adjacent rectangular spaces into larger ones.

	Args:
		spaces (list): A list of space tuples [((x0,y0,x1,y1), (cx,cy)), ...].
		tolerance (float): A small value to handle floating point inaccuracies.

	Returns:
		list: A new list with the merged spaces in the same format.
	"""
	# Convert to a mutable list of bounds for easier processing
	rects = [list(s[0]) for s in spaces]

	while True:
		merged_in_pass = False
		i = 0
		while i < len(rects):
			j = i + 1
			while j < len(rects):
				r1 = rects[i]
				r2 = rects[j]

				# Check for horizontal merge
				# r1 is to the left of r2, and they are vertically aligned
				if (abs(r1[2] - r2[0]) < tolerance and 
					abs(r1[1] - r2[1]) < tolerance and 
					abs(r1[3] - r2[3]) < tolerance):
					# Merge r2 into r1
					r1[2] = r2[2]
					rects.pop(j)
					merged_in_pass = True
					continue # Restart inner loop with the now-larger r1

				# Check for vertical merge
				# r1 is below r2, and they are horizontally aligned
				if (abs(r1[3] - r2[1]) < tolerance and 
					abs(r1[0] - r2[0]) < tolerance and 
					abs(r1[2] - r2[2]) < tolerance):
					# Merge r2 into r1
					r1[3] = r2[3]
					rects.pop(j)
					merged_in_pass = True
					continue # Restart inner loop with the now-larger r1
				
				j += 1
			i += 1
		
		if not merged_in_pass:
			break

	# Convert back to original format with calculated centers
	merged_spaces = []
	for r in rects:
		bounds = tuple(r)
		center = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
		merged_spaces.append((bounds, center))
		
	return merged_spaces

def find_free_spaces_grid(bounds, obstacles):
	"""
	Finds free space by creating a grid from all object boundaries
	and testing the center of each resulting grid cell.
	
	Args:
		bounds (tuple): A tuple (x0, y0, x1, y1) for the overall area.
		obstacles (list): A list of shapely box objects.
		
	Returns:
		list: A list of tuples in the format [((x0,y0,x1,y1), (cx,cy)), ...].
	"""
	x0, y0, x1, y1 = bounds

	# 1. Collect all unique X and Y coordinates
	all_x = {x0, x1}
	all_y = {y0, y1}
	for obs in obstacles:
		ox_min, oy_min, ox_max, oy_max = obs.bounds
		all_x.add(ox_min)
		all_x.add(ox_max)
		all_y.add(oy_min)
		all_y.add(oy_max)

	# Create sorted lists of the grid lines
	sorted_x = sorted(list(all_x))
	sorted_y = sorted(list(all_y))

	free_spaces = []

	# 2. Iterate through each cell in the grid
	for i in range(len(sorted_x) - 1):
		for j in range(len(sorted_y) - 1):
			# Define the grid cell
			cell_x0, cell_x1 = sorted_x[i], sorted_x[i+1]
			cell_y0, cell_y1 = sorted_y[j], sorted_y[j+1]
			
			# 3. Test the center point of the cell
			center_x = (cell_x0 + cell_x1) / 2
			center_y = (cell_y0 + cell_y1) / 2
			center_point = Point(center_x, center_y)
			
			# 4. Check if the center is inside any obstacle
			is_occupied = any(obs.contains(center_point) for obs in obstacles)
			
			if not is_occupied:
				# This cell is free space. Format it to match the required output.
				bounds_tuple = (cell_x0, cell_y0, cell_x1, cell_y1)
				center_tuple = (center_x, center_y)
				free_spaces.append((bounds_tuple, center_tuple))
				
	return free_spaces

def make_walls_from_bounds(world_bounds, wall_material="wall_material", height=3.0, thickness=0.1):
	x_min, y_min, x_max, y_max = world_bounds
	width = x_max - x_min
	depth = y_max - y_min

	wall_01 = f"""wall_01 = AssetBaseCfg(
	prim_path="{{ENV_REGEX_NS}}/wall_01",
	init_state=AssetBaseCfg.InitialStateCfg(
		pos=({(x_min + x_max)/2:.3f}, {y_min:.3f}, {height/2:.3f}),
		rot=(0.70711, 0.70711, 0.0, 0.0),
	),
	spawn=sim_utils.UsdFileCfg(
		usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/floor.usd",
		scale=({width:.3f}, {height}, {thickness}),
		visual_material=MdlFileCfg(mdl_path={wall_material}),
	),
	collision_group=-1,
)\n"""

	wall_02 = f"""wall_02 = AssetBaseCfg(
	prim_path="{{ENV_REGEX_NS}}/wall_02",
	init_state=AssetBaseCfg.InitialStateCfg(
		pos=({x_max:.3f}, {(y_min + y_max)/2:.3f}, {height/2:.3f}),
		rot=(0.5, 0.5, 0.5, 0.5),
	),
	spawn=sim_utils.UsdFileCfg(
		usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/floor.usd",
		scale=({depth:.3f}, {height}, {thickness}),
		visual_material=MdlFileCfg(mdl_path={wall_material}),
	),
	collision_group=-1,
)\n"""
	wall_03 = f"""wall_03 = AssetBaseCfg(
	prim_path="{{ENV_REGEX_NS}}/wall_03",
	init_state=AssetBaseCfg.InitialStateCfg(
		pos=({x_min:.3f}, {(y_min + y_max)/2:.3f}, {height/2:.3f}),
		rot=(0.5, 0.5, 0.5, 0.5),
	),
	spawn=sim_utils.UsdFileCfg(
		usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/floor.usd",
		scale=({depth:.3f}, {height}, {thickness}),
		visual_material=MdlFileCfg(mdl_path={wall_material}),
	),
	collision_group=-1,
)\n"""
	wall_04 = f"""wall_04 = AssetBaseCfg(
		prim_path="{{ENV_REGEX_NS}}/wall_04",
		init_state=AssetBaseCfg.InitialStateCfg(
			pos=({(x_min + x_max)/2:.3f}, {y_max:.3f}, {height/2:.3f}),
			rot=(0.70711, 0.70711, 0.0, 0.0),
		),
		spawn=sim_utils.UsdFileCfg(
			usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/floor.usd",
			scale=({width:.3f}, {height}, {thickness}),
			visual_material=MdlFileCfg(mdl_path={wall_material}),
		),
		collision_group=-1,
	)\n"""


	return wall_01, wall_02, wall_03, wall_04

def generate_arc_points(center, radius, start_angle_deg, end_angle_deg, num_points=10):
	"""
	Calculates points on a circular arc.

	Args:
		center (tuple): The (x, y) coordinates of the circle's center.
		radius (float): The circle's radius.
		start_angle_deg (float): The starting angle in degrees (0 is right, 90 is up).
		end_angle_deg (float): The ending angle in degrees.
		num_points (int): The number of points to generate along the arc.

	Returns:
		list: A list of (x, y) tuples representing the points on the arc.
	"""
	# 1. Extract center coordinates
	cx, cy = center
	
	# 2. Convert the start and end angles from degrees to radians
	start_angle_rad = math.radians(start_angle_deg)
	end_angle_rad = math.radians(end_angle_deg)

	# 3. Generate evenly spaced angles in the range
	angles = np.linspace(start_angle_rad, end_angle_rad, num_points)

	# 4. Use the parametric circle equations to find the (x, y) for each angle
	x_coords = cx + radius * np.cos(angles)
	y_coords = cy + radius * np.sin(angles)

	# 5. Combine the coordinates into a list of points
	points = list(zip(x_coords, y_coords))
	
	return points

def pick_random_grasp_file(input_path):
	# Replace "use_data" with "graspdata" and add "/floating"
	base_dir = input_path.split("/use_data/")[0]   # /root/IsaacLab/BODex_obj
	obj_name = input_path.split("/use_data/")[1].split("/")[0]	# core_bottle_3d

	# Build graspdata floating dir
	grasp_dir = os.path.join(base_dir, "graspdata", obj_name, "floating")
	
	# List .npy files
	files = [f for f in os.listdir(grasp_dir) if f.endswith("010_grasp.npy")]
	if not files:
		files = [f for f in os.listdir(grasp_dir) if f.endswith(".npy")]
	chosen = random.choice(files)
	return os.path.join(grasp_dir, chosen)

def find_first_free_direction(position, free_squares, step=0.03, max_steps=50):
	"""
	position: (x,y) of the object (float)
	free_squares: [((x0,y0,x1,y1),(cx,cy)), ...]
	step: step size in meters (0.03 = 3cm)
	max_steps: how many steps outward to check (step*max_steps is max distance)

	Returns: ("N"|"S"|"E"|"W", (px,py)) or None if nothing found
	"""
	x, y = position[0], position[1]

	directions = {
		"E": (0, +1),
		"W": (0, -1),
		"S": (+1, 0),
		"N": (-1, 0)
	}
	# Preextract bounds for speed
	bounds_list = [b for b, _ in free_squares]

	for i in range(1, max_steps + 1):
		dist = step * i
		for dname, (dx, dy) in directions.items():
			px = x + dx * dist
			py = y + dy * dist

			# check if (px,py) is inside any free square
			for (x0, y0, x1, y1) in bounds_list:
				if x0 <= px <= x1 and y0 <= py <= y1:
					return dname, (px, py)

	return None  # no free space found


# Save Kitchen data in JSON format
kitchen_data = {
		"task_name": "",
		"kitchen_num": 0,
		"kitchen_type": "",
		"island_bound": 0,
		"kitchen_sub_num": 0,
		"initial_pos_ranges": [],
		"initial_rot_yaw_range": "",
		"goals": []
	}

# Task name
task_name = input("What task do you want to generate synthetic data? ")
kitchen_data["task_name"] = task_name
print("-" * 80)

# Kitchen setup
kitchen_num = int(input("What Kitchen setup are you going to use? You can tell me by number of the kitchen. \nEnter: "))
kitchen_data["kitchen_num"] = kitchen_num
print("-" * 80)
kitchen_sub_num = input("What is the subnumber? \nEnter: ")
kitchen_data["kitchen_sub_num"] = kitchen_sub_num 

# Get the USD context
usd_context = omni.usd.get_context()
usd_file_path = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}_{kitchen_sub_num}.usd"
success = usd_context.open_stage(usd_file_path)

if success:
	print(f"Kitchen{kitchen_num}_{kitchen_sub_num} usd is opened.")
	stage = usd_context.get_stage()
	print([prim.GetPath() for prim in stage.Traverse()])
	
	# 1. Initial pos range and wall 
	text = [str(prim.GetPath()) for prim in stage.Traverse()]
	first_level = {"/world/" + p.split("/")[2] for p in text if p.startswith("/world/") and len(p.split("/")) > 2}
	first_level_list = sorted(first_level)
	furniture = []

	for obj_prim_path in first_level_list:
		try:
			name = obj_prim_path.split("/")[-1]
			obj_prim = stage.GetPrimAtPath(obj_prim_path)
			xform = UsdGeom.Xformable(obj_prim)
			time = Usd.TimeCode.Default()
			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
			position: Gf.Vec3d = world_transform.ExtractTranslation()
			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(obj_prim)   # returns GfBBox3d
			bbox_range = bbox3d.ComputeAlignedRange()
			furniture_min_bound = bbox_range.GetMin()
			furniture_max_bound = bbox_range.GetMax()
			furniture.append(box(furniture_min_bound[0], furniture_min_bound[1], furniture_max_bound[0], furniture_max_bound[1]))
		
		except Exception as e:
			print("Error on", obj_prim_path, e)
	# Compute world bounds automatically
	min_x = min(obs.bounds[0] for obs in furniture)
	min_y = min(obs.bounds[1] for obs in furniture)
	max_x = max(obs.bounds[2] for obs in furniture)
	max_y = max(obs.bounds[3] for obs in furniture)
	bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
	json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")

	# Load dictionary
	with open(json_path, "r") as f:
		k_data = json.load(f)
		kitchen_data["kitchen_type"] = k_data["kitchen_type"]
		if k_data["kitchen_type"] == "island":
			island_prim = stage.GetPrimAtPath("/world/kitchen_island")
			xform = UsdGeom.Xformable(island_prim)
			time = Usd.TimeCode.Default()
			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
			position: Gf.Vec3d = world_transform.ExtractTranslation()
			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(island_prim)	# returns GfBBox3d
			bbox_range = bbox3d.ComputeAlignedRange()
			furniture_min_bound = bbox_range.GetMin()
			furniture_max_bound = bbox_range.GetMax()
			kitchen_data["island_bound"] = (furniture_min_bound[0], furniture_min_bound[1], furniture_max_bound[0], furniture_max_bound[1])
		if k_data["kitchen_type"] == "single_wall":
			entry = [
				["x", -0.3, 3.7],
				["y",-1.5, -1],
			]
			kitchen_data["initial_pos_ranges"].append(entry)
			world_bounds = (-0.5, -2, 4, 0.5) 
			free_squares = [((-0.5, -1.5, 3.7, -1), (1.6, -1.25))]
		else:
			margin = 0.2
			world_bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
			merge_squares = merge_free_spaces(find_free_spaces_grid(world_bounds, furniture))
			free_squares = []
			robot_radius = 0.23 
			safe = 0.02
			for bounds, center in merge_squares:
				# Unpack the coordinates from the bounds tuple
				x0, y0, x1, y1 = bounds
				
				# Calculate the width (x space) and height (y space)
				width = x1 - x0
				height = y1 - y0
				min_size = robot_radius * 2
				# Check if both dimensions are larger than your minimum size
				if width > min_size and height > min_size:
					print(f"Found qualifying space: {bounds}, Center: {center}")
					entry = [
						["x", x0 + robot_radius + safe, x1 - robot_radius - safe],
						["y", y0 + robot_radius + safe, y1 - robot_radius - safe]
					]
					kitchen_data["initial_pos_ranges"].append(entry)
					free_squares.append((bounds, center))
	# 2. Auto generate env config file
	source_file = "/root/IsaacLab/scripts/simvla/kitchen_env_cfg_source.py" 
	out_dir = "/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen"
	output_file = os.path.join(out_dir, f"kitchen_{kitchen_num}_{kitchen_sub_num}.py")
	
	kitchen_usd = f'''
# Kitchen
kitchen = AssetBaseCfg(
	prim_path="{{ENV_REGEX_NS}}/Kitchen",
	# Make sure to set the correct path to the generated scene
	spawn=sim_utils.UsdFileCfg(usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}_{kitchen_sub_num}.usd"),
)'''

	generated_blocks = []
	for obj_prim_path in first_level_list:
		try:
			name = obj_prim_path.split("/")[-1]
			obj_prim = stage.GetPrimAtPath(obj_prim_path)

			# World transform
			xform = UsdGeom.Xformable(obj_prim)
			world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
			pos = tuple(round(v, 4) for v in world_transform.ExtractTranslation())
			quat = (
				round(world_transform.ExtractRotationQuat().GetReal(), 4),
				round(world_transform.ExtractRotationQuat().GetImaginary()[0], 4),
				round(world_transform.ExtractRotationQuat().GetImaginary()[1], 4),
				round(world_transform.ExtractRotationQuat().GetImaginary()[2], 4),
			)

			# Joints
			joint_attr = obj_prim.GetAttribute("joint_names")
			joint_names = joint_attr.Get() if joint_attr else []
			joint_pos = {j: 0.0 for j in joint_names}

			# Actuators
			actuator_block = ""
			if joint_names:
				joint_names_list = [f'"{j}"' for j in joint_names]
				actuator_block = f'''
						actuators={{
							"default": ImplicitActuatorCfg(
								joint_names_expr=[{", ".join(joint_names_list)}],
								effort_limit=87.0,
								velocity_limit=100.0,
								stiffness=0.0,
								damping=1.0,
							)
						}},'''
			if joint_names:  # has joints â articulation
				block = f"""{name} = ArticulationCfg(
					prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",
					spawn=None,
					init_state=ArticulationCfg.InitialStateCfg(
						pos={pos},
						rot={quat},
						joint_pos={joint_pos},
					),{actuator_block}
				)\n"""
			else:  # rigid object, no joints
				block = f"""{name} = RigidObjectCfg(
					prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",
					spawn=None,
					init_state=RigidObjectCfg.InitialStateCfg(
						pos={pos},
						rot={quat},
					),
				)\n"""
			generated_blocks.append(block)

		except Exception as e:
			print("Error on", obj_prim_path, e)
	wall_01, wall_02, wall_03, wall_04 = make_walls_from_bounds(world_bounds)
	generated_blocks.insert(0, wall_01)
	generated_blocks.insert(1, wall_02)
	generated_blocks.insert(2, wall_03)
	generated_blocks.insert(3, wall_04)
	generated_blocks.insert(4, kitchen_usd)

	indented_blocks = ["\n".join("\t" + line for line in block.splitlines()) for block in generated_blocks]
	generated_code = "\n".join(indented_blocks)

	with open(source_file, "r") as f:
		config_text = f.read()
	
	new_text = re.sub(
		r"# -------Change-------.*?# -------Stop-------",
		"# -------Change-------\n" + generated_code + "# -------Stop-------",
		config_text,
		flags=re.DOTALL
	)
	with open(output_file, "w") as f:
		f.write(new_text)

	print(f"â Created {output_file} with {len(generated_blocks)} kitchen configs.")

print("-" * 80)
yaw_min_deg = float(input("3. Enter minimum yaw angle (in degrees): "))
yaw_max_deg = float(input("4. Enter maximum yaw angle (in degrees): "))

yaw_min_rad = math.radians(yaw_min_deg)
yaw_max_rad = math.radians(yaw_max_deg)

kitchen_data["initial_rot_yaw_range"] = [["yaw", yaw_min_rad, yaw_max_rad]]
print("-" * 80)

# 3. Goal generator
if success:
	stage = usd_context.get_stage()
	print([prim.GetPath() for prim in stage.Traverse()])

goal_dir = os.path.expanduser("~/IsaacLab/scripts/simvla/goals")
json_file = os.path.join(goal_dir, f"Isaac-Kitchen-v{kitchen_num}-{kitchen_sub_num}.json")
while True:
	print("-" * 80)
	print(f"For the task {task_name}, we would like to get your prior knowledge. As the scene can change, the subtask in sense on language is constant. For example, task PUT BOWL TO SINK can be decomposed as following. \n 1. Move robot to where the bowl is. \n 2. Move both arm to the bowl to grasp. \n 3. Grasp with both arms for safetly. \n 4. Move both arm to original position \n 5. Move robot to where the sink is. \n 6. Move the arm to place into the sink. \n 7. Release both arm.")

	print("-" * 80)

	subtask_language_list = []
	subtask_action_list = []
	subtask_num = 1

	while True:
		subtask = input(f"Enter subtask{subtask_num} (enter 'done' if you are done): ")
		if subtask.lower() == "done":
			break

		subtask_language_list.append(subtask)
		subtask_num += 1

	print("-" * 80)

	reset_arm_r = torch.tensor([999, 0.0, 1.1020, 0.6322, -0.5870, 0.3344, -0.3793])
	reset_arm_l = torch.tensor([999, 0.0, 1.1020, 0.7244, -0.6568, -0.1491, 0.1470])

	for sub_language in subtask_language_list:
		sub_action = input(f"For this task ({sub_language}) what is primitive action do you think among \n N: navigation \n N_s: Rotating and go straight \n A_r: Moving right arm \n A_l: Moving left arm \n A_b: Moving both arms \n G_r: Grasping right arm \n G_l: Grasping left arm \n G_b: Grasping both arm \n Enter: ")
		
		print("-" * 80)

		# Navigation api
		if sub_action == "N":
			# Goal of navigation
			where_N_prim_path = input("I see. So where do you want to navigate? Among above can you specify the prim path? ")
			where_N_prim = stage.GetPrimAtPath(where_N_prim_path)
			xform = UsdGeom.Xformable(where_N_prim)
			time = Usd.TimeCode.Default()
			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
			position: Gf.Vec3d = world_transform.ExtractTranslation()
			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(where_N_prim)   # returns GfBBox3d
			bbox_range = bbox3d.ComputeAlignedRange()
			object_min_bound = bbox_range.GetMin()
			object_max_bound = bbox_range.GetMax()

			# Which arm
			which_arm = input("After Navigation which arm are you going to use? Left or Right or Both. ")

			# Bias for easy arm movement
			if which_arm.lower() == "left":
				arm_bias = 0.15
			elif which_arm.lower() == "right":
				arm_bias = -0.15
			elif which_arm.lower() == "both":
				arm_bias = 0.0
		
			# Goal state! 
			N_pos = torch.zeros(3)
			safety = 0.1
			wheel_r = 0.23
			N_dir = None
			# For furniture
			if position[-1] < 0.01:
				# Get the orientation of furniture
				w = round(orientation_quat.GetReal(),4)
				x = round(orientation_quat.GetImaginary()[0], 4)
				y = round(orientation_quat.GetImaginary()[1], 4)
				z = round(orientation_quat.GetImaginary()[2], 4)
				quat_wxyz = [w, x, y, z]

				furniture_min_bound = bbox_range.GetMin()
				furniture_max_bound = bbox_range.GetMax()
				
				if quat_wxyz == [-0.7071, 0.0, 0.0, 0.7071]: 
					N_pos[1] = position[1] + arm_bias
					N_pos[0] = furniture_min_bound[0] - safety - wheel_r
					N_pos[2] = math.radians(0.0)
					N_dir = "N" 
				elif quat_wxyz == [0.7071, 0.0, 0.0, 0.7071]:
					N_pos[1] = position[1] + arm_bias
					N_pos[0] = furniture_max_bound[0] + safety + wheel_r
					N_pos[2] = math.radians(180.0)
					N_dir = "S"
				elif quat_wxyz == [1.0, 0.0, 0.0, 0.0]: 
					N_pos[0] = position[0] + arm_bias
					N_pos[1] = furniture_min_bound[1] - safety - wheel_r
					N_pos[2] = math.radians(90.0)
					N_dir = "W"  #"E"
				elif quat_wxyz == [0.0, 0.0, 0.0, 1.0]: 
					N_pos[0] = position[0] + arm_bias
					N_pos[1] = furniture_max_bound[1] + safety + wheel_r
					N_pos[2] = math.radians(-90.0)
					N_dir = "E"
			# For objects
			else:
				physx_interface = omni.physx.get_physx_interface()
				physx_interface.start_simulation()
				scene_query_interface = omni.physx.get_physx_scene_query_interface()
				object_pos = position
				ray_origin = object_pos + Gf.Vec3d(0, 0, 0.2)
				ray_distance = 1
				direction = ["N", "E", "S", "W"]

				furniture_prim = None
				furniture_min_bound = None
				furniture_max_bound = None
				for dir in direction:
					if dir == "E":
						ray_direction = Gf.Vec3d(0, 1, -1)
					elif dir == "S":
						ray_direction = Gf.Vec3d(1, 0, -1)
					elif dir == "W":
						ray_direction = Gf.Vec3d(0, -1, -1)
					elif dir == "N":
						ray_direction = Gf.Vec3d(-1, 0, -1)
					hit = scene_query_interface.raycast_closest(ray_origin, ray_direction, ray_distance)
					if hit["hit"]:
						if furniture_prim is None:
							furniture_prim = stage.GetPrimAtPath(hit['rigidBody'])
							xform = UsdGeom.Xformable(furniture_prim)
							time = Usd.TimeCode.Default()
							world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
							position: Gf.Vec3d = world_transform.ExtractTranslation()
							orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
							bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
							bbox3d = bbox_cache.ComputeWorldBound(furniture_prim)   # returns GfBBox3d
							bbox_range = bbox3d.ComputeAlignedRange()
							furniture_min_bound = bbox_range.GetMin()
							furniture_max_bound = bbox_range.GetMax()
						continue
				if furniture_min_bound is None:
					user = input("Current ray trace is not complete, so where is the object? Provide the prim path. \nEnter: ")
					print("-" * 80)
					furniture_prim = stage.GetPrimAtPath(user)
					xform = UsdGeom.Xformable(furniture_prim)
					time = Usd.TimeCode.Default()
					world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
					position: Gf.Vec3d = world_transform.ExtractTranslation()													
					orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()												
					bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
					bbox3d = bbox_cache.ComputeWorldBound(furniture_prim)	# returns GfBBox3d
					bbox_range = bbox3d.ComputeAlignedRange()
					furniture_min_bound = bbox_range.GetMin()
					furniture_max_bound = bbox_range.GetMax()
				object_pos_2d = (object_pos[0], object_pos[1]) 
				dir, point = find_first_free_direction(object_pos_2d, free_squares, step=0.03, max_steps=100)
				N_dir = dir	
				if dir == "W":
					N_pos[0] = position[0] + arm_bias
					N_pos[1] = furniture_min_bound[1] - safety - wheel_r
					N_pos[2] = math.radians(90.0)
				elif dir == "E":
					N_pos[0] = position[0] - arm_bias
					N_pos[1] = furniture_max_bound[1] + safety + wheel_r
					N_pos[2] = math.radians(-90.0)
				elif dir == "S":
					N_pos[1] = position[1] + arm_bias
					N_pos[0] = furniture_max_bound[0] + safety + wheel_r
					N_pos[2] = math.radians(180.0)
				elif dir == "N":
					N_pos[1] = position[1] - arm_bias
					N_pos[0] = furniture_min_bound[0] - safety - wheel_r
					N_pos[2] = math.radians(0.0)
		
			# Check for island
			if k_data["kitchen_type"] == "island":
				subtask_action_list.append((sub_action, N_pos.tolist()))
			subtask_action_list.append((sub_action, N_pos.tolist()))

		elif sub_action == "A_r":
			usage = int(input("Before we generate, we have a few list on how the arm should move. \n 1. Move arm to grasp \n 2. Move arm to reset. \n 3. Move arm to place. \n Choose a number: "))
			print("-" * 80)
			grasp_pos = torch.zeros(7)	
			# Grasp
			if usage == 1:
				where_A_prim_path = input("Gotchu. Then what object do you wanna grasp? The prim path should be precise like not just /world/mug but /world/mug/handle \n Enter: ")
				where_A_prim = stage.GetPrimAtPath(where_A_prim_path)
				xform = UsdGeom.Xformable(where_A_prim)
				time = Usd.TimeCode.Default()
				world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
				position: Gf.Vec3d = world_transform.ExtractTranslation()
				orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
				bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
				bbox3d = bbox_cache.ComputeWorldBound(where_A_prim)   # returns GfBBox3d
				bbox_range = bbox3d.ComputeAlignedRange()  
				object_min_bound = bbox_range.GetMin()
				object_max_bound = bbox_range.GetMax()
				print("-" * 80)
				# Read BODex grasp pose
				bodex_attr = where_A_prim.GetAttribute("BODex_path")
				if bodex_attr and bodex_attr.HasAuthoredValue():
					value = bodex_attr.Get()
					grasp_file_path = pick_random_grasp_file(value)
					grasp_data = np.load(grasp_file_path, allow_pickle=True).tolist()
					eef_data = grasp_data["robot_pose"][0][:,1,:7]
					# Mask out nan values
					mask = np.isfinite(eef_data)
					valid_rows_mask = np.all(mask, axis=1)
					valid_eef_data = eef_data[valid_rows_mask]
					# Get human preference
					thumbnail_path = f"{os.path.splitext(grasp_file_path)[0]}_segments_thumbnails"					
					prefer_list = select_thumbnails_cached(thumbnail_path)
					prefer_eef_data = valid_eef_data[prefer_list]
					if not prefer_list:
						ipdb.set_trace()  # fallback if user picked nothing
					# Rotate 90 w.r.t. x axis
					xyz = prefer_eef_data[:, 0:3]
					quat_wxyz = prefer_eef_data[:, 3:7]
					quat_xyzw = np.hstack([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]])
					rot_x = R.from_rotvec([np.pi/2, 0, 0])
					rotated_xyz = rot_x.apply(xyz)
					original_rotations = R.from_quat(quat_xyzw)
					new_rotations = rot_x * original_rotations
					# Rotate w.r.t y axis
					angle_deg = int(kitchen_sub_num) * 30
					angle_rad = np.radians(angle_deg)
					rot_y = R.from_rotvec([0, angle_rad, 0])
					rotated_xyz = rot_y.apply(rotated_xyz)
					new_rotations = rot_y * new_rotations
					rotated_quats_xyzw = new_rotations.as_quat()  # xyzw
					rotated_quats_wxyz = np.hstack([rotated_quats_xyzw[:, 3:4],
													rotated_quats_xyzw[:, 0:3]])
					rotated_data = np.hstack((rotated_xyz, rotated_quats_wxyz))					
						
					reference_frame = torch.tensor((object_min_bound + object_max_bound)/2)
					# Get available grasp
					if N_dir == "N":
						avail_data = rotated_data[rotated_data[:,0] < 0]
					elif N_dir == "S":
						avail_data = rotated_data[rotated_data[:,0] > 0]
					elif N_dir == "E":
						avail_data = rotated_data[rotated_data[:,1] > 0]
					elif N_dir == "W":
						avail_data = rotated_data[rotated_data[:,1] < 0]
					
					rand_data = random.choice(avail_data)
					pose_idx = prefer_list[np.where(np.isclose(rotated_data , random.choice(avail_data),atol=1e-6 ))[0][0]]
					mean_contact_point = torch.tensor(grasp_data["contact_point"][0])[pose_idx].mean(dim=1).squeeze(0)
					mean_contact_point_np = mean_contact_point.numpy()	# to NumPy if needed

					rot_x = R.from_rotvec([np.pi/2, 0, 0])
					angle_deg = int(kitchen_sub_num) * 30
					angle_rad = np.radians(angle_deg)
					rot_y = R.from_rotvec([0, angle_rad, 0])
					rotated_point = rot_x.apply(mean_contact_point_np)
					rotated_point = rot_y.apply(rotated_point)
					transformed_mean_contact_point = torch.tensor(rotated_point, dtype=mean_contact_point.dtype)
					grasp_pos[0:3] = torch.tensor(reference_frame + transformed_mean_contact_point)
					grasp_pos[3:] = torch.tensor(rand_data[3:]) 
				else:
					ipdb.set_trace()
			# Reset
			elif usage == 2:
				grasp_pos = torch.tensor(reset_arm_r)

			# Place
			elif usage == 3:
				where_A_prim_path = input("So then where do you want to place? Give me the exact prim path like /world/refrigerator/corpus/freezer_separator, /world/sink_cabinet/corpus/sink. \n Enter:")		
				where_A_prim = stage.GetPrimAtPath(where_A_prim_path)
				xform = UsdGeom.Xformable(where_A_prim)
				time = Usd.TimeCode.Default()
				world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
				position: Gf.Vec3d = world_transform.ExtractTranslation()
				orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
				bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
				bbox3d = bbox_cache.ComputeWorldBound(where_A_prim)   # returns GfBBox3d
				bbox_range = bbox3d.ComputeAlignedRange()
				object_min_bound = bbox_range.GetMin()
				object_max_bound = bbox_range.GetMax()
					
				offset = 0.1
				put_above = 0.15
				if N_dir == "N":
					grasp_pos[0] = object_min_bound[0] + offset
					grasp_pos[1] = position[1]
					grasp_pos[2] = object_max_bound[2] + put_above
				elif N_dir == "E":
					grasp_pos[0] = position[0]
					grasp_pos[1] = object_max_bound[1] - offset
					grasp_pos[2] = object_max_bound[2] + put_above
				elif N_dir == "S":
					grasp_pos[0] = object_max_bound[0] - offset
					grasp_pos[1] = position[1]
					grasp_pos[2] = object_max_bound[2] + put_above
				elif N_dir == "W":
					grasp_pos[0] = position[0]
					grasp_pos[1] = object_min_bound[1] + offset
					grasp_pos[2] = object_max_bound[2] + put_above
			subtask_action_list.append((sub_action,grasp_pos.tolist()))		
		
		elif sub_action == "G_r":
			grasp = input("True for grasp, False for releasing.\n Enter: ").strip().lower() == "true"
			subtask_action_list.append((sub_action, grasp))

	kitchen_data["goals"].append(subtask_action_list)
	print("-" * 80)

	more = input("True for more goals, False for done. \n Enter: ")
	if more == True:
		with open(json_file, "w") as f:
			json.dump(kitchen_data, f, indent=4)
		print(f" Progress saved to {json_file}")
	else:
		# final save before break
		with open(json_file, "w") as f:
			json.dump(kitchen_data, f, indent=4)
		print(f"Final save to {json_file}")
		break
		
