import json

# 1. Task name
task_name = input("What task do you want to generate synthetic data? ")
print("-" * 80)

# 2. Kitchen setup
kitchen_num = int(input("What Kitchen setup are you going to use? You can tell me by number of the kitchen. "))
print("-" * 80)

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
from pxr import UsdGeom, Gf, Usd
import omni.usd
import omni.physx
import torch
import math
import random
import ipdb

# Get the USD context
usd_context = omni.usd.get_context()
#usd_file_path = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/Kitchen_0{kitchen_num}.usd"  # Replace with your actual path
usd_file_path = "/root/IsaacLab/scripts/isaacsim/kitchen_bodex.usd"
success = usd_context.open_stage(usd_file_path)

if success:
	print(f"Kitchen{kitchen_num} usd is opened.")
	stage = usd_context.get_stage()
	print([prim.GetPath() for prim in stage.Traverse()])

print("-" * 80)
print(f"For the task {task_name}, we would like to get your prior knowledge. As the scene can change, the subtask in sense on language is constant. For example, task PUT BOWL TO SINK can be decomposed as following. \n 1. Move robot to where the bowl is. \n 2. Move both arm to the bowl to grasp. \n 3. Grasp with both arms for safetly. \n 4. Move both arm to original position \n 5. Move robot to where the sink is. \n 6. Move the arm to place into the sink. \n 7. Release both arm.")

print("-" * 80)

import numpy as np
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

reset_arm_r = torch.tensor([0.0, 0.0, 1.1020, 0.7244, -0.6568, -0.1491, 0.1470])
reset_arm_l = torch.tensor([0.0, 0.0, 1.1020, 0.7244, -0.6568, -0.1491, 0.1470])

for sub_language in subtask_language_list:
	sub_action = input(f"For this task ({sub_language}) what is primitive action do you think among \n N: navigation \n N_s: Rotating and go straight \n A_r: Moving right arm \n A_l: Moving left arm \n A_b: Moving both arms \n G_r: Grasping right arm \n G_l: Grasping left arm \n G_b: Grasping both arm \n Enter: ")
	
	print("-" * 80)
	if sub_action == "N":
		# Currently only based on object. Make it also for furnitures like sink_cabinet. Their position z is 0. Using that we can detact if it is a furniture
		where_N_prim_path = input("I see. So where do you want to navigate? Among above can you specify the prim path? ")
		where_N_prim = stage.GetPrimAtPath(where_N_prim_path)
		xform = UsdGeom.Xformable(where_N_prim)
		# Get the current time for non-animated scenes
		time = Usd.TimeCode.Default()

		# Compute the transformation matrix of the prim in world space
		world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
		sx = Gf.Vec3d(world_transform.GetRow3(0)).GetLength()
		sy = Gf.Vec3d(world_transform.GetRow3(1)).GetLength()
		sz = Gf.Vec3d(world_transform.GetRow3(2)).GetLength()
		print("world scale:", sx, sy, sz)	
		# Extract position (translation) and orientation (rotation) from the matrix		
		position: Gf.Vec3d = world_transform.ExtractTranslation()
		orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
		
		print(f"Object Path: {where_N_prim.GetPath()}")
		print(f"Position (X, Y, Z): {position}")
		print(f"Orientation (Quat Real, i, j, k): {orientation_quat}")

		# The real part of the quaternion is first
		print(f"Quaternion Real Part: {orientation_quat.GetReal()}")
		# The imaginary part is a 3D vector
		print(f"Quaternion Imaginary Part: {orientation_quat.GetImaginary()}")
		which_arm = input("After Navigation which arm are you going to use? Left or Right or Both. ")

		if which_arm.lower() == "left":
			arm_bias = 0.18
		elif which_arm.lower() == "right":
			arm_bias = -0.18
		elif which_arm.lower() == "both":
			arm_bias = 0.0
		
		N_pos = torch.zeros(3)
		safety = 0.1
		wheel_r = 0.23
		N_dir = None
		ee_link1_v = torch.tensor([0.2166, -0.1001]).unsqueeze(1)
		ee_link2_v = torch.tensor([0.2166, 0.1001]).unsqueeze(1)
		if position[-1] < 0.01: # Furniture
			# Get the orientation of furniture
			w = round(orientation_quat.GetReal(),4)
			x = round(orientation_quat.GetImaginary()[0], 4)
			y = round(orientation_quat.GetImaginary()[1], 4)
			z = round(orientation_quat.GetImaginary()[2], 4)
			quat_wxyz = [w, x, y, z]
			# Get the size of furniture
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(where_N_prim)			
			bbox_range = bbox3d.ComputeAlignedRange()  

			# Now you can call the methods you want on the resulting range object
			furniture_min_bound = bbox_range.GetMin()
			furniture_max_bound = bbox_range.GetMax()
			furniture_size = bbox_range.GetSize()
			
			if quat_wxyz == [-0.7071, 0.0, 0.0, 0.7071]: 
				N_pos[1] = position[1] - arm_bias
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
				N_pos[0] = position[0] - arm_bias
				N_pos[1] = furniture_max_bound[1] + safety + wheel_r
				N_pos[2] = math.radians(-90.0)
				N_dir = "E"

		else:
			physx_interface = omni.physx.get_physx_interface()

			# 3. CRITICAL: Manually start and step the simulation
			print("Starting simulation...")
			physx_interface.start_simulation()
		
			scene_query_interface = omni.physx.get_physx_scene_query_interface()

			object_pos = position
			ray_origin = object_pos + Gf.Vec3d(0, 0, 0.2)
			ray_distance = 0.5
			
			direction = ["N", "E", "S", "W"]
				
			furniture_prim = None
			furniture_min_bound = None
			furniture_max_bound = None
			furniture_size = None

			for dir in direction:
				if dir == "E":
					ray_direction = Gf.Vec3d(0, -1, -1)
				elif dir == "S":
					ray_direction = Gf.Vec3d(1, 0, -1)
				elif dir == "W":
					ray_direction = Gf.Vec3d(0, 1, -1)
				elif dir == "N":
					ray_direction = Gf.Vec3d(-1, 0, -1)
				hit = scene_query_interface.raycast_closest(ray_origin, ray_direction, ray_distance)
				if hit["hit"]:
					if furniture_prim is None:
						furniture_prim = stage.GetPrimAtPath(hit['rigidBody'])
						bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_])
						world_bbox: Gf.Range3d = bbox_cache.ComputeWorldBound(furniture_prim)
						bbox_range = world_bbox.GetRange()

						# Now you can call the methods you want on the resulting range object
						furniture_min_bound = bbox_range.GetMin()
						furniture_max_bound = bbox_range.GetMax()
						furniture_size = bbox_range.GetSize()
					continue
				else:
					print(f"{dir} direction is empty!")	
					N_dir = dir
			if N_dir == "W":
				N_pos[0] = position[0] + arm_bias
				N_pos[1] = furniture_max_bound[1] + safety + wheel_r
				N_pos[2] = math.radians(90.0)
			elif N_dir == "E":
				N_pos[0] = position[0] - arm_bias
				N_pos[1] = furniture_min_bound[1] - safety - wheel_r
				N_pos[2] = math.radians(-90.0)
			elif N_dir == "S":
				N_pos[1] = position[1] + arm_bias
				N_pos[0] = furniture_max_bound[0] + safety + wheel_r
				N_pos[2] = math.radians(180.0)
			elif N_dir == "N":
				N_pos[1] = position[1] - arm_bias
				N_pos[0] = furniture_min_bound[0] - safety - wheel_r
				N_pos[2] = math.radians(0.0)
			
		subtask_action_list.append((sub_action, N_pos))
		M = torch.tensor([[-math.sin(N_pos[2]), -math.cos(N_pos[2])],
								[math.cos(N_pos[2]), -math.sin(N_pos[2])]])
		reset_arm_r[:2] = N_pos[:2] + (M @ ee_link1_v).squeeze(1)
		reset_arm_l[:2] = N_pos[:2] + (M @ ee_link2_v).squeeze(1)	

	elif sub_action == "A_r":
		usage = int(input("Before we generate, we have a few list on how the arm should move. \n 1. Move arm to grasp \n 2. Move arm to reset. \n 3. Move arm to place. \n Choose a number: "))
		print("-" * 80)
		grasp_pos = torch.zeros(7)	
		# Grasp
		if usage == 1:
			where_A_prim_path = input("Gotchu. Then what object do you wanna grasp? The prim path should be precise like not just /world/mug but /world/mug/handle \n Enter: ")
			where_A_prim = stage.GetPrimAtPath(where_A_prim_path)
			xform = UsdGeom.Xformable(where_A_prim)
			# Get the current time for non-animated scenes
			time = Usd.TimeCode.Default()

			# Compute the transformation matrix of the prim in world space
			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
			sx = Gf.Vec3d(world_transform.GetRow3(0)).GetLength()
			sy = Gf.Vec3d(world_transform.GetRow3(1)).GetLength()
			sz = Gf.Vec3d(world_transform.GetRow3(2)).GetLength()
			print("world scale:", sx, sy, sz)	
			# Extract position (translation) and orientation (rotation) from the matrix		
			position: Gf.Vec3d = world_transform.ExtractTranslation()
			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
			
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(where_A_prim)   # returns GfBBox3d
			bbox_range = bbox3d.ComputeAlignedRange()  

			# Now you can call the methods you want on the resulting range object
			object_min_bound = bbox_range.GetMin()
			object_max_bound = bbox_range.GetMax()
			object_size = bbox_range.GetSize()
				
			print(f"Object Path: {where_A_prim.GetPath()}")
			print(f"Position (X, Y, Z): {position}")
			print(f"Orientation (Quat Real, i, j, k): {orientation_quat}")

			# The real part of the quaternion is first
			print(f"Quaternion Real Part: {orientation_quat.GetReal()}")
			# The imaginary part is a 3D vector
			print(f"Quaternion Imaginary Part: {orientation_quat.GetImaginary()}")
			
			print("-" * 80)
			container = bool(input("Is the object something like a container? "))
			
			if container:		
				physx_interface = omni.physx.get_physx_interface()

				print("Starting simulation...")
				physx_interface.start_simulation()
				scene_query_interface = omni.physx.get_physx_scene_query_interface()
				
				ray_origin_top = Gf.Vec3d(position[0], position[1], object_max_bound[-1] + 0.1)
				ray_direction = Gf.Vec3d(0, 0, -1)
				ray_distance = 0.2
				pre_ray_origin_top = 0.0
				pre_pre_ray_origin_top = 0.0
				start_ang = 0.0
				end_ang = 0.0
				hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
				center = (hit["position"][0], hit["position"][1])
				if N_dir == "N":
					start_ang = 90.0
					end_ang = 0.0
					while True:
						ray_origin_top[1] -= 0.01
						hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
						if hit["hit"]:
							if hit["rigidBody"] == where_A_prim_path:
								pre_ray_origin_top = hit["position"]
							else:
								ray_origin_top[1] += 0.02
								hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
								pre_pre_ray_origin_top = hit["position"]
								break
						else:
							break
				elif N_dir == "S":
					start_ang = 270.0
					end_ang = 180.0
					while True:
						ray_origin_top[1] += 0.01
						hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
						if hit["hit"]:
							if hit["rigidBody"] == where_A_prim_path:
								pre_ray_origin_top = hit["position"]
							else:
								ray_origin_top[1] -= 0.02
								hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
								pre_pre_ray_origin_top = hit["position"]
								break
						else:
							break

				elif N_dir == "E":
					start_ang = 0.0
					end_ang = 270.0
					while True:
						ray_origin_top[0] += 0.01
						hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
						if hit["hit"]:
							if hit["rigidBody"] == where_A_prim_path:
								pre_ray_origin_top = hit["position"]
							else:
								ray_origin_top[0] -= 0.02
								hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
								pre_pre_ray_origin_top = hit["position"]
								break
						else:
							break

				elif N_dir == "W":
					start_ang = 180.0
					end_ang = 90.0
					while True:
						ray_origin_top[0] -= 0.01
						hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
						if hit["hit"]:
							if hit["rigidBody"] == where_A_prim_path:
								pre_ray_origin_top = hit["position"]
							else:
								ray_origin_top[0] += 0.02
								hit = scene_query_interface.raycast_closest(ray_origin_top, ray_direction, ray_distance)
								pre_pre_ray_origin_top = hit["position"]
								break
						else:
							break

				radius = (pre_ray_origin_top[1] - center[1])
				arc_points = generate_arc_points(
						center=center,
						radius=radius,
						start_angle_deg=start_ang,
						end_angle_deg=end_ang,
						num_points=10
					)	
				grasp_pos[0] = arc_points[5][0]
				grasp_pos[1] = arc_points[5][1]
				grasp_pos[2] = object_max_bound[2]
			else: 
				grasp_pos[:3] = torch.tensor(position)
			#TODO Use BODex info	
			grasp_pos[3:] = torch.tensor([0.7244, -0.6568, -0.1491, 0.1470]) 

		# Reset
		elif usage == 2:
			grasp_pos = torch.tensor(reset_arm_r)

		# Place
		elif usage == 3:
			where_A_prim_path = input("So then where do you want to place? Give me the exact prim path like /world/refrigerator/corpus/freezer_separator, /world/sink_cabinet/corpus/sink. \n Enter:")		
			where_A_prim = stage.GetPrimAtPath(where_A_prim_path)
			xform = UsdGeom.Xformable(where_A_prim)
			# Get the current time for non-animated scenes
			time = Usd.TimeCode.Default()

			# Compute the transformation matrix of the prim in world space
			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
			sx = Gf.Vec3d(world_transform.GetRow3(0)).GetLength()
			sy = Gf.Vec3d(world_transform.GetRow3(1)).GetLength()
			sz = Gf.Vec3d(world_transform.GetRow3(2)).GetLength()
			
			# Extract position (translation) and orientation (rotation) from the matrix		
			position: Gf.Vec3d = world_transform.ExtractTranslation()
			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
			
			bbox_cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_], useExtentsHint=False)
			bbox3d = bbox_cache.ComputeWorldBound(where_A_prim)   # returns GfBBox3d
			bbox_range = bbox3d.ComputeAlignedRange()

			# Now you can call the methods you want on the resulting range object
			object_min_bound = bbox_range.GetMin()
			object_max_bound = bbox_range.GetMax()
			object_size = bbox_range.GetSize()
				
			print(f"Position (X, Y, Z): {position}")
			print(f"Orientation (Quat Real, i, j, k): {orientation_quat}")
			offset = 0.1
			put_above = 0.15
			if N_dir == "N":
				grasp_pos[0] = object_min_bound[0] + offset
				grasp_pos[1] = position[1]
				grasp_pos[2] = object_max_bound[2] + put_above
				grasp_pos[3:] = torch.tensor(reset_arm_r[3:])

			elif N_dir == "E":
				grasp_pos[0] = position[0]
				grasp_pos[1] = object_max_bound[1] - offset
				grasp_pos[2] = object_max_bound[2] + put_above
				grasp_pos[3:] = torch.tensor(reset_arm_r[3:])

			elif N_dir == "S":
				grasp_pos[0] = object_max_bound[0] - offset
				grasp_pos[1] = position[1]
				grasp_pos[2] = object_max_bound[2] + put_above
				grasp_pos[3:] = torch.tensor(reset_arm_r[3:])
			elif N_dir == "W":
				grasp_pos[0] = position[0]
				grasp_pos[1] = object_min_bound[1] + offset
				grasp_pos[2] = object_max_bound[2] + put_above
				grasp_pos[3:] = torch.tensor(reset_arm_r[3:])

		subtask_action_list.append((sub_action,grasp_pos))		
	
	elif sub_action == "G_r":
		grasp = bool(input("True for grasp, False for releasing."))
		subtask_action_list.append((sub_action, grasp))

	# For quat
	print(subtask_action_list)
	print("-" * 80)



