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
usd_file_path = "./kitchen_bodex.usd"
success = usd_context.open_stage(usd_file_path)

if success:
	print(f"Kitchen{kitchen_num} usd is opened.")
	stage = usd_context.get_stage()
	print([prim.GetPath() for prim in stage.Traverse()])
	ipdb.set_trace()
	text = [str(prim.GetPath()) for prim in stage.Traverse()]
	first_level = {"/world/" + p.split("/")[2] for p in text if p.startswith("/world/") and len(p.split("/")) > 2}
	first_level_list = sorted(first_level)
	output_file = "kitchen_cfg.py"

	with open(output_file, "w") as f:
		f.write("from omni.isaac.lab.sim import ArticulationCfg, ImplicitActuatorCfg\n\n")

		for obj_prim_path in first_level_list:
			try:
				# name
				name = obj_prim_path.split("/")[-1]
				obj_prim = stage.GetPrimAtPath(obj_prim_path)

				# Transform
				xform = UsdGeom.Xformable(obj_prim)
				time = Usd.TimeCode.Default()
				world_transform = xform.ComputeLocalToWorldTransform(time)

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

				# Write block to file
				f.write(f"""{name} = ArticulationCfg(
		prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos={pos},
			rot={quat},
			joint_pos={joint_pos},
		),{actuator_block}
	)\n\n""")

			except Exception as e:
				print("Error on", obj_prim_path, e)

#	for obj_prim_path in first_level_list:
#		try:
#			# 1. name, prim_path
#			name = obj_prim_path.split("/")[-1]
#			
#			# 2. init_state, actuators if have
#			obj_prim = stage.GetPrimAtPath(obj_prim_path)
#			xform = UsdGeom.Xformable(obj_prim)
#			time = Usd.TimeCode.Default()
#			# Compute the transformation matrix of the prim in world space
#			world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
#			# Extract position (translation) and orientation (rotation) from the matrix
#			position: Gf.Vec3d = world_transform.ExtractTranslation()
#			orientation_quat: Gf.Quatd = world_transform.ExtractRotationQuat()
#			w = round(orientation_quat.GetReal(),4)
#			 x = round(orientation_quat.GetImaginary()[0], 4)
#			 y = round(orientation_quat.GetImaginary()[1], 4)
#			 z = round(orientation_quat.GetImaginary()[2], 4)
#			 quat_wxyz = (w, x, y, z)
#			joint_names = obj_prim.GetAttribute("joint_names").Get()
#			joint_list = []
#			if len(joint_names) != 0:
#				for joint in joint_names:
#					joint_list.append(joint)
#			
