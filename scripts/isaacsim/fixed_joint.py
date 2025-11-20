from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd

import ipdb
import glob
import os
from pxr import Usd, UsdGeom, Sdf, PhysxSchema, UsdPhysics
# Read folder where the usd are located
folder_path = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen"
files = glob.glob(f"{folder_path}/kitchen_1119_04.usd")

# Get the USD context
for file in files:
	usd_context = omni.usd.get_context()
	success = usd_context.open_stage(file)
	if len(file.split('/')[-1].split('_')) != 3:
		continue
	new_target_path = "/Root/"+file.split('/')[-1].removesuffix(".usd")
	if success:
		stage = usd_context.get_stage()
		if stage.GetPrimAtPath(new_target_path).GetPath().isEmpty:
			new_target_path = '/world'
			print(file)
		else:
			print(f"{file} is not in an appropriate state.")
			continue
		prims = [prim.GetPath() for prim in stage.Traverse()]
		fixed_joints = [prim for prim in prims if 'fixed' in prim.name]
		for fixed_joint in fixed_joints:
			new_target_path = '/world'
			fixed_prim = stage.GetPrimAtPath(fixed_joint)
			if fixed_prim.GetRelationships()[0].GetTargets() == []:
				print(f"{fixed_joint} missing target 0.")
				prim = stage.GetPrimAtPath(fixed_joint.GetParentPath())
				rigid_api = UsdPhysics.RigidBodyAPI(prim)
				rigid_api.CreateKinematicEnabledAttr(True)
				stage.RemovePrim(Sdf.Path(fixed_joint))

		omni.usd.get_context().save_as_stage(file)
	else:
		print(f"Failed to open {file} USD stage.")

