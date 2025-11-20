from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd

import ipdb
import glob
import os
from pxr import Usd, UsdGeom, Sdf

# Read folder where the usd are located
folder_path = input("Enter the folder path: ")
files = glob.glob(f"{folder_path}/kitchen_1024_05.usd")

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
#if fixed_prim.GetRelationships()[0].GetTargets() != []:
#				fixed_prim.GetRelationships()[0].RemoveTarget(fixed_prim.GetRelationships()[0].GetTargets()[0])	
#			if 'countertop' in fixed_prim.GetPath().pathString or 'corner' in fixed_prim.GetPath().pathString or 'hood' in fixed_prim.GetPath().pathString:
			if '/corner' in fixed_prim.GetPath().pathString:
				ipdb.set_trace()
				layer = stage.GetRootLayer()
				src_path0 = Sdf.Path("/world/corner/geometry_0")
#src_path1 = Sdf.Path("/world/corner/geometry_1")
				dst_path0 = Sdf.Path("/world/corner/corpus/geometry_0")
#				dst_path1 = Sdf.Path("/world/corner/corpus/geometry_1")
				Sdf.CopySpec(layer, src_path0, layer, dst_path0)
#				Sdf.CopySpec(layer, src_path1, layer, dst_path1)
				stage.RemovePrim(src_path0)	
#				stage.RemovePrim(src_path1)	

				temp_target_path = stage.GetPrimAtPath(fixed_prim.GetRelationships()[1].GetTargets()[0].pathString+"/corpus")
				if fixed_prim.GetRelationships()[1].GetTargets() != []:
					fixed_prim.GetRelationships()[1].RemoveTarget(fixed_prim.GetRelationships()[1].GetTargets()[0])	
				relationship = fixed_prim.CreateRelationship('physics:body1')
				relationship.SetTargets([temp_target_path.GetPath()])
#				stage.RemovePrim(fixed_prim.GetPath())
#				continue
#				ipdb.set_trace()
#				new_target_path = new_target_path + "/" + fixed_prim.GetPath().pathString.split('/')[2].removeprefix("countertop_")
			if fixed_prim.GetRelationships()[0].GetTargets() == []:
				relationship = fixed_prim.CreateRelationship('physics:body0')
				relationship.SetTargets([stage.GetPrimAtPath(new_target_path).GetPath()])

		omni.usd.get_context().save_as_stage(file)
	else:
		print(f"Failed to open {file} USD stage.")

