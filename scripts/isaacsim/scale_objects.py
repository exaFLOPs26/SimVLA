from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
from pxr import UsdGeom, Gf

import omni.usd

import ipdb
# Get the USD context
usd_context = omni.usd.get_context()
usd_file_path = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen/Kitchen_01.usd"  # Replace with your actual path
success = usd_context.open_stage(usd_file_path)

if success:
	print("Stage opened successfully!")
	stage = usd_context.get_stage()
	print([prim.GetPath() for prim in stage.Traverse()])
	object_prim_path = "/world/obj0"
	object_prim = stage.GetPrimAtPath(object_prim_path)
	xform = UsdGeom.Xformable(object_prim)
	
	scale_op = None
	for op in xform.GetOrderedXformOps():
		if op.GetOpType() == UsdGeom.XformOp.TypeScale:
			scale_op = op
			break

	# If the scale op does not exist, add a new one
	if not scale_op:
		scale_op = xform.AddScaleOp()	
	# Define the new scale as a 3D vector (x, y, z)
	# A value of (2.0, 2.0, 2.0) will double the object's size uniformly
	new_scale = Gf.Vec3d(0.7, 0.7, 0.7)

	# Set the scale value
	scale_op.Set(new_scale)

	omni.usd.get_context().save_as_stage(usd_file_path)
else:
	print("Failed to open USD stage.")
