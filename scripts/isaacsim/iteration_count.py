from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd

import ipdb
import glob
import os
from pxr import Usd, UsdGeom, Sdf, PhysxSchema, UsdPhysics
# Read folder where the usd are located
folder_path = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen"
files = glob.glob(f"{folder_path}/kitchen_1103_00.usd")

# Get the USD context
for file in files:
	usd_context = omni.usd.get_context()
	success = usd_context.open_stage(file)

	if success:
		stage = usd_context.get_stage()
		n_checked = n_pos_set = n_vel_set = 0

		for prim in stage.Traverse():
			if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
				continue
			n_checked += 1
			rb = PhysxSchema.PhysxRigidBodyAPI(prim)

			# Position iterations: author if missing, then set to 4
			pos_attr = rb.GetSolverPositionIterationCountAttr()
			if not pos_attr or not pos_attr.HasAuthoredValue():  # authorship check
				pos_attr = rb.CreateSolverPositionIterationCountAttr()

			ipdb.set_trace()
			pos_attr.Set(4)
			n_pos_set += 1

			# Velocity iterations: author if missing, then set to 1
			vel_attr = rb.GetSolverVelocityIterationCountAttr()
			if not vel_attr or not vel_attr.HasAuthoredValue():
				vel_attr = rb.CreateSolverVelocityIterationCountAttr()
			vel_attr.Set(1)
			n_vel_set += 1

		print(f"Checked rigid-body prims: {n_checked}")
		print(f"Set position-iteration attrs: {n_pos_set}")
		print(f"Set velocity-iteration attrs: {n_vel_set}")


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

