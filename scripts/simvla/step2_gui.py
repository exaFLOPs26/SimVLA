from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})	# must be False to render!
app_launcher.app

import os
import json
from pxr import UsdGeom, Gf, Usd, Sdf
import omni.usd
import ipdb
kitchen_num = int(input("Assign a number for the kitchen. Enter: "))

# Path to JSON
bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")

if kitchen_num < 10:
	json_path = os.path.join(bodex_dir, f"kitchen_data_0{kitchen_num}.json")
else:
	json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")

# Load dictionary
with open(json_path, "r") as f:
	kitchen_data = json.load(f)

# Base USD (this will be opened first)
base_usd = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}.usd"

# Get the USD context
usd_context = omni.usd.get_context()
success = usd_context.open_stage(base_usd)
init_file = "/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/__init__.py"
if success:
	print(f"Kitchen {kitchen_num} usd is opened.")
	stage = usd_context.get_stage()

	# Add metadata back into the USD for each object
	for obj, fname in kitchen_data.items():
		if obj == "kitchen_type":
			continue
		# Change names	  
		old_path = f"/world/{obj}0"
		new_path = f"/world/{obj}"
		if stage.GetPrimAtPath(old_path):
			Sdf.CopySpec(stage.GetRootLayer(), old_path,
						 stage.GetRootLayer(), new_path)
			stage.RemovePrim(old_path)
		# BODex grasp info	
		obj_prim = stage.GetPrimAtPath(f"/world/{obj}")
		if obj_prim:
			attr = obj_prim.CreateAttribute("BODex_path", Sdf.ValueTypeNames.String)
			attr.Set(fname)

	rotate_obj = input("Which object? Enter prim path.\nEnter: ")
	obj_prim = stage.GetPrimAtPath(rotate_obj)
	xformable = UsdGeom.Xformable(obj_prim)

	# --- Get existing ops ---
	xform_ops = xformable.GetOrderedXformOps()

	# Find an existing translate op (if any)
	translate_ops = [op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
	translate_op = translate_ops[0] if translate_ops else None

	# Find an existing rotateXYZ op (if any)
	rotate_ops = [op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
	rotate_op = rotate_ops[0] if rotate_ops else None

	# Add rotate op if missing
	if rotate_op is None:
		rotate_op = xformable.AddRotateXYZOp()

	# Make sure the op order is translate first then rotate
	if translate_op:
		xformable.SetXformOpOrder([translate_op, rotate_op])
	else:
		xformable.SetXformOpOrder([rotate_op])

	# --- Loop to export rotated versions ---
	for i in range(12):
		angle = i * 30.0
		
		# Set only rotation; translation stays untouched automatically
		rotate_op.Set(Gf.Vec3f(0.0, 0.0, angle))

		# create a new usd file name
		usd_out = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num:02d}_{i:02d}.usd"
		stage.GetRootLayer().Export(usd_out)
		print(f"Saved rotated kitchen USD: {usd_out}")

		with open(init_file, "r") as f:
			lines = f.readlines()
		lines.append(f"""
gym.register(
    id="Isaac-Kitchen-v{kitchen_num:02d}-{i:02d}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.kitchen_{kitchen_num:02d}_{i:02d}:AnubisKitchenEnvCfg",
    }},
)
""")	
		with open(init_file, "w") as f:
			f.writelines(lines)

	print("All 12 rotated USDs created and register gym env.")
else:
	print(f"Failed to open {base_usd}")

