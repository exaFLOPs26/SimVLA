from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": False})  # must be False to render!
app_launcher.app

import os
import json
from pxr import UsdGeom, Gf, Usd, Sdf
import omni.usd
import omni.kit.viewport.utility as vp
import ipdb
import math
kitchen_num = int(input("Assign a number for the kitchen. Enter: "))

# Path to JSON
bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")

# Output USD
usd_filename = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}.usd"

# Load dictionary
with open(json_path, "r") as f:
	kitchen_data = json.load(f)

# Get the USD context
usd_context = omni.usd.get_context()
success = usd_context.open_stage(usd_filename)

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
		Sdf.CopySpec(stage.GetRootLayer(), old_path,
				stage.GetRootLayer(), new_path)
		stage.RemovePrim(old_path)
		# BODex grasp info  
		obj_prim = stage.GetPrimAtPath(f"/world/{obj}")
		if obj_prim:
			attr = obj_prim.CreateAttribute("BODex_path", Sdf.ValueTypeNames.String)
			attr.Set(fname)
		# TODO Make 12 different kitchen usd with different orientation.		
		for i in range(12):



	# ---- Handle ----
#	handle = input("Please select a type of handle for cabinets. \n 1. long \n 2. Small circle \n 3.\n")
#	handle_path = "/world/base_cabinet/door_1_2/door_handle_3"
## --- 1. Read old handle transform before removing it ---
#	old_prim = stage.GetPrimAtPath(handle_path)
#
## --- 2. Remove old handle prim ---
#	stage.RemovePrim(handle_path)
#
## --- 3. Create new handle prim ---
#	new_handle = stage.DefinePrim(handle_path, "Xform")
#	new_handle.GetReferences().AddReference("/root/IsaacLab/knob.usd")
#
## --- 4. Apply the old transform to new handle ---
#	xformable = UsdGeom.Xformable(new_handle)
#
## --- 5. Save stage ---
	stage.GetRootLayer().Save()
	usd_context.save_as_stage(usd_filename)
#	# Get the active viewport window
#	viewport = vp.get_active_viewport()
#	# Save a screenshot
#	screenshot_path = f"/root/IsaacLab/scripts/simvla/scene/kitchen_{kitchen_num}.png"
#	vp.capture_viewport_to_file(viewport, screenshot_path)
#	print(f"Screenshot saved: {screenshot_path}")

else:
	print(f"Failed to open {usd_filename}")

