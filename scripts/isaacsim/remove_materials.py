from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd
import ipdb

usd_file_path = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen/Kitchen_01.usd"  # Replace with your actual path
success = usd_context.open_stage(usd_file_path)

if success:
    print("Stage opened successfully!")
    stage = usd_context.get_stage()
    print([prim.GetPath() for prim in stage.Traverse()])

    for prim in stage.Traverse():
		if prim.HasProperty('material:binding'):
			rel = prim.GetRelationship('material:binding')
			rel.ClearTargets(False)
			# To be extra sure, you can also remove the property itself
			prim.RemoveProperty('material:binding')
			print(f"Cleared material binding on: {prim.GetPath()}")
    
	print("Removing '/world/Looks' prim...")
	looks_prim_path = '/world/Looks'
	looks_prim = stage.GetPrimAtPath(looks_prim_path)
	if looks_prim.IsValid():
		stage.RemovePrim(looks_prim_path)
		print(f"Successfully removed '{looks_prim_path}'.")
	else:
		print(f"Prim '{looks_prim_path}' not found. No action taken.")

		# Step 3: Save the modified stage
	usd_context.save_as_stage(output_usd_path)
