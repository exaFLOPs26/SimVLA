from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd
import ipdb
from pxr import Usd, UsdPhysics

# --- Open the stage ---
usd_context = omni.usd.get_context()
usd_context.open_stage("/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_1103_00.usd")
stage = usd_context.get_stage()

# --- Config ---
TARGET_SUBSTR = "bottle0"
NEW_PATH = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_1103_00.usd"

def ensure_sdf_approx_on_prim(prim) -> bool:
	"""
	Ensure MeshCollisionAPI.approximation is 'sdf' on the given prim.
	Returns True if it was changed (or created as 'sdf'), False if already 'sdf' or not applicable.
	"""
	# Apply MeshCollisionAPI if missing
	if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
		# Some prims (e.g., Xforms) may not be valid mesh colliders, but we still try to apply.
		UsdPhysics.MeshCollisionAPI.Apply(prim)

	mesh_api = UsdPhysics.MeshCollisionAPI(prim)
	if not mesh_api:
		return False  # couldn't attach/use the API

	approx_attr = mesh_api.GetApproximationAttr()
	if not approx_attr or not approx_attr.HasAuthoredValueOpinion():
		approx_attr = mesh_api.CreateApproximationAttr()  # creates the token-typed attr

	current = approx_attr.Get()
	if current != "sdf":
		print(f"[UPDATE] {prim.GetPath()}: approximation {current!r} -> 'sdf'")
		approx_attr.Set("sdf")
		return True
	else:
		print(f"[OK]	 {prim.GetPath()}: already 'sdf'")
		return False

changed = 0
checked = 0

# Traverse every prim and match any with 'bottle0' in name or path
for prim in stage.Traverse():
	name = prim.GetName()
	path_str = str(prim.GetPath())
	if TARGET_SUBSTR in name or TARGET_SUBSTR in path_str:
		checked += 1
		if ensure_sdf_approx_on_prim(prim):
			changed += 1

print(f"\nSummary: checked {checked} prim(s) containing '{TARGET_SUBSTR}', updated {changed} to 'sdf'.")

# --- Save to a new file (non-destructive) ---
usd_context.save_as_stage(NEW_PATH)
print(f"✅ Saved updated stage to: {NEW_PATH}")

# Optional: drop into debugger right before closing if you want to inspect interactively
# ipdb.set_trace()

# Clean shutdown (especially helpful when running many times)
try:
	app_launcher.close()
except Exception:
	pass

