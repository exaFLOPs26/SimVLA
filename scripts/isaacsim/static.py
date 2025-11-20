from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

import sys
from pxr import Usd, UsdPhysics, UsdGeom
import omni.usd

# usage:
#	python make_kinematic_existing_rb.py /path/to/src.usd [/path/to/out.usd]
SRC = sys.argv[1]
DST = sys.argv[2] if len(sys.argv) > 2 else None

usd = omni.usd.get_context()
usd.open_stage(SRC)
stage: Usd.Stage = usd.get_stage()

# --- 1) collect & delete joints ---
joint_paths = []
for prim in stage.Traverse():
	t = prim.GetTypeName() or ""
	if t.endswith("Joint"):
		joint_paths.append(prim.GetPath())

for p in joint_paths:
	if 'sink_cabinet' in str(p):
		continue
	stage.RemovePrim(p)
print(f"Removed {len(joint_paths)} joint(s).")

# --- 2) set kinematicEnabled=True ONLY where RigidBodyAPI already exists ---
changed = 0
for prim in stage.Traverse():
	# only touch prims that already have the API
	if "mug0" in str(prim.GetPath()) or "sink_cabinet" in str(prim.GetPath()):
		continue
	if prim.HasAPI(UsdPhysics.RigidBodyAPI):
		# (Do not Apply; construct API wrapper over existing schema.)
		rb = UsdPhysics.RigidBodyAPI(prim)
		# Only set kinematicEnabled; DO NOT modify rigidBodyEnabled
		rb.CreateKinematicEnabledAttr(True)
		changed += 1

print(f"Set kinematicEnabled=True on {changed} existing rigid bodies.")

# --- 3) save ---
if DST:
	omni.usd.get_context().save_as_stage(DST)
	print(f"Saved to {DST}")
else:
	omni.usd.get_context().save_as_stage(SRC)
	print(f"Overwrote {SRC}")

