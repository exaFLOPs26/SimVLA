from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

import sys
from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
import omni.usd

# usage:
#   python make_collision_only_except_two.py /path/to/src.usd [/path/to/out.usd]
SRC = sys.argv[1]
DST = sys.argv[2] if len(sys.argv) > 2 else None

usd = omni.usd.get_context()
usd.open_stage(SRC)
stage: Usd.Stage = usd.get_stage()

EXCEPT_KEYS = ("bottle0", "sink_cabinet")

def is_exception_path(s: str) -> bool:
    s = s.lower()
    return any(k in s for k in EXCEPT_KEYS)

# --- 1) Remove joints unless they connect to either exception ---
removed_joints = 0
for prim in list(stage.Traverse()):
    t = prim.GetTypeName() or ""
    if not t.endswith("Joint"):
        continue

    keep = False
    try:
        j = UsdPhysics.Joint(prim)
        body0 = j.GetBody0Rel().GetTargets()
        body1 = j.GetBody1Rel().GetTargets()
        if any(is_exception_path(tgt.pathString) for tgt in body0):
            keep = True
        if any(is_exception_path(tgt.pathString) for tgt in body1):
            keep = True
        # fallback: joint prim path contains exception name
        if is_exception_path(str(prim.GetPath())):
            keep = True
    except Exception:
        keep = False

    if not keep:
        stage.RemovePrim(prim.GetPath())
        removed_joints += 1

print(f"Removed {removed_joints} joint(s).")

# --- 2) For non-exceptions: remove rigid body APIs, keep collisions as-is ---
removed_rb = 0
removed_physx_rb = 0
removed_artic = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if is_exception_path(p):
        continue  # handled below

    # Remove USD RigidBody (makes it non-simulated)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        removed_rb += 1

    # Remove PhysX RigidBody (provider-specific)
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        removed_physx_rb += 1

    # If this object was an articulation root, remove that too (prevents solver work)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        removed_artic += 1
    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        removed_artic += 1

print(f"Removed UsdPhysics.RigidBodyAPI from {removed_rb} prim(s).")
print(f"Removed PhysxSchema.PhysxRigidBodyAPI from {removed_physx_rb} prim(s).")
print(f"Removed articulation APIs from {removed_artic} prim(s).")
print("Kept any existing CollisionAPI / PhysxCollisionAPI (so they remain colliders).")

# --- 3) Ensure exceptions are dynamic (interactive) ---
ensured_dynamic = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not is_exception_path(p):
        continue
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateKinematicEnabledAttr().Set(False)
        ensured_dynamic += 1
    # (Optional) ensure they actually have collision bound
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)

print(f"Ensured exceptions dynamic (kinematicEnabled=False) on {ensured_dynamic} prim(s).")

# --- 4) save ---
if DST:
    omni.usd.get_context().save_as_stage(DST)
    print(f"Saved to {DST}")
else:
    omni.usd.get_context().save_as_stage(SRC)
    print(f"Overwrote {SRC}")

