from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

import sys
from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
import omni.usd

# usage:
#   python make_collision_only_except_four.py /path/to/src.usd [/path/to/out.usd]
SRC = sys.argv[1]
DST = sys.argv[2] if len(sys.argv) > 2 else None

usd = omni.usd.get_context()
usd.open_stage(SRC)
stage: Usd.Stage = usd.get_stage()

# Exceptions that must RETAIN COLLISION:
EXCEPT_COLLISION_KEYS = ("mug0", "sink_cabinet", "dishwasher", "countertop_dishwasher")
# Exceptions that must remain DYNAMIC (non-kinematic):
EXCEPT_DYNAMIC_KEYS   = ("mug0", "sink_cabinet")

def path_has_any(p: str, keys) -> bool:
    p = p.lower()
    return any(k in p for k in keys)

def is_collision_exception(p: str) -> bool:
    return path_has_any(p, EXCEPT_COLLISION_KEYS)

def is_dynamic_exception(p: str) -> bool:
    return path_has_any(p, EXCEPT_DYNAMIC_KEYS)

# --- 1) Remove joints unless they connect to a dynamic exception (bottle0/sink_cabinet) ---
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
        if any(is_dynamic_exception(tgt.pathString) for tgt in body0):
            keep = True
        if any(is_dynamic_exception(tgt.pathString) for tgt in body1):
            keep = True
        # fallback: match joint prim path itself
        if is_dynamic_exception(str(prim.GetPath())):
            keep = True
    except Exception:
        keep = False

    if not keep:
        stage.RemovePrim(prim.GetPath())
        removed_joints += 1

print(f"Removed {removed_joints} joint(s).")

# --- 2) Strip rigid bodies from ALL non-dynamic-exception prims ---
removed_rb = removed_physx_rb = removed_artic = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())

    # If NOT a dynamic exception, remove rigid body & articulation APIs
    if not is_dynamic_exception(p):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_rb += 1
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            removed_physx_rb += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed_artic += 1
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            removed_artic += 1

print(f"Removed UsdPhysics.RigidBodyAPI from {removed_rb} prim(s).")
print(f"Removed PhysxSchema.PhysxRigidBodyAPI from {removed_physx_rb} prim(s).")
print(f"Removed articulation APIs from {removed_artic} prim(s).")

# --- 3) Remove collisions from everything EXCEPT the four collision exceptions ---
removed_collision = removed_physx_collision = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if is_collision_exception(p):
        continue  # keep collisions here

    if prim.HasAPI(UsdPhysics.CollisionAPI):
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        removed_collision += 1
    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
        removed_physx_collision += 1
    # (Optional) You could also remove UsdPhysics.MeshCollisionAPI, but without
    # CollisionAPI it won't create colliders. Usually safe to leave as metadata.

print(f"Removed UsdPhysics.CollisionAPI from {removed_collision} prim(s).")
print(f"Removed PhysxSchema.PhysxCollisionAPI from {removed_physx_collision} prim(s).")
print("Kept collisions only on: bottle0, sink_cabinet, dishwasher, countertop_dishwasher.")

# --- 4) Ensure dynamic exceptions are actually dynamic & colliding ---
ensured_dynamic = ensured_collision = 0
for prim in stage.Traverse():
    p = str(prim.GetPath())

    if is_dynamic_exception(p):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(False)
            ensured_dynamic += 1
        # Ensure they have collision bound
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            ensured_collision += 1
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            ensured_collision += 1

    elif is_collision_exception(p):
        # For dishwasher/countertop_dishwasher: keep as static colliders
        # (no rigid body). Ensure collision exists.
        need = False
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim); need = True
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim); need = True
        if need:
            ensured_collision += 1

print(f"Ensured kinematicEnabled=False on {ensured_dynamic} dynamic-exception rigid bodies.")
print(f"Ensured collision present on {ensured_collision} exception prim(s).")

# --- 5) save ---
if DST:
    omni.usd.get_context().save_as_stage(DST)
    print(f"Saved to {DST}")
else:
    omni.usd.get_context().save_as_stage(SRC)
    print(f"Overwrote {SRC}")

