from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
import omni.usd
from pxr import UsdGeom, Gf, Usd
import omni.usd
import omni.physx
import torch
import math
import random
import ipdb

usd_context = omni.usd.get_context()
usd_file_path = "/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_12.usd"   # Replace with your actual path
success = usd_context.open_stage(usd_file_path)

stage = usd_context.get_stage()
handle_path = "/world/base_cabinet/drawer_0_0/door_handle"
handle = stage.GetPrimAtPath(handle_path)
transform = UsdGeom.Xformable(handle).GetLocalTransformation()
world_transform = UsdGeom.Xformable(handle).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
stage.RemovePrim(handle_path)

new_handle = stage.DefinePrim("/world/base_cabinet/drawer_0_0/door_handle", "Xform")
new_handle.GetReferences().AddReference("/root/IsaacLab/knob.usd")
UsdGeom.Xformable(new_handle).AddTransformOp().Set(transform)
stage.GetRootLayer().Save()


output = "/root/IsaacLab/no_handle.usd"
omni.usd.get_context().save_as_stage(output)


