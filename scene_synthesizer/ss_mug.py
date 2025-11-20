import random
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer.usd_import import get_scene_paths
from scene_synthesizer.exchange.usd_export import add_mdl_material, bind_material_to_prims
from scene_synthesizer import utils
import itertools
import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import datasets
import json
kitchen_num = int(input("Assign a number for the kitchen. Enter: "))
# Output file
usd_filename = f"/home/exaFLOPs/Documents/IsaacLab/kitchen_{kitchen_num}.usd"
seed = None
random.seed(seed)

# -------- Generate Kitchen Scene -------- #
kitchen = ps.kitchen(seed=seed)
# Generate UV coordinates for certain primitives
kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')

# -------- Label support -------#
kitchen.label_support(
    label="base_cabinet",
    geom_ids="countertop_base_cabinet"
    )
kitchen.label_support(
    label="dishwasher",
    geom_ids="countertop_dishwasher"
    )

# -------- Insert Objects -------- #
mug = pa.MugAsset(origin=("com", "com", "bottom"))
bowl = pa.BowlAsset(origin=("com", "com", "bottom"))
plate = pa.PlateAsset(origin=("com", "com", "bottom"), height= 0.06, thickness= 0.01)
data = datasets.load_dataset("BODex")
mesh_files = data.get_filenames()
import ipdb
ipdb.set_trace()
selected_mesh_fnames = random.sample(mesh_files, 1)
fname = selected_mesh_fnames[0] # .replace('.obj', '_simple.obj')

data = {}
data["bottle"] = fname


# Place mug using a regex for cabinet countertop
kitchen.place_object(
    obj_id="mug",
    obj_asset=mug,
    support_id="base_cabinet",
    obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04),
    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
)
kitchen.place_objects(
    obj_id_iterator=utils.object_id_generator("obj"),
    obj_asset_iterator= synth.assets.asset_generator(
            itertools.repeat(fname, 1),
            scale=0.1,
            up=(0, 1, 0),
            front=(0, 0, -1),
            origin=("com", "bottom", "com"),
            align=True,
        ),
    obj_support_id_iterator=kitchen.support_generator(support_ids="base_cabinet"),
    obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04),
    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
)
kitchen.place_object(
    obj_id="plate",
    obj_asset=plate,
    support_id="dishwasher",
    obj_position_iterator=utils.PositionIteratorGrid(step_x=0.01, step_y=0.01, noise_std_x=0.04, noise_std_y=0.04),
    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
)
kitchen.place_object(
    obj_id="bowl",
    obj_asset=bowl,
    support_id="dishwasher",
    obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04),
    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
)
# -------- Export USD Stage -------- #
stage = kitchen.export(file_type='usd')

# Save USD file
stage.Export(usd_filename)
print(f"Kitchen with mug exported to {usd_filename}")

# Load Usd file
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
from pxr import UsdGeom, Gf, Usd
import omni.usd
import omni.physx

# open sim assign the file name show pic
