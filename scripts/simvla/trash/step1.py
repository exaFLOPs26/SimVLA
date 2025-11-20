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
import ipdb
import os
kitchen_num = int(input("Assign a number for the kitchen. Enter: "))
# Output file
usd_filename = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}.usd"
seed = None
random.seed(seed)

# Data save for BODex 
kitchen_data = {}

# -------- Generate Kitchen Scene -------- #
kitchen_options = {
    0: (ps.kitchen_island, "island"),
    1: (ps.kitchen_l_shaped, "l_shaped"),
    2: (ps.kitchen_peninsula, "peninsula"),
    3: (ps.kitchen_u_shaped, "u_shaped"),
    4: (ps.kitchen_single_wall, "single_wall"),
}

kitchen_type = random.choice(list(kitchen_options.keys()))  # pick random type

# unpack the function + string from dict
kitchen_func, kitchen_name = kitchen_options[kitchen_type]

kitchen = kitchen_func(seed=seed, counter_height = 0.95)               # call the correct function
kitchen_data["kitchen_type"] = kitchen_name     # store the name

# Generate UV coordinates for certain primitives
kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')

# -------- Object -------- #
data = datasets.load_dataset("BODex")
mesh_files = data.get_filenames()

# -------- Label support --------- #
if kitchen_name == "island":
	kitchen.label_support(
		label="base_cabinet",
		geom_ids="countertop_base_cabinet"
		)
	kitchen.label_support(
		label="dishwasher",
		geom_ids="countertop_dishwasher"
		)
	kitchen.label_support(
		label="island",
		geom_ids="kitchen_island/countertop"
		)
elif kitchen_name == "l_shaped":
	kitchen.label_support(
		label="base_cabinet",
		geom_ids="countertop_base_cabinet"
		)
	kitchen.label_support(
		label="dishwasher",
		geom_ids="countertop_dishwasher"
		)

elif kitchen_name == "peninsula":
	kitchen.label_support(
		label="base_cabinet",
		geom_ids="countertop_base_cabinet"
		)
	kitchen.label_support(
		label="base_cabinet_0",
		geom_ids="countertop_base_cabinet_0"
		)
	kitchen.label_support(
		label="base_cabinet_1",
		geom_ids="countertop_base_cabinet_1"
		)
	kitchen.label_support(
		label="dishwasher",
		geom_ids="countertop_dishwasher"
		)
	
elif kitchen_name == "u_shaped":
	kitchen.label_support(
		label="base_cabinet",
		geom_ids="countertop_base_cabinet"
		)
	kitchen.label_support(
		label="base_cabinet_0",
		geom_ids="countertop_base_cabinet_0"
		)
	kitchen.label_support(
		label="dishwasher",
		geom_ids="countertop_dishwasher"
		)
	
elif kitchen_name == "single_wall":
	kitchen.label_support(
		label="base_cabinet",
		geom_ids="countertop_base_cabinet"
		)
	kitchen.label_support(
		label="dishwasher",
		geom_ids="countertop_dishwasher"
		)
# Refrigerator
kitchen.label_support(
		label="refrigerator_1st",
		geom_ids="refrigerator/shelf_3",
		)
kitchen.label_support(
		label="refrigerator_2nd",
		geom_ids="refrigerator/shelf_2",
		)
kitchen.label_support(
		label="refrigerator_3nd",
		geom_ids="refrigerator/shelf_1",
		)
kitchen.label_support(
		label="refrigerator_4nd",
		geom_ids="refrigerator/shelf_0",
		)

obj_dict = {
	"bottle": 0,
	"bowl": 0,
	"apple": 0,
	"sodacan": 0,
	"nutella": 0,
	"mug": 0,
}

while True:
	obj = input("Which object do you want to put in the kitchen? Choose among the list. If you are done, enter DONE.  \n - bottle \n - bowl \n - apple \n - SodaCan \n - nutella \n - mug \n Enter: ")
	obj = obj.lower()
	if obj == "done":
		break
	else:
		# Random choose object 
		obj_list = [elmt for elmt in mesh_files if obj in elmt.lower()]
		selected_mesh_fnames = random.sample(obj_list, 1)
		fname = selected_mesh_fnames[0]
		# Assign numbers 
		obj_n = obj+str(obj_dict[obj])
		kitchen_data[obj_n] = fname
		obj_dict[obj] += 1
		
		# Orientation
		if obj in ["apple", "sodacan"]:
			up = (0,0,1)
			front = (0,1,0)
			origin = ("com", "com", "bottom")
		elif obj in ["bottle", "bowl"]:
			up = (0,1,0)
			front = (0,0,-1)
			origin = ("com", "bottom", "com")

		# where to place 
		loc = int(input(f"Where do you want to put {obj}? Choose a number among the list. \n - 1.Above cabinet \n - 2.Above dishwasher \n - 3.In refrigerator \n - 4.Above Island. If there is one. \n Enter: "))
		if loc == 1:
			if kitchen_name in ["island", "l_shaped", "single_wall"]:
				place = "base_cabinet"
			elif kitchen_name == "peninsula":
				place = random.choice(["base_cabinet", "base_cabinet_0", "base_cabinet_1"])
			elif kitchen_name == "u_shaped":
				place = random.choice(["base_cabinet", "base_cabinet_0"])
		elif loc == 2:
			place = "dishwasher"
		elif loc == 3:
			place = random.choice(["refrigerator_1st", "refrigerator_2nd"])
		elif loc == 4:
			if kitchen_name == "island":
				place = "island"
			else:
				print("The kitchen you generated have no island.")
				continue

		kitchen.place_objects(
			obj_id_iterator=utils.object_id_generator(obj_n),
			obj_asset_iterator= synth.assets.asset_generator(
					itertools.repeat(fname, 1),
					scale=0.1,
					up=up,
					front=front,
					origin=origin,
					align=True,
				),
			obj_support_id_iterator=kitchen.support_generator(support_ids=place),
			obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04),
			obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
		)

# -------- Export USD Stage -------- #
stage = kitchen.export(file_type='usd')
# Save USD file
stage.Export(usd_filename)
print(f"Kitchen exported to {usd_filename}")

bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")

with open(json_path, "w") as f:
    json.dump(kitchen_data, f, indent=4)

print(f"kitchen_data saved to {json_path}")
