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

# Output file
usd_file_name = "./Kitchen/Kichen_01.usd"
seed = 42  # Fix number
random.seed(seed)

task = input("Hello, I found out that you want to generate a kitchen. What task you want to try? \n Examples: \n 1. Put bowl into sink. \n 2. Restock water to refrigerator. \n Enter: ")
print("-" * 80)
objects = input("I see. Then what objects will you need enter like (bowl, water bottle, plate, apple) \n Enter: ")

obj_list = objects.strip("()").replace(" ", "").split(",")

for obj in obj_list:
    place = int(input(f"Where do you want to put {obj} in the kicthen scene? You can put into the followings. \n 1. Inside refrigerator \n 2. Above cabinet \n 3. Above island \n 4. Above range \n 4. Above dishwasheri \n Enter number:  "))

    if place == 1:


# Generate Kitchen scene
kitchen = ps.kitchen(seed=seed)
kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')

#

