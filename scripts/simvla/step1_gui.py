import tkinter as tk
from tkinter import ttk, messagebox, Listbox
import random
import itertools
import numpy as np
import json
import os
import ipdb # Keep this if you use it for debugging

# --- Import your project's specific modules ---
# Make sure these modules are accessible in your Python environment
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer.usd_import import get_scene_paths
from scene_synthesizer.exchange.usd_export import add_mdl_material, bind_material_to_prims
from scene_synthesizer import utils
import scene_synthesizer as synth
from scene_synthesizer import datasets


DEFAULT_TEXTURE_SCALE = 0.25

url_mdl_material = 'http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/'

# A manually defined dictionary mapping specific materials to types of objects and parts
materials = {
    'sink': [
        ('Base/Stone/Ceramic_Smooth_Fired.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'countertop': [
        ('vMaterials_2/Metal/Copper_Hammered.mdl', 'Copper_Hammered_Shiny', 0.5),
        ('Base/Stone/Marble.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Dark.mdl', 'Granite_Dark', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Light.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Metal/Stainless_Steel_Milled.mdl', 'Stainless_Steel_Milled_Worn', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Slate.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4_Linen.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Ceramic_Tile_12.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Smooth.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Stone_Natural_Black.mdl', 'Stone_Natural_Black_Shiny', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Steel_Grey.mdl', 'Steel_Grey_Bright', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Basaltite.mdl', 'Basaltite_Worn', DEFAULT_TEXTURE_SCALE),
    ],
    'glass': [
        ('Base/Glass/Tinted_Glass_R85.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'tinted glass': [
        ('Base/Glass/Tinted_Glass_R75.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'cabinet': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Ash', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Denim', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Light_Denim', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Bamboo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Cherry.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
    ],
    'rusted metal': [
        ('Base/Metals/RustedMetal.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'glossy black': [
        ('vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl', 'Black', DEFAULT_TEXTURE_SCALE)
    ],
    'handle': [
        ('vMaterials_2/Metal/Silver_Foil.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'appliances': [
        ('vMaterials_2/Metal/Aluminum_Brushed.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'wall': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Pale_Rose', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Lime', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Mint', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Red_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Graphite_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Glazed_Subway', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_Yellow', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Sloppy_Paint_Job', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall_Cracked', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Cappucino', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Antique_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Lime_Green_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Graphite_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Mint_Varied', DEFAULT_TEXTURE_SCALE),
    ],
    'floor': [
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Parquet_Floor.mdl', 'Parquet_Floor', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Concrete/Concrete_Floor_Damage.mdl', 'Concrete_Floor_Damage', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Beech.mdl', 'Wood_Tiles_Beech_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Adobe_Octagon_Dots.mdl', 'Adobe_Octagon_Dots', None),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Mosaic', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terracotta.mdl', 'Terracotta', None),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Square.mdl', 'Ceramic_Tiles_Square_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Paseo.mdl', 'Ceramic_Tiles_Paseo_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
    ],
    'white plastic': [
        ('vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl', 'Plastic_Thick_Translucent', DEFAULT_TEXTURE_SCALE),
    ],
}

# A dictionary mapping scene object/part names to types the general types of objects/parts defined above
# The keys are regular expressions of prim_paths in the USD
geometry2material = {
    "(.*cabinet.*corpus.*|.*cabinet.*door|.*drawer.*board.*|.*cabinet.*closed.*|/world/kitchen_island/.*)|/world/corner.*": "cabinet",
    "(.*refrigerator.*|.*range_hood.*|.*range.*|.*dishwasher.*)": "appliances",
    "/world/range/corpus/heater.*": 'rusted metal',
    "/world/range/corpus/top": 'glossy black',
    ".*/corpus/sink": 'sink',
    ".*countertop.*": "countertop",
    ".*handle.*": "handle",
    "/world/range/.*window": 'tinted glass',
    "(.*glass|.*_window)": 'glass',
    "/world/plate.*": 'sink',
    "(/world/wall/geometry.*|/world/wall_(x|y|_y|_x)/geometry_0)": 'wall',
    "/world/floor/geometry.*": 'floor',
    ".*dishwasher.*basket": 'white plastic',
}


class KitchenBuilderApp(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Kitchen Scene Generator")
		self.geometry("600x480") # Made window a bit taller

		# --- Data and State Management ---
		self.objects_to_place = []
		self.obj_dict = {"bottle": 0, "bowl": 0, "apple": 0, "sodacan": 0, "nutella": 0, "mug": 0}
		
		# --- NEW: Define kitchen options as an instance variable ---
		self.kitchen_options = {
			0: (ps.kitchen_island, "island"), 1: (ps.kitchen_l_shaped, "l_shaped"),
			2: (ps.kitchen_peninsula, "peninsula"), 3: (ps.kitchen_u_shaped, "u_shaped"),
			4: (ps.kitchen_single_wall, "single_wall"),
		}
		self.kitchen_name = None # Will store the selected kitchen name

		try:
			self.bodex_data = datasets.load_dataset("BODex")
			self.mesh_files = self.bodex_data.get_filenames()
		except Exception as e:
			messagebox.showerror("Error", f"Could not load BODex dataset. Make sure it's installed correctly.\n\n{e}")
			self.destroy()
			return

		self.create_widgets()
		
		# --- NEW: Select the first kitchen type when the app starts ---
		self.select_new_kitchen()

	def select_new_kitchen(self):
		"""NEW: Randomly selects a new kitchen type and updates the GUI."""
		kitchen_type_key = random.choice(list(self.kitchen_options.keys()))
		_, self.kitchen_name = self.kitchen_options[kitchen_type_key]
		self.kitchen_type_var.set(f"Current Kitchen: {self.kitchen_name.replace('_', ' ').capitalize()}")
		
		# Clear any previously added objects since the kitchen is new
		self.objects_to_place.clear()
		self.objects_listbox.delete(0, tk.END)
		# Reset object counters
		self.obj_dict = {key: 0 for key in self.obj_dict}

	def create_widgets(self):
		main_frame = ttk.Frame(self, padding="10")
		main_frame.pack(fill=tk.BOTH, expand=True)

		top_frame = ttk.LabelFrame(main_frame, text="1. Kitchen Setup", padding="10")
		top_frame.pack(fill=tk.X, pady=5)
		
		# Kitchen Number Entry
		ttk.Label(top_frame, text="Kitchen Number:").grid(row=0, column=0, padx=5, sticky="w")
		self.kitchen_num_var = tk.StringVar()
		ttk.Entry(top_frame, textvariable=self.kitchen_num_var, width=10).grid(row=0, column=1, sticky="w")
		
		# --- NEW: Label to display kitchen type and a button to change it ---
		self.kitchen_type_var = tk.StringVar()
		ttk.Label(top_frame, textvariable=self.kitchen_type_var, font=("Helvetica", 10, "bold")).grid(row=1, column=0, pady=(10,0), sticky="w")
		ttk.Button(top_frame, text="Change Kitchen Type", command=self.select_new_kitchen).grid(row=1, column=1, pady=(10,0), sticky="w")

		middle_frame = ttk.LabelFrame(main_frame, text="2. Object Placement", padding="10")
		middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

		controls_frame = ttk.Frame(middle_frame)
		controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

		ttk.Label(controls_frame, text="Object Type:").pack(anchor=tk.W, pady=(0, 2))
		self.object_var = tk.StringVar()
		object_choices = ["bottle", "bowl", "apple", "sodacan", "nutella", "mug"]
		object_menu = ttk.Combobox(controls_frame, textvariable=self.object_var, values=object_choices, state="readonly", width=25)
		object_menu.pack(fill=tk.X, pady=(0, 10))
		object_menu.set(object_choices[0])

		ttk.Label(controls_frame, text="Placement Location:").pack(anchor=tk.W, pady=(0, 2))
		self.location_var = tk.StringVar()
		location_choices = ["1. Above cabinet", "2. Above dishwasher", "3. In refrigerator", "4. Above Island (if available)"]
		location_menu = ttk.Combobox(controls_frame, textvariable=self.location_var, values=location_choices, state="readonly", width=25)
		location_menu.pack(fill=tk.X, pady=(0, 15))
		location_menu.set(location_choices[0])

		ttk.Button(controls_frame, text="Add Object to List", command=self.add_object_to_list).pack(fill=tk.X)
		
		listbox_frame = ttk.Frame(middle_frame)
		listbox_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
		ttk.Label(listbox_frame, text="Objects to be Added:").pack(anchor=tk.W)
		self.objects_listbox = Listbox(listbox_frame, height=10)
		self.objects_listbox.pack(fill=tk.BOTH, expand=True)

		bottom_frame = ttk.Frame(main_frame)
		bottom_frame.pack(fill=tk.X, pady=(10, 0))
		generate_button = ttk.Button(bottom_frame, text="Generate Kitchen Scene", command=self.run_generation_process, style="Accent.TButton")
		generate_button.pack(pady=5)
		
		style = ttk.Style()
		style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))

	def add_object_to_list(self):
		obj = self.object_var.get()
		loc_str = self.location_var.get()
		loc_num = int(loc_str.split('.')[0])

		# --- NEW: Add a check for the island location ---
		if loc_num == 4 and self.kitchen_name != "island":
			messagebox.showwarning("Invalid Location", "The 'Above Island' location is only available for the 'Island' kitchen type.")
			return

		if not obj or not loc_str:
			messagebox.showwarning("Warning", "Please select both an object and a location.")
			return
			
		obj_list = [elmt for elmt in self.mesh_files if obj in elmt.lower()]
		if not obj_list:
			messagebox.showerror("Error", f"No mesh files found for object type: {obj}")
			return
		fname = random.choice(obj_list)

		obj_n = f"{obj}{self.obj_dict[obj]}"
		self.obj_dict[obj] += 1
		
		self.objects_to_place.append({"obj_n": obj_n, "type": obj, "fname": fname, "loc": loc_num})
		self.objects_listbox.insert(tk.END, f"{obj_n} -> placing on '{loc_str}'")

	def run_generation_process(self):
		kitchen_num_str = self.kitchen_num_var.get()
		if not kitchen_num_str.isdigit():
			messagebox.showerror("Error", "Please enter a valid integer for the kitchen number.")
			return
		
		kitchen_num = int(kitchen_num_str)
		
		if not self.objects_to_place:
			messagebox.showwarning("Warning", "Please add at least one object before generating the scene.")
			return

		try:
			usd_filename = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num:02d}.usd"
			seed = None
			random.seed(seed)
			kitchen_data = {}

			# --- MODIFIED: Use the pre-selected kitchen type ---
			kitchen_name = self.kitchen_name
			kitchen_data["kitchen_type"] = kitchen_name
			
			# Find the correct kitchen generation function
			kitchen_func = None
			for func, name in self.kitchen_options.values():
				if name == kitchen_name:
					kitchen_func = func
					break
			
			if kitchen_func is None:
				raise ValueError("Could not find the kitchen generation function.")

			kitchen = kitchen_func(seed=seed, counter_height=0.95)
			kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')
			
			# (The rest of this function is the same as before...)

			# 3. Label support
			if kitchen_name == "island":
				kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
				kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
				kitchen.label_support(label="island", geom_ids="kitchen_island/countertop")
			elif kitchen_name == "l_shaped":
				kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
				kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
			elif kitchen_name == "peninsula":
				kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
				kitchen.label_support(label="base_cabinet_0", geom_ids="countertop_base_cabinet_0")
				kitchen.label_support(label="base_cabinet_1", geom_ids="countertop_base_cabinet_1")
				kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
			elif kitchen_name == "u_shaped":
				kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
				kitchen.label_support(label="base_cabinet_0", geom_ids="countertop_base_cabinet_0")
				kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
			elif kitchen_name == "single_wall":
				kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
				kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
			kitchen.label_support(label="refrigerator_1st", geom_ids="refrigerator/shelf_3")
			kitchen.label_support(label="refrigerator_2nd", geom_ids="refrigerator/shelf_2")
			kitchen.label_support(label="refrigerator_3nd", geom_ids="refrigerator/shelf_1")
			kitchen.label_support(label="refrigerator_4nd", geom_ids="refrigerator/shelf_0")

			# 4. Place objects
			for obj_info in self.objects_to_place:
				# (This part is unchanged)
				obj_n, obj_type, fname, loc = obj_info["obj_n"], obj_info["type"], obj_info["fname"], obj_info["loc"]
				kitchen_data[obj_n] = fname
				if obj_type in ["apple", "sodacan"]: up, front, origin = (0,0,1), (0,1,0), ("com", "com", "bottom")
				elif obj_type in ["bottle", "bowl", "mug", "nutella"]: up, front, origin = (0,1,0), (0,0,-1), ("com", "bottom", "com")
				place = ""
				if loc == 1:
					if kitchen_name in ["island", "l_shaped", "single_wall"]: place = "base_cabinet"
					elif kitchen_name == "peninsula": place = random.choice(["base_cabinet", "base_cabinet_0", "base_cabinet_1"])
					elif kitchen_name == "u_shaped": place = random.choice(["base_cabinet", "base_cabinet_0"])
				elif loc == 2: place = "dishwasher"
				elif loc == 3: place = random.choice(["refrigerator_1st", "refrigerator_2nd"])
				elif loc == 4:
					if kitchen_name == "island": place = "island"
					else: continue
				if not place: continue
				kitchen.place_objects(obj_id_iterator=utils.object_id_generator(obj_n), obj_asset_iterator=synth.assets.asset_generator(itertools.repeat(fname, 1), scale=0.08, up=up, front=front, origin=origin, align=True), obj_support_id_iterator=kitchen.support_generator(support_ids=place), obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04), obj_orientation_iterator=utils.orientation_generator_uniform_around_z(lower=0.0, upper=0.0))

			stage = kitchen.export(file_type='usd')

			for geom_regex, material_group in geometry2material.items():
				# Find all geometry prims of a particular category
				paths = get_scene_paths(
					stage=stage,
					prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
					scene_path_regex=geom_regex,
				)

				# select random material
				if len(materials[material_group]) == 0:
					print(f"Warning: No materials for {material_group}")
					continue
				
				mtl_url, mtl_name, texture_scale = random.choice(materials[material_group])
				
				# add material to USD stage and bind to geometry prims
				mtl = add_mdl_material(
					stage=stage,
					mtl_url=url_mdl_material + mtl_url,
					mtl_name=mtl_name,
					texture_scale=texture_scale
					)
				bind_material_to_prims(
					stage=stage,
					material=mtl,
					prim_paths=paths
					)

			# 5. Export
			stage.Export(usd_filename)
			bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
			os.makedirs(bodex_dir, exist_ok=True)
			json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num:02d}.json")
			with open(json_path, "w") as f: json.dump(kitchen_data, f, indent=4)
			
			messagebox.showinfo("Success!", f"Scene generation complete!\n\nKitchen Type: {kitchen_name.replace('_', ' ').capitalize()}\n\nUSD saved to: {usd_filename}\nJSON saved to: {json_path}")

		except Exception as e:
			messagebox.showerror("An Error Occurred", f"Failed during scene generation:\n\n{e}")

if __name__ == "__main__":
	app = KitchenBuilderApp()
	app.mainloop()
