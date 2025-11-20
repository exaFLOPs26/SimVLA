# step3: Generate goal states

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})

from pxr import UsdGeom, Gf, Usd
import omni.usd, omni.physx
import torch, math, random, json, numpy as np, re, os, ipdb
from shapely.geometry import box, Point
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
from isaaclab.simvla.utils import select_thumbnails_cached, select_thumbnails, merge_free_spaces, find_free_spaces_grid, make_walls_from_bounds, load_grasp_file, find_first_free_direction
# ===============================================================
# MAIN GUI APPLICATION CLASS
# ===============================================================


class GoalGeneratorApp:
	def __init__(self, root):
		self.root = root; self.root.title("Kitchen Goal Generator")
		self.stage, self.free_squares, self.N_dir, self.goal_steps = None, [], "N", []
		self.subtask_cache = self._load_subtask_cache()
		self.all_prim_paths = [] 
		# -- Main layout frames --
		top_frame = ttk.Frame(root)
		top_frame.pack(fill="x", padx=10, pady=5)
		
		main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
		main_paned_window.pack(fill="both", expand=True, padx=10, pady=5)
		
		left_frame = ttk.Frame(main_paned_window)
		main_paned_window.add(left_frame, weight=1)

		right_frame = ttk.Frame(main_paned_window)
		main_paned_window.add(right_frame, weight=2)
		
		log_frame = ttk.LabelFrame(root, text="Status Log")
		log_frame.pack(fill="x", padx=10, pady=5)

		# -- Top frame: Setup --
		setup_frame = ttk.LabelFrame(top_frame, text="1. Kitchen Setup")
		setup_frame.pack(fill="x")
		ttk.Label(setup_frame, text="Task Name:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
		self.task_name_var = tk.StringVar(value="PUT BOWL TO SINK"); ttk.Entry(setup_frame, textvariable=self.task_name_var).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
		ttk.Label(setup_frame, text="Kitchen Number:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
		self.kitchen_num_var = tk.IntVar(value=1); ttk.Entry(setup_frame, textvariable=self.kitchen_num_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=2)
		ttk.Label(setup_frame, text="Sub-Number:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
		self.sub_num_var = tk.StringVar(value="01"); ttk.Entry(setup_frame, textvariable=self.sub_num_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=2)
		self.process_btn = ttk.Button(setup_frame, text="Load Kitchen and Generate Config", command=self.process_kitchen); self.process_btn.grid(row=3, column=0, columnspan=2, pady=10)
		setup_frame.columnconfigure(1, weight=1)
		# -- Left frame: Scene Inspector --
		inspector_frame = ttk.LabelFrame(left_frame, text="Scene Inspector (Double-click to copy)")
		inspector_frame.pack(fill="both", expand=True, padx=5, pady=5)
		
		# ADD: Search bar for Scene Inspector
		self.search_var = tk.StringVar()
		self.search_var.trace_add("write", self._filter_prim_list) # This calls the filter function as you type
		search_entry = ttk.Entry(inspector_frame, textvariable=self.search_var)
		search_entry.pack(fill="x", padx=5, pady=(5, 0))

		prim_list_frame = ttk.Frame(inspector_frame); prim_list_frame.pack(fill="both", expand=True, padx=5, pady=5)
		self.prim_listbox = tk.Listbox(prim_list_frame); self.prim_scrollbar = ttk.Scrollbar(prim_list_frame, orient=tk.VERTICAL, command=self.prim_listbox.yview)
		self.prim_listbox.config(yscrollcommand=self.prim_scrollbar.set)
		self.prim_scrollbar.pack(side="right", fill="y"); self.prim_listbox.pack(side="left", fill="both", expand=True)
		self.prim_listbox.bind("<Double-1>", self._copy_selected_prim_path)
		# -- Right frame: Goal Definition --
		goal_frame = ttk.LabelFrame(right_frame, text="2. Goal Definition")
		goal_frame.pack(fill="both", expand=True)
		self.goal_listbox = tk.Listbox(goal_frame, height=10); self.goal_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
		goal_btn_frame = ttk.Frame(goal_frame); goal_btn_frame.pack(side="left", fill="y", padx=5)
		self.add_goal_btn = ttk.Button(goal_btn_frame, text="Add Subtask", command=self.add_subtask, state="disabled"); self.add_goal_btn.pack(pady=2, fill="x")
		self.load_goals_btn = ttk.Button(goal_btn_frame, text="Load Goals...", command=self.load_goals_from_file, state="disabled"); self.load_goals_btn.pack(pady=2, fill="x")
		self.remove_goal_btn = ttk.Button(goal_btn_frame, text="Remove Selected", command=self.remove_subtask, state="disabled"); self.remove_goal_btn.pack(pady=2, fill="x")
		self.save_goals_btn = ttk.Button(goal_btn_frame, text="Save Goal File", command=self.save_goal_file, state="disabled"); self.save_goals_btn.pack(pady=20, fill="x")

		# -- Bottom frame: Log --
		self.log_text = tk.Text(log_frame, height=6, wrap="word", relief="sunken", borderwidth=1); self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
		
	def log(self, message):
		self.log_text.insert(tk.END, str(message) + "\n"); self.log_text.see(tk.END); self.root.update_idletasks()
		
	def _load_subtask_cache(self):
		self.cache_file = "subtask_cache.json"
		if os.path.exists(self.cache_file):
			try:
				with open(self.cache_file, "r") as f: return json.load(f)
			except json.JSONDecodeError: return {}
		return {}

	def _save_subtask_cache(self):
		with open(self.cache_file, "w") as f: json.dump(self.subtask_cache, f, indent=2)
			
	def format_num(self, num):
		try: return f"{int(num):02d}"
		except (ValueError, TypeError): return str(num)
	def _update_prim_list(self):
		self.prim_listbox.delete(0, tk.END)
		self.all_prim_paths = []
		if not self.stage: return
		try:
			paths = [str(prim.GetPath()) for prim in self.stage.Traverse()]
			self.all_prim_paths = sorted(paths)
			self.search_var.set("") # Clear search bar
			self._filter_prim_list() # Populate listbox with full, unfiltered list
			self.log(f"Scene Inspector updated with {len(paths)} prims.")
		except Exception as e:
			self.log(f"Error updating prim list: {e}")
	def _copy_selected_prim_path(self, event=None):
		selected_indices = self.prim_listbox.curselection()
		if not selected_indices: return
		selected_path = self.prim_listbox.get(selected_indices[0])
		self.root.clipboard_clear()
		self.root.clipboard_append(selected_path)
		self.log(f"Copied to clipboard: {selected_path}")
	def _filter_prim_list(self, *args):
		query = self.search_var.get().lower()
		self.prim_listbox.delete(0, tk.END)
		if not query:
			for path in self.all_prim_paths:
				self.prim_listbox.insert(tk.END, path)
		else:
			filtered_paths = [path for path in self.all_prim_paths if query in path.lower()]
			for path in filtered_paths:
				self.prim_listbox.insert(tk.END, path)
	def process_kitchen(self):
		task_name, kitchen_num, sub_num = self.task_name_var.get(), self.kitchen_num_var.get(), self.sub_num_var.get()
		if not task_name: messagebox.showerror("Error", "Task Name cannot be empty."); return
		kitchen_num_str, sub_num_str = self.format_num(kitchen_num), self.format_num(sub_num)
		self.log(f"Processing Kitchen {kitchen_num_str}_{sub_num_str} for task: '{task_name}'")
		self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END)
		self.kitchen_data = {"task_name": task_name, "kitchen_num": kitchen_num, "kitchen_type": "", "island_bound": [], "kitchen_sub_num": sub_num, "initial_pos_ranges": [], "initial_rot_yaw_range": [["yaw", math.radians(-10.0), math.radians(10.0)]], "goals": []}
		usd_file_path = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num_str}_{sub_num_str}.usd"
		if not os.path.exists(usd_file_path): messagebox.showerror("File Not Found", f"Could not find USD file:\n{usd_file_path}"); return
		if omni.usd.get_context().open_stage(usd_file_path):
			self.stage = omni.usd.get_context().get_stage()
			self.log(f"Successfully opened {usd_file_path}")
			self._update_prim_list() # Update the scene inspector
			self._generate_env_config(kitchen_num_str, sub_num_str)
			for btn in [self.add_goal_btn, self.remove_goal_btn, self.save_goals_btn, self.load_goals_btn]: btn.config(state="normal")
		else:
			self.log(f"ERROR: Failed to open {usd_file_path}"); self.stage = None; self._update_prim_list(); messagebox.showerror("Error", f"Failed to open USD file:\n{usd_file_path}")
			
	def _generate_env_config(self, kitchen_num_str, sub_num_str):
		try:
			first_level_set = {"/world/" + p.split("/")[-1] for p in [str(prim.GetPath()) for prim in self.stage.Traverse()] if p.startswith("/world/") and len(p.split("/")) == 3}
			first_level_list = sorted(list(first_level_set))
			json_path = os.path.join(os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex"), f"kitchen_data_{kitchen_num_str}.json")
			with open(json_path, "r") as f: k_data = json.load(f)
			self.kitchen_data["kitchen_type"] = k_data["kitchen_type"]
			self.free_squares.clear(); self.kitchen_data["initial_pos_ranges"].clear()
			world_bounds = None
			if k_data["kitchen_type"] == "single_wall":
				self.log("Applying special case for 'single_wall' kitchen type.")
				self.kitchen_data["initial_pos_ranges"].append([["x", -0.3, 3.7], ["y", -1.5, -1]])
				world_bounds = (-0.5, -2, 4, 0.5)
				self.free_squares = [((-0.5, -1.5, 3.7, -1), (1.6, -1.25))]
			else:
				self.log("Calculating free space automatically.")
				furniture = []
				for path in first_level_list:
					try:
						prim_name = path.split("/")[-1].lower()
						prim = self.stage.GetPrimAtPath(path)
						bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(prim).ComputeAlignedRange()
						if bbox.IsEmpty(): 
							continue
						furniture.append(box(bbox.GetMin()[0], bbox.GetMin()[1], bbox.GetMax()[0], bbox.GetMax()[1]))
					except Exception as e: self.log(f"Warning: Could not process prim {path}: {e}")
				if not furniture: self.log("Warning: No furniture obstacles found!"); min_x, min_y, max_x, max_y = -5.0, -5.0, 5.0, 5.0
				else: min_x, min_y, max_x, max_y = min(o.bounds[0] for o in furniture), min(o.bounds[1] for o in furniture), max(o.bounds[2] for o in furniture), max(o.bounds[3] for o in furniture)
				world_bounds = (min_x - 0.2, min_y - 0.2, max_x + 0.2, max_y + 0.2)
				merged_spaces = merge_free_spaces(find_free_spaces_grid(world_bounds, furniture))
				robot_radius, safe = 0.23, 0.02
				for bounds, center in merged_spaces:
					if (bounds[2] - bounds[0]) > robot_radius * 2 and (bounds[3] - bounds[1]) > robot_radius * 2:
						self.kitchen_data["initial_pos_ranges"].append([["x", bounds[0] + robot_radius + safe, bounds[2] - robot_radius - safe], ["y", bounds[1] + robot_radius + safe, bounds[3] - robot_radius - safe]])
						self.free_squares.append((bounds, center))
			self.log(f"Found {len(self.free_squares)} free spaces for robot placement.")
			self._write_env_config_file(kitchen_num_str, sub_num_str, first_level_list, world_bounds)
		except Exception as e: self.log(f"ERROR during config generation: {e}"); messagebox.showerror("Processing Error", f"An error occurred: {e}")

	def _write_env_config_file(self, kitchen_num_str, sub_num_str, first_level_list, world_bounds):
		source_file, out_dir = "/root/IsaacLab/scripts/simvla/kitchen_env_cfg_source.py", "/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen"
		output_file = os.path.join(out_dir, f"kitchen_{kitchen_num_str}_{sub_num_str}.py")
		os.makedirs(out_dir, exist_ok=True)
		kitchen_usd = f'''\nkitchen = AssetBaseCfg(\n\tprim_path="{{ENV_REGEX_NS}}/Kitchen",\n\tspawn=sim_utils.UsdFileCfg(usd_path="file:/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num_str}_{sub_num_str}.usd"),\n)'''
		generated_blocks = []
		generated_blocks.append(kitchen_usd)
		for path in first_level_list:
			try:
				name, prim = path.split("/")[-1], self.stage.GetPrimAtPath(path)
				if name == "Looks":
					continue
				xform = UsdGeom.Xformable(prim); world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
				pos, quat_obj = tuple(round(v, 4) for v in world_transform.ExtractTranslation()), world_transform.ExtractRotationQuat()
				quat = (round(quat_obj.GetReal(), 4), *[round(v, 4) for v in quat_obj.GetImaginary()])
				joint_attr = prim.GetAttribute("joint_names"); joint_names = joint_attr.Get() if joint_attr and joint_attr.HasAuthoredValue() else []
				joint_pos, actuator_block = {j: 0.0 for j in joint_names}, ""
				if joint_names:
					actuator_block = f''',\n\t\t\t\t\tactuators={{\n\t\t\t\t\t\t"default": ImplicitActuatorCfg(joint_names_expr=[{", ".join(f'"{j}"' for j in joint_names)}],effort_limit=87.0,velocity_limit=100.0,stiffness=0.0,damping=1.0)\n\t\t\t\t\t\t}}'''
				block = f"""{name} = ArticulationCfg(prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos={pos}, rot={quat}, joint_pos={joint_pos}){actuator_block})""" if joint_names else f"""{name} = RigidObjectCfg(prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos={pos}, rot={quat}))"""

				generated_blocks.append(block)
			except Exception as e: self.log(f"Warning: Could not generate config for {path}: {e}")
		if world_bounds: generated_blocks.extend(make_walls_from_bounds(world_bounds))
		indented_blocks = ["\n".join("\t" + line for line in block.splitlines()) for block in generated_blocks]
		generated_code = "\n".join(indented_blocks)
		with open(source_file, "r") as f: config_text = f.read()
		new_text = re.sub(r"# -------Change-------.*?# -------Stop-------", "# -------Change-------\n" + generated_code + "\n\t\t# -------Stop-------", config_text, flags=re.DOTALL)
		with open(output_file, "w") as f: f.write(new_text)
		self.log(f"✅ Created environment config: {output_file}")
	
	def add_subtask(self):
		dialog = SubtaskDialog(self.root, "Add Subtask", self.subtask_cache)
		if dialog.result:
			lang, action, params = dialog.result
			self.goal_steps.append({"lang": lang, "action": action, "params": params})
			self.goal_listbox.insert(tk.END, f"{len(self.goal_steps)}. [{action}] {lang}")
			if lang not in self.subtask_cache: self.subtask_cache[lang] = action; self._save_subtask_cache()

	def remove_subtask(self):
		selected_indices = self.goal_listbox.curselection()
		if not selected_indices: return
		for index in sorted(selected_indices, reverse=True): self.goal_listbox.delete(index); del self.goal_steps[index]
			
	def load_goals_from_file(self):
		goal_dir = os.path.expanduser("~/IsaacLab/scripts/simvla/goals")
		filepath = filedialog.askopenfilename(initialdir=goal_dir, title="Select Goal File", filetypes=(("JSON files", "*.json*"), ("all files", "*.*")))
		if not filepath: return
		try:
			with open(filepath, 'r') as f: loaded_data = json.load(f)
			if "goals" not in loaded_data or not loaded_data["goals"]: messagebox.showwarning("Load Warning", "No 'goals' found in file."); return
			self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END)
			goal_sequence = loaded_data["goals"][0]
			for i, step_data in enumerate(goal_sequence):
				if isinstance(step_data, dict) and "action" in step_data and "parameters" in step_data:
					lang, action, params = step_data.get("language", f"Loaded Step {i+1}"), step_data["action"], step_data["parameters"]
				elif isinstance(step_data, list) and len(step_data) == 2 and isinstance(step_data[1], dict):
					action, params = step_data; lang = f"Loaded: {action} to {params.get('prim_path', '...')}"
				else:
					messagebox.showerror("Load Error", "Cannot load file. It uses an old, processed-only format."); self.goal_steps.clear(); self.goal_listbox.delete(0, tk.END); return
				self.goal_steps.append({"lang": lang, "action": action, "params": params})
				self.goal_listbox.insert(tk.END, f"{len(self.goal_steps)}. [{action}] {lang}")
			self.log(f"Successfully loaded {len(self.goal_steps)} goal steps from {os.path.basename(filepath)}")
		except Exception as e: self.log(f"Failed to load file: {e}"); messagebox.showerror("Load Error", f"An error occurred:\n{e}")

	def save_goal_file(self):
		if not self.goal_steps: messagebox.showwarning("Warning", "No goal steps defined."); return
		final_goal_sequence_for_sim, rich_goal_sequence_for_reload = [], []
		for step in self.goal_steps:
			try:
				processed_step_dict = self._process_goal_step(step)
				if processed_step_dict:
					rich_goal_sequence_for_reload.append(processed_step_dict)
					final_goal_sequence_for_sim.append(processed_step_dict['processed_goal'])
					if processed_step_dict["action"] in ["N", "N_s"] and self.kitchen_data.get("kitchen_type") == "island":
						final_goal_sequence_for_sim.append(processed_step_dict['processed_goal'])
			except Exception as e: self.log(f"Error processing step '{step['lang']}': {e}"); messagebox.showerror("Processing Error", f"Could not process step '{step['lang']}':\n{e}"); return
		self.kitchen_data["goals"] = [final_goal_sequence_for_sim]
		kitchen_num_str, sub_num_str = self.format_num(self.kitchen_num_var.get()), self.format_num(self.sub_num_var.get())
		goal_dir = os.path.expanduser("~/IsaacLab/scripts/simvla/goals")
		os.makedirs(goal_dir, exist_ok=True)
		json_file = os.path.join(goal_dir, f"Isaac-Kitchen-v{kitchen_num_str}-{sub_num_str}.json")
		reloadable_json_file = os.path.join(goal_dir, f"Isaac-Kitchen-v{kitchen_num_str}-{sub_num_str}.reloadable.json")
		try:
			with open(reloadable_json_file, "w") as f: json.dump({"goals": [rich_goal_sequence_for_reload]}, f, indent=4)
			self.log(f"SUCCESS: Saved reloadable goal file to {reloadable_json_file}")
			with open(json_file, "w") as f: json.dump(self.kitchen_data, f, indent=4)
			self.log(f"SUCCESS: Goal file for simulation saved to {json_file}")
			messagebox.showinfo("Success", f"Saved simulation file to:\n{json_file}\n\nSaved reloadable file to:\n{reloadable_json_file}")
		except Exception as e: self.log(f"ERROR saving file: {e}"); messagebox.showerror("Save Error", f"Could not save file:\n{e}")

	def _process_goal_step(self, step):
		action, params = step["action"], step["params"]
		processed_value = None
		if action in ["N", "N_s"]:
			prim_path, which_arm = params.get("prim_path"), params.get("which_arm")
			if not prim_path: raise ValueError("Prim path required for navigation")
			prim = self.stage.GetPrimAtPath(prim_path)
			if not prim: raise ValueError(f"Prim not found: {prim_path}")
			
			xform = UsdGeom.Xformable(prim); world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
			position, orientation_quat = world_transform.ExtractTranslation(), world_transform.ExtractRotationQuat()
			bbox_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(prim).ComputeAlignedRange()
			
			arm_bias = 0.05 if which_arm == "Left" else -0.05 if which_arm == "Right" else 0.0
			N_pos, safety, wheel_r = torch.zeros(3), 0.12, 0.23
			if position[2] < 0.01: # It's a furniture piece
				min_b, max_b = bbox_range.GetMin(), bbox_range.GetMax()
				quat_wxyz = [orientation_quat.GetReal(), *orientation_quat.GetImaginary()]
				if np.allclose(quat_wxyz, [-0.7071, 0, 0, 0.7071], atol=1e-3): self.N_dir, N_pos[:] = "N", torch.tensor([min_b[0]-safety-wheel_r, position[1]-arm_bias, math.radians(0.0)])
				elif np.allclose(quat_wxyz, [0.7071, 0, 0, 0.7071], atol=1e-3): self.N_dir, N_pos[:] = "S", torch.tensor([max_b[0]+safety-wheel_r, position[1]+arm_bias, math.radians(180.0)])
				elif np.allclose(quat_wxyz, [1.0, 0, 0, 0], atol=1e-3): self.N_dir, N_pos[:] = "W", torch.tensor([position[0]+arm_bias, min_b[1]-safety-wheel_r, math.radians(90.0)])
				elif np.allclose(quat_wxyz, [0.0, 0, 0, 1.0], atol=1e-3): self.N_dir, N_pos[:] = "E", torch.tensor([position[0]+arm_bias, max_b[1]+safety-wheel_r, math.radians(-90.0)])
			else:
				physx_interface = omni.physx.get_physx_interface()
				physx_interface.start_simulation()
				scene_query_interface = omni.physx.get_physx_scene_query_interface()

				object_pos = position
				ray_origin = object_pos + Gf.Vec3d(0, 0, 0.4)
				ray_distance = 1.0
				directions_to_check = ["N", "E", "S", "W"]

				furniture_prim = None
				for dir_str in directions_to_check:
					if dir_str == "E": ray_direction = Gf.Vec3d(0, 1, -1)
					elif dir_str == "S": ray_direction = Gf.Vec3d(1, 0, -1)
					elif dir_str == "W": ray_direction = Gf.Vec3d(0, -1, -1)
					elif dir_str == "N": ray_direction = Gf.Vec3d(-1, 0, -1)
					hit = scene_query_interface.raycast_closest(ray_origin, ray_direction, ray_distance)
					if hit["hit"] == False:
						continue
					if prim_path.split('/')[-1] in hit['rigidBody']:
						continue
					if hit["hit"]:
						hit_path = hit['rigidBody']
						if hit_path and hit_path.startswith("/world/"):
							top_level_path = "/world/" + hit_path.split("/")[2]
							furniture_prim = self.stage.GetPrimAtPath(top_level_path)
							self.log(f"Raycast hit and identified furniture: {top_level_path}")
							break
				
				if furniture_prim is None:
					self.log("Raycast failed to find supporting furniture.")
					user_prim_path = simpledialog.askstring("Input Required", 
						"Raycast failed. Please provide the prim path of the furniture the object is on:", 
						parent=self.root)
					if not user_prim_path: raise ValueError("Supporting furniture prim path is required.")
					furniture_prim = self.stage.GetPrimAtPath(user_prim_path)
					if not furniture_prim: raise ValueError(f"Could not find furniture prim at path: {user_prim_path}")

				f_xform = UsdGeom.Xformable(furniture_prim)
				f_world_transform = f_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
				f_position = f_world_transform.ExtractTranslation()
				f_bbox_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(furniture_prim).ComputeAlignedRange()
				f_min_bound, f_max_bound = f_bbox_range.GetMin(), f_bbox_range.GetMax()

				object_pos_2d = (object_pos[0], object_pos[1])
				dir_point = find_first_free_direction(object_pos_2d, self.free_squares, step=0.03, max_steps=100)
				if not dir_point: raise RuntimeError(f"Could not find any free space near object {prim_path}")
				
				self.N_dir, _ = dir_point
				if self.N_dir == "W":
					N_pos[0] = object_pos_2d[0] + arm_bias
					N_pos[1] = f_min_bound[1] - safety - wheel_r
					N_pos[2] = math.radians(90.0)
				elif self.N_dir == "E":
					N_pos[0] = object_pos_2d[0] + arm_bias
					N_pos[1] = f_max_bound[1] + safety + wheel_r
					N_pos[2] = math.radians(-90.0)
				elif self.N_dir == "S":
					N_pos[1] = object_pos_2d[1] + arm_bias
					N_pos[0] = f_max_bound[0] + safety + wheel_r
					N_pos[2] = math.radians(180.0)
				elif self.N_dir == "N":
					N_pos[1] = object_pos_2d[1] + arm_bias
					N_pos[0] = f_min_bound[0] - safety - wheel_r
					N_pos[2] = math.radians(0.0)

			processed_value = N_pos.tolist()

		elif action in ["A_r", "A_l", "A_b"]:
			usage, prim_path = params.get("usage"), params.get("prim_path")
			if usage == "Move arm to reset":
				processed_value = [999.0, 0.0, 1.102, 0.6322, -0.587, 0.3344, -0.3793]
			else:
				if not prim_path: raise ValueError("Prim path required for grasp/place")
				prim = self.stage.GetPrimAtPath(prim_path)
				if not prim: raise ValueError(f"Prim not found: {prim_path}")
				
				bodex_attr = prim.GetAttribute("BODex_path")
				if usage == "Move arm to grasp" and bodex_attr and bodex_attr.HasAuthoredValue():
					self.log("Using full graspdata logic for 'Move arm to grasp'")
					grasp_pos = torch.zeros(7)
					bbox_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(prim).ComputeAlignedRange()
					object_min_bound, object_max_bound = bbox_range.GetMin(), bbox_range.GetMax() # read object bounding box
					robot_name = "sim_parallel"
					# 1. Get the BASE path (e.g., .../floating/scale010_grasp)
					base_grasp_path = load_grasp_file(bodex_attr.Get(), robot_name)

					if not base_grasp_path:
						raise FileNotFoundError(f"No grasp base path found for {bodex_attr.Get()}")
					else:
						print(f"Found grasp base path: {base_grasp_path}")

# 2. Define paths for both _right and _left .npy files
					grasp_file_right = f"{base_grasp_path}_right.npy"
					grasp_file_left = f"{base_grasp_path}_left.npy"

					if not os.path.exists(grasp_file_right):
						raise FileNotFoundError(f"Missing grasp file: {grasp_file_right}")
					if not os.path.exists(grasp_file_left):
						raise FileNotFoundError(f"Missing grasp file: {grasp_file_left}")

# 3. Load data from BOTH .npy files using np.load
					print(f"Loading RIGHT grasps: {os.path.basename(grasp_file_right)}")
# --- USED NP.LOAD ---
# .item() extracts the dictionary from the 0-dimensional array
					grasp_data_right = np.load(grasp_file_right, allow_pickle=True).item()
					eef_data_right = grasp_data_right["robot_pose"][0, :, 0, :7]

					print(f"Loading LEFT grasps: {os.path.basename(grasp_file_left)}")
# --- USED NP.LOAD ---
					grasp_data_left = np.load(grasp_file_left, allow_pickle=True).item()
					eef_data_left = grasp_data_left["robot_pose"][0, :, 0, :7]

# 4. Define thumbnail path
					thumbnail_path = f"{base_grasp_path}_segments_thumbnails"
					if not os.path.exists(thumbnail_path):
						raise FileNotFoundError(f"Thumbnail folder not found: {thumbnail_path}")

# 5. Get the list of preferred grasps
#	 This line calls your (now modified) cached function
					prefer_list = select_thumbnails_cached(thumbnail_path)

					if not prefer_list:
						raise ValueError("No grasp preference selected from thumbnails.")

					print(f"Found {len(prefer_list)} preferred grasps to load.")

# 6. Build the final data list based on preferences
					valid_eef_data_list = []
					for segment_index, hand in prefer_list:

						if hand == 'right':
							# Select the pose from the 'right' data array
							valid_eef_data_list.append(eef_data_right[segment_index])

						elif hand == 'left':
							# Select the pose from the 'left' data array
							valid_eef_data_list.append(eef_data_left[segment_index])

# 7. Stack the list into a single numpy array
					valid_eef_data = np.stack(valid_eef_data_list, axis=0)

					print(f"Successfully loaded and combined {valid_eef_data.shape[0]} poses.")

# ... rest of your code continues here ...
# 'valid_eef_data' now holds the combined, selected data
					prefer_eef_data = valid_eef_data

					print("--- Loading complete ---")
#					grasp_file_path = load_grasp_file(bodex_attr.Get(), robot_name) # read scale 010 folder
#					if not grasp_file_path: raise FileNotFoundError(f"No grasp file found for {bodex_attr.Get()}")
#					grasp_data = np.load(grasp_file_path, allow_pickle=True).tolist()
#					eef_data = grasp_data["robot_pose"][0][:,1,:7] # robot pose : right gripperbase link
#					valid_eef_data = eef_data[np.all(np.isfinite(eef_data), axis=1)]
#					thumbnail_path = f"{os.path.splitext(grasp_file_path)[0]}_segments_thumbnails"
#					prefer_list = select_thumbnails_cached(thumbnail_path)
#					if not prefer_list: raise ValueError("No grasp preference selected from thumbnails.")
#					prefer_eef_data = valid_eef_data[prefer_list]
					xyz, quat_wxyz = prefer_eef_data[:, 0:3], prefer_eef_data[:, 3:7]
					quat_xyzw = np.hstack([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]])
					# Rotate based on sub_num
					angle_deg = int(self.kitchen_data["kitchen_sub_num"]) * 30

					# Create rotation about Z (Euler z)
					rot_z = R.from_euler('z', angle_deg, degrees=True)

					# Rotate position (xyz)
					rotated_xyz = rot_z.apply(xyz)

					# Convert input quat from wxyz to xyzw
					quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

					# Apply Z-rotation in local frame
					new_rotations = R.from_quat(quat_xyzw)
					new_rotations = rot_z * new_rotations

					# Convert back to wxyz
					rotated_quats_xyzw = new_rotations.as_quat()
					rotated_quats_wxyz = np.hstack([rotated_quats_xyzw[:, 3:4], rotated_quats_xyzw[:, 0:3]])

					# Combine rotated positions and orientations
					rotated_data = np.hstack((rotated_xyz, rotated_quats_wxyz))
					if not self.N_dir: raise ValueError("Navigation direction 'N_dir' must be set by an 'N' action before grasping.")
					if action == "A_r":
						if self.N_dir == "N": avail_data = rotated_data[(rotated_data[:,0] < 0) & (rotated_data[:,1] < 0.01)]
						elif self.N_dir == "S": avail_data = rotated_data[(rotated_data[:,0] > 0) & (rotated_data[:,1] > -0.01)]
						elif self.N_dir == "E": avail_data = rotated_data[(rotated_data[:,1] > 0) & (rotated_data[:,0] < 0.01)]
						elif self.N_dir == "W": avail_data = rotated_data[(rotated_data[:,1] < 0) & (rotated_data[:,0] > -0.01)]
						else: raise ValueError(f"Invalid navigation direction: {self.N_dir}")
					elif action == "A_l":
						if self.N_dir == "N": avail_data = rotated_data[(rotated_data[:,0] < 0) & (rotated_data[:,1] > -0.01)]
						elif self.N_dir == "S": avail_data = rotated_data[(rotated_data[:,0] > 0) & (rotated_data[:,1] < 0.01)]
						elif self.N_dir == "E": avail_data = rotated_data[(rotated_data[:,1] > 0) & (rotated_data[:,0] > -0.01)]
						elif self.N_dir == "W": avail_data = rotated_data[(rotated_data[:,1] < 0) & (rotated_data[:,0] < 0.01)]
						else: raise ValueError(f"Invalid navigation direction: {self.N_dir}")
					elif action == "A_b":
						if self.N_dir == "N": avail_data = rotated_data[(rotated_data[:,0] < 0)]
						elif self.N_dir == "S": avail_data = rotated_data[(rotated_data[:,0] > 0)]
						elif self.N_dir == "E": avail_data = rotated_data[(rotated_data[:,1] > 0)]
						elif self.N_dir == "W": avail_data = rotated_data[(rotated_data[:,1] < 0)]
						else: raise ValueError(f"Invalid navigation direction: {self.N_dir}")
					if len(avail_data) == 0:
						raise ValueError(f"No valid grasps found for approach direction '{self.N_dir}'")

					all_grasp_poses = []
					for grasp_pose_data in avail_data:
						# Find the index of the current pose to map back to the prefer_list
						pose_idx_mask = np.all(np.isclose(rotated_data, grasp_pose_data, atol=1e-6), axis=1)
						# Skip if the pose cannot be uniquely identified
						if np.sum(pose_idx_mask) != 1:
							self.log(f"Warning: Could not uniquely identify a grasp pose. Skipping.")
							continue
							
						original_pose_idx = np.where(pose_idx_mask)[0][0]
						pose_idx = prefer_list[original_pose_idx]
						print(pose_idx)
						# Get the mean contact point for EEF
						if pose_idx[1] == 'left':
							grasp_data = grasp_data_left
						elif pose_idx[1] == 'right':
							grasp_data = grasp_data_right
						mean_contact_point = torch.tensor(grasp_data["contact_point"][0])[pose_idx[0]].squeeze(0).mean(dim=0)
						rotated_point = rot_z.apply(mean_contact_point.numpy())
# TODO  check for N S E 
						if self.N_dir == "N": rotated_point = rotated_point[[1, 0, 2]]
						elif self.N_dir == "S": rotated_point = rotated_point[[1, 0, 2]] * [-1 , 1, 1]
						elif self.N_dir == "E": rotated_point = rotated_point[[0, 1, 2]] * [-1 , -1, 1]
						elif self.N_dir == "W": rotated_point = rotated_point
						

						# Calculate the final grasp pose
						current_grasp_pos = torch.zeros(7)
						if 'handle' in prim.GetName():
							reference_frame = (torch.tensor(object_min_bound) + torch.tensor(object_max_bound)) / 2
						else:
							xform = UsdGeom.Xformable(prim); world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
							pos = tuple(round(v, 4) for v in world_transform.ExtractTranslation())
							reference_frame = torch.tensor(list(pos))
							reference_frame[-1] = ((torch.tensor(object_min_bound) + torch.tensor(object_max_bound)) / 2)[-1]
					   
						current_grasp_pos[0:3] = reference_frame + torch.tensor(rotated_point, dtype=torch.float32)
						quat = torch.tensor(grasp_pose_data[3:7], dtype=torch.float32)  # (w, x, y, z)

						def quat_mul(a, b):
							"""Hamilton product a ⊗ b"""
							w1, x1, y1, z1 = a
							w2, x2, y2, z2 = b
							return torch.tensor([
								w1*w2 - x1*x2 - y1*y2 - z1*z2,
								w1*x2 + x1*w2 + y1*z2 - z1*y2,
								w1*y2 - x1*z2 + y1*w2 + z1*x2,
								w1*z2 + x1*y2 - y1*x2 + z1*w2
							], dtype=torch.float32)

						def euler_to_quat(roll, pitch, yaw):
							"""Convert Euler angles (deg) to quaternion (w, x, y, z) using XYZ order"""
							r = math.radians(roll) / 2
							p = math.radians(pitch) / 2
							y = math.radians(yaw) / 2

							cr, sr = math.cos(r), math.sin(r)
							cp, sp = math.cos(p), math.sin(p)
							cy, sy = math.cos(y), math.sin(y)

							w = cr*cp*cy + sr*sp*sy
							x = sr*cp*cy - cr*sp*sy
							y = cr*sp*cy + sr*cp*sy
							z = cr*cp*sy - sr*sp*cy
							return torch.tensor([w, x, y, z], dtype=torch.float32)

# --- define your local Euler rotation ---
						rot_euler = (90.0, 90.0, 0.0)  # roll (X), pitch (Y), yaw (Z)
# rot_euler = (90.0, 0.0, -90.0)  # try another if needed

# convert Euler to quaternion
						q_rot = euler_to_quat(*rot_euler)

# rotate in LOCAL frame  (post-multiply)
						quat_new = quat_mul(quat, q_rot)
						quat_new = quat_new / torch.norm(quat_new)

						current_grasp_pos[3:] = quat_new
#current_grasp_pos[3:] = torch.tensor(grasp_pose_data[3:7])
						all_grasp_poses.append(current_grasp_pos.tolist())

					if not all_grasp_poses:
						raise ValueError("Failed to process any of the available grasps after filtering.")

					processed_value = all_grasp_poses
				else:
					if usage == "Move arm to grasp": self.log("WARNING: No BODex_path found. Using simplified grasp logic.")
					bbox_range = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False).ComputeWorldBound(prim).ComputeAlignedRange()
					center = (torch.tensor(bbox_range.GetMin()) + torch.tensor(bbox_range.GetMax())) / 2
					if usage == "Move arm to grasp": processed_value = (center + torch.tensor([0, 0, 0.1])).tolist() + [0.0, 0.0, 0.0, 1.0]
					elif usage == "Move arm to place":
						grasp_pos_place = torch.zeros(7)
						offset, put_above = 0.1, 0.15
						object_min_bound, object_max_bound = bbox_range.GetMin(), bbox_range.GetMax()
						position = (torch.tensor(object_min_bound) + torch.tensor(object_max_bound)) / 2
						if not self.N_dir: raise ValueError("Navigation direction 'N_dir' must be set before placing.")
						if self.N_dir == "N": grasp_pos_place[0:3] = torch.tensor([object_min_bound[0] + offset, position[1], object_max_bound[2] + put_above])
						elif self.N_dir == "E": grasp_pos_place[0:3] = torch.tensor([position[0], object_max_bound[1] - offset, object_max_bound[2] + put_above])
						elif self.N_dir == "S": grasp_pos_place[0:3] = torch.tensor([object_max_bound[0] - offset, position[1], object_max_bound[2] + put_above])
						elif self.N_dir == "W": grasp_pos_place[0:3] = torch.tensor([position[0], object_min_bound[1] + offset, object_max_bound[2] + put_above])
						grasp_pos_place[3] = 999
						processed_value = grasp_pos_place.tolist()

		elif action in ["G_r", "G_l", "G_b"]:
			processed_value = bool(params.get("grasp"))
		
		return {"language": step["lang"], "action": action, "parameters": params, "processed_goal": [action, processed_value]}	
# ===============================================================
# CUSTOM DIALOG FOR ADDING SUBTASKS
# ===============================================================
class SubtaskDialog(simpledialog.Dialog):
	def __init__(self, parent, title, cache):
		self.cache = cache; self.result = None; super().__init__(parent, title)
	def body(self, master):
		self.action_var, self.lang_var = tk.StringVar(), tk.StringVar()
		ttk.Label(master, text="Subtask Language:").pack(anchor="w"); self.lang_combo = ttk.Combobox(master, textvariable=self.lang_var, values=list(self.cache.keys())); self.lang_combo.pack(fill="x", expand=True, padx=5, pady=2); self.lang_combo.bind("<<ComboboxSelected>>", self.on_lang_select)
		ttk.Label(master, text="Action Primitive:").pack(anchor="w", pady=(10,0)); actions = ["N", "N_s", "A_r", "A_l", "A_b", "G_r", "G_l", "G_b"]; self.action_menu = ttk.OptionMenu(master, self.action_var, "Select Action", *actions, command=self.on_action_select); self.action_menu.pack(fill="x", padx=5, pady=2)
		self.param_frame = ttk.Frame(master); self.param_frame.pack(fill="x", expand=True, pady=10, padx=5); self.params = {}
		return self.lang_combo
	def on_lang_select(self, event=None):
		lang = self.lang_var.get()
		if lang in self.cache: self.action_var.set(self.cache[lang]); self.on_action_select(self.cache[lang])
	def on_action_select(self, selected_action):
		for widget in self.param_frame.winfo_children(): widget.destroy()
		self.params = {}
		if selected_action in ["N", "N_s"]:
			ttk.Label(self.param_frame, text="Target Prim Path:").pack(anchor="w"); self.params["prim_path"] = tk.StringVar(); ttk.Entry(self.param_frame, textvariable=self.params["prim_path"]).pack(fill="x")
			ttk.Label(self.param_frame, text="Arm to Use Afterwards:").pack(anchor="w"); self.params["which_arm"] = tk.StringVar(value="Both"); ttk.OptionMenu(self.param_frame, self.params["which_arm"], "Both", "Left", "Right", "Both").pack(fill="x")
		elif selected_action in ["A_r", "A_l", "A_b"]:
			ttk.Label(self.param_frame, text="Arm Usage:").pack(anchor="w"); self.params["usage"] = tk.StringVar(value="Move arm to grasp"); usages = ["Move arm to grasp", "Move arm to reset", "Move arm to place"]; ttk.OptionMenu(self.param_frame, self.params["usage"], usages[0], *usages).pack(fill="x")
			ttk.Label(self.param_frame, text="Target Prim Path (for grasp/place):").pack(anchor="w"); self.params["prim_path"] = tk.StringVar(); ttk.Entry(self.param_frame, textvariable=self.params["prim_path"]).pack(fill="x")
		elif selected_action in ["G_r", "G_l", "G_b"]:
			self.params["grasp"] = tk.BooleanVar(value=True); ttk.Checkbutton(self.param_frame, text="Grasp (uncheck to release)", variable=self.params["grasp"]).pack(anchor="w")
	def apply(self):
		lang, action = self.lang_var.get(), self.action_var.get()
		if not lang or not action or action == "Select Action": messagebox.showwarning("Input Error", "Please provide a subtask language and select an action.", parent=self); return
		self.result = (lang, action, {key: var.get() for key, var in self.params.items()})

if __name__ == "__main__":
	root = tk.Tk()
	root.option_add("*Font", ("DejaVu Sans", 10))
	app = GoalGeneratorApp(root)
	root.mainloop()
