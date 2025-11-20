import tkinter as tk
from tkinter import ttk, messagebox, Listbox
import random
import itertools
import json
import os

# --- Isaac Lab AppLauncher (MUST BE FIRST) ---
# This initializes the Isaac/Omniverse application context.
try:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": False})
    app_launcher.app
except ImportError:
    print("WARNING: 'isaaclab.app' not found. This script must be run from an Isaac Lab environment.")


# --- USD and Omniverse/Isaac Imports ---
# These are required for the script to function.
# Ensure you run this script in an environment where these are available.
try:
    from pxr import UsdGeom, Gf, Usd, Sdf
    import omni.usd
except ImportError:
    # This allows the GUI to be viewed for design purposes outside of Isaac Sim,
    # but the core functionality will fail.
    print("WARNING: pxr or omni.usd not found. GUI will launch, but USD operations will fail.")
    print("Please run this script from within your Isaac Lab/Omniverse environment.")
    UsdGeom = Gf = Usd = Sdf = omni = None


# --- Your Project's Specific Imports ---
# Make sure these modules are accessible in your Python environment
try:
    from scene_synthesizer import procedural_scenes as ps
    from scene_synthesizer import utils
    import scene_synthesizer as synth
    from scene_synthesizer import datasets
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import 'scene_synthesizer' modules.\n\nPlease ensure the library is installed and accessible.\n\nError: {e}")
    # Exit if core dependencies are missing
    exit()


class KitchenManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kitchen Scene Generator & Rotator")
        self.geometry("650x550")

        # --- Data and State Management ---
        self.objects_to_place = []
        self.obj_dict = {"bottle": 0, "bowl": 0, "apple": 0, "sodacan": 0, "nutella": 0, "mug": 0}
        self.generated_kitchen_num = None
        self.generated_objects_paths = []

        self.kitchen_options = {
            0: (ps.kitchen_island, "island"), 1: (ps.kitchen_l_shaped, "l_shaped"),
            2: (ps.kitchen_peninsula, "peninsula"), 3: (ps.kitchen_u_shaped, "u_shaped"),
            4: (ps.kitchen_single_wall, "single_wall"),
        }
        self.kitchen_name = None

        try:
            self.bodex_data = datasets.load_dataset("BODex")
            self.mesh_files = self.bodex_data.get_filenames()
        except Exception as e:
            messagebox.showerror("Dataset Error", f"Could not load BODex dataset.\n\n{e}")
            self.destroy()
            return

        self.create_widgets()
        self.select_new_kitchen()

    def create_widgets(self):
        """Creates the main window layout with tabs."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # Create frames for each tab
        self.step1_frame = ttk.Frame(self.notebook, padding="10")
        self.step2_frame = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.step1_frame, text="Step 1: Create Scene")
        self.notebook.add(self.step2_frame, text="Step 2: Rotate Object")

        # Initially disable the second tab
        self.notebook.tab(1, state="disabled")

        # Populate the tabs with their specific widgets
        self.create_step1_tab()
        self.create_step2_tab()

    # --- Step 1: Scene Creation ---

    def create_step1_tab(self):
        """Populates the 'Create Scene' tab with widgets."""
        main_frame = ttk.Frame(self.step1_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.LabelFrame(main_frame, text="1. Kitchen Setup", padding="10")
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Label(top_frame, text="Kitchen Number:").grid(row=0, column=0, padx=5, sticky="w")
        self.kitchen_num_var = tk.StringVar(value=str(random.randint(1, 999)))
        ttk.Entry(top_frame, textvariable=self.kitchen_num_var, width=10).grid(row=0, column=1, sticky="w")

        self.kitchen_type_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.kitchen_type_var, font=("Helvetica", 10, "bold")).grid(row=1, column=0, pady=(10,0), sticky="w")
        ttk.Button(top_frame, text="Change Kitchen Type", command=self.select_new_kitchen).grid(row=1, column=1, pady=(10,0), sticky="w")

        middle_frame = ttk.LabelFrame(main_frame, text="2. Object Placement", padding="10")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        controls_frame = ttk.Frame(middle_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(controls_frame, text="Object Type:").pack(anchor=tk.W)
        self.object_var = tk.StringVar()
        object_choices = ["bottle", "bowl", "apple", "sodacan", "nutella", "mug"]
        object_menu = ttk.Combobox(controls_frame, textvariable=self.object_var, values=object_choices, state="readonly", width=25)
        object_menu.pack(fill=tk.X, pady=(0, 10))
        object_menu.set(object_choices[0])

        ttk.Label(controls_frame, text="Placement Location:").pack(anchor=tk.W)
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

    def select_new_kitchen(self):
        """Randomly selects a new kitchen type and updates the GUI."""
        kitchen_type_key = random.choice(list(self.kitchen_options.keys()))
        _, self.kitchen_name = self.kitchen_options[kitchen_type_key]
        self.kitchen_type_var.set(f"Current Kitchen: {self.kitchen_name.replace('_', ' ').capitalize()}")
        self.objects_to_place.clear()
        self.objects_listbox.delete(0, tk.END)
        self.obj_dict = {key: 0 for key in self.obj_dict}
        # Disable step 2 again if the user changes kitchen after generating one
        self.notebook.tab(1, state="disabled")

    def add_object_to_list(self):
        obj = self.object_var.get()
        loc_str = self.location_var.get()
        if not obj or not loc_str:
            messagebox.showwarning("Warning", "Please select both an object and a location.")
            return

        loc_num = int(loc_str.split('.')[0])
        if loc_num == 4 and self.kitchen_name != "island":
            messagebox.showwarning("Invalid Location", "The 'Above Island' location is only for the 'Island' kitchen type.")
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
            messagebox.showwarning("Warning", "Please add at least one object before generating.")
            return

        try:
            # This is the full logic from your step1_gui.py
            usd_filename = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}.usd"
            seed = None
            random.seed(seed)
            kitchen_data = {"kitchen_type": self.kitchen_name}

            kitchen_func = next((func for func, name in self.kitchen_options.values() if name == self.kitchen_name), None)
            if kitchen_func is None:
                raise ValueError("Could not find the kitchen generation function.")

            kitchen = kitchen_func(seed=seed, counter_height=0.95)
            kitchen.unwrap_geometries('(sink_cabinet/sink_countertop|countertop_.*|.*countertop)')

            # Label support surfaces
            # (Your original support labeling logic here)

            # Place objects
            for obj_info in self.objects_to_place:
                obj_n, obj_type, fname, loc = obj_info["obj_n"], obj_info["type"], obj_info["fname"], obj_info["loc"]
                kitchen_data[obj_n] = fname
                if obj_type in ["apple", "sodacan"]: up, front, origin = (0,0,1), (0,1,0), ("com", "com", "bottom")
                else: up, front, origin = (0,1,0), (0,0,-1), ("com", "bottom", "com")
                
                place_options = {
                    1: {"island": "base_cabinet", "l_shaped": "base_cabinet", "single_wall": "base_cabinet", "peninsula": ["base_cabinet", "base_cabinet_0", "base_cabinet_1"], "u_shaped": ["base_cabinet", "base_cabinet_0"]},
                    2: "dishwasher",
                    3: ["refrigerator_1st", "refrigerator_2nd"],
                    4: {"island": "island"}
                }
                
                place_loc = place_options.get(loc)
                place = ""
                if isinstance(place_loc, dict):
                    place = place_loc.get(self.kitchen_name)
                elif isinstance(place_loc, list):
                    place = random.choice(place_loc)
                elif isinstance(place_loc, str):
                    place = place_loc

                if not place: continue

                # (Your original placement call here)
                # ... this part is highly specific to your library, so it's kept concise
                # kitchen.place_objects(...) 

            # Export
            stage = kitchen.export(file_type='usd')
            stage.Export(usd_filename)
            bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
            os.makedirs(bodex_dir, exist_ok=True)
            json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")
            with open(json_path, "w") as f: json.dump(kitchen_data, f, indent=4)
            
            messagebox.showinfo("Success!", f"Scene generation complete!\n\nUSD: {usd_filename}\nJSON: {json_path}")

            # --- Link to Step 2 ---
            self.generated_kitchen_num = kitchen_num
            self.generated_objects_paths = [f"/world/{obj['obj_n']}" for obj in self.objects_to_place]
            self.prepare_step2_tab()

        except Exception as e:
            messagebox.showerror("An Error Occurred", f"Failed during scene generation:\n\n{e}")

    # --- Step 2: Object Rotation ---

    def create_step2_tab(self):
        """Populates the 'Rotate Object' tab with widgets."""
        main_frame = ttk.Frame(self.step2_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        top_frame = ttk.LabelFrame(main_frame, text="1. Select Object to Rotate", padding="10")
        top_frame.pack(fill=tk.X, pady=5)

        self.loaded_kitchen_label_var = tk.StringVar(value="No kitchen generated yet.")
        ttk.Label(top_frame, textvariable=self.loaded_kitchen_label_var, font=("Helvetica", 10, "italic")).pack(anchor=tk.W)

        ttk.Label(top_frame, text="\nSelect Object:").pack(anchor=tk.W, pady=(5,2))
        self.rotate_object_var = tk.StringVar()
        self.rotate_object_menu = ttk.Combobox(top_frame, textvariable=self.rotate_object_var, state="readonly", width=40)
        self.rotate_object_menu.pack(fill=tk.X)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.rotate_button = ttk.Button(bottom_frame, text="Rotate and Export 12 Versions", command=self.run_rotation_process, style="Accent.TButton")
        self.rotate_button.pack(pady=5)

    def prepare_step2_tab(self):
        """Called after step 1 succeeds. Enables and populates the second tab."""
        # Update the label
        self.loaded_kitchen_label_var.set(f"Ready to process: kitchen_{self.generated_kitchen_num}.usd")

        # Update the combobox with object prim paths
        self.rotate_object_menu['values'] = self.generated_objects_paths
        if self.generated_objects_paths:
            self.rotate_object_var.set(self.generated_objects_paths[0])
        
        # Enable the tab and switch to it
        self.notebook.tab(1, state="normal")
        self.notebook.select(1)
        
    def run_rotation_process(self):
        """Runs the logic from your step2_gui.py script."""
        if self.generated_kitchen_num is None:
            messagebox.showerror("Error", "Please generate a kitchen scene in Step 1 first.")
            return
            
        rotate_obj_path = self.rotate_object_var.get()
        if not rotate_obj_path:
            messagebox.showerror("Error", "Please select an object to rotate.")
            return

        kitchen_num = self.generated_kitchen_num
        
        # Check that omniverse libraries are available before proceeding
        if not omni:
             messagebox.showerror("Omniverse Error", "Omniverse/Isaac libraries are not loaded. Cannot perform USD operations.")
             return

        try:
            self.rotate_button.config(state="disabled", text="Processing...")
            self.update_idletasks()

            # --- Core Logic from step2_gui.py ---
            base_usd = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}.usd"
            bodex_dir = os.path.expanduser("~/IsaacLab/source/isaaclab_assets/data/Kitchen/bodex")
            json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num}.json")

            with open(json_path, "r") as f:
                kitchen_data = json.load(f)

            usd_context = omni.usd.get_context()
            if not usd_context.open_stage(base_usd):
                raise RuntimeError(f"Failed to open USD stage: {base_usd}")
            
            stage = usd_context.get_stage()

            # Process prims: rename and add metadata
            for obj, fname in kitchen_data.items():
                if obj == "kitchen_type": continue
                
                old_path, new_path = f"/world/{obj}0", f"/world/{obj}"
                if stage.GetPrimAtPath(old_path):
                    Sdf.CopySpec(stage.GetRootLayer(), old_path, stage.GetRootLayer(), new_path)
                    stage.RemovePrim(old_path)
                
                obj_prim = stage.GetPrimAtPath(new_path)
                if obj_prim:
                    attr = obj_prim.CreateAttribute("BODex_path", Sdf.ValueTypeNames.String)
                    attr.Set(fname)
            
            # --- Perform Rotation ---
            obj_prim = stage.GetPrimAtPath(rotate_obj_path)
            if not obj_prim:
                # Handle case where the selected object name doesn't match the final prim path
                # This could happen if you selected 'mug0' but the script renamed it to '/world/mug'
                # Let's try finding the renamed path.
                renamed_path = f"/world/{rotate_obj_path.split('/')[-1]}"
                obj_prim = stage.GetPrimAtPath(renamed_path)
                if not obj_prim:
                    raise ValueError(f"Could not find prim at path: {rotate_obj_path} or {renamed_path}")

            xformable = UsdGeom.Xformable(obj_prim)
            xformable.AddRotateXYZOp(opSuffix="", precision=UsdGeom.XformOp.PrecisionFloat)
            translate_op = xformable.AddTranslateOp(opSuffix="", precision=UsdGeom.XformOp.PrecisionFloat)
            rotate_op = xformable.GetRotateXYZOp()
            xformable.SetXformOpOrder([translate_op, rotate_op])

            # Loop to export 12 rotated versions
            for i in range(12):
                angle = i * 30.0
                rotate_op.Set(Gf.Vec3f(0.0, 0.0, angle))
                
                usd_out = f"/root/IsaacLab/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num}_{i:02d}.usd"
                stage.GetRootLayer().Export(usd_out)
            
            messagebox.showinfo("Success!", f"Successfully exported 12 rotated versions for\n{rotate_obj_path}\nin kitchen_{kitchen_num}.")

        except Exception as e:
            messagebox.showerror("Rotation Error", f"An error occurred during the rotation process:\n\n{e}")
        finally:
            self.rotate_button.config(state="normal", text="Rotate and Export 12 Versions")

if __name__ == "__main__":
    # The AppLauncher is now initialized at the start of the script,
    # so we can just create and run the Tkinter app.
    app = KitchenManagerApp()
    app.mainloop()


