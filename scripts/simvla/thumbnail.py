import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import subprocess
import threading
import shutil
import platform # Import platform to check the OS
import open3d as o3d
import ipdb
class USDSelector:
	def __init__(self, master, usd_path):
		self.master = master
		master.title("USD File Selector")

		self.usd_folder = usd_path
		self.selected_file = None
		self.thumbnail_dir = f"{os.path.dirname(os.path.abspath(__file__))}/usd_thumbnails"
		os.makedirs(self.thumbnail_dir, exist_ok=True)
		
		self.status_label = tk.Label(master, text=f"Rendering previews for: {self.usd_folder}...")
		self.status_label.pack(pady=5)
		
		self.frame = tk.Frame(master)
		self.frame.pack(expand=True, fill=tk.BOTH)
		
		self.canvas = tk.Canvas(self.frame)
		self.scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
		self.scrollable_frame = tk.Frame(self.canvas)
		
		self.scrollable_frame.bind(
			"<Configure>",
			lambda e: self.canvas.configure(
				scrollregion=self.canvas.bbox("all")
			)
		)
		self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
		self.canvas.configure(yscrollcommand=self.scrollbar.set)
		
		self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		
		threading.Thread(target=self.generate_and_display).start()



	def generate_and_display(self):
		"""
		Generate thumbnails for USD files by converting them to OBJ first,
		then rendering offscreen with Open3D.
		"""
		from open3d.visualization import rendering
		import subprocess, tempfile

		# Clear any previous widgets
		for widget in self.scrollable_frame.winfo_children():
			widget.destroy()

		usd_files = [f for f in os.listdir(self.usd_folder)
					 if f.endswith(('.usd', '.usda'))]

		for i, filename in enumerate(usd_files):
			usd_path = os.path.join(self.usd_folder, filename)
			thumb_path = os.path.join(self.thumbnail_dir, f"thumb_{i}.png")
			ipdb.set_trace()
			try:
				# --- Convert USD to OBJ ---
				with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmpfile:
					obj_path = tmpfile.name
				# call usdcat to convert
				subprocess.run(
					["usdcat", usd_path, "-o", obj_path],
					check=True,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE
				)

				# --- Load OBJ into Open3D ---
				mesh = o3d.io.read_triangle_mesh(obj_path)
				if not mesh.has_triangles():
					print(f"Warning: Could not load mesh from {filename}, skipping.")
					continue

				# --- Render offscreen ---
				renderer = rendering.OffscreenRenderer(512, 512)

				# Setup material
				material = rendering.MaterialRecord()
				material.shader = "defaultLit"

				renderer.scene.clear_geometry()
				renderer.scene.add_geometry("mesh", mesh, material)

				# Auto camera view based on mesh bounds
				center = mesh.get_center()
				extent = mesh.get_max_bound() - mesh.get_min_bound()
				cam_pos = center + [0, 0, max(extent) * 2.5]
				up_dir = [0, 1, 0]
				renderer.scene.camera.look_at(center, cam_pos, up_dir)

				img = renderer.render_to_image()

				# Save image to file
				o3d.io.write_image(thumb_path, img)

			except Exception as e:
				self.status_label.config(text=f"Error rendering {filename}: {e}")
				continue

			# Display the generated thumbnail in the GUI
			self.display_thumbnail(thumb_path, filename)

		self.status_label.config(text="All previews generated. Select a file.")


	
	def display_thumbnail(self, thumb_path, filename):
		# The display logic can be simplified slightly to handle potential errors
		try:
			img = Image.open(thumb_path)
			img = img.resize((256, 256), Image.Resampling.LANCZOS)
			photo = ImageTk.PhotoImage(img)

			card_frame = tk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=1)
			card_frame.pack(pady=5, padx=5, side=tk.LEFT)

			label = tk.Label(card_frame, image=photo)
			label.image = photo
			label.pack()

			name_label = tk.Label(card_frame, text=filename, wraplength=200)
			name_label.pack()
			
			select_btn = tk.Button(card_frame, text="Select", command=lambda f=filename: self.on_select(f))
			select_btn.pack(pady=5)
		except Exception as e:
			print(f"Failed to display thumbnail for {filename}: {e}")


	def on_select(self, filename):
		self.selected_file = os.path.join(self.usd_folder, filename)
		self.status_label.config(text=f"Selected: {self.selected_file}")
		print(f"Selected file path: {self.selected_file}")
		self.master.destroy()
		
	def run(self):
		self.master.mainloop()
		
	def __del__(self):
		if os.path.exists(self.thumbnail_dir):
			shutil.rmtree(self.thumbnail_dir)

if __name__ == "__main__":
	# --- IMPORTANT ---
	# Change this path to the folder containing your USD files
	FOLDER_PATH = "/Users/your_user/Documents/your_usd_folder" 
	
	if not os.path.isdir(FOLDER_PATH):
		print(f"Error: The folder '{FOLDER_PATH}' does not exist.")
		print("Please update the FOLDER_PATH variable in the script.")
	else:
		root = tk.Tk()
		app = USDSelector(root, FOLDER_PATH)
		app.run()
