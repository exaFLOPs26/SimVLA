import os
import subprocess

storage_dir = "assets/StorageFurniture"
output_dir = "source/isaaclab_assets/data/Cabinet"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

for folder_name in os.listdir(storage_dir):
    folder_path = os.path.join(storage_dir, folder_name)
    urdf_path = os.path.join(folder_path, "mobility.urdf")
    usd_path = os.path.join(output_dir, f"{folder_name}.usd")

    if os.path.isdir(folder_path) and os.path.isfile(urdf_path):
        cmd = [
            "./isaaclab.sh",
            "-p",
            "scripts/tools/convert_urdf.py",
            urdf_path,
            usd_path,
            "--fix-base",
            "--merge-joints",
            "--headless",
            "--joint-stiffness", "0.0",
            "--joint-damping", "0.0",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)

