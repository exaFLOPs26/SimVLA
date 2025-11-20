import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

if __name__ == "__main__":
	# === Define dataset format ===
	features = {
		"index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
		"episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
		"frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
		"timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
		"is_first": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": null
        },
        "is_last": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": null
        },
		"language_instruction": {
            "dtype": "string",
            "shape": [
                1
            ],
            "names": null
        },
		"sub_task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
		"sub_task_class": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
		"sub_task_language_instruction": {
            "dtype": "string",
            "shape": [
                1
            ],
            "names": null
        },
		"observation.state.ee_pose": {
			"dtype": "float32", 
			"shape": (22,),
			"names": {
				"motors" : [
					"l_x",
                    "l_y",
                    "l_z",
                    "l_r1",
                    "l_r2",
                    "l_r3",
                    "l_r4",
                    "l_r5",
                    "l_r6",
                    "r_x",
                    "r_y",
                    "r_z",
                    "r_r1",
                    "r_r2",
                    "r_r3",
                    "r_r4",
                    "r_r5",
                    "r_r6",
                    "l_gripper",
                    "l_gripperR",
                    "r_gripper",
                    "r_gripperR"
				]
			}
		},
		"observation.images.front": {
            "dtype": "video",
            "shape": [
                320,
                240,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 320,
                "video.width": 240,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
		"observation.images.back": {
            "dtype": "video",
            "shape": [
                320,
                240,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 320,
                "video.width": 240,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
		"observation.images.wrist_left": {
            "dtype": "video",
            "shape": [
                320,
                240,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 320,
                "video.width": 240,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
		"observation.images.wrist_right": {
            "dtype": "video",
            "shape": [
                320,
                240,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 320,
                "video.width": 240,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
		"action.original":{
			"dtype": "float32",
            "shape": [
                20
            ],
            "names": {
                "motors": [
                    "l_x",
                    "l_y",
                    "l_z",
                    "l_r1",
                    "l_r2",
                    "l_r3",
                    "l_r4",
                    "l_r5",
                    "l_r6",
                    "r_x",
                    "r_y",
                    "r_z",
                    "r_r1",
                    "r_r2",
                    "r_r3",
                    "r_r4",
                    "r_r5",
                    "r_r6",
                    "l_gripper",
                    "r_gripper"
                ]
            }
		}
		"action_base" : {"dtype": "float32", "shape": (3,)} 
	}
	# === Load Input files ===
	hdf5_path = "/root/IsaacLab/datasets/anubis/Isaac_Kitchen_v1_1.hdf5"
	goal_json_file_path = "/root/IsaacLab/scripts/simvla/goals/Isaac-Kitchen-v1103-00.reloadable.json"

	f = h5py.File(hdf5_path, "r")
	# for all the demo
	demo = f["data"]["demo_0"]

	actions = demo["actions"]["ee_6D_pos"][:]# shape: (N, 6 or 7)
	actions_base = demo["actions"]["base"][:]
	observations = demo["observations"]["ee_6d_pos"][:]
    
	# Handle instruction
	instruction = demo["observations"]["language_instruction"][()]
	if isinstance(instruction, bytes):
		instruction = instruction.decode("utf-8")

	assert len(actions) == len(images_front)
	import shutil

	# === Set dataset path and clear it if already exists ===
	dataset_path = "/root/IsaacLab/datasets/lerobot/final_format"
	if Path(dataset_path).exists():
		shutil.rmtree(dataset_path)

	# === Create LeRobot dataset ===
	dataset = LeRobotDataset.create(
		repo_id="converted_hdf5_lerobot",
		root=dataset_path,
		fps=30,
		use_videos=False,
		features=features,
		image_writer_processes=5,
		image_writer_threads=10,
	)

	# === Add frames ===
	for i in range(len(actions)):
		sample = {
			"observation.state.ee_pose": observations[i],
			"action": actions[i],
			"action_base": actions_base[i],
		}
		dataset.add_frame(sample)

	# === Save episode ===
	print(">>> Saving episode...")
	dataset.save_episode()
	print("✅ Done: LeRobot dataset saved from HDF5.")


