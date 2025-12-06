## What can SimVLA do?

In short, SimVLA is meant to train VLA using simulation data. For having simulation data, we need to generate a large-scale, diverse dataset. This is the repo that can generate goal states for each task and based on the goal states it runs motion planning(cuRobo) to collect an expert demo.

## Pipeline
- kitchen_scene_generator.py (/root/IsaacLab/scripts/simvla/kitchen_scene_generator.py): Generate kitchen scene
   - 5 different types of kitchen (Based on [scene-synthesizer](https://scene-synthesizer.github.io/index.html))
   - For the same type of kitchen, the arrangement, materials of furniture change.
   - Diverse objects can be put inside the scene.
   - Objects are rotated by 30 degrees (in total of 12 scenes made) if needed. (To grasp a mug, we grasp the handle if it is pointing to us, but if not, just grasp the cup.)
   - Unused furniture can be just visually existing but have no physical existence in simulation for lighter computation.
 
- goal_generator.py (/root/IsaacLab/scripts/simvla/goal_generator.py): Generate goal pose for motion planning
   - Based on the scripted skill api, it can do diverse tasks. Grasping pose is based on [BODex](https://github.com/JYChen18/BODex).
 
- generate.py (/root/IsaacLab/scripts/simvla/generate.py)
   - Using IsaacLab to generate demos in parallel
   - Currently using 50 envs at once to collect demo. 

## How to use it

1. Install the Docker image tar file
   ```bash
   docker load -i simvla.tar
   ```
3. Activate conda env
   ```bash
   conda activate env_isaaclab
   ```
4. Generate kitchen USD
   <img width="602" height="499" alt="image" src="https://github.com/user-attachments/assets/8f89423f-0b80-45e6-8046-45c3d686b048" />

   ```bash
   ./isaaclab.sh -p scripts/simvla/kitchen_scene_generator.py
   ```
5. Goal generator
   <img width="760" height="532" alt="image" src="https://github.com/user-attachments/assets/967f824e-3083-4074-bb5e-09c1ec53bde8" />

   ```bash
   ./isaaclab.sh -p scripts/simvla/goal_generator.py
   ```
6. Demo generate
   ```bash
   ./isaaclab.sh -p scripts/simvla/generate.py \
	--task Isaac-Kitchen-v1103-00 \
	--enable_cameras \
	--num_envs 35 \
	--num_demos 100 \
	--headless \
	--fix_init False \
	--robot anubis \
	--task_type NavManipulation \
	--task_language "Put bottle to sink"
   ```
7. Demo replay
   ```bash
   ./isaaclab.sh -p scripts/simvla/demo_replay.py \
	--task Isaac-Kitchen-v1103-00 \
	--enable_cameras \
	--num_envs 1 \
	--headless \
	--task_type NavManipulation
   ```
8. LeRobot Training
   ```bash
   lerobot-train \
     --dataset.repo_id={repo_id} \
     --policy.type=act \
     --output_dir= \
     --job_name= \
     --policy.device=cuda \
     --wandb.enable=true \
     --policy.repo_id= \
     --steps=100000 \
     --batch_size=8 \
     --wandb.disable_artifact=True
   ```
9. LeRobot evaluation in IsaacLab
   ```bash
   ./isaaclab.sh -p scripts/simvla/async_eval.py \
	--task Isaac-Kitchen-v1103-00 \
	--data Isaac-Kitchen-v1103-00 \
	--task_language  \
	--model ACT \
	--policy_path "" \
	--horizon 700 \
	--num_rollouts 5 \
	--log_dir  \
	--headless \
	--enable_cameras \
	--OOD False
   
   ```
