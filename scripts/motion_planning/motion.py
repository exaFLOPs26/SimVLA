try:
	# Third Party
	import isaacsim
except ImportError:
	pass

# Third Party
import torch
a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
	"--headless_mode",
	type=str,
	default=None,
	help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
	"--robot",
	type=str,
	default="anubis_left_arm.yml",
	help="robot configuration to load"
	)
parser.add_argument(
	"--external_asset_path",
	type=str,
	default=None,
	help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
	"--external_robot_configs_path",
	type=str,
	default=None,
	help="Path to external robot config when loading an external robot",
)

parser.add_argument(
	"--visualize_spheres",
	action="store_true",
	help="When True, visualizes robot spheres",
	default=False,
)
parser.add_argument(
	"--reactive",
	action="store_true",
	help="When True, runs in reactive mode",
	default=False,
)

parser.add_argument(
	"--constrain_grasp_approach",
	action="store_true",
	help="When True, approaches grasp with fixed orientation and motion only along z axis.",
	default=False,
)

parser.add_argument(
	"--reach_partial_pose",
	nargs=6,
	metavar=("qx", "qy", "qz", "x", "y", "z"),
	help="Reach partial pose",
	type=float,
	default=None,
)
parser.add_argument(
	"--hold_partial_pose",
	nargs=6,
	metavar=("qx", "qy", "qz", "x", "y", "z"),
	help="Hold partial pose while moving to goal",
	type=float,
	default=None,
)

args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
	{
		"headless": args.headless_mode is not None,
		"width": "1920",
		"height": "1080",
	}
)

# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions#, add_robot_to_scene
from isaacsim.core.api.world.world import World
from isaacsim.core.api.objects import cuboid, sphere
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api.robots.robot import Robot

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import (
	get_assets_path,
	get_filename,
	get_path_of_dir,
	get_robot_configs_path,
	get_world_configs_path,
	join_path,
	load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
	MotionGen,
	MotionGenConfig,
	MotionGenPlanConfig,
	PoseCostMetric,
)

############################################################

def add_robot_to_scene_mod(
	robot_config: Dict,
	my_world: World,
	load_from_usd: bool = False,
	subroot: str = "",
	robot_name: str = "robot",
	position: np.array = np.array([0, 0, 0]),
	orientation: np.array = np.array([1.0, 0.0, 0.0, 0.0]),
):

	urdf_interface = _urdf.acquire_urdf_interface()
	# Set the settings in the import config
	import_config = _urdf.ImportConfig()
	import_config.merge_fixed_joints = False
	import_config.convex_decomp = False
	import_config.fix_base = True
	import_config.make_default_prim = True
	import_config.self_collision = False
	import_config.create_physics_scene = True
	import_config.import_inertia_tensor = False
	import_config.default_drive_strength = 1047.19751
	import_config.default_position_drive_damping = 52.35988
	import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
	import_config.distance_scale = 1
	import_config.density = 0.0

	asset_path = get_assets_path()
	if (
		"external_asset_path" in robot_config["kinematics"]
		and robot_config["kinematics"]["external_asset_path"] is not None
	):
		asset_path = robot_config["kinematics"]["external_asset_path"]
	full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
	robot_path = get_path_of_dir(full_path)
	filename = get_filename(full_path)
	imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
	dest_path = subroot
	robot_path = urdf_interface.import_robot(
		robot_path,
		filename,
		imported_robot,
		import_config,
		dest_path,
	)

	base_link_name = robot_config["kinematics"]["base_link"]

	robot_p = Robot(
		prim_path=robot_path + "/" + base_link_name,
		name=robot_name,
	)

	robot_prim = robot_p.prim
	stage = robot_prim.GetStage()
	linkp = stage.GetPrimAtPath(robot_path)
	set_prim_transform(linkp, [position[0], position[1], position[2],
							   orientation[0], orientation[1], orientation[2], orientation[3]])

	robot = my_world.scene.add(robot_p)
	return robot, robot_path

def main():
	# create a curobo motion gen instance:
	num_targets = 0
	# assuming obstacles are in objects_path:
	my_world = World(stage_units_in_meters=1.0)
	stage = my_world.stage

	xform = stage.DefinePrim("/World", "Xform")
	stage.SetDefaultPrim(xform)
	stage.DefinePrim("/curobo", "Xform")
	stage = my_world.stage


	# Make a target to follow
	target = cuboid.VisualCuboid(
		"/World/target",
		position=np.array([0.2,  0.0, 1.0]),
		orientation=np.array([1.0, 0.0, 0.0,0.0]),
		color=np.array([1.0, 0, 0]),
		size=0.1,
	)

	setup_curobo_logger("warn")
	past_pose = None
	n_obstacle_cuboids = 30
	n_obstacle_mesh = 100

	# warmup curobo instance
	usd_help = UsdHelper()
	target_pose = None

	tensor_args = TensorDeviceType()
	robot_cfg_path = get_robot_configs_path()
	if args.external_robot_configs_path is not None:
		robot_cfg_path = args.external_robot_configs_path
	robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

	if args.external_asset_path is not None:
		robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
	if args.external_robot_configs_path is not None:
		robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
	j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
	default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

	robot, robot_prim_path = add_robot_to_scene_mod(robot_config=robot_cfg,
													my_world=my_world,
													position=np.array([0.0, 0.0, 0.0], float),
													orientation=np.array([1.0, 0.0, 0.0, 0.0], float)
												   )

	articulation_controller = None

	world_cfg_table = WorldConfig.from_dict(
		load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
	)
	world_cfg_table.cuboid[0].pose[2] -= 0.02
	world_cfg1 = WorldConfig.from_dict(
		load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
	).get_mesh_world()
	world_cfg1.mesh[0].name += "_mesh"
	world_cfg1.mesh[0].pose[2] = -10.5

	world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

	trajopt_dt = None
	optimize_dt = True
	trajopt_tsteps = 32
	trim_steps = None
	max_attempts = 2
	interpolation_dt = 0.05
	enable_finetune_trajopt = True
	if args.reactive:
		trajopt_tsteps = 40
		trajopt_dt = 0.04
		optimize_dt = False
		max_attempts = 1
		trim_steps = [1, None]
		interpolation_dt = trajopt_dt
		enable_finetune_trajopt = False
	motion_gen_config = MotionGenConfig.load_from_robot_config(
		robot_cfg,
		world_cfg,
		tensor_args,
		rotation_threshold = 1,
		position_threshold =1 ,
		collision_checker_type=CollisionCheckerType.MESH,
		num_trajopt_seeds=12,
		num_graph_seeds=12,
		interpolation_dt=interpolation_dt,
		collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
		optimize_dt=optimize_dt,
		trajopt_dt=trajopt_dt,
		trajopt_tsteps=trajopt_tsteps,
		trim_steps=trim_steps,
	)
	motion_gen = MotionGen(motion_gen_config)
	if not args.reactive:
		print("warming up...")
		motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

	print("Curobo is Ready")

	add_extensions(simulation_app, args.headless_mode)

	plan_config = MotionGenPlanConfig(
		enable_graph=False,
		enable_graph_attempt=4,
		max_attempts=max_attempts,
		enable_finetune_trajopt=enable_finetune_trajopt,
		time_dilation_factor=0.5,
	)

	usd_help.load_stage(my_world.stage)
	usd_help.add_world_to_stage(world_cfg, base_frame="/World")

	cmd_plan = None
	cmd_idx = 0
	my_world.scene.add_default_ground_plane()
	i = 0
	spheres = None
	past_cmd = None
	target_orientation = None
	past_orientation = None
	pose_metric = None
	initialized = False
	while simulation_app.is_running():
		my_world.step(render=True)
		if not my_world.is_playing():
			if i % 100 == 0:
				print("**** Click Play to start simulation *****")
			i += 1
			continue

		step_index = my_world.current_time_step_index

		if articulation_controller is None:
			# robot.initialize()
			articulation_controller = robot.get_articulation_controller()

		if initialized == False and step_index < 20:
			my_world.reset()
			robot._articulation_view.initialize()
			idx_list = [robot.get_dof_index(x) for x in j_names]
			robot.set_joint_positions(default_config, idx_list)

			robot._articulation_view.set_max_efforts(
				values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
			)
			initialized = True
			print(f"initialized:{initialized}")

		if step_index < 20:
			continue
		else:
			initialized = False

		# position and orientation of target virtual cube:
		cube_position, cube_orientation = target.get_world_pose()
		if past_pose is None:
			past_pose = cube_position
		if target_pose is None:
			target_pose = cube_position
		if target_orientation is None:
			target_orientation = cube_orientation
		if past_orientation is None:
			past_orientation = cube_orientation

		sim_js = robot.get_joints_state()
		sim_js_names = robot.dof_names
		if np.any(np.isnan(sim_js.positions)):
			log_error("isaac sim has returned NAN joint position values.")
		cu_js = JointState(
			position=tensor_args.to_device(sim_js.positions),
			velocity=tensor_args.to_device(sim_js.velocities),	# * 0.0,
			acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
			jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
			joint_names=sim_js_names,
		)

		if not args.reactive:
			cu_js.velocity *= 0.0
			cu_js.acceleration *= 0.0

		if args.reactive and past_cmd is not None:
			cu_js.position[:] = past_cmd.position
			cu_js.velocity[:] = past_cmd.velocity
			cu_js.acceleration[:] = past_cmd.acceleration
		cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

		if args.visualize_spheres and step_index % 2 == 0:
			sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

			if spheres is None:
				spheres = []
				# create spheres:

				for si, s in enumerate(sph_list[0]):
					sp = sphere.VisualSphere(
						prim_path="/curobo/robot_sphere_" + str(si),
						position=np.ravel(s.position),
						radius=float(s.radius),
						color=np.array([0, 0.8, 0.2]),
					)
					spheres.append(sp)
			else:
				for si, s in enumerate(sph_list[0]):
					if not np.isnan(s.position[0]):
						spheres[si].set_world_pose(position=np.ravel(s.position))
						spheres[si].set_radius(float(s.radius))

		if (
			(
				np.linalg.norm(cube_position - target_pose) > 1e-3
				or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
			)
			and np.linalg.norm(past_pose - cube_position) == 0.0
			and np.linalg.norm(past_orientation - cube_orientation) == 0.0
		):
			print("here")
			ee_translation_goal = cube_position
			ee_orientation_teleop_goal = cube_orientation
			ik_goal = Pose(
				position=tensor_args.to_device(ee_translation_goal),
				quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
			)
			plan_config.pose_cost_metric = pose_metric
			import ipdb
			ipdb.set_trace()
			result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
			succ = result.success.item()
			print(f"succ:{succ}")
			if succ:
				num_targets += 1
				cmd_plan = result.get_interpolated_plan()
				cmd_plan = motion_gen.get_full_js(cmd_plan)
				# get only joint names that are in both:
				idx_list = []
				common_js_names = []
				for x in sim_js_names:
					if x in cmd_plan.joint_names:
						idx_list.append(robot.get_dof_index(x))
						common_js_names.append(x)

				cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

				cmd_idx = 0

			else:
				carb.log_warn("Plan did not converge to a solution: " + str(result.status))
			target_pose = cube_position
			target_orientation = cube_orientation
		past_pose = cube_position
		past_orientation = cube_orientation
		if cmd_plan is not None:
			cmd_state = cmd_plan[cmd_idx]
			past_cmd = cmd_state.clone()
			# get full dof state
			art_action = ArticulationAction(
				cmd_state.position.cpu().numpy(),
				cmd_state.velocity.cpu().numpy(),
				joint_indices=idx_list,
			)
			print(art_action)
			# ['arm1_base_link_joint', 'link11_joint', 'link12_joint', 'link13_joint', 'link14_joint', 'link15_joint']
			# For 6 joints art_action will print joint position, velocity
			# {'joint_positions': [-0.00017660381854511797, -0.6920557022094727, 2.343322277069092, -0.056345272809267044, 0.523541271686554, -0.17770053446292877], 'joint_velocities': [-0.0020027693826705217, -0.005713684484362602, 0.0044566309079527855, 0.0022359408903867006, -0.004965614061802626, 0.00461201649159193], 'joint_efforts': None}


			# set desired joint angles obtained from IK:
			articulation_controller.apply_action(art_action)
			cmd_idx += 1
			for _ in range(2):
				my_world.step(render=False)
			if cmd_idx >= len(cmd_plan.position):
				cmd_idx = 0
				cmd_plan = None
				past_cmd = None
	simulation_app.close()


if __name__ == "__main__":
	main()

