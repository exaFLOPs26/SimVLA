from omni.isaac.lab.sim import ArticulationCfg, ImplicitActuatorCfg

base_cabinet = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.0, 0.0),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_drawer_0_0': 0.0, 'corpus_to_drawer_0_1': 0.0, 'corpus_to_drawer_0_2': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_drawer_0_0", "corpus_to_drawer_0_1", "corpus_to_drawer_0_2"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

base_cabinet_0 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet_0",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(2.2758, -2.0792, 0.0),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={'corpus_to_door_0_2': 0.0, 'corpus_to_door_1_2': 0.0, 'corpus_to_door_2_2': 0.0, 'corpus_to_drawer_0_0': 0.0, 'corpus_to_drawer_0_1': 0.0, 'corpus_to_drawer_1_0': 0.0, 'corpus_to_drawer_1_1': 0.0, 'corpus_to_drawer_2_0': 0.0, 'corpus_to_drawer_2_1': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_2", "corpus_to_door_1_2", "corpus_to_door_2_2", "corpus_to_drawer_0_0", "corpus_to_drawer_0_1", "corpus_to_drawer_1_0", "corpus_to_drawer_1_1", "corpus_to_drawer_2_0", "corpus_to_drawer_2_1"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

base_cabinet_1 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/base_cabinet_1",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.6073, -2.0792, 0.0),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={'corpus_to_door_0_2': 0.0, 'corpus_to_door_1_2': 0.0, 'corpus_to_door_2_2': 0.0, 'corpus_to_drawer_0_0': 0.0, 'corpus_to_drawer_0_1': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_2", "corpus_to_door_1_2", "corpus_to_door_2_2", "corpus_to_drawer_0_0", "corpus_to_drawer_0_1"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

bowl = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/bowl",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(-0.2614, -0.3062, 0.7521),
			rot=(0.0701, 0.0, 0.0, 0.9975),
			joint_pos={},
		),
	)

countertop_base_cabinet = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/countertop_base_cabinet",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.0, 0.7297),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={},
		),
	)

countertop_base_cabinet_0 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/countertop_base_cabinet_0",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(2.2758, -2.0792, 0.7297),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={},
		),
	)

countertop_base_cabinet_1 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/countertop_base_cabinet_1",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.6073, -2.0792, 0.7297),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={},
		),
	)

countertop_dishwasher = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/countertop_dishwasher",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.0, 0.0, 0.7297),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={},
		),
	)

dishwasher = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/dishwasher",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.0, 0.0, 0.0),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpse_to_bottom_basket': 0.0, 'corpse_to_top_basket': 0.0, 'corpus_to_door_0_1': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpse_to_bottom_basket", "corpse_to_top_basket", "corpus_to_door_0_1"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

mug = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/mug",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.033, -0.3313, 0.7521),
			rot=(0.9952, 0.0, 0.0, 0.0976),
			joint_pos={},
		),
	)

obj0 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/obj0",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.1952, -0.3206, 0.7531),
			rot=(0.8439, 0.0, 0.0, 0.5365),
			joint_pos={},
		),
	)

plate = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/plate",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.1425, -0.3305, 0.7521),
			rot=(0.8981, 0.0, 0.0, 0.4398),
			joint_pos={},
		),
	)

range = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/range",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.4415, -2.0792, 0.0),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_drawer_0_2': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_1", "corpus_to_drawer_0_2"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

range_hood = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/range_hood",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.4415, -2.2044, 1.2558),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={},
		),
	)

refrigerator = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/refrigerator",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(2.3932, -0.0698, 0.0),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'door_joint': 0.0, 'freezer_door_joint': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["door_joint", "freezer_door_joint"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

sink_cabinet = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/sink_cabinet",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.6623, 0.0, 0.0),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_1': 0.0, 'corpus_to_door_1_1': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_1", "corpus_to_door_1_1"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(2.2758, -2.2593, 1.2558),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0, 'corpus_to_door_2_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0", "corpus_to_door_2_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_0 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_0",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.6073, -2.2593, 1.2558),
			rot=(0.0, 0.0, 0.0, 1.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0, 'corpus_to_door_2_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0", "corpus_to_door_2_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_1 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_1",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.1801, 1.2558),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_2 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_2",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.1801, 1.2558),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_3 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_3",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.1801, 1.2558),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_4 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_4",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(1.5106, 0.1801, 1.2558),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0, 'corpus_to_door_2_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0", "corpus_to_door_2_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

wall_cabinet_5 = ArticulationCfg(
		prim_path="{ENV_REGEX_NS}/Kitchen/wall_cabinet_5",
		spawn=None,
		init_state=ArticulationCfg.InitialStateCfg(
			pos=(0.8482, 0.1801, 1.2558),
			rot=(1.0, 0.0, 0.0, 0.0),
			joint_pos={'corpus_to_door_0_0': 0.0, 'corpus_to_door_1_0': 0.0, 'corpus_to_door_2_0': 0.0, 'corpus_to_door_3_0': 0.0, 'corpus_to_door_4_0': 0.0, 'corpus_to_door_5_0': 0.0},
		),
		actuators={
			"default": ImplicitActuatorCfg(
				joint_names_expr=["corpus_to_door_0_0", "corpus_to_door_1_0", "corpus_to_door_2_0", "corpus_to_door_3_0", "corpus_to_door_4_0", "corpus_to_door_5_0"],
				effort_limit=87.0,
				velocity_limit=100.0,
				stiffness=0.0,
				damping=1.0,
			)
		},
	)

