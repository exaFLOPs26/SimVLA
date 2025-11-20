import scene_synthesizer as synth
from scene_synthesizer import utils
from scene_synthesizer import procedural_assets as pa

import trimesh.transformations as tra

obj_position_iterator = [
    utils.PositionIteratorUniform(),
    utils.PositionIteratorGaussian(params=[0, 0, 0.08, 0.08]),
    utils.PositionIteratorPoissonDisk(k=30, r=0.1),
    utils.PositionIteratorGrid(step_x=0.02, step_y=0.02, noise_std_x=0.04, noise_std_y=0.04),
    utils.PositionIteratorGrid(step_x=0.2, step_y=0.02, noise_std_x=0.0, noise_std_y=0.0),
    utils.PositionIteratorFarthestPoint(sample_count=1000),
]

mug = pa.MugAsset(origin=('com', 'com', 'bottom'))
table = pa.TableAsset(1.0, 1.4, 0.7)

s = synth.Scene()

cnt = 0
for x in range(3):
    for y in range(2):
        s.add_object(table, f'table{cnt}', transform=tra.translation_matrix((x * 1.5, y * 1.5, 0.0)))
        s.label_support(f'support{cnt}', obj_ids=[f'table{cnt}'])
        s.place_objects(
            obj_id_iterator=utils.object_id_generator(f"Mug{cnt}_"),
            obj_asset_iterator=(mug for _ in range(20)),
            obj_support_id_iterator=s.support_generator(f'support{cnt}'),
            obj_position_iterator=obj_position_iterator[cnt],
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(),
        )
        cnt += 1

s.colorize()
s.colorize(specific_objects={f'table{i}': [123, 123, 123] for i in range(6)})
s.show()
