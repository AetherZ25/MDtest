import os
from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version='v1.0-trainval',
                dataroot='../data/nuscenes', verbose=True)

#### find scene by name ####
scene_name = "scene-0329"
for idx, scene in enumerate(nusc.scene):
    if scene['name'] == scene_name:
        print(idx)
        break
my_scene = nusc.scene[idx]


#### visualiza sample with token ####
sample_token = "edd33328448f40b49f0f7e1da07b9ca8"
sensor = 'CAM_FRONT'

# render camera with bboxes
my_sample = nusc.get('sample', sample_token)
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
nusc.render_sample_data(cam_front_data['token'])
nusc.render_sample_data(cam_front_data['token'], out_path="./temp.png")

# render each annotation with lidar
os.makedirs("./anno")
my_annotation_tokens = my_sample['anns']
for idx, anno in enumerate(my_annotation_tokens):
    nusc.render_annotation(anno, out_path=f"./anno/{idx}.png")

# render full lidar
nusc.render_sample_data(
    my_sample['data']['LIDAR_TOP'],
    nsweeps=5, underlay_map=True, out_path="./anno/lidar.png")
