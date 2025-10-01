import time
import os
import open3d as o3d
import numpy as np
import yaml
import re
from scipy.spatial import KDTree
from sklearn import linear_model
from tqdm import tqdm
import scipy
from scipy.spatial import Delaunay
from functools import partial
import multiprocessing
import math
from scipy.spatial import ConvexHull


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = np.array((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3).astype(float)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_registration_angle(mat):
    cos_theta, sin_theta = mat[0, 0], mat[1, 0]
    cos_theta = np.clip(cos_theta, -1, 1)  
    theta_cos = np.arccos(cos_theta)
    return theta_cos if sin_theta >= 0 else 2 * np.pi - theta_cos


import random


def remove_ground_points(point_cloud, max_iterations=100, distance_threshold=0.2):
    ground_points = []
    non_ground_points = []

    for _ in range(max_iterations):
        indices = np.random.choice(len(point_cloud), 3, replace=False)
        candidate_points = point_cloud[indices]

        model = linear_model.RANSACRegressor()
        model.fit(candidate_points[:, :2], candidate_points[:, 2])

        distances = np.abs(model.predict(point_cloud[:, :2]) - point_cloud[:, 2])

        if np.sum(distances < distance_threshold) > len(ground_points):
            ground_points = point_cloud[distances < distance_threshold]
            non_ground_points = point_cloud[distances >= distance_threshold]

    return non_ground_points

def max_consecutive_zeros(lst):
    max_count = 0  
    current_count = 0  
    for num in lst:
        if num == 0:
            current_count += 1 
            max_count = max(max_count, current_count)  
        else:
            current_count = 0  
    return max_count




def classify_state(inter_points_number_total, convex_hull_number_total, distance_total):
    

    c1 = 0
    c2 = 0


    for i in range(len(inter_points_number_total)):
        score_r_1 = ( inter_points_number_total[i][0] - inter_points_number_total[i][1] ) / inter_points_number_total[i][1]
        score_r_2 = ( inter_points_number_total[i][1] - inter_points_number_total[i][2] ) / inter_points_number_total[i][2]
        score_r = ( score_r_1 + score_r_2 ) / 2

        score_0_1 = ( convex_hull_number_total[i][2] - convex_hull_number_total[i][3] ) / convex_hull_number_total[i][2]
        score_0_2 = ( convex_hull_number_total[i][3] - convex_hull_number_total[i][4] ) / convex_hull_number_total[i][3]
        score_0 = ( score_r_1 + score_r_2 ) / 2

        score_d = distance_total[i]/ sum(distance_total)

        c1 += score_r * score_d
        c2 += score_0 * score_d

    if c1 < 0.1 and c2 > 0.7:
        return 1 

    return 0 

def box_filter(pseduo_labels, multi_agent_point, ok, now_timestamp):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []
    # kdtree_points = [KDTree(multi_frame_points[i][:, :2]) for i in range(len(multi_frame_points))]

    for j in range(num_box):

        distance_total = []
        for car_ok in range(len(ok)):
            po = ok[car_ok].reshape(1,3)
            distance_total.append(np.linalg.norm(pseduo_labels[j][:2] - po[:, :2]))



        inter_points_number_total = []
        convex_hull_number_total = []
        scale_var = [1.5, 1.2, 1.0, 0.8, 0.5]
        # color_list = ['red', 'blue','green']
        for car_num in range(len(ok)):

            inter_points_number_val_scal = []
            convex_hull_number_val_scal = []

            for scale in range(len(scale_var)):
                # scale_box = pseduo_labels[j][:7].copy()
                scale_box = np.ones(7)
                scale_box[:3] = pseduo_labels[j][:3]
                scale_box[3:6] = pseduo_labels[j][3:6] * scale_var[scale]
                scale_box[6] = pseduo_labels[j][6]
                # vi.add_3D_boxes(scale_box.reshape(-1, 7), color='red')
                inter_mask_scale = in_hull(multi_agent_point[car_num][now_timestamp][:, :3],
                                 boxes_to_corners_3d(scale_box.reshape(-1, 7)).reshape(-1, 3))
                inter_points_scale = multi_agent_point[car_num][now_timestamp][:, :3][inter_mask_scale]
                convex_hull_scale = inter_points_scale[ConvexHull(inter_points_scale).vertices]


                # vi.add_points(inter_points_scale[:, :3], color='red', radius=10)
                inter_points_number_val_scal.append(inter_points_scale.shape[0])
                convex_hull_number_val_scal.append(convex_hull_scale.shape[0])

            inter_points_number_total.append(inter_points_number_val_scal)
            convex_hull_number_total.append(convex_hull_number_val_scal)

        state = classify_state(inter_points_number_total, convex_hull_number_total, distance_total)
        #
        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)


    return new_boxes

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def multi_pt2world(points_path, poses):
    points = []
    for point_path, pose in zip(points_path, poses):
        point = pcd_to_np(point_path)
        point_homogeneous = np.hstack((point[:, :3], np.ones((point.shape[0], 1))))
        pose_ = x_to_world(pose)
        point_ = np.dot(pose_, point_homogeneous.T).T
        point__ = remove_ground_points(point_)
        points.append(point__)
    return points


def pc_2_world(points, poses):
    point_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    pose_ = x_to_world(poses)
    point_ = np.dot(pose_, point_homogeneous.T).T
    # points.append(point_)
    return point_


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param

def return_pl_frome_single_scenario(count, node_timestamp_lsit):
  
    path = '/mnt/32THHD/xhe/datasets/OPV2V/train'
    scenario_folders = sorted([os.path.join(path, x)  
                               for x in os.listdir(path) if
                               os.path.isdir(os.path.join(path, x))])
    cav_list = sorted([x for x in os.listdir(scenario_folders[count]) 
                       if os.path.isdir(
            os.path.join(scenario_folders[count], x))])
    for cav_id in cav_list:
        cav_path = os.path.join(scenario_folders[count], cav_id)
        yaml_files = \
            sorted([os.path.join(cav_path, x)  
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml') and 'additional' not in x])
        break

    
    
    cur_timestamps = node_timestamp_lsit[count] 
    node_timestamp= node_timestamp_lsit[count+1] 

    multi_agent_point = []
    poses = []
    yaml_file_list = []
    for cav_id in cav_list:
        # if count == 0:
        #     continue
        cav_path = os.path.join(scenario_folder, cav_id)
        single_agent_point = []
        pose = []
        yaml_file_name_list = []
        for timestamp in timestamps:
            pcd_file_name = timestamp + ".pcd"
            point_path = os.path.join(cav_path, pcd_file_name)
            points_local = pcd_to_np(point_path)
    
            yaml_file_name = timestamp + ".yaml"
            yaml_path = os.path.join(cav_path, yaml_file_name)
            lidar_pose = load_yaml(yaml_path)['lidar_pose']
    
    
            points_gloabal = pc_2_world(points_local, lidar_pose)
            points_gloabal_ = remove_ground_points(points_gloabal)
            single_agent_point.append(points_gloabal_)
            pose.append(lidar_pose)
            yaml_file_name_list.append(yaml_path)
    
        poses.append(pose)
        yaml_file_list.append(yaml_file_name_list)
    
        single_agent_points = np.concatenate(single_agent_point, axis=0)
        multi_agent_point.append(single_agent_point)
   

    for num_timestamp in tqdm(range(cur_timestamps, node_timestamp)): #tqdm(range(node_timestamp-len(timestamps), node_timestamp))
        

            
        pseduo_labels = np.load(f'/mnt/32THHD/lwk/codes/OpenCOOD/pseduo_label_moma_1/pre_{num_timestamp}.npy')
       
        pseduo_labels_ = pseduo_labels.copy()

       

        box_center = pseduo_labels[:, :3].copy()
        box_center_new = pc_2_world(box_center, poses[0][num_timestamp - cur_timestamps])
        dif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - cur_timestamps]))
        pseduo_labels[:, :3] = box_center_new[:, :3]
        pseduo_labels[:, 6] = pseduo_labels[:, 6] + dif_ang


        ok = []
        for m in range(len(cav_list)):
            ok.append(np.array(poses[m][num_timestamp - cur_timestamps])[:3].reshape(1, 3))

        now_timestamp = num_timestamp - cur_timestamps
        out_pseduo_labels = box_filter(pseduo_labels, multi_agent_point, ok, now_timestamp)

        inverted_list = [not x for x in out_pseduo_labels]

        np.save(f'/mnt/32THHD/lwk/datas/OPV2V/out_xqm_moma_1_plus_ONLY_DENSITY/out_pseduo_labels_v1_{num_timestamp}.npy',
                pseduo_labels_[out_pseduo_labels])
        np.save(f'/mnt/32THHD/lwk/datas/OPV2V/out_xqm_moma_1_plus_ONLY_DENSITY/out_pseduo_labels_noise_v1_{num_timestamp}.npy',
                pseduo_labels_[inverted_list])
    
    return True

import itertools

if __name__ == '__main__':


    path = "/mnt/32THHD/xhe/datasets/OPV2V/train"

    scenario_folders = sorted([os.path.join(path, x)  # 单个元素的例：.../OPV2V/train/2021_08_16_22_26_54，为一个场景
                               for x in os.listdir(path) if
                               os.path.isdir(os.path.join(path, x))])
    count = 0
    node_timestamp_lsit = []
    for scenario_folder in tqdm(scenario_folders):
        cav_list = sorted([x for x in os.listdir(scenario_folder)  # scenario_folder下每个文件夹都代表一辆车，如641，650，659；单个元素例：641
                           if os.path.isdir(
                os.path.join(scenario_folder, x))])
        for cav_id in cav_list:
            cav_path = os.path.join(scenario_folder, cav_id)
            yaml_files = \
                sorted([os.path.join(cav_path, x)  # 例：将...\OPV2V\train\2021_08_16_22_26_54\641下'000069.yaml'这样的文件路径升序排序
                        for x in os.listdir(cav_path) if
                        x.endswith('.yaml') and 'additional' not in x])
            break
        timestamps = []
        for file in yaml_files:
            res = file.split(os.path.sep)[-1]
            timestamp = res.replace('.yaml', '')  # 如'000069.yaml'变成'000069'
            timestamps.append(timestamp)
        node_timestamp_lsit.append(len(timestamps))
    

    
    node_timestamp_lsit = [0]+ node_timestamp_lsit
    new_list = []
    current_sum = 0

    for num in range(len(node_timestamp_lsit)):
        current_sum += node_timestamp_lsit[num]
        new_list.append(current_sum) 

    sample_sequence_file_list = [i for i in range(43)]


    process_single_sequence = partial(
        return_pl_frome_single_scenario,
        node_timestamp_lsit=new_list,
    )

    with multiprocessing.Pool(16) as p:
        sequence_infos = list(
            tqdm(p.imap(process_single_sequence, sample_sequence_file_list), total=len(sample_sequence_file_list)))

