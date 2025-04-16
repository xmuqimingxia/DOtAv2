import time
from viewer.viewer import Viewer
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

from outline_utils import OutlineFitter, drop_cls, voxel_sampling

def max_consecutive_zeros(lst):
    max_count = 0  # 初始化最大连续0的数量
    current_count = 0  # 初始化当前连续0的数量

    for num in lst:
        if num == 0:
            current_count += 1  # 如果是0，增加当前计数
            max_count = max(max_count, current_count)  # 更新最大计数
        else:
            current_count = 0  # 如果不是0，重置当前计数

    return max_count

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
    cos_theta = np.clip(cos_theta, -1, 1)  # 限制cos_theta在-1到1之间
    theta_cos = np.arccos(cos_theta)
    return theta_cos if sin_theta >= 0 else 2 * np.pi - theta_cos


import random


def remove_ground_points(point_cloud, max_iterations=100, distance_threshold=0.2):
    ground_points = []
    non_ground_points = []

    for _ in range(max_iterations):
        # 随机选择三个点作为地面模型的候选点
        indices = np.random.choice(len(point_cloud), 3, replace=False)
        candidate_points = point_cloud[indices]

        # 拟合地面模型
        model = linear_model.RANSACRegressor()
        model.fit(candidate_points[:, :2], candidate_points[:, 2])

        # 计算所有点到地面模型的垂直距离
        distances = np.abs(model.predict(point_cloud[:, :2]) - point_cloud[:, 2])

        # 判断当前地面模型是否合理
        if np.sum(distances < distance_threshold) > len(ground_points):
            ground_points = point_cloud[distances < distance_threshold]
            non_ground_points = point_cloud[distances >= distance_threshold]

    return non_ground_points


def classify_state_v1(inter_points, key):
    if not inter_points:  # 检查inter_points是否为空
        return 0  # 如果为空，返回状态0
    num_zeros = np.sum(np.array(inter_points) == 0)
    # print(inter_points, num_zeros, len(inter_points))
    if num_zeros == len(inter_points):
        return 0
    if inter_points[key] < 50:
        return 0
    threshold = 0.5 * len(inter_points)
    return 1 if num_zeros > 1 else 0

def classify_state_v2(inter_points_number_total, key, sorted_indices, distance_total):

    #规则一：需要框内点云数量大于零
    cur_frame_points = 0
    for car in range(len(inter_points_number_total)):
        cur_frame_points += inter_points_number_total[car][key]

    if cur_frame_points == 0:
        return 0
    else:
        True

    # print(inter_points_number_total[sorted_indices[0]][key], distance_total)
    #规则二：对于近处物体要求当前帧点云数量大于50，连续出现0
    result = [sum(elements) for elements in zip(*inter_points_number_total)]
    if distance_total[sorted_indices[0]] < 20:
        # print(inter_points_number_total[sorted_indices[0]][key]) #sorted_indices[0]最近的那个智能体是哪个智能体
        # input()
        densest = inter_points_number_total[sorted_indices[0]]
        # print('000000000000000000000000000', densest)
        if densest[key] < 100:
            return 0
        elif max_consecutive_zeros(densest) < 3:
            return 0
    elif 20 < distance_total[sorted_indices[0]] < 30:
        # print('111111111111111111111111111', result)
        if result[key] < 50:
            return 0
        elif max_consecutive_zeros(result) < 5:
            return 0
    elif 30 < distance_total[sorted_indices[0]] < 50:
        # print('****************************', result)
        if result[key] < 25:
            return 0
        elif max_consecutive_zeros(result) < 5:
            return 0
    else:
        # print('22222222222222222222222222222', result)
        if result[key] < 5:
            return 0
        elif max_consecutive_zeros(result) < 10:
            return 0

    # for inter_points_number_single in inter_points_number_total:
    #     number_sum = sum(inter_points_number_single)
    #     # for i in range(len(inter_points_number_total[car])):
    #     #     ppscore.append(inter_points_number_total[car][i] / number_sum)
    #     ppscore = [(inter_points_number_single[i] / number_sum if number_sum else 0) for i in range(len(inter_points_number_single))]
    #     ppscore_differences = [0] + [ppscore[i] - ppscore[i - 1] for i in
    #                                         range(1, len(ppscore))]
    #     print('######################', ppscore, ppscore_differences)
    #
    #     ppscore_all_car.append(ppscore)
    # exit()


    # if not inter_points:  # 检查inter_points是否为空
    #     return 0  # 如果为空，返回状态0
    # num_zeros = np.sum(np.array(inter_points) == 0)
    # # print(inter_points, num_zeros, len(inter_points))
    # if num_zeros == len(inter_points):
    #     return 0
    # if inter_points[key] < 50:
    #     return 0
    # threshold = 0.5 * len(inter_points)
    return 1


def classify_state(inter_points, key, inter_points_threshold):
    if not inter_points:  # 检查inter_points是否为空
        return 0  # 如果为空，返回状态0
    num_zeros = np.sum(np.array(inter_points) == 0)
    # print(inter_points, num_zeros, len(inter_points))
    if num_zeros == len(inter_points):
        return 0
    if inter_points[key] < inter_points_threshold:
        return 0
    threshold = 0.5 * len(inter_points)
    return 1 if num_zeros > 1 else 0


def box_filter(pseduo_labels, multi_frame_points, key, ok):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []
    kdtree_points = [KDTree(multi_frame_points[i][:, :2]) for i in range(len(multi_frame_points))]

    for j in range(num_box):
        # kdtree_points = [KDTree(multi_frame_points[i][:, :2]) for i in range(len(multi_frame_points))]
        center_annotion = pseduo_labels[j, :2]
        pose_center = ok[0, :2]
        disance_with_ok = np.linalg.norm(center_annotion - pose_center)
        if disance_with_ok < 20:
            inter_points_threshold = 20
        elif 20 < disance_with_ok < 40:
            inter_points_threshold = 10
        else:
            inter_points_threshold = 5
        radiu = 2
        inter_points = []

        for i, kdtree in enumerate(kdtree_points):
            indices = kdtree.query_ball_point(center_annotion, radiu)
            proposal_points = multi_frame_points[i][indices]
            points_num = proposal_points.shape[0]
            inter_points.append(points_num)

        state = classify_state(inter_points, key, inter_points_threshold)
        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)

    return new_boxes


def box_filter_v2(pseduo_labels, multi_frame_points, key, ok):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []

    # multi_frame_points_remove_ground = []
    # for f in range(len(multi_frame_points)):
    #     multi_frame_points_remove_ground.append(remove_ground_points(multi_frame_points[f]))

    for j in range(num_box):
        center_annotion = pseduo_labels[j, :2]
        pose_center = ok[0, :2]
        disance_with_ok = np.linalg.norm(center_annotion - pose_center)
        if disance_with_ok < 20:
            inter_points_threshold = 20
        elif 20 < disance_with_ok < 40:
            inter_points_threshold = 10
        else:
            inter_points_threshold = 5

        inter_points_number = []
        for i in range(len(multi_frame_points)):
            inter_mask = in_hull(multi_frame_points[i][:, :3],
                                 boxes_to_corners_3d(pseduo_labels[j][:7].reshape(-1, 7)).reshape(-1, 3))
            inter_points = multi_frame_points[i][inter_mask]
            inter_points_number.append(inter_points.shape[0])

        state = classify_state(inter_points_number, key, inter_points_threshold)
        # print(inter_points_number, inter_points_number[key], len(inter_points_number), state)
        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)

    return new_boxes


def box_filter_v3(pseduo_labels, multi_frame_points, key, ok):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []

    # multi_frame_points_remove_ground = []
    # for f in range(len(multi_frame_points)):
    #     multi_frame_points_remove_ground.append(remove_ground_points(multi_frame_points[f]))

    for j in range(num_box):
        center_annotion = pseduo_labels[j, :2]
        # pose_center = ok[0, :2]
        pose_center = [0, 0, 0]
        for cav in range(len(ok)):
            pose_center = ok[cav][0, :] + pose_center
        pose_center = pose_center / len(ok)
        disance_with_ok = np.linalg.norm(center_annotion - pose_center[:2])  # 距离中心点的距离

        r = 0
        for cav in range(len(ok)):
            r = np.linalg.norm(ok[cav][0, :2] - pose_center[:2]) + r
        r = r / len(ok)
        if disance_with_ok < r:
            inter_points_threshold = 200
        elif r < disance_with_ok < 2 * r:
            inter_points_threshold = 100
        elif 2 * r < disance_with_ok < 2.5 * r:
            inter_points_threshold = 80
        else:
            inter_points_threshold = 15

        inter_points_number = []
        for i in range(len(multi_frame_points)):
            inter_mask = in_hull(multi_frame_points[i][:, :3],
                                 boxes_to_corners_3d(pseduo_labels[j][:7].reshape(-1, 7)).reshape(-1, 3))
            inter_points = multi_frame_points[i][inter_mask]
            inter_points_number.append(inter_points.shape[0])

        state = classify_state(inter_points_number, key, inter_points_threshold)
        # print('################', inter_points_number, state, inter_points_threshold, disance_with_ok, r)
        # # #
        # vi.add_points(multi_frame_points[key][:, :3])
        # vi.add_points(pose_center[:3].reshape(1, 3), radius=10, color='red')
        # # vi.add_3D_boxes(gt, color='green')
        # vi.add_3D_boxes(pseduo_labels[j].reshape(1, 7), color='red')
        # vi.show_3D()

        # print(inter_points_number, inter_points_number[key], len(inter_points_number), state)

        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)

    return new_boxes


def box_filter_v4(pseduo_labels, multi_frame_points, key, ok, begin_frame, end_frame): #box_filter_v4(pseduo_labels, multi_frame_points, key, ok, begin_frame, end_frame)
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []
    # kdtree_points = [KDTree(multi_frame_points[i][:, :2]) for i in range(len(multi_frame_points))]

    for j in range(num_box):
        # kdtree_points = [KDTree(multi_frame_points[i][:, :2]) for i in range(len(multi_frame_points))]
        # center_annotion = pseduo_labels[j, :2]
        # radiu = 2

        # inter_points = []
        # for i, kdtree in enumerate(kdtree_points):
        #     indices = kdtree.query_ball_point(center_annotion, radiu)
        #     proposal_points = multi_frame_points[i][indices]
        #     points_num = proposal_points.shape[0]
        #     inter_points.append(points_num)

        distance_total = []
        for car_ok in range(len(ok)):
            po = ok[car_ok].reshape(1,3)
            distance_total.append(np.linalg.norm(pseduo_labels[j][:2] - po[:, :2]))
            # vi.add_points(po, color='red', radius= 10)

        sorted_indices = [index for index, value in sorted(enumerate(distance_total), key=lambda x: x[1])]
        # print(distance_total[sorted_indices[0]])

        inter_points_number_total = []
        # color_list = ['red', 'blue','green']
        for car_num in range(len(ok)):
            inter_points_number = []
            for i in range(begin_frame, end_frame):
                inter_mask = in_hull(multi_frame_points[car_num][i][:, :3],
                                     boxes_to_corners_3d(pseduo_labels[j][:7].reshape(-1, 7)).reshape(-1, 3))
                inter_points = multi_frame_points[car_num][i][inter_mask]
                # vi.add_points(inter_points[:, :3], color=color_list[car_num])
                inter_points_number.append(inter_points.shape[0])
            inter_points_number_total.append(inter_points_number)
        # vi.add_3D_boxes(pseduo_labels[j][:7].reshape(-1, 7))
        # vi.show_3D()

        # print('###############len(inter_points)###############', len(inter_points_number_total[0]))

        state = classify_state_v2(inter_points_number_total, key, sorted_indices, distance_total)
        # print('state', state)
        # if 1 == 1:
        #     vi.add_3D_boxes(pseduo_labels[j][:7].reshape(-1, 7))
        #     vi.add_points(multi_frame_points[sorted_indices[0]][key][:, :3])
        #     vi.show_3D()
        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)

    # new_boxes_ = np.vstack(new_boxes) if new_boxes else np.array([])
    # print('###################', new_boxes_.shape)

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
    '''

    :param scenario_folder: 选的是哪个场景
    :param node_timestamp: 该场景的起始帧号
    :return:
    '''
    path = "F:\\OPV2V\\OPV2V\\train"
    scenario_folders = sorted([os.path.join(path, x)  # 单个元素的例：.../OPV2V/train/2021_08_16_22_26_54，为一个场景
                               for x in os.listdir(path) if
                               os.path.isdir(os.path.join(path, x))])
    cav_list = sorted([x for x in os.listdir(scenario_folders[count])  # scenario_folder下每个文件夹都代表一辆车，如641，650，659；单个元素例：641
                       if os.path.isdir(
            os.path.join(scenario_folders[count], x))])
    for cav_id in cav_list:
        cav_path = os.path.join(scenario_folders[count], cav_id)
        yaml_files = \
            sorted([os.path.join(cav_path, x)  # 例：将...\OPV2V\train\2021_08_16_22_26_54\641下'000069.yaml'这样的文件路径升序排序
                    for x in os.listdir(cav_path) if
                    x.endswith('.yaml') and 'additional' not in x])
        break

    cur_timestamps = node_timestamp_lsit[count]
    node_timestamp= node_timestamp_lsit[count+1]


    multi_agent_point = np.load(f'F:\\OPV2V\\OPV2V\\multi_agent_point_remove_ground\\multi_agent_point{count}.npy',
                                allow_pickle=True)
    poses = np.load(f'F:\\OPV2V\\OPV2V\\multi_agent_point_pose\\multi_agent_point_pose{count}.npy',
                    allow_pickle=True)

    for num_timestamp in tqdm(range(cur_timestamps, node_timestamp)): #tqdm(range(node_timestamp-len(timestamps), node_timestamp))
        # if os.path.exists(f'F:\\OPV2V\\unsupervised_dbscan\\unsupervised_dbscan_{num_timestamp}.npy'):
        #     continue


        # gt = np.load(f'F:\\OPV2V\\OPV2V\\gt_box\\gt_{num_timestamp}.npy')
        #
        # gt_ = gt.copy()
        #
        # gtbox_center = gt[:, :3].copy()
        # gtbox_center_new = pc_2_world(gtbox_center, poses[0][num_timestamp - cur_timestamps])
        gtdif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - cur_timestamps]))
        # gt[:, :3] = gtbox_center_new[:, :3]
        # gt[:, 6] = gt[:, 6] + gtdif_ang


        for m in range(len(cav_list)):
            if m == 0:
                dense_points = multi_agent_point[m][num_timestamp-cur_timestamps]
            else:
                dense_points = np.concatenate((dense_points, multi_agent_point[m][num_timestamp-cur_timestamps]), 0)

        outline_estimator = OutlineFitter(sensor_height=0,
                                          ground_min_threshold=[0.2, -0.5, -0.5],
                                          ground_min_distance=[0, 20, 40, 100],
                                          cluster_dis=0.5,
                                          cluster_min_points=5,
                                          discard_max_height=4,
                                          min_box_volume=0.1,
                                          min_box_height=0.3,
                                          max_box_volume=200,
                                          max_box_len=10)

        clusters, labels = outline_estimator.clustering(dense_points)

        boxes = outline_estimator.box_fit(clusters)

        if boxes != []:
            pose_inverse = np.linalg.inv(x_to_world(poses[0][num_timestamp - cur_timestamps]))
            box_center = np.hstack((boxes[:, :3], np.ones((boxes.shape[0], 1))))
            gtbox_center_original = np.dot(pose_inverse, box_center.T).T
            gtdif_ang_inverse = -gtdif_ang
            gt_angle_original = boxes[:, 6] - gtdif_ang_inverse
            boxes[:, :3] = gtbox_center_original[:, :3]
            boxes[:, 6] = gt_angle_original

            mask = np.logical_and(np.logical_and(7 > boxes[:, 3], boxes[:, 3] >2),
                                  np.logical_and(4 > boxes[:, 4], boxes[:, 4] > 1))
            boxes = boxes[mask]

        np.save(f'F:\\OPV2V\\unsupervised_modest_FILTER\\unsupervised_dbscan_{num_timestamp}.npy',
                boxes)


        # vi.add_3D_boxes(pseduo_labels, color='black')
        # vi.add_points(ok_0, radius=10, color='red')
        # vi.add_3D_boxes(gt_, color='green')
        # vi.add_3D_boxes(boxes, color='red')
        # vi.add_points(dense_points[:, :3])


        vi.show_3D()
    return True


if __name__ == '__main__':

    vi = Viewer()

    path = "F:\\OPV2V\\OPV2V\\train"

    # print(os.listdir(path))
    scenario_folders = sorted([os.path.join(path, x)  # 单个元素的例：.../OPV2V/train/2021_08_16_22_26_54，为一个场景
                               for x in os.listdir(path) if
                               os.path.isdir(os.path.join(path, x))])
    count = 0
    # node_timestamp = 0
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

    #return_pl_frome_single_scenario(count, node_timestamp_lsit)

    node_timestamp_lsit = [0] + node_timestamp_lsit
    new_list = []
    current_sum = 0

    for num in range(len(node_timestamp_lsit)):
        current_sum += node_timestamp_lsit[num]
        new_list.append(current_sum)
        # return_pl_frome_single_scenario(count, node_timestamp_lsit)

    multiprocessing_index = False

    if multiprocessing_index == True:

        sample_sequence_file_list = [i for i in range(43)]

        process_single_sequence = partial(
            return_pl_frome_single_scenario,
            node_timestamp_lsit=new_list,
        )

        with multiprocessing.Pool(16) as p:
            sequence_infos = list(
                tqdm(p.imap(process_single_sequence, sample_sequence_file_list), total=len(sample_sequence_file_list)))
    else:

        for count in range(43):
            return_pl_frome_single_scenario(count, new_list)
            # print(count)
