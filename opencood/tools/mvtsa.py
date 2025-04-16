import time
# from Viewer.viewer.viewer import Viewer
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
import pandas as pd

import math


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


def classify_state(inter_points, key, inter_points_threshold):
    if not inter_points:  # 检查inter_points是否为空
        return 0  # 如果为空，返回状态0
    num_zeros = np.sum(np.array(inter_points) == 0)
    # print(inter_points, num_zeros, len(inter_points))
    if num_zeros == len(inter_points):
        return 0
    if inter_points[key] < 5:
        return 0
    if max(inter_points) < inter_points_threshold:
        return 0
    if max_consecutive_zeros(inter_points) / len(inter_points) < 0.3:
        return 0
    # threshold = 0.5 * len(inter_points)
    return 1 if num_zeros > 1 else 0


def box_filter(pseduo_labels, multi_frame_points, key, ok):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []
    new_boxes_support = []
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

        if inter_points[key] > 5:
            new_boxes_support.append(True)
        else:
            new_boxes_support.append(False)

        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
        else:
            new_boxes.append(False)

    return new_boxes, new_boxes_support


def box_filter_v2(pseduo_labels, multi_frame_points, key, ok):
    if pseduo_labels.ndim != 2 or pseduo_labels.shape[1] < 2:
        raise ValueError("pseduo_labels must be a 2D array with at least 2 columns")

    num_box = pseduo_labels.shape[0]
    new_boxes = []
    new_boxes_support = []
    # multi_frame_points_remove_ground = []
    # for f in range(len(multi_frame_points)):
    #     multi_frame_points_remove_ground.append(remove_ground_points(multi_frame_points[f]))
    # data = []
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
        # # data.append([state, inter_points_threshold, inter_points_number])
        # # # #
        # vi.add_points(multi_frame_points[key][:, :3])
        # vi.add_points(pose_center[:3].reshape(1, 3), radius=10, color='red')
        # vi.add_3D_boxes(gt, color='green')
        # vi.add_3D_boxes(pseduo_labels[j].reshape(1, 7), color='red')
        # vi.show_3D()

        # print(inter_points_number, inter_points_number[key], len(inter_points_number), state)

        if state == 1:
            # new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
            new_boxes_support.append(False)
        else:
            new_boxes.append(False)
            if inter_points_number[key] > 0:
                new_boxes_support.append(True)
            else:
                new_boxes_support.append(False)

    # df = pd.DataFrame(data)
    # print(data)
    # df.to_csv('output.csv', index=False)
    # exit()
    return new_boxes, new_boxes_support


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


# 计算IOU
import numpy as np
import os
from scipy.spatial import ConvexHull


def box_to_corners(center, dimensions, yaw):
    """
    将框的中心坐标、尺寸和yaw角度转换为8个角点的坐标
    """
    x, y, z = center
    l, w, h = dimensions
    yaw_rad = np.radians(yaw)

    # 计算角点
    corners = np.array([
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2]
    ])

    # 旋转矩阵
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    rotated_corners = np.dot(corners, R.T)
    rotated_corners += np.array(center)

    return rotated_corners


def compute_iou_3d(box1, box2):
    """
    计算两个三维框的 IoU (Intersection over Union)。
    """
    center1, dimensions1, yaw1 = box1[:3], box1[3:6], box1[6]
    center2, dimensions2, yaw2 = box2[:3], box2[3:6], box2[6]

    # 获取每个框的角点
    corners1 = box_to_corners(center1, dimensions1, yaw1)
    corners2 = box_to_corners(center2, dimensions2, yaw2)

    # 计算交集体积的简化方法
    def intersect_volume(corners1, corners2):
        """
        使用包围盒的简单方法计算体积交集
        """
        def overlap(a, b):
            """计算两个区间的重叠长度"""
            return max(0, min(a[1], b[1]) - max(a[0], b[0]))

        # 计算包围盒的边界
        def bounding_box(corners):
            min_corner = np.min(corners, axis=0)
            max_corner = np.max(corners, axis=0)
            return min_corner, max_corner

        min1, max1 = bounding_box(corners1)
        min2, max2 = bounding_box(corners2)

        # 计算交集体积
        dx = overlap([min1[0], max1[0]], [min2[0], max2[0]])
        dy = overlap([min1[1], max1[1]], [min2[1], max2[1]])
        dz = overlap([min1[2], max1[2]], [min2[2], max2[2]])

        return dx * dy * dz

    volume1 = np.prod(dimensions1)
    volume2 = np.prod(dimensions2)
    inter_volume = intersect_volume(corners1, corners2)
    union_volume = volume1 + volume2 - inter_volume

    return inter_volume / union_volume if union_volume > 0 else 0


def compute_precision_recall(gt_folder, pred_folder, iou_threshold=0.1):
    """
    计算文件夹中所有三维框的 Precision 和 Recall
    gt_folder: Ground Truth 框文件夹
    pred_folder: 预测框文件夹
    iou_threshold: IoU 阈值，决定 True Positive 的标准
    """
    tp, fp, fn = 0, 0, 0  # 初始化 True Positive, False Positive, False Negative

    # 获取文件夹中所有的文件名
    gt_files = sorted(os.listdir(gt_folder))
    pred_files_ = os.listdir(pred_folder)
    pred_files = [s for s in pred_files_ if 'noise' not in s]
    pred_files = sorted(pred_files)
    # 确保文件数量一致
    assert len(gt_files) == len(pred_files), "Ground Truth 和预测框文件数量不一致！"

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files)):
        # 加载 npy 文件中的 3D 框
        gt_boxes = np.load(os.path.join(gt_folder, gt_file))  # N x 7 的数组
        pred_boxes = np.load(os.path.join(pred_folder, pred_file))  # M x 7 的数组
        # print(gt_boxes.shape, pred_boxes.shape)
        # 标记哪些 Ground Truth 框已经匹配上
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        # 遍历预测框，计算 IoU 并统计 True Positive, False Positive
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            # 找到 IoU 最大的 Ground Truth 框
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                current_iou = compute_iou_3d(pred_box, gt_box)
                # print(current_iou)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = i

            # 根据 IoU 阈值判断是 TP 还是 FP
            # print(best_iou)
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt[best_gt_idx] = True
            else:
                fp += 1

        # 统计 False Negative
        fn += len(gt_boxes) - np.sum(matched_gt)

    # 计算 Precision 和 Recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall


if __name__ == '__main__':

    # 文件夹路径
    gt_folder = '/mnt/32THHD/lwk/datas/OPV2V/gt_box'  # Ground Truth 框文件夹路径
    pred_folder = '/mnt/32THHD/lwk/datas/OPV2V/out_xqm'  # 预测框文件夹路径
    
    # 计算 Precision 和 Recall
    precision, recall = compute_precision_recall(gt_folder, pred_folder, iou_threshold=0.5)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    exit()

    # vi = Viewer()

    path = "E:\\OPV2V\\train"

    # print(os.listdir(path))
    scenario_folders = sorted([os.path.join(path, x)  # 单个元素的例：.../OPV2V/train/2021_08_16_22_26_54，为一个场景
                               for x in os.listdir(path) if
                               os.path.isdir(os.path.join(path, x))])
    count = 0
    node_timestamp = 0
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
        node_timestamp = node_timestamp + len(timestamps)

        ##################################################################################
        # multi_agent_point = []
        # poses = []
        # for cav_id in cav_list:
        #     # if count == 0:
        #     #     continue
        #     cav_path = os.path.join(scenario_folder, cav_id)
        #     single_agent_point = []
        #     pose = []
        #     for timestamp in timestamps:
        #         pcd_file_name = timestamp + ".pcd"
        #         point_path = os.path.join(cav_path, pcd_file_name)
        #         points_local = pcd_to_np(point_path)
        #
        #         yaml_file_name = timestamp + ".yaml"
        #         yaml_path = os.path.join(cav_path, yaml_file_name)
        #         lidar_pose = load_yaml(yaml_path)['lidar_pose']
        #
        #         points_gloabal = pc_2_world(points_local, lidar_pose)
        #         points_gloabal_ = remove_ground_points(points_gloabal)
        #         single_agent_point.append(points_gloabal_)
        #         pose.append(lidar_pose)
        #     poses.append(pose)
        #
        #     # single_agent_points = np.concatenate(single_agent_point, axis=0)
        #     multi_agent_point.append(single_agent_point)
        #
        # np.save(f'F:\\OPV2V\\OPV2V\\multi_agent_point_remove_ground\\multi_agent_point{count}.npy', multi_agent_point)
        # np.save(f'F:\\OPV2V\\OPV2V\\multi_agent_point_pose\\multi_agent_point_pose{count}.npy', poses)
        ##################################################################################
        multi_agent_point = np.load(f'E:\\OPV2V\\multi_agent_point_remove_ground\\multi_agent_point{count}.npy',
                                    allow_pickle=True)
        poses = np.load(f'E:\\OPV2V\\multi_agent_point_pose\\multi_agent_point_pose{count}.npy', allow_pickle=True)
        table_sum = np.zeros((9, 22), float)
        time_num = 0
        for num_timestamp in tqdm(range(node_timestamp - len(timestamps), node_timestamp)):
            # if count < 2:
            #     continue
            # pseduo_labels = np.load(f'E:\\OPV2V\\pre_box\\pre_{num_timestamp}.npy')
            pseduo_labels = np.load(f'E:\\OPV2V\\out_v6\\out_pseduo_labels_v6_{num_timestamp}.npy')
            gt = np.load(f'E:\\OPV2V\\gt_box\\gt_{num_timestamp}.npy')  # n*7 [x,y,z,l,w,h,yaw]
            # exit()
            # pseduo_labels = gt.copy()
            pseduo_labels_ = pseduo_labels.copy()

            gtbox_center = gt[:, :3].copy()
            gtbox_center_new = pc_2_world(gtbox_center, poses[0][num_timestamp - node_timestamp + len(timestamps)])
            gtdif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - node_timestamp + len(timestamps)]))
            gt[:, :3] = gtbox_center_new[:, :3]
            gt[:, 6] = gt[:, 6] + gtdif_ang

            box_center = pseduo_labels[:, :3].copy()
            box_center_new = pc_2_world(box_center, poses[0][num_timestamp - node_timestamp + len(timestamps)])
            dif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - node_timestamp + len(timestamps)]))
            pseduo_labels[:, :3] = box_center_new[:, :3]
            pseduo_labels[:, 6] = pseduo_labels[:, 6] + dif_ang

            if num_timestamp - (node_timestamp - len(timestamps)) > 25:
                a = num_timestamp - 25
            else:
                a = node_timestamp - len(timestamps)

            if node_timestamp - num_timestamp > 25:
                b = num_timestamp + 25
            else:
                b = node_timestamp

            key = num_timestamp - a  # 表达当前帧在传入序列中的相对位置

            # ok_0 = np.array(poses[0][num_timestamp - node_timestamp + len(timestamps)])[:3].reshape(1, 3)  # 自车位置
            ok = []
            mask = [False] * pseduo_labels.shape[0]
            mask_support = [False] * pseduo_labels.shape[0]
            dense_points_multi_frame = []
            for frame in range(a - node_timestamp + len(timestamps), b - node_timestamp + len(timestamps)):
                dense_points = 0
                for m in range(len(cav_list)):
                    if m == 0:
                        dense_points = multi_agent_point[m][frame]
                    else:
                        dense_points = np.concatenate((dense_points, multi_agent_point[m][frame]), 0)
                dense_points_multi_frame.append(dense_points)

            for m in range(len(cav_list)):
                ok.append(np.array(poses[m][num_timestamp - node_timestamp + len(timestamps)])[:3].reshape(1, 3))

            out_pseduo_labels, out_pseduo_labels_support = box_filter_v2(pseduo_labels, dense_points_multi_frame, key,
                                                                         ok)

            # inverted_list = [not x for x in out_pseduo_labels]

            np.save(f'E:\\OPV2V\\out_v7\\out_pseduo_labels_v7_{num_timestamp}.npy',
                    pseduo_labels_[out_pseduo_labels])

            # # 匹配gt和预测框
            # # 加载npy文件
            # gt_boxes = np.load('gt_boxes.npy')  # Ground Truth 框
            # pred_boxes = np.load('pred_boxes.npy')  # 预测框
            #
            # # 计算precision和recall
            # precision, recall = compute_precision_recall(gt_boxes, pred_boxes)
            #
            # print(f"Precision: {precision:.4f}")
            # print(f"Recall: {recall:.4f}")

            # np.save(f'E:\\OPV2V\\out_v4\\out_pseduo_labels_noise_v4_{num_timestamp}.npy',
            #         pseduo_labels_[inverted_list])
            # np.save(f'F:\\OPV2V\\OPV2V\\out_v3\\out_pseduo_labels_v3_{num_timestamp}.npy', pseduo_labels_[mask])
            #################################################################################
            #     # clear_points = remove_ground_points(multi_agent_point[m][key][:, :3])
            #     # vi.add_points(clear_points)
            #     vi.add_points(multi_agent_point[m][key][:, :3])
            #     # vi.add_3D_boxes(pseduo_labels, color='black')
            #     vi.add_points(ok_0, radius=10, color='red')
            #     vi.add_3D_boxes(gt, color='green')
            #     vi.add_3D_boxes(pseduo_labels[out_pseduo_labels], color='red')
            #     vi.show_3D()
            #
            # for m in range(len(cav_list)):
            #     vi.add_points(np.array(poses[m][num_timestamp - node_timestamp + len(timestamps)])[:3].reshape(1, 3), radius=10, color='red')
            # pose_center = [0, 0, 0]
            # for cav in range(len(ok)):
            #     pose_center = ok[cav][0, :] + pose_center
            # pose_center = pose_center / len(ok)
            # vi.add_points(pose_center[:3].reshape(1, 3), radius=15, color='red')
            # vi.add_points(dense_points_multi_frame[key][:, :3])
            # vi.add_3D_boxes(gt, color='green')
            # vi.add_3D_boxes(pseduo_labels, color='black')
            # vi.add_3D_boxes(pseduo_labels[out_pseduo_labels], color='red')
            # vi.show_3D()
        print("单个场景生成伪标签")
        exit()
        count += 1
