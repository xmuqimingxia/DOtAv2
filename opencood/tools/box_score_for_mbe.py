import numpy as np
import torch
import copy
from scipy.spatial import ConvexHull


def get_transform(boxs: torch.Tensor) -> torch.Tensor:
    # Extract box center coordinates, dimensions and rotation angle
    # print(boxs.shape)
    boxs = boxs.reshape(1, 7)
    center_xyz = boxs[:, :3]
    dimensions = boxs[:, 3:6]
    rotation_xy = boxs[:, 6]

    # Compute rotation matrix around the z-axis
    cos, sin = torch.cos(rotation_xy), torch.sin(rotation_xy)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rotation_z = torch.stack([cos, -sin, zero, zero,
                              sin, cos, zero, zero,
                              zero, zero, one, zero,
                              zero, zero, zero, one], dim=-1).view(-1, 4, 4)

    # Compute translation matrix to move the center of the box to the origin
    translation = torch.eye(4).to(boxs.device).unsqueeze(0).repeat(center_xyz.shape[0], 1, 1)
    translation[:, 3, :3] = -center_xyz
    return torch.matmul(translation.float(), rotation_z.float())

def get_scale(
    ptc: torch.Tensor,  # (N_points, 4)
    boxs: torch.Tensor  # (M, 7)
) -> torch.Tensor:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc.float(), trs.float())[:, :, :3]  # [M, N_points, 3]
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc / (boxs[:, 3:6].unsqueeze(dim=1) * 0.5)
    scale = torch.max(scale, dim=2).values
    return scale

def get_distance_score(
    ptc: torch.Tensor,  # (N_points, 4)
    boxs: torch.Tensor  # (M, 7)
) -> torch.Tensor:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc.float(), trs.float())[:, :, :3]  # [M, N_points, 3]
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc[:, :, :2] / (boxs[:, 3:5].unsqueeze(dim=1) * 0.5)
    scale = torch.max(scale, dim=2).values
    return scale

def KL_entropy_score(x, y, max_dif = 0.05):
    KL = 0.0
    for i in range(len(x)):
        KL += x[i] * np.log(x[i] / y[i])

    if KL>max_dif:
        KL = max_dif
    return (max_dif-KL)/max_dif

def compute_confidence(points, box, parts=6):
    x, y, z, l, w, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    cloud = np.zeros(shape=(points.shape[0], 4))
    cloud[:, 0:3] = points[:, 0:3]
    cloud[:, 3] = 1

    trans_mat = np.eye(4, dtype=np.float32)
    trans_mat[0, 0] = np.cos(yaw)
    trans_mat[0, 1] = -np.sin(yaw)
    trans_mat[0, 3] = x
    trans_mat[1, 0] = np.sin(yaw)
    trans_mat[1, 1] = np.cos(yaw)
    trans_mat[1, 3] = y
    trans_mat[2, 3] = z

    trans_mat_i = np.linalg.inv(trans_mat)
    cloud = np.matmul(cloud, trans_mat_i.T)

    delta_l = l/parts
    delta_w = w/parts

    valid_vol = 0

    for i in range(parts):
        for j in range(parts):
            mask_x_l = -l/2+i*delta_l<=cloud[:, 0]
            mask_x_r = cloud[:,0]<-l/2+(i+1)*delta_l
            mask_y_l = -w/2+j*delta_w<=cloud[:, 1]
            mask_y_r = cloud[:, 1]<-w/2+(j+1)*delta_w

            mask = mask_x_l*mask_x_r*mask_y_l*mask_y_r

            this_pts = cloud[mask]

            if len(this_pts)>1:
                valid_vol+=1

    return valid_vol/(parts**2)
def hierarchical_occupancy_score(points, box, parts=[7,5,3]):
    all_confi = 0
    for part in parts:
        all_confi+=compute_confidence(points,box,part)
    return all_confi/len(parts)

def compute_css(points, box):

    predefined_size = [4.7, 2.0, 1.5]  # (3)

    dis_dis = np.linalg.norm(box[0:3])
    if dis_dis > 80:
        dis_dis = 80
    dis_score = 1 - dis_dis / 80
    mlo_parts = [9, 7, 5]
    mlo_score = hierarchical_occupancy_score(points, box, mlo_parts)


    new_box = copy.deepcopy(box)
    this_size_norm = new_box[3:6] / new_box[3:6].sum()
    this_temp_norm = np.array(predefined_size)
    this_temp_norm = this_temp_norm / this_temp_norm.sum()
    size_score = KL_entropy_score(this_size_norm, this_temp_norm)
    if new_box[3] < predefined_size[0]+0.5:
        new_s = np.linalg.norm(new_box[3:6] - np.array(predefined_size), axis=0)
        if new_s>3:
            new_s=3
        new_s = 1 - new_s/3

        size_score = (size_score + new_s)/2
    weights_ = np.array([1, 1, 1])

    weights = np.array(weights_) / np.sum(weights_)
    # print('mlo_score', dis_score, mlo_score, size_score)

    final_score = dis_score * weights[0] + mlo_score * weights[1] + size_score * weights[2]
    print('******************',mlo_score)
    # final_score = size_score


    return final_score

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

import math

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
        cosa,  sina, zeros,
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
    if inter_points[key] < inter_points_threshold:
        return 0
    threshold = 0.5 * len(inter_points)
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
            #new_boxes.append(pseduo_labels[j])
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

    for j in range(num_box):
        center_annotion = pseduo_labels[j, :2]
        # pose_center = ok[0, :2]
        pose_center = [0, 0, 0]
        for cav in range(len(ok)):
            pose_center = ok[cav][0, :] + pose_center
        pose_center = pose_center / len(ok)
        disance_with_ok = np.linalg.norm(center_annotion - pose_center[:2]) #距离中心点的距离

        r = 0
        for cav in range(len(ok)):
            r = np.linalg.norm(ok[cav][0, :2] - pose_center[:2]) + r
        r = r / len(ok)
        if disance_with_ok < 2 * r:
            inter_points_threshold = 200
        elif 2 * r < disance_with_ok < 2.5 * r:
            inter_points_threshold = 80
        else:
            inter_points_threshold = 5

        inter_points_number = []
        for i in range(len(multi_frame_points)):
            inter_mask = in_hull(multi_frame_points[i][:, :3], boxes_to_corners_3d(pseduo_labels[j][:7].reshape(-1, 7)).reshape(-1, 3))
            inter_points = multi_frame_points[i][inter_mask]
            inter_points_number.append(inter_points.shape[0])

        state = classify_state(inter_points_number, key, inter_points_threshold)
        # print('################', inter_points_number, state, inter_points_threshold, disance_with_ok)
        #
        # vi.add_points(multi_frame_points[key][:, :3])
        # vi.add_points(pose_center[:3].reshape(1, 3), radius=10, color='red')
        # # vi.add_3D_boxes(gt, color='green')
        # vi.add_3D_boxes(pseduo_labels[j].reshape(1, 7), color='red')
        # vi.show_3D()

        # print(inter_points_number, inter_points_number[key], len(inter_points_number), state)




        if state == 1:
            #new_boxes.append(pseduo_labels[j])
            new_boxes.append(True)
            new_boxes_support.append(False)
        else:
            new_boxes.append(False)
            if inter_points_number[key] > 0:
                new_boxes_support.append(True)
            else:
                new_boxes_support.append(False)

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

if __name__ == '__main__':

    vi = Viewer()

    path = "F:\\OPV2V\\OPV2V\\train"

    # print(os.listdir(path))
    scenario_folders = sorted([os.path.join(path, x)    # 单个元素的例：.../OPV2V/train/2021_08_16_22_26_54，为一个场景
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
                    sorted([os.path.join(cav_path, x)   # 例：将...\OPV2V\train\2021_08_16_22_26_54\641下'000069.yaml'这样的文件路径升序排序
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
            break
        timestamps = []
        for file in yaml_files:
            res = file.split(os.path.sep)[-1]
            timestamp = res.replace('.yaml', '')    # 如'000069.yaml'变成'000069'
            timestamps.append(timestamp)
        node_timestamp = node_timestamp + len(timestamps)

        multi_agent_point = np.load(f'F:\\OPV2V\\OPV2V\\multi_agent_point_remove_ground\\multi_agent_point{count}.npy', allow_pickle=True)
        poses = np.load(f'F:\\OPV2V\\OPV2V\\multi_agent_point_pose\\multi_agent_point_pose{count}.npy', allow_pickle=True)

        for num_timestamp in tqdm(range(node_timestamp-len(timestamps), node_timestamp)):
            # if count < 2:
            #     continue
            gt = np.load(f'F:\\OPV2V\\OPV2V\\gt_box\\gt_{num_timestamp}.npy')
            # pseduo_labels = gt.copy()

            gtbox_center = gt[:, :3].copy()
            gtbox_center_new = pc_2_world(gtbox_center, poses[0][num_timestamp - node_timestamp + len(timestamps)])
            gtdif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - node_timestamp + len(timestamps)]))
            gt[:, :3] = gtbox_center_new[:, :3]
            gt[:, 6] = gt[:, 6] + gtdif_ang


            # np.save(f'F:\\OPV2V\\OPV2V\\out_v4\\out_pseduo_labels_v4_{num_timestamp}.npy', pseduo_labels_[out_pseduo_labels])
            # np.save(f'F:\\OPV2V\\OPV2V\\out_v4\\out_pseduo_labels_noise_v4_{num_timestamp}.npy', pseduo_labels_[inverted_list])
#################################################################################
            #     # clear_points = remove_ground_points(multi_agent_point[m][key][:, :3])
            #     # vi.add_points(clear_points)
            #     vi.add_points(multi_agent_point[m][key][:, :3])
            #     # vi.add_3D_boxes(pseduo_labels, color='black')
            #     vi.add_points(ok_0, radius=10, color='red')
            #     vi.add_3D_boxes(gt, color='green')
            #     vi.add_3D_boxes(pseduo_labels[out_pseduo_labels], color='red')
            #     vi.show_3D()
            if num_timestamp - (node_timestamp-len(timestamps)) > 25:
                a = num_timestamp -25
            else:
                a = node_timestamp-len(timestamps)

            if node_timestamp - num_timestamp > 25:
                b = num_timestamp + 25
            else:
                b = node_timestamp

            key = num_timestamp - a #表达当前帧在传入序列中的相对位置

            dense_points_multi_frame = []
            for frame in range(a-node_timestamp+len(timestamps), b-node_timestamp+len(timestamps)):
                dense_points = 0
                for m in range(len(cav_list)):
                    if m == 0:
                        dense_points = multi_agent_point[m][frame]
                    else:
                        dense_points = np.concatenate((dense_points, multi_agent_point[m][frame]), 0)
                dense_points_multi_frame.append(dense_points)


            pseduo_labels = np.load(f'C:\\Users\\Administrator\\Desktop\\新建文件夹 (6)\\新建文件夹\\out_xqm\\out_pseduo_labels_v1_{num_timestamp}.npy')
            pseduo_labels_error = np.load(f'C:\\Users\\Administrator\\Desktop\\新建文件夹 (6)\\新建文件夹\\out_xqm\\out_pseduo_labels_noise_v1_{num_timestamp}.npy')
            pseduo_labels_ = pseduo_labels.copy()
            pseduo_labels_error_ = pseduo_labels_error.copy()

            box_center = pseduo_labels[:, :3].copy()
            box_center_new = pc_2_world(box_center, poses[0][num_timestamp - node_timestamp + len(timestamps)])
            dif_ang = get_registration_angle(x_to_world(poses[0][num_timestamp - node_timestamp + len(timestamps)]))
            pseduo_labels[:, :3] = box_center_new[:, :3]
            pseduo_labels[:, 6] = pseduo_labels[:, 6] + dif_ang

            box_center_error = pseduo_labels_error[:, :3].copy()
            box_center_error_new = pc_2_world(box_center_error, poses[0][num_timestamp - node_timestamp + len(timestamps)])
            dif_ang_error = get_registration_angle(x_to_world(poses[0][num_timestamp - node_timestamp + len(timestamps)]))
            pseduo_labels_error[:, :3] = box_center_error_new[:, :3]
            pseduo_labels_error[:, 6] = pseduo_labels_error[:, 6] + dif_ang_error


            # for m in range(len(cav_list)):
            #     vi.add_points(np.array(poses[m][num_timestamp - node_timestamp + len(timestamps)])[:3].reshape(1, 3), radius=10, color='red')
                # vi.add_3D_boxes(pseduo_labels, color='black')
                # vi.add_points(ok_0, radius=10, color='red')

            total_score = []
            for i in range(pseduo_labels.shape[0]):
                inter_mask = in_hull(dense_points_multi_frame[key][:, :3],
                                     boxes_to_corners_3d(pseduo_labels[i][:7].reshape(-1, 7)).reshape(-1, 3))
                forground_instance_points = dense_points_multi_frame[key][:, :3][inter_mask]
                pl = np.array(pseduo_labels[i]).reshape(1, 7)

                points = forground_instance_points[:, [0, 1]]
                # new_points = np.array([points[:, 1], points[:, 0]]).T
                corner_points = points[ConvexHull(points).vertices]
                three_D_corner_points = np.zeros((corner_points.shape[0], 3))
                three_D_corner_points[:, :2] = corner_points

                distance_score = get_distance_score(torch.tensor(three_D_corner_points), torch.tensor(pl))
                final_score = distance_score.mean()
                total_score.append(final_score)

            total_score = np.array(total_score)
            pseduo_labels_with_score = np.hstack((pseduo_labels_, total_score.reshape(-1, 1)))
            np.save(f'F:\\OPV2V\\OPV2V\\out_xqm_score\\out_pseduo_labels_with_score_v4_{num_timestamp}.npy', pseduo_labels_with_score)


            total_score_error = []
            for i in range(pseduo_labels_error.shape[0]):
                inter_mask = in_hull(dense_points_multi_frame[key][:, :3],
                                     boxes_to_corners_3d(pseduo_labels_error[i][:7].reshape(-1, 7)).reshape(-1, 3))
                forground_instance_points = dense_points_multi_frame[key][:, :3][inter_mask]
                pl = np.array(pseduo_labels_error[i]).reshape(1, 7)

                points = forground_instance_points[:, [0, 1]]

                # new_points = np.array([points[:, 1], points[:, 0]]).T
                try:
                    corner_points = points[ConvexHull(points).vertices]
                except:
                    corner_points = points
                three_D_corner_points = np.zeros((corner_points.shape[0], 3))
                three_D_corner_points[:, :2] = corner_points

                distance_score = get_distance_score(torch.tensor(three_D_corner_points), torch.tensor(pl))
                final_score = distance_score.mean()
                total_score_error.append(final_score)

            total_score_error = np.array(total_score_error)
            pseduo_labels_noise_with_score = np.hstack((pseduo_labels_error_, total_score_error.reshape(-1, 1)))
            np.save(f'F:\\OPV2V\\OPV2V\\out_xqm_score\\out_pseduo_labels_noise_with_score_v4_{num_timestamp}.npy', pseduo_labels_noise_with_score)

        count += 1


