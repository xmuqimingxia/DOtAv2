# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE
        self.point_cloud_range = args['lidar_range']
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def pair_wise(self, foregound_feature, background_feature):
        '''
        loss_1 : positive-positive
        loss_2 : positive-negative
        '''
        if len(foregound_feature) == 0 or len(background_feature) == 0:
            return 0.1

        foregound_feature = torch.cat(foregound_feature, dim=0).reshape(-1, 384)
        background_feature = torch.cat(background_feature, dim=0).reshape(-1, 384)
        
        # if positive_feature.shape[0] > negative_feature.shape[0]:
        #     positive_feature = positive_feature[:negative_feature.shape[0], :]

        # positive_feature = F.normalize(positive_feature, dim=1, p=2)
        # negative_feature = F.normalize(negative_feature, dim=1, p=2)

        feat_foregound_postive = foregound_feature.repeat(foregound_feature.shape[0]-1, 1)
        feat_foregound_negtive = torch.tensor([]).cuda()
        feat_foregound_negtive_sub = foregound_feature.clone()
        for i in range(foregound_feature.shape[0]-1):
            feat_foregound_negtive_sub = torch.roll(feat_foregound_negtive_sub, 1, 0)
            feat_foregound_negtive = torch.cat((feat_foregound_negtive, feat_foregound_negtive_sub), 0)

        num_of_foregound = feat_foregound_postive.shape[0]
        if num_of_foregound == 0:
            return 0.1

        feat_background_postive = background_feature.repeat(background_feature.shape[0]-1, 1)
        feat_background_negtive = torch.tensor([]).cuda()
        feat_background_negtive_sub = background_feature.clone()
        for i in range(background_feature.shape[0]-1):
            feat_background_negtive_sub = torch.roll(feat_background_negtive_sub, 1, 0)
            feat_background_negtive = torch.cat((feat_background_negtive, feat_background_negtive_sub), 0)

        num_of_background = feat_background_postive.shape[0]
        if num_of_background == 0:
            return 0.1

        dims_min = feat_foregound_postive.shape[0] if feat_foregound_postive.shape[0] < feat_background_postive.shape[0] else feat_background_postive.shape[0]

        # feat_foregound_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_foregound_postive.shape[0]))(feat_foregound_postive), 0).reshape(1, -1)
        # feat_background_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_background_postive.shape[0]))(feat_background_postive), 0).reshape(1, -1)

        # revert_feat_foregound_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_foregound_negtive.shape[0]))(feat_foregound_negtive), 0).reshape(1, -1)
        # revert_feat_background_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_background_negtive.shape[0]))(feat_background_negtive), 0).reshape(1, -1)

        feat_foregound_flatten = feat_foregound_postive[:dims_min, :]
        feat_background_flatten = feat_background_postive[:dims_min, :]

        revert_feat_foregound_flatten = feat_foregound_negtive[:dims_min, :]
        revert_feat_background_flatten = feat_background_negtive[:dims_min, :]

        q = torch.cat((feat_foregound_flatten, feat_background_flatten), 0)
        k = torch.cat((revert_feat_foregound_flatten, revert_feat_background_flatten), 0)

        

        # logits = torch.mm(positive_feature, negative_feature.transpose(0, 1)) / 0.07

        # # # out = logits.squeeze().contiguous()
        # out = logits.contiguous()

        # n = out.size(0) #if out.size(0) < out.size(1) else out.size(1)

        # labels = torch.arange(n).cuda().long() #.to(logits.device)

        # criterion = nn.CrossEntropyLoss().cuda()
        # loss = criterion(out, labels)
        n = q.size(0)

        logits = torch.mm(q, k.transpose(1, 0))


        logits = logits/ 0.07
        labels = torch.arange(n).cuda().long()
        out = logits.squeeze().contiguous()
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(out, labels)


        return loss


    def create_bev_mask(self, bev_map, box):
        """
        输入:
        - bev_map: tensor, 尺寸为 (C, H, W)
        - x, y, z: 坐标框的中心点 (以BEV的中心为原点)
        - h, w, l: 坐标框的高度、宽度、长度
        - yaw: 坐标框的旋转角度（绕z轴）

        输出:
        - mask: tensor, 与bev_map形状相同的掩膜，框内为1，框外为0
        """

        bounding = self.point_cloud_range
        
        begin_w = bounding[3] - bounding[0]

        begin_h = bounding[4] - bounding[1]

        x, y, z, h, w, l, yaw = box[:7]

        _, H, W = bev_map.shape

        bev_map_ = F.normalize(bev_map, dim=0, p=2)

        # 将坐标框中心转换为BEV地图的索引
        center_x = ((x-bounding[0]) / begin_w) * W
        center_y = ((y-bounding[1]) / begin_h) * H


        # # 计算框的半径
        # half_w = (w / begin_w) * W / 2
        # half_l = (l / begin_h) * H / 2

        # # 生成初始掩膜
        # mask = torch.zeros_like(bev_map)

        # # 计算框的范围
        # x_min = max(center_x - half_w, 0)
        # x_max = min(center_x + half_w, W)
        # y_min = max(center_y - half_l, 0)
        # y_max = min(center_y + half_l, H)

        # # 旋转框（如果需要）
        # # 这里只处理矩形的旋转，如果需要更精确的旋转，请考虑使用仿射变换
        # if yaw != 0:
        #     # 创建一个旋转矩阵
        #     rot_matrix = torch.tensor([
        #         [torch.cos(yaw), -torch.sin(yaw)],
        #         [torch.sin(yaw), torch.cos(yaw)]
        #     ])

        #     # 计算框的四个角点
        #     corners = torch.tensor([
        #         [x_min, y_min],
        #         [x_max, y_min],
        #         [x_max, y_max],
        #         [x_min, y_max]
        #     ]) - torch.tensor([center_x, center_y])

        #     rotated_corners = torch.matmul(rot_matrix, corners.T).T + torch.tensor([center_x, center_y])

        #     # 创建旋转后的框
        #     x_min_rot, y_min_rot = torch.min(rotated_corners, dim=0)[0]
        #     x_max_rot, y_max_rot = torch.max(rotated_corners, dim=0)[0]

        #     # 填充掩膜
        #     mask[:, int(y_min_rot):int(y_max_rot), int(x_min_rot):int(x_max_rot)] = 1
        # else:
        #     # 如果不旋转，直接填充掩膜
        #     mask[:, int(y_min):int(y_max), int(x_min):int(x_max)] = 1

        # instance_feature = bev_map * mask
        # count_totle = 

        # return torch.sum(instance_feature, dim=(1, 2))



        return bev_map_[:, int(center_y), int(center_x)]



    def forward(self, data_dict):

        if data_dict['iterative_training']:
            object_bbx_center_noise = data_dict['object_bbx_center_noise']
            object_bbx_center = data_dict['object_bbx_center']
            object_bbx_mask = data_dict['object_bbx_mask']
            pure_mask = object_bbx_mask == 1
        else:
            object_bbx_center = data_dict['object_bbx_center']

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']


        if data_dict['iterative_training']:
            batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'object_bbx_center_noise': object_bbx_center_noise}
        else:
            batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']


        if data_dict['iterative_training']:
            out_contrast_loss = 0
            for ba in range(object_bbx_center_noise.shape[0]): # batch_sized的数量
                positive_feature_list = []
                object_bbx_center_pure = object_bbx_center[ba, :, :][pure_mask[ba, :]]
                # print('********************', object_bbx_center_pure.shape)
                for num_gt in range(object_bbx_center_pure.shape[0]): # gt的数量
                    # if object_bbx_center[ba, num_gt, 0] != 0:
                    positive_feature = self.create_bev_mask(spatial_features_2d[ba, :, :, :], object_bbx_center_pure[num_gt, :])
                    positive_feature_list.append(positive_feature.reshape(1, -1))
                    
                negative_feature_list = []
                for num_noise in range(min(object_bbx_center_noise.shape[1], 100)): # gt的数量
                    if  0.8 > object_bbx_center_noise[ba, num_noise, 7] > 0.2: 
                        negative_feature = self.create_bev_mask(spatial_features_2d[ba, :, :, :], object_bbx_center_noise[ba, num_noise, :])
                        negative_feature_list.append(negative_feature.reshape(1, -1))
                
                out_contrast_loss = self.pair_wise(positive_feature_list, negative_feature_list) + out_contrast_loss

            out_contrast_loss_ = (out_contrast_loss / object_bbx_center_noise.shape[0]) * 0.1
        else:
            out_contrast_loss_ = 0


        # out_contrast_loss_ = 0.
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        
        output_dict = {'psm': psm,
                        'rm': rm,
                        'out_contrast_loss': out_contrast_loss_}

        return output_dict