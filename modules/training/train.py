"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import argparse
import os
import time
import sys

from kornia.geometry.camera.pinhole import homography_i_H_ref


def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=170_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader
import torch


def compute_homography(params):
    """
    将 7 个参数转换为单应性矩阵。
    输入：
        params (torch.Tensor): 参数，形状 [batch_size, 7]
                              [tx, ty, theta, sx, sy, px, py]
    输出：
        hom (torch.Tensor): 单应性矩阵，形状 [batch_size, 3, 3]
    """
    batch_size = params.shape[0]
    device = params.device

    # 提取参数并调整范围
    tx, ty = params[:, 0], params[:, 1]  # 平移：[-8, 8]，适应 8x8 patch
    theta = params[:, 2]  # 旋转：[-pi, pi]
    sx, sy = params[:, 3], params[:, 4]  # 缩放：[0, 2]
    px, py = params[:, 5], params[:, 6]  # 透视：[-0.1, 0.1]

    # 初始化单位矩阵
    hom = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    # 透视矩阵 P
    persp = torch.zeros_like(hom)
    persp[:, 0, 0] = 1
    persp[:, 1, 1] = 1
    persp[:, 2, 0] = px
    persp[:, 2, 1] = py
    persp[:, 2, 2] = 1

    # 缩放矩阵 S
    scale = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    scale[:, 0, 0] = sx
    scale[:, 1, 1] = sy

    # 旋转矩阵 R
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    rot[:, 0, 0] = cos_theta
    rot[:, 0, 1] = -sin_theta
    rot[:, 1, 0] = sin_theta
    rot[:, 1, 1] = cos_theta

    # 平移矩阵 T
    trans = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    trans[:, 0, 2] = tx
    trans[:, 1, 2] = ty

    # 组合：H = T * R * S * P
    hom = torch.bmm(scale, persp)  # 先应用透视和缩放
    hom = torch.bmm(rot, hom)  # 再应用旋转
    hom = torch.bmm(trans, hom)  # 最后应用平移

    return hom



# def compute_homography(src_corners, tgt_corners):
#     """
#     使用直接线性变换（DLT）从角点计算单应性矩阵 (批量计算)。
#
#     :param src_corners: 源角点，形状为 [batch_size, 4, 2]
#     :param tgt_corners: 目标角点，形状为 [batch_size, 4, 2]
#     :return: 单应性矩阵，形状为 [batch_size, 3, 3]
#     """
#     batch_size = src_corners.shape[0]
#
#     # 展开角点信息，方便批量计算
#     src_x = src_corners[:, :, 0]
#     src_y = src_corners[:, :, 1]
#     tgt_x = tgt_corners[:, :, 0]
#     tgt_y = tgt_corners[:, :, 1]
#
#     # 构造矩阵 A，形状为 [batch_size, 8, 8]
#     A = torch.zeros(batch_size, 8, 8, device=src_corners.device)
#
#     # 填充 A 矩阵
#     A[:, 0::2, 0] = src_x
#     A[:, 0::2, 1] = src_y
#     A[:, 0::2, 2] = 1
#     A[:, 1::2, 3] = src_x
#     A[:, 1::2, 4] = src_y
#     A[:, 1::2, 5] = 1
#     A[:, 0::2, 6] = -tgt_x * src_x
#     A[:, 0::2, 7] = -tgt_x * src_y
#     A[:, 1::2, 6] = -tgt_y * src_x
#     A[:, 1::2, 7] = -tgt_y * src_y
#
#     # 执行 SVD 求解最小奇异值对应的特征向量
#     _, _, V = torch.linalg.svd(A, full_matrices=False)
#
#     # 获取最后一列特征向量，它就是 h
#     h = V[:, -1]
#
#     # 拼接 1，使得 h 变为 9 维，并重塑为 [batch_size, 3, 3] 的形状
#     h = torch.cat([h, torch.ones(batch_size, 1, device=h.device)], dim=1).view(batch_size, 3, 3)
#
#     return h


class Trainer():
    """
        Class for training XFeat with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_default',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5,
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)

        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader = DataLoader(data, 
                                          batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name


    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MD batch
                            d = next(self.data_iter)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                #Join megadepth & synthetic data
                with torch.inference_mode():
                    #RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    #Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                #Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                #Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)
                # feats1,hmap1 = self.net(p1)
                # feats2,hmap2 = self.net(p2)

                # 假设 p1 和 p2 是形状为 [B, C, H, W] 的张量
                # H, W = p1.shape[2], p1.shape[3]

                loss_items = []
                # print("positives_c:",positives_c.shape)
                #列表，batchsize，[N,4,4]张量
                for b in range(len(positives_c)):
                    #Get positive correspondencies
                    # pts11, pts22 = positives_c[b][:,:,:2], positives_c[b][:,:,2:]   #[M, 4, 2]
                    # pts1 = pts11.view(-1, 2)  # 形状是 [M * 4, 2][13280,2]
                    # pts2 = pts22.view(-1, 2)  # 形状是 [M * 4, 2][13280,2]

                    # print("pts1:",pts1.shape)
                    # print("pts2:", pts2.shape)
                    # 假设每个网格的四个角点顺序是：[左上, 右上, 左下, 右下]
                    # tgt_points = pts22.reshape(positives_c[b].shape[0],-1)
                    # src_points = pts11.reshape(positives_c[b].shape[0],-1)
                    # print("tgt_points:", tgt_points.shape)
                    # print("src_points:", src_points.shape)   # 形状是 [M,8]

                    # pts1 = src_points[:, :2]  # 提取前两个维度
                    # pts2 = tgt_points[:, :2]  # 提取前两个维度
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]
                    # gt_offsets=tgt_points-src_points
                    # gt_offsets[:, 0::2] = (gt_offsets[:, 0::2] / W) * 2 - 1  # dx 归一化到 [-1, 1]
                    #
                    # # 对于所有的dy，gt_offsets[:, 1::2] 选择的是每个patch的 dy1, dy2, dy3, dy4
                    # gt_offsets[:, 1::2] = (gt_offsets[:, 1::2] / H) * 2 - 1  # dy 归一化到 [-1, 1]

                    # print("gt_offsets:", gt_offsets.shape)

                    #Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    # #grab heatmaps at corresponding idxs
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    # coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    H_pred = self.net.geometry_estimator(m1, m2)
                    # rotation, s, Q, p_t = self.net.geometry_estimator(m1, m2)
                    # H_pred=compute_homography(H_pred)
                    #Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    # loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    # 损失计算
                    # loss_corner = corner_loss(corner_offsets, gt_offsets)  # 角点偏移损失
                    loss_reprojection = reprojection_loss(H_pred, pts2, pts1,conf)  # 重投影误差
                    homography_loss = loss_reprojection

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(homography_loss.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = homography_loss.item()
                # loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')

                # pbar.set_description(
                #     'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_kp: {:.3f} #matches_c: {:d}'.format(
                #         loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_l1, nb_coarse))

                # pbar.set_description(
                #     'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d}'.format(
                #         loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_coord,loss_l1, nb_coarse))
                # pbar.set_description(
                #     'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_f: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                #         loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_coord, loss_l1, nb_coarse,
                #         loss_kp_pos, acc_pos))

                pbar.set_description(
                    'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f} acc_kp_pos: {:.3f}'.format(
                        loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_l1, nb_coarse,
                        loss_kp_pos, acc_pos))

                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                # self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                # self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)


if __name__ == '__main__':

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path, 
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )

    #The most fun part
    trainer.train()
